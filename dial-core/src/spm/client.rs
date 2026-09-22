use anyhow::{anyhow, Result};
use async_trait::async_trait;
use candle_core::{Device, Tensor};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, Once, OnceLock},
    time::Instant,
};
use tokio::{net::TcpStream, sync::Mutex as AsyncMutex};

use crate::models::llama3::{Cache, Config};

use super::{CompactBatch, CompactRangeBatch, Message, SamplingRequest, SessionId, WorkerInfo};

tokio::task_local! {
    static REMOTE_SAMPLING_REQUEST: SamplingRequest;
    static REMOTE_SESSION_ID: SessionId;
    static REMOTE_PROFILE: Arc<Mutex<DistributedProfile>>;
}

pub async fn with_remote_session<F>(session_id: SessionId, future: F) -> F::Output
where F: std::future::Future {
    REMOTE_SESSION_ID.scope(session_id, future).await
}

pub async fn with_remote_profile<F>(profile: Arc<Mutex<DistributedProfile>>, future: F) -> F::Output
where F: std::future::Future {
    REMOTE_PROFILE.scope(profile, future).await
}

fn update_distributed_profile<F>(update: F)
where
    F: Fn(&mut DistributedProfile),
{
    if REMOTE_PROFILE
        .try_with(|p| {
            if let Ok(mut p) = p.lock() {
                update(&mut p);
                true
            } else {
                false
            }
        })
        .unwrap_or(false)
    {
        return;
    }
    if let Ok(mut p) = distributed_profile().lock() {
        update(&mut p);
    }
}

fn current_session_id() -> SessionId {
    REMOTE_SESSION_ID.try_with(|id| *id).unwrap_or(0)
}

pub async fn with_remote_sampling_request<F>(sampling: SamplingRequest, future: F) -> F::Output
where
    F: std::future::Future,
{
    REMOTE_SAMPLING_REQUEST.scope(sampling, future).await
}

fn current_sampling_request() -> Option<SamplingRequest> {
    REMOTE_SAMPLING_REQUEST.try_with(Clone::clone).ok()
}

#[derive(Debug, Clone, Default)]
pub struct DistributedProfile {
    pub remote_requests: usize,
    pub remote_total_s: f64,
    pub remote_compute_s: f64,
    pub remote_write_s: f64,
    pub remote_read_s: f64,
    pub remote_write_bytes: usize,
    pub remote_read_bytes: usize,
}

// 记录分布式统计信息：远程请求数、远程总耗时、远程计算耗时（秒）。
impl DistributedProfile {
    pub fn distributed_overhead_s(&self) -> f64 {
        (self.remote_total_s - self.remote_compute_s).max(0.0)
    }
}
//
fn distributed_profile() -> &'static Mutex<DistributedProfile> {
    static PROFILE: OnceLock<Mutex<DistributedProfile>> = OnceLock::new();
    PROFILE.get_or_init(|| Mutex::new(DistributedProfile::default()))
}

pub fn reset_distributed_profile() {
    if let Ok(mut profile) = distributed_profile().lock() {
        *profile = DistributedProfile::default();
    }
}

pub fn snapshot_distributed_profile() -> DistributedProfile {
    distributed_profile()
        .lock()
        .map(|profile| profile.clone())
        .unwrap_or_default()
}

/// The transport state of one persistent Master -> Worker connection.
#[derive(Debug)]
struct ClientConnection {
    stream: TcpStream,
    request_seq: u64,
}

/// A layer-specific view over one persistent worker connection.
///
/// The model keeps one `Forwarder` per layer, while all layers assigned to the
/// same worker address share the TCP stream and request sequence.
#[derive(Debug, Clone)]
pub struct Client {
    device: Device,
    address: String,
    layer_name: String,
    info: Arc<WorkerInfo>,
    compact_batch: bool,
    compact_range_batch: bool,
    remote_sampling: bool,
    connection: Arc<AsyncMutex<ClientConnection>>,
    /// Extra persistent lanes used by concurrent inference sessions. A session is
    /// deterministically mapped to one lane, so odd and even concurrency counts work alike.
    session_connections: Arc<AsyncMutex<HashMap<SessionId, Arc<AsyncMutex<ClientConnection>>>>>,
}

/// Reuses one persistent connection for every layer assigned to a worker
/// address. A model owns one pool, so separate Master processes still get
/// independent connections and KV-cache sessions on the Worker.
#[derive(Debug, Default)]
pub struct ClientPool {
    clients_by_address: HashMap<String, Client>,
}

/// Opens a short-lived connection and returns the Worker's handshake metadata.
/// This is used by the Master's read-only topology status endpoint and does not
/// allocate model layers or reuse the inference connection pool.
pub async fn probe_worker(address: &str) -> Result<WorkerInfo> {
    let mut stream = TcpStream::connect(address)
        .await
        .map_err(|e| anyhow!("can't connect to {address}: {e}"))?;
    if let Err(e) = stream.set_nodelay(true) {
        log::warn!(
            "failed to set TCP_NODELAY for topology probe {}: {}",
            address,
            e
        );
    }

    Message::Hello.to_writer(&mut stream).await?;
    let (_, response) = Message::from_reader(&mut stream).await?;
    match response {
        Message::WorkerInfo(info) => Ok(info),
        other => Err(anyhow!(
            "unexpected topology probe response from {address}: {:?}",
            other
        )),
    }
}

impl ClientPool {
    pub async fn client_for_layer(
        &mut self,
        device: Device,
        address: &str,
        layer_name: &str,
    ) -> Result<Client> {
        if let Some(client) = self.clients_by_address.get(address) {
            return Ok(client.for_layer(layer_name));
        }

        let client = Client::new(device, address, layer_name).await?;
        self.clients_by_address
            .insert(address.to_string(), client.clone());
        Ok(client)
    }
}

impl Client {
    fn compact_batch_enabled() -> bool {
        matches!(
            std::env::var("SPM_COMPACT_BATCH").ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
        )
    }

    fn log_compact_batch_mode(enabled: bool, range_enabled: bool) {
        static LOGGED: Once = Once::new();
        LOGGED.call_once(|| {
            if range_enabled {
                log::info!("spm compact range batch enabled by SPM_COMPACT_BATCH");
            } else if enabled {
                log::info!("spm compact batch enabled by SPM_COMPACT_BATCH");
            } else {
                log::info!("spm compact batch disabled; set SPM_COMPACT_BATCH=1 to test it");
            }
        });
    }

    fn max_session_lanes() -> usize {
        std::env::var("DIAL_MAX_SESSION_LANES").ok().and_then(|v| v.parse().ok()).filter(|v: &usize| *v > 0).unwrap_or(64)
    }

    fn transfer_trace_enabled() -> bool {
        matches!(
            std::env::var("SPM_TRACE_TRANSFER").ok().as_deref(),
            Some("1") | Some("true") | Some("TRUE") | Some("yes") | Some("YES")
        )
    }

    fn message_summary(message: &Message) -> String {
        match message {
            Message::Hello => "hello".to_string(),
            Message::WorkerInfo(_) => "worker_info".to_string(),
            Message::ReleaseSession { session_id } => format!("release_session session={}", session_id),
            Message::SingleOp {
                session_id,
                layer_name,
                x,
                index_pos,
                block_idx,
                sampling,
            } => format!(
                "single session={} layer={} index_pos={} block_idx={} sampling={} shape={:?} dtype={:?}",
                session_id, layer_name, index_pos, block_idx, sampling.is_some(), x.shape, x.dtype
            ),
            Message::Batch { session_id, x, batch, sampling } => {
                let first = batch
                    .first()
                    .map(|(name, _, _)| name.as_str())
                    .unwrap_or("-");
                let last = batch
                    .last()
                    .map(|(name, _, _)| name.as_str())
                    .unwrap_or("-");
                format!(
                    "batch session={} ops={} first={} last={} sampling={} shape={:?} dtype={:?}",
                    session_id, batch.len(),
                    first,
                    last,
                    sampling.is_some(),
                    x.shape,
                    x.dtype
                )
            }
            Message::CompactBatch { session_id, x, batch, sampling } => format!(
                "compact_batch session={} ops={} first={} index_pos={} block_idx={} sampling={} shape={:?} dtype={:?}",
                session_id, batch.num_layers,
                batch.first_layer_name,
                batch.index_pos,
                batch.first_block_idx,
                sampling.is_some(),
                x.shape,
                x.dtype
            ),
            Message::CompactRangeBatch { session_id, x, batch, sampling } => format!(
                "compact_range_batch session={} ops={} index_pos={} block_idx={} sampling={} shape={:?} dtype={:?}",
                session_id, batch.num_layers, batch.index_pos, batch.first_block_idx, sampling.is_some(), x.shape, x.dtype
            ),
            Message::Tensor { x, compute_us } => format!(
                "tensor shape={:?} dtype={:?} compute={:.3}ms",
                x.shape,
                x.dtype,
                *compute_us as f64 / 1000.0
            ),
            Message::Error { message } => format!("error {message}"),
        }
    }

    /// Connects to the given worker address.
    /// NOTE: device and layer_name here are only passed for std::fmt::Display.
    pub async fn new(device: Device, address: &str, layer_name: &str) -> Result<Self> {
        let address = address.to_string();
        let layer_name = layer_name.to_string();
        let stream = TcpStream::connect(&address)
            .await
            .map_err(|e| anyhow!("can't connect to {address}: {e}"))?;
        if let Err(e) = stream.set_nodelay(true) {
            log::warn!("failed to set TCP_NODELAY for {}: {}", address, e);
        }
        let mut client = Self {
            address,
            device,
            layer_name,
            info: Arc::new(WorkerInfo::default()),
            compact_batch: false,
            compact_range_batch: false,
            remote_sampling: false,
            connection: Arc::new(AsyncMutex::new(ClientConnection {
                stream,
                request_seq: 0,
            })),
            session_connections: Arc::new(AsyncMutex::new(HashMap::new())),
        };

        let resp = client.request(Message::Hello).await?;
        client.info = if let Message::WorkerInfo(info) = resp {
            Arc::new(info)
        } else {
            return Err(anyhow!("unexpected worker info message: {:?}", &resp));
        };
        let compact_allowed = Self::compact_batch_enabled();
        client.compact_range_batch =
            compact_allowed && client.info.supports("compact_range_batch_v1");
        client.compact_batch = compact_allowed
            && (client.compact_range_batch || client.info.supports("compact_batch_v1"));
        Self::log_compact_batch_mode(client.compact_batch, client.compact_range_batch);
        client.remote_sampling = client.info.supports("gguf_sample_token_v1");
        if client.remote_sampling {
            log::info!(
                "remote GGUF sampling capability enabled for {}",
                client.address
            );
        }

        Ok(client)
    }

    /// Create a layer view which reuses this client's persistent connection.
    pub fn for_layer(&self, layer_name: &str) -> Self {
        let mut view = self.clone();
        view.layer_name = layer_name.to_string();
        view
    }


    async fn connection_for_session(&self, session_id: SessionId) -> Result<Arc<AsyncMutex<ClientConnection>>> {
        if session_id == 0 {
            return Ok(self.connection.clone());
        }
        {
            let lanes = self.session_connections.lock().await;
            if let Some(connection) = lanes.get(&session_id) {
                return Ok(connection.clone());
            }
        }
        let mut stream = TcpStream::connect(&self.address)
            .await
            .map_err(|e| anyhow!("can't open inference lane to {}: {e}", self.address))?;
        if let Err(e) = stream.set_nodelay(true) {
            log::warn!("failed to set TCP_NODELAY for inference lane {}: {}", self.address, e);
        }
        Message::Hello.to_writer(&mut stream).await?;
        let (_, response) = Message::from_reader(&mut stream).await?;
        if !matches!(response, Message::WorkerInfo(_)) {
            return Err(anyhow!("unexpected inference-lane handshake from {}: {:?}", self.address, response));
        }
        let connection = Arc::new(AsyncMutex::new(ClientConnection { stream, request_seq: 0 }));
        let mut lanes = self.session_connections.lock().await;
        if let Some(existing) = lanes.get(&session_id) {
            return Ok(existing.clone());
        }
        if lanes.len() >= Self::max_session_lanes() {
            return Err(anyhow!(
                "concurrent session lane limit reached for {}: {} active lanes (DIAL_MAX_SESSION_LANES={})",
                self.address,
                lanes.len(),
                Self::max_session_lanes()
            ));
        }
        lanes.insert(session_id, connection.clone());
        Ok(connection)
    }

    pub async fn release_session(&self, session_id: SessionId) -> Result<()> {
        if session_id == 0 { return Ok(()); }
        let lane = { self.session_connections.lock().await.remove(&session_id) };
        let Some(lane) = lane else { return Ok(()); };
        let mut connection = lane.lock().await;
        let release = async {
            Message::ReleaseSession { session_id }
                .to_writer(&mut connection.stream)
                .await
                .map_err(|e| anyhow!("failed to send release_session {} to {}: {}", session_id, self.address, e))?;
            let (_, ack) = Message::from_reader(&mut connection.stream)
                .await
                .map_err(|e| anyhow!("failed to receive release_session {} ack from {}: {}", session_id, self.address, e))?;
            match ack {
                Message::ReleaseSession { session_id: ack_id } if ack_id == session_id => Ok(()),
                other => Err(anyhow!("unexpected release_session response from {}: {:?}", self.address, other)),
            }
        };
        tokio::time::timeout(std::time::Duration::from_secs(2), release)
            .await
            .map_err(|_| anyhow!("release_session {} to {} timed out", session_id, self.address))??;
        Ok(())
    }

    /// Send a Message to the worker and return a response.
    async fn request(&self, req: Message) -> Result<Message> {
        let session_id = match &req {
            Message::SingleOp { session_id, .. }
            | Message::Batch { session_id, .. }
            | Message::CompactBatch { session_id, .. }
            | Message::CompactRangeBatch { session_id, .. } => *session_id,
            _ => 0,
        };
        let lane = self.connection_for_session(session_id).await?;
        let mut connection = lane.lock().await;
        connection.request_seq += 1;
        let req_id = connection.request_seq;
        let req_summary = Self::message_summary(&req);
        let total_start = Instant::now();

        let write_start = Instant::now();
        let written = req
            .to_writer(&mut connection.stream)
            .await
            .map_err(|e| anyhow!("error sending {}: {}", req_summary, e))?;
        let write_time = write_start.elapsed();

        let read_start = Instant::now();
        let (read_size, msg) = super::Message::from_reader(&mut connection.stream)
            .await
            .map_err(|e| anyhow!("error receiving response for {}: {}", req_summary, e))?;
        let read_time = read_start.elapsed();
        let total_time = total_start.elapsed();

        if let Message::Tensor { compute_us, .. } = &msg {
            let compute_s = *compute_us as f64 / 1_000_000.0;
            update_distributed_profile(|profile| {
                profile.remote_requests += 1;
                profile.remote_total_s += total_time.as_secs_f64();
                profile.remote_compute_s += compute_s;
                profile.remote_write_s += write_time.as_secs_f64();
                profile.remote_read_s += read_time.as_secs_f64();
                profile.remote_write_bytes += written;
                profile.remote_read_bytes += read_size;
            });
        }

        if Self::transfer_trace_enabled() {
            log::info!(
                "[transfer][client {}#{}] {} -> write={}B/{:.3}ms read={}B/{:.3}ms total={:.3}ms resp={}",
                self.address,
                req_id,
                req_summary,
                written,
                write_time.as_secs_f64() * 1000.0,
                read_size,
                read_time.as_secs_f64() * 1000.0,
                total_time.as_secs_f64() * 1000.0,
                Self::message_summary(&msg)
            );
        }
        Ok(msg)
    }

    async fn forward_request(&self, req: Message) -> Result<Tensor> {
        let resp = self.request(req).await?;
        match resp {
            Message::Tensor { x, .. } => Ok(x.to_tensor(&self.device)?),
            Message::Error { message } => Err(anyhow!("worker error: {message}")),
            _ => Err(anyhow!("unexpected response {:?}", &resp)),
        }
    }
}

impl std::fmt::Display for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}@{} [{}<{}> {}-{} latency={}ms]",
            &self.layer_name,
            &self.address,
            &self.info.device,
            &self.info.device_idx,
            &self.info.os,
            &self.info.arch,
            self.info.latency
        )
    }
}

#[async_trait]
impl super::Forwarder for Client {
    fn load(_: String, _: candle_nn::VarBuilder, _: &Config) -> Result<Box<Self>> {
        Err(anyhow!("load should never be called on spm::Client"))
    }

    async fn forward(&self, _: &Tensor, _: usize, _: usize, _: &mut Cache) -> Result<Tensor> {
        Err(anyhow!(
            "immutable forward should never be called on spm::Client"
        ))
    }

    /// Executes the worker's pipeline for this tensor.
    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        _: &mut Cache,
    ) -> Result<Tensor> {
        self.forward_request(super::Message::single_op(
            current_session_id(),
            &self.layer_name,
            x,
            index_pos,
            block_idx,
            self.remote_sampling
                .then(current_sampling_request)
                .flatten(),
        ))
        .await
    }

    async fn forward_batch_shared(
        &self,
        x: &Tensor,
        batch: Vec<(String, usize, usize)>,
        _: &mut Cache,
    ) -> Result<Tensor> {
        let sampling = self.remote_sampling.then(current_sampling_request).flatten();
        if self.compact_batch {
            if let Some((first_layer_name, index_pos, first_block_idx)) = batch.first().cloned() {
                let contiguous = batch.iter().enumerate().all(|(off, (_, idx, block))| *idx == index_pos && *block == first_block_idx + off);
                if contiguous {
                    if self.compact_range_batch {
                        return self.forward_request(super::Message::from_compact_range_batch(current_session_id(), x, CompactRangeBatch { index_pos, first_block_idx, num_layers: batch.len() }, sampling)).await;
                    }
                    return self.forward_request(super::Message::from_compact_batch(current_session_id(), x, CompactBatch { first_layer_name, index_pos, first_block_idx, num_layers: batch.len() }, sampling)).await;
                }
            }
        }
        self.forward_request(super::Message::from_batch(current_session_id(), x, batch, sampling)).await
    }

    /// 一次性发送多个层的批量计算任务，给远端服务器发一批算子一起执行，减少网络往返。
    async fn forward_batch(
        &mut self,
        x: &Tensor,                         //输入张量
        batch: Vec<(String, usize, usize)>, //批量算子信息：层名、位置、块号
        _: &mut Cache,
    ) -> Result<Tensor> {
        let sampling = self
            .remote_sampling
            .then(current_sampling_request)
            .flatten();
        if self.compact_batch {
            let compact_meta =
                batch
                    .first()
                    .and_then(|(first_layer_name, index_pos, first_block_idx)| {
                        let same_index = batch.iter().all(|(_, idx, _)| idx == index_pos);
                        let contiguous_blocks =
                            batch.iter().enumerate().all(|(offset, (_, _, block_idx))| {
                                *block_idx == first_block_idx + offset
                            });
                        if same_index && contiguous_blocks {
                            Some((
                                first_layer_name.clone(),
                                *index_pos,
                                *first_block_idx,
                                batch.len(),
                            ))
                        } else {
                            None
                        }
                    });
            if let Some((first_layer_name, index_pos, first_block_idx, num_layers)) = compact_meta {
                if self.compact_range_batch {
                    return self
                        .forward_request(super::Message::from_compact_range_batch(
                            current_session_id(),
                            x,
                            CompactRangeBatch {
                                index_pos,
                                first_block_idx,
                                num_layers,
                            },
                            sampling,
                        ))
                        .await;
                }
                return self
                    .forward_request(super::Message::from_compact_batch(
                        current_session_id(),
                        x,
                        CompactBatch {
                            first_layer_name,
                            index_pos,
                            first_block_idx,
                            num_layers,
                        },
                        sampling,
                    ))
                    .await;
            }
        }
        // 打包成Batch信息->发送远程请求->返回结果
        self.forward_request(super::Message::from_batch(current_session_id(), x, batch, sampling))
            .await
    }

    async fn release_remote_session(&self, session_id: SessionId) -> Result<()> {
        Client::release_session(self, session_id).await
    }

    fn requires_remote_sampling(&self) -> bool {
        self.remote_sampling
    }

    // 返回客户端的唯一身份标识=服务器地址
    fn ident(&self) -> &str {
        &self.address
    }
    // 返回整个客户端负责的模型层名字
    fn layer_name(&self) -> &str {
        &self.layer_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::{
        net::TcpListener,
        time::{timeout, Duration},
    };

    #[tokio::test]
    async fn client_pool_reuses_one_connection_per_worker_address() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let accepted = Arc::new(AtomicUsize::new(0));
        let accepted_by_server = accepted.clone();

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            accepted_by_server.fetch_add(1, Ordering::SeqCst);

            let (_, hello) = Message::from_reader(&mut socket).await.unwrap();
            assert!(matches!(hello, Message::Hello));
            Message::WorkerInfo(WorkerInfo::default())
                .to_writer(&mut socket)
                .await
                .unwrap();

            assert!(timeout(Duration::from_millis(100), listener.accept())
                .await
                .is_err());
        });

        let mut pool = ClientPool::default();
        let layer0 = pool
            .client_for_layer(Device::Cpu, &address, "model.layers.0")
            .await
            .unwrap();
        let layer1 = pool
            .client_for_layer(Device::Cpu, &address, "model.layers.1")
            .await
            .unwrap();

        assert!(Arc::ptr_eq(&layer0.connection, &layer1.connection));
        assert_eq!(layer0.layer_name, "model.layers.0");
        assert_eq!(layer1.layer_name, "model.layers.1");

        server.await.unwrap();
        assert_eq!(accepted.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn release_session_closes_and_removes_dedicated_lane() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap().to_string();

        let server = tokio::spawn(async move {
            let (mut base, _) = listener.accept().await.unwrap();
            let (_, hello) = Message::from_reader(&mut base).await.unwrap();
            assert!(matches!(hello, Message::Hello));
            Message::WorkerInfo(WorkerInfo::default())
                .to_writer(&mut base)
                .await
                .unwrap();

            let (mut lane, _) = listener.accept().await.unwrap();
            let (_, hello) = Message::from_reader(&mut lane).await.unwrap();
            assert!(matches!(hello, Message::Hello));
            Message::WorkerInfo(WorkerInfo::default())
                .to_writer(&mut lane)
                .await
                .unwrap();

            let (_, release) = Message::from_reader(&mut lane).await.unwrap();
            assert!(matches!(release, Message::ReleaseSession { session_id: 42 }));
            Message::ReleaseSession { session_id: 42 }
                .to_writer(&mut lane)
                .await
                .unwrap();
        });

        let client = Client::new(Device::Cpu, &address, "model.layers.0")
            .await
            .unwrap();
        let _lane = client.connection_for_session(42).await.unwrap();
        assert!(client.session_connections.lock().await.contains_key(&42));

        client.release_session(42).await.unwrap();
        assert!(!client.session_connections.lock().await.contains_key(&42));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn probe_worker_returns_handshake_info() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap().to_string();

        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let (_, hello) = Message::from_reader(&mut socket).await.unwrap();
            assert!(matches!(hello, Message::Hello));
            Message::WorkerInfo(WorkerInfo {
                version: "test-v1".to_string(),
                dtype: "F16".to_string(),
                os: "linux".to_string(),
                arch: "aarch64".to_string(),
                device: "NPU".to_string(),
                device_idx: 2,
                latency: 7,
                cpu_usage_percent: Some(25.0),
                memory_usage_percent: Some(50.0),
                npu_usage_percent: Some(75.0),
                temperature_c: Some(48.0),
                system_model: Some("test-board".to_string()),
            })
            .to_writer(&mut socket)
            .await
            .unwrap();
        });

        let info = probe_worker(&address).await.unwrap();
        assert_eq!(info.version, "test-v1");
        assert_eq!(info.device, "NPU");
        assert_eq!(info.device_idx, 2);
        assert_eq!(info.latency, 7);
        server.await.unwrap();
    }
}

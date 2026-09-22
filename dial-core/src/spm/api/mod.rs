use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use actix_web::web;
use actix_web::App;
use actix_web::HttpRequest;
use actix_web::HttpResponse;
use actix_web::HttpServer;
use actix_web::Responder;
use serde::Deserialize;
use serde::Serialize;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

use crate::models::chat::Message;
use crate::models::Generator;

use super::worker::collect_worker_metrics;
use super::{configured_pipeline_depth, probe_worker, EngineEvent, EngineRequest, Master, PipelineEngine, Topology, Context};

#[derive(Deserialize)]
struct Request {
    pub messages: Vec<Message>,
    /// OpenAI-style streaming responses over SSE.
    #[serde(default)]
    pub stream: bool,
}

#[derive(Serialize)]
struct Choice {
    pub index: usize,
    pub message: Message,
}

#[derive(Serialize)]
struct Response {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    /// （新增）首 token 延迟（TTFT, time-to-first-token），单位秒。
    /// 说明：包含多模态图片编码、prefill，以及生成出第一个 token 的总耗时。
    pub ttft_s: Option<f64>,
    /// （新增）本次请求总耗时，单位秒（从开始生成到结束）。
    pub total_s: f64,
    /// （新增）平均生成速率（tokens/s），使用生成的 token 数 / total_s 计算。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_second: Option<f64>,
    /// （新增）解码阶段平均生成速率（tokens/s），使用 (生成token数-1) / (total_s-ttft_s) 计算。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_tokens_per_second: Option<f64>,
    /// Number of tokens emitted before EOS or the configured sample limit.
    pub generated_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distributed_overhead_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote_compute_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote_requests: Option<usize>,
}

#[derive(Serialize)]
struct StreamDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Serialize)]
struct StreamChoice {
    pub index: usize,
    pub delta: StreamDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Serialize)]
struct StreamResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<StreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_tokens_per_second: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distributed_overhead_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote_compute_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote_requests: Option<usize>,
}

#[derive(Serialize)]
struct WorkerStatus {
    role: String,
    name: String,
    host: String,
    description: Option<String>,
    layers: Vec<String>,
    active: bool,
    online: bool,
    state: String,
    version: Option<String>,
    dtype: Option<String>,
    os: Option<String>,
    arch: Option<String>,
    device: Option<String>,
    device_idx: Option<usize>,
    latency_ms: Option<u64>,
    worker_latency_ms: Option<u64>,
    cpu_usage_percent: Option<f32>,
    memory_usage_percent: Option<f32>,
    npu_usage_percent: Option<f32>,
    temperature_c: Option<f32>,
    system_model: Option<String>,
    error: Option<String>,
}

#[derive(Serialize)]
struct MasterStatus {
    role: String,
    name: String,
    host: String,
    description: String,
    active: bool,
    online: bool,
    state: String,
    version: String,
    dtype: String,
    os: String,
    arch: String,
    device: String,
    device_idx: usize,
    cpu_usage_percent: Option<f32>,
    memory_usage_percent: Option<f32>,
    npu_usage_percent: Option<f32>,
    temperature_c: Option<f32>,
    system_model: Option<String>,
}

#[derive(Serialize)]
struct TopologyStatus {
    master_api: String,
    master: MasterStatus,
    topology_path: String,
    configured_workers: usize,
    active_workers: usize,
    online_workers: usize,
    reload_required: bool,
    workers: Vec<WorkerStatus>,
}

async fn topology(state: web::Data<Context>) -> impl Responder {
    let (
        master_api,
        topology_path,
        active_topology,
        master_dtype,
        master_device,
        master_device_idx,
    ) = {
        (
            state.args.api.clone().unwrap_or_default(),
            state.args.topology.clone(),
            state.topology.clone(),
            format!("{:?}", state.dtype),
            if state.device.is_cuda() {
                "cuda".to_string()
            } else if state.device.is_metal() {
                "metal".to_string()
            } else {
                "cpu".to_string()
            },
            state.args.device,
        )
    };
    let master_metrics = collect_worker_metrics(master_device_idx);
    let master_status = MasterStatus {
        role: "master".to_string(),
        name: "Master".to_string(),
        host: master_api.clone(),
        description: "DIAL 调度与本地推理节点".to_string(),
        active: true,
        online: true,
        state: "online".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        dtype: master_dtype,
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        device: master_device,
        device_idx: master_device_idx,
        cpu_usage_percent: master_metrics.cpu_usage_percent,
        memory_usage_percent: master_metrics.memory_usage_percent,
        npu_usage_percent: master_metrics.npu_usage_percent,
        temperature_c: master_metrics.temperature_c,
        system_model: master_metrics.system_model,
    };

    // Reload the configured topology for observability. Newly configured
    // workers appear immediately, while `active` tells callers whether the
    // running Master already loaded the same worker at startup.
    let configured_topology = match Topology::from_path_silent(&topology_path) {
        Ok(topology) => topology,
        Err(error) => {
            return HttpResponse::InternalServerError().json(serde_json::json!({
                "error": format!("failed to load topology: {error}"),
                "topology_path": topology_path,
            }));
        }
    };

    let mut configured = configured_topology
        .iter()
        .map(|(name, node)| (name.clone(), node.clone()))
        .collect::<Vec<_>>();
    configured.sort_by(|left, right| left.0.cmp(&right.0));

    let mut probes = Vec::with_capacity(configured.len());
    for (name, node) in configured {
        let active = active_topology
            .get(&name)
            .map(|active_node| active_node.host == node.host && active_node.layers == node.layers)
            .unwrap_or(false);
        probes.push(tokio::spawn(async move {
            let started = Instant::now();
            let probe =
                tokio::time::timeout(Duration::from_millis(900), probe_worker(&node.host)).await;
            let latency_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;

            match probe {
                Ok(Ok(info)) => WorkerStatus {
                    role: "worker".to_string(),
                    name,
                    host: node.host,
                    description: node.description,
                    layers: node.layers,
                    active,
                    online: true,
                    state: if active {
                        "online".to_string()
                    } else {
                        "standby".to_string()
                    },
                    version: Some(info.version),
                    dtype: Some(info.dtype),
                    os: Some(info.os),
                    arch: Some(info.arch),
                    device: Some(info.device),
                    device_idx: Some(info.device_idx),
                    latency_ms: Some(latency_ms),
                    worker_latency_ms: Some(info.latency.min(u64::MAX as u128) as u64),
                    cpu_usage_percent: info.cpu_usage_percent,
                    memory_usage_percent: info.memory_usage_percent,
                    npu_usage_percent: info.npu_usage_percent,
                    temperature_c: info.temperature_c,
                    system_model: info.system_model,
                    error: None,
                },
                Ok(Err(error)) => WorkerStatus {
                    role: "worker".to_string(),
                    name,
                    host: node.host,
                    description: node.description,
                    layers: node.layers,
                    active,
                    online: false,
                    state: "offline".to_string(),
                    version: None,
                    dtype: None,
                    os: None,
                    arch: None,
                    device: None,
                    device_idx: None,
                    latency_ms: None,
                    worker_latency_ms: None,
                    cpu_usage_percent: None,
                    memory_usage_percent: None,
                    npu_usage_percent: None,
                    temperature_c: None,
                    system_model: None,
                    error: Some(error.to_string()),
                },
                Err(_) => WorkerStatus {
                    role: "worker".to_string(),
                    name,
                    host: node.host,
                    description: node.description,
                    layers: node.layers,
                    active,
                    online: false,
                    state: "offline".to_string(),
                    version: None,
                    dtype: None,
                    os: None,
                    arch: None,
                    device: None,
                    device_idx: None,
                    latency_ms: None,
                    worker_latency_ms: None,
                    cpu_usage_percent: None,
                    memory_usage_percent: None,
                    npu_usage_percent: None,
                    temperature_c: None,
                    system_model: None,
                    error: Some("worker probe timed out after 900 ms".to_string()),
                },
            }
        }));
    }

    let mut workers = Vec::with_capacity(probes.len());
    for probe in probes {
        match probe.await {
            Ok(status) => workers.push(status),
            Err(error) => log::warn!("topology probe task failed: {}", error),
        }
    }

    let active_workers = workers.iter().filter(|worker| worker.active).count();
    let online_workers = workers.iter().filter(|worker| worker.online).count();
    let reload_required = workers.iter().any(|worker| !worker.active);
    HttpResponse::Ok().json(TopologyStatus {
        master_api,
        master: master_status,
        topology_path,
        configured_workers: workers.len(),
        active_workers,
        online_workers,
        reload_required,
        workers,
    })
}

impl Response {
    pub fn from_assistant_response(
        model: String,
        message: String,
        ttft_s: Option<f64>,
        total_s: f64,
        tokens_per_second: Option<f64>,
        decode_tokens_per_second: Option<f64>,
        generated_tokens: usize,
        distributed_overhead_s: Option<f64>,
        remote_compute_s: Option<f64>,
        remote_requests: Option<usize>,
    ) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        let object = String::from("chat.completion");
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let choices = vec![Choice {
            index: 0,
            message: Message::assistant(message),
        }];

        Self {
            id,
            object,
            created,
            model,
            choices,
            ttft_s,
            total_s,
            tokens_per_second,
            decode_tokens_per_second,
            generated_tokens,
            distributed_overhead_s,
            remote_compute_s,
            remote_requests,
        }
    }
}

async fn chat<G>(
    engine: web::Data<mpsc::UnboundedSender<EngineRequest>>,
    _req: HttpRequest,
    body: web::Json<Request>,
) -> impl Responder
where
    G: Generator + Send + Sync + 'static,
{
    let Request { messages, stream } = body.into_inner();
    let (event_tx, mut event_rx) = mpsc::unbounded_channel();
    let (accepted_tx, accepted_rx) = oneshot::channel();
    if engine.send(EngineRequest { messages, events: event_tx, accepted: accepted_tx }).is_err() {
        return HttpResponse::ServiceUnavailable().body("pipeline engine is not running");
    }
    let session_id = match accepted_rx.await {
        Ok(id) => id,
        Err(_) => return HttpResponse::InternalServerError().body("pipeline admission failed"),
    };

    if !stream {
        let start = Instant::now();
        let mut text = String::new();
        let mut ttft_s = None;
        let mut generated = 0usize;
        while let Some(event) = event_rx.recv().await {
            match event {
                EngineEvent::Token(token) => { if ttft_s.is_none() { ttft_s = Some(start.elapsed().as_secs_f64()); } generated += 1; text.push_str(&token); }
                EngineEvent::Finished { generated_tokens, elapsed_s, profile } => {
                    let total_s = elapsed_s;
                    let tps = if total_s > 0.0 { Some(generated_tokens as f64 / total_s) } else { None };
                    let decode_tps = ttft_s.and_then(|ttft| { let dt = total_s - ttft; if generated_tokens > 1 && dt > 0.0 { Some((generated_tokens - 1) as f64 / dt) } else { None } });
                    return HttpResponse::Ok().json(Response::from_assistant_response(G::MODEL_NAME.to_string(), text, ttft_s, total_s, tps, decode_tps, generated, Some(profile.distributed_overhead_s()), Some(profile.remote_compute_s), Some(profile.remote_requests)));
                }
                EngineEvent::Error(e) => return HttpResponse::InternalServerError().body(e),
            }
        }
        return HttpResponse::InternalServerError().body(format!("pipeline session {session_id} closed unexpectedly"));
    }

    let (tx, rx) = mpsc::unbounded_channel::<String>();
    let model = G::MODEL_NAME.to_string();
    tokio::spawn(async move {
        let id = format!("dial-{session_id}");
        let created = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        while let Some(event) = event_rx.recv().await {
            match event {
                EngineEvent::Token(token) => {
                    let chunk = StreamResponse { id: id.clone(), object: "chat.completion.chunk".into(), created, model: model.clone(), choices: vec![StreamChoice { index: 0, delta: StreamDelta { content: Some(token) }, finish_reason: None }], ttft_s: None, total_s: None, tokens_per_second: None, decode_tokens_per_second: None, generated_tokens: None, distributed_overhead_s: None, remote_compute_s: None, remote_requests: None };
                    if let Ok(j) = serde_json::to_string(&chunk) { let _ = tx.send(format!("data: {j}\\n\\n")); }
                }
                EngineEvent::Finished { generated_tokens, elapsed_s, profile } => {
                    let final_chunk = StreamResponse { id: id.clone(), object: "chat.completion.chunk".into(), created, model: model.clone(), choices: vec![StreamChoice { index: 0, delta: StreamDelta { content: None }, finish_reason: Some("stop".into()) }], ttft_s: None, total_s: Some(elapsed_s), tokens_per_second: if elapsed_s > 0.0 { Some(generated_tokens as f64 / elapsed_s) } else { None }, decode_tokens_per_second: None, generated_tokens: Some(generated_tokens), distributed_overhead_s: Some(profile.distributed_overhead_s()), remote_compute_s: Some(profile.remote_compute_s), remote_requests: Some(profile.remote_requests) };
                    if let Ok(j) = serde_json::to_string(&final_chunk) { let _ = tx.send(format!("data: {j}\\n\\n")); }
                    let _ = tx.send("data: [DONE]\\n\\n".into()); break;
                }
                EngineEvent::Error(e) => { let _ = tx.send(format!("data: {{\\"error\\":{}}}\\n\\n", serde_json::to_string(&e).unwrap_or_else(|_| "\\"generation error\\"".into()))); let _ = tx.send("data: [DONE]\\n\\n".into()); break; }
            }
        }
    });
    let body = UnboundedReceiverStream::new(rx).map(|s| Ok::<web::Bytes, actix_web::Error>(web::Bytes::from(s)));
    HttpResponse::Ok().insert_header(("Content-Type", "text/event-stream")).insert_header(("Cache-Control", "no-cache")).insert_header(("X-DIAL-Session", session_id.to_string())).streaming(body)
}

async fn not_found() -> actix_web::Result<HttpResponse> {
    Ok(HttpResponse::NotFound().body("nope"))
}

async fn web_chat() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../web_chat/index.html"
        )))
}

async fn web_chat_styles() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/css; charset=utf-8")
        .body(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../web_chat/styles.css"
        )))
}

async fn web_chat_script() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("application/javascript; charset=utf-8")
        .body(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../web_chat/app.js"
        )))
}

pub(crate) async fn start<G>(master: Master<G>) -> anyhow::Result<()>
where
    G: Generator + Send + Sync + 'static,
{
    let address = master.ctx.args.api.as_ref().unwrap().to_string();
    // Base64 expands video bytes by roughly 4/3. Keep room for JSON, text and history.
    let json_limit = master
        .ctx
        .args
        .video_max_bytes
        .saturating_mul(4)
        .saturating_div(3)
        .saturating_add(8 * 1024 * 1024);

    log::info!("starting api on http://{} ...", &address);

    // Topology is immutable runtime metadata and does not need ownership of the model.
    let topology_snapshot = master.ctx.clone();
    let (engine_tx, engine) = PipelineEngine::channel(master, configured_pipeline_depth());
    tokio::spawn(engine.run());

    HttpServer::new(
        move || {
            App::new()
                .app_data(web::Data::new(engine_tx.clone()))
                .app_data(web::Data::new(topology_snapshot.clone()))
                .app_data(web::JsonConfig::default().limit(json_limit))
                .route("/", web::get().to(web_chat))
                .route("/chat", web::get().to(web_chat))
                .route("/web-chat", web::get().to(web_chat))
                .route("/web-chat/styles.css", web::get().to(web_chat_styles))
                .route("/web-chat/app.js", web::get().to(web_chat_script))
                .route("/api/v1/chat/completions", web::post().to(chat::<G>))
                .route("/api/v1/topology", web::get().to(topology))
                .default_service(web::route().to(not_found))
        }, //.wrap(actix_web::middleware::Logger::default()))
    )
    .bind(&address)
    .map_err(|e| anyhow!(e))?
    .run()
    .await
    .map_err(|e| anyhow!(e))
}

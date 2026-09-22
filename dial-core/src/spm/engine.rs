use std::collections::VecDeque;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;

use tokio::sync::{mpsc, oneshot, Mutex, Semaphore};
use tokio::task::JoinSet;

use crate::models::{chat::Message, Generator, PipelineStageOutput};
use super::{
    with_remote_profile, with_remote_session, DistributedProfile, Master, SessionId,
};

pub struct EngineRequest {
    pub messages: Vec<Message>,
    pub events: mpsc::UnboundedSender<EngineEvent>,
    pub accepted: oneshot::Sender<SessionId>,
}

#[derive(Debug)]
pub enum EngineEvent {
    Token(String),
    Finished {
        generated_tokens: usize,
        elapsed_s: f64,
        profile: DistributedProfile,
    },
    Error(String),
}

struct ActiveRequest {
    session_id: SessionId,
    step: usize,
    generated: usize,
    started: Instant,
    events: mpsc::UnboundedSender<EngineEvent>,
    profile: Arc<StdMutex<DistributedProfile>>,
    stage: usize,
    hidden: Option<Box<dyn std::any::Any + Send>>,
}

type StageTaskResult = (ActiveRequest, anyhow::Result<PipelineStageOutput>);

/// Multi-request pipeline scheduler.
///
/// Prefill keeps the mature monolithic path for correctness. Decode requests are
/// split into owner-contiguous transformer stages. Each stage has one execution
/// slot, while different stages may be in flight simultaneously:
/// Master(A0,B0,C0) overlaps Worker1(A1,B1,...) and Worker2(A2,...).
pub struct PipelineEngine<G> {
    master: Arc<Mutex<Master<G>>>,
    rx: mpsc::UnboundedReceiver<EngineRequest>,
    ready: VecDeque<ActiveRequest>,
    in_flight: JoinSet<StageTaskResult>,
    stage_slots: Vec<Arc<Semaphore>>,
    max_active: usize,
    sample_len: usize,
    next_session: SessionId,
}

impl<G: Generator + Send + Sync + 'static> PipelineEngine<G> {
    pub fn channel(
        master: Master<G>,
        max_active: usize,
    ) -> (mpsc::UnboundedSender<EngineRequest>, Self) {
        let (tx, rx) = mpsc::unbounded_channel();
        let sample_len = master.ctx.args.sample_len;
        let stage_count = master.pipeline_stage_count().max(1);
        let stage_slots = (0..stage_count)
            .map(|_| Arc::new(Semaphore::new(1)))
            .collect();
        (
            tx,
            Self {
                master: Arc::new(Mutex::new(master)),
                rx,
                ready: VecDeque::new(),
                in_flight: JoinSet::new(),
                stage_slots,
                max_active: max_active.max(1),
                sample_len,
                next_session: 1,
            },
        )
    }

    fn active_count(&self) -> usize {
        self.ready.len() + self.in_flight.len()
    }

    async fn admit(&mut self, request: EngineRequest) {
        let id = self.next_session;
        self.next_session = self.next_session.wrapping_add(1).max(1);
        let mut master = self.master.lock().await;
        match master.create_session(id, request.messages) {
            Ok(()) => {
                drop(master);
                let _ = request.accepted.send(id);
                self.ready.push_back(ActiveRequest {
                    session_id: id,
                    step: 0,
                    generated: 0,
                    started: Instant::now(),
                    events: request.events,
                    profile: Arc::new(StdMutex::new(DistributedProfile::default())),
                    stage: 0,
                    hidden: None,
                });
            }
            Err(e) => {
                let _ = request.events.send(EngineEvent::Error(e.to_string()));
            }
        }
    }

    async fn fill_slots(&mut self) {
        while self.active_count() < self.max_active {
            match self.rx.try_recv() {
                Ok(req) => self.admit(req).await,
                Err(_) => break,
            }
        }
    }

    fn finish_event(request: &ActiveRequest) -> EngineEvent {
        let profile = request
            .profile
            .lock()
            .map(|p| p.clone())
            .unwrap_or_default();
        EngineEvent::Finished {
            generated_tokens: request.generated,
            elapsed_s: request.started.elapsed().as_secs_f64(),
            profile,
        }
    }

    async fn release_request(&self, request: &ActiveRequest) {
        let mut master = self.master.lock().await;
        master.release_session(request.session_id).await;
    }

    async fn handle_stage_completion(&mut self, completed: StageTaskResult) {
        let (mut request, result) = completed;
        match result {
            Ok(output) => {
                let attach_result = {
                    let master = self.master.lock().await;
                    match request.hidden.as_mut() {
                        Some(hidden) => master.attach_session_stage(hidden, output),
                        None => Err(anyhow::anyhow!("pipeline stage completed without hidden state")),
                    }
                };
                match attach_result {
                    Ok(()) => {
                        request.stage += 1;
                        self.ready.push_back(request);
                    }
                    Err(e) => {
                        self.release_request(&request).await;
                        let _ = request.events.send(EngineEvent::Error(e.to_string()));
                    }
                }
            }
            Err(e) => {
                self.release_request(&request).await;
                let _ = request.events.send(EngineEvent::Error(e.to_string()));
            }
        }
    }

    async fn schedule_ready(&mut self, mut request: ActiveRequest) {
        if request.events.is_closed() {
            self.release_request(&request).await;
            return;
        }

        if request.step >= self.sample_len {
            self.release_request(&request).await;
            let _ = request.events.send(Self::finish_event(&request));
            return;
        }

        // Prepare a decode work item. Index 0 / multimodal prefill deliberately
        // returns None and stays on the legacy path.
        if request.hidden.is_none() {
            let prepared = {
                let mut master = self.master.lock().await;
                master
                    .prepare_session_step(request.session_id, request.step)
                    .await
            };
            match prepared {
                Ok(Some(hidden)) => {
                    request.hidden = Some(hidden);
                    request.stage = 0;
                }
                Ok(None) => {
                    // Correctness-first prefill fallback. This happens once per
                    // request; subsequent decode tokens use detached stages.
                    let token_result = {
                        let mut master = self.master.lock().await;
                        with_remote_profile(
                            request.profile.clone(),
                            master.step_session(request.session_id, request.step),
                        )
                        .await
                    };
                    match token_result {
                        Ok(token) if token.is_end_of_stream => {
                            self.release_request(&request).await;
                            let _ = request.events.send(Self::finish_event(&request));
                        }
                        Ok(token) => {
                            request.generated += 1;
                            request.step += 1;
                            let _ = request.events.send(EngineEvent::Token(token.to_string()));
                            self.ready.push_back(request);
                        }
                        Err(e) => {
                            self.release_request(&request).await;
                            let _ = request.events.send(EngineEvent::Error(e.to_string()));
                        }
                    }
                    return;
                }
                Err(e) => {
                    self.release_request(&request).await;
                    let _ = request.events.send(EngineEvent::Error(e.to_string()));
                    return;
                }
            }
        }

        let stage_count = self.stage_slots.len();
        if request.stage >= stage_count {
            let hidden = request.hidden.take().expect("pipeline hidden state");
            let token_result = {
                let mut master = self.master.lock().await;
                master
                    .finish_session_step(request.session_id, hidden)
                    .await
            };
            match token_result {
                Ok(token) if token.is_end_of_stream => {
                    self.release_request(&request).await;
                    let _ = request.events.send(Self::finish_event(&request));
                }
                Ok(token) => {
                    request.generated += 1;
                    request.step += 1;
                    request.stage = 0;
                    let _ = request.events.send(EngineEvent::Token(token.to_string()));
                    self.ready.push_back(request);
                }
                Err(e) => {
                    self.release_request(&request).await;
                    let _ = request.events.send(EngineEvent::Error(e.to_string()));
                }
            }
            return;
        }

        // Detach the actual activation + Master KV from the model while holding
        // the lock only briefly. The expensive local/remote await happens below
        // in JoinSet without holding Master.
        let job = {
            let master = self.master.lock().await;
            let hidden = request.hidden.as_mut().expect("pipeline hidden state");
            master.detach_session_stage(request.stage, hidden)
        };

        let job = match job {
            Ok(job) => job,
            Err(e) => {
                self.release_request(&request).await;
                let _ = request.events.send(EngineEvent::Error(e.to_string()));
                return;
            }
        };

        let slot = self.stage_slots[request.stage].clone();
        let session_id = request.session_id;
        let profile = request.profile.clone();

        self.in_flight.spawn(async move {
            let result = async {
                let _permit = slot
                    .acquire_owned()
                    .await
                    .map_err(|_| anyhow::anyhow!("pipeline stage semaphore closed"))?;
                let output = with_remote_profile(
                    profile,
                    with_remote_session(session_id, job.execute()),
                )
                .await;
                output
            }
            .await;
            (request, result)
        });
    }

    pub async fn run(mut self) {
        loop {
            self.fill_slots().await;

            // Drain already-completed stages before scheduling more ready work.
            // This prevents a legacy/RKNN fallback request that immediately
            // requeues itself from starving completed pipeline stages.
            while let Some(joined) = self.in_flight.try_join_next() {
                match joined {
                    Ok(completed) => self.handle_stage_completion(completed).await,
                    Err(e) => log::error!("pipeline stage task failed: {e}"),
                }
            }

            if let Some(request) = self.ready.pop_front() {
                self.schedule_ready(request).await;
                tokio::task::yield_now().await;
                continue;
            }

            if !self.in_flight.is_empty() {
                match self.in_flight.join_next().await {
                    Some(Ok(completed)) => self.handle_stage_completion(completed).await,
                    Some(Err(e)) => log::error!("pipeline stage task failed: {e}"),
                    None => {}
                }
                continue;
            }

            match self.rx.recv().await {
                Some(req) => self.admit(req).await,
                None => break,
            }
        }

        self.in_flight.abort_all();
    }
}

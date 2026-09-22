use std::collections::VecDeque;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;

use tokio::sync::{mpsc, oneshot, Mutex};

use crate::models::{chat::Message, Generator};
use super::{with_remote_profile, DistributedProfile, Master, SessionId};

pub struct EngineRequest {
    pub messages: Vec<Message>,
    pub events: mpsc::UnboundedSender<EngineEvent>,
    pub accepted: oneshot::Sender<SessionId>,
}

#[derive(Debug)]
pub enum EngineEvent {
    Token(String),
    Finished { generated_tokens: usize, elapsed_s: f64, profile: DistributedProfile },
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

/// Single model owner + request-level round-robin. The model weights remain one copy;
/// request-local state is swapped by Master::step_session between token steps.
pub struct PipelineEngine<G> {
    master: Arc<Mutex<Master<G>>>,
    rx: mpsc::UnboundedReceiver<EngineRequest>,
    active: VecDeque<ActiveRequest>,
    max_active: usize,
    sample_len: usize,
    next_session: SessionId,
}

impl<G: Generator + Send + Sync + 'static> PipelineEngine<G> {
    pub fn channel(master: Master<G>, max_active: usize) -> (mpsc::UnboundedSender<EngineRequest>, Self) {
        let (tx, rx) = mpsc::unbounded_channel();
        {
            let sample_len = master.ctx.args.sample_len;
            (tx, Self { master: Arc::new(Mutex::new(master)), rx, active: VecDeque::new(), max_active: max_active.max(1), sample_len, next_session: 1 })
        }
    }

    async fn admit(&mut self, request: EngineRequest) {
        let id = self.next_session;
        self.next_session = self.next_session.wrapping_add(1).max(1);
        let mut master = self.master.lock().await;
        match master.create_session(id, request.messages) {
            Ok(()) => {
                drop(master);
                let _ = request.accepted.send(id);
                self.active.push_back(ActiveRequest { session_id: id, step: 0, generated: 0, started: Instant::now(), events: request.events, profile: Arc::new(StdMutex::new(DistributedProfile::default())), stage: 0, hidden: None });
            }
            Err(e) => { let _ = request.events.send(EngineEvent::Error(e.to_string())); }
        }
    }

    async fn fill_slots(&mut self) {
        while self.active.len() < self.max_active {
            match self.rx.try_recv() { Ok(req) => self.admit(req).await, Err(_) => break }
        }
    }

    pub async fn run(mut self) {
        loop {
            self.fill_slots().await;
            if let Some(mut request) = self.active.pop_front() {
                if request.step >= self.sample_len {
                    let mut master = self.master.lock().await;
                    master.release_session(request.session_id);
                    let profile = request.profile.lock().map(|p| p.clone()).unwrap_or_default();
                    let _ = request.events.send(EngineEvent::Finished { generated_tokens: request.generated, elapsed_s: request.started.elapsed().as_secs_f64(), profile });
                    continue;
                }
                let mut master = self.master.lock().await;
                let stages = master.pipeline_stage_count();
                let outcome: anyhow::Result<Option<crate::models::Token>> = with_remote_profile(request.profile.clone(), async {
                    if request.hidden.is_none() {
                        request.hidden = Some(master.prepare_session_step(request.session_id, request.step).await?);
                        request.stage = 0;
                    }
                    if request.stage < stages {
                        let hidden = request.hidden.take().expect("pipeline hidden state");
                        request.hidden = Some(master.run_session_stage(request.session_id, request.stage, hidden).await?);
                        request.stage += 1;
                        return Ok(None);
                    }
                    let hidden = request.hidden.take().expect("pipeline hidden state");
                    let token = master.finish_session_step(request.session_id, hidden).await?;
                    Ok(Some(token))
                }).await;
                match outcome {
                    Ok(None) => self.active.push_back(request),
                    Ok(Some(token)) if token.is_end_of_stream => {
                        master.release_session(request.session_id);
                        let profile = request.profile.lock().map(|p| p.clone()).unwrap_or_default();
                        let _ = request.events.send(EngineEvent::Finished { generated_tokens: request.generated, elapsed_s: request.started.elapsed().as_secs_f64(), profile });
                    }
                    Ok(Some(token)) => {
                        request.generated += 1; request.step += 1; request.stage = 0;
                        let _ = request.events.send(EngineEvent::Token(token.to_string()));
                        self.active.push_back(request);
                    }
                    Err(e) => { master.release_session(request.session_id); let _ = request.events.send(EngineEvent::Error(e.to_string())); }
                }
                drop(master);
                tokio::task::yield_now().await;
                continue;
            }
            match self.rx.recv().await { Some(req) => self.admit(req).await, None => break }
        }
    }
}

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::{mpsc, oneshot};

use crate::models::chat::Message;

use super::SessionId;

/// One logical inference request. Request count is unrestricted: 1, 3, 5, ...
/// are scheduled exactly like even counts; pipeline depth only bounds in-flight work.
#[derive(Debug)]
pub struct PipelineRequest {
    pub session_id: SessionId,
    pub messages: Vec<Message>,
    pub stream: bool,
    pub reply: oneshot::Sender<PipelineAdmission>,
}

#[derive(Debug, Clone)]
pub struct PipelineAdmission {
    pub session_id: SessionId,
    pub queue_position: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestPhase { Prefill, Decode, Finished }

#[derive(Debug)]
pub struct RequestState {
    pub session_id: SessionId,
    pub phase: RequestPhase,
    pub step_id: usize,
}

/// vLLM-style control-plane skeleton: waiting requests are admitted independently
/// of model execution and ready sessions are round-robin scheduled. A session may
/// have at most one autoregressive step in flight, while different sessions overlap.
pub struct PipelineScheduler {
    next_session: AtomicU64,
    waiting: VecDeque<PipelineRequest>,
    ready: VecDeque<SessionId>,
    in_flight: HashMap<SessionId, RequestState>,
    max_in_flight: usize,
}

impl PipelineScheduler {
    pub fn new(max_in_flight: usize) -> Self {
        Self {
            next_session: AtomicU64::new(1),
            waiting: VecDeque::new(),
            ready: VecDeque::new(),
            in_flight: HashMap::new(),
            max_in_flight: max_in_flight.max(1),
        }
    }

    pub fn allocate_session_id(&self) -> SessionId {
        self.next_session.fetch_add(1, Ordering::Relaxed)
    }

    pub fn enqueue(&mut self, request: PipelineRequest) { self.waiting.push_back(request); }

    pub fn admit_ready(&mut self) {
        while self.in_flight.len() < self.max_in_flight {
            let Some(request) = self.waiting.pop_front() else { break };
            let session_id = request.session_id;
            let queue_position = self.ready.len();
            self.in_flight.insert(session_id, RequestState { session_id, phase: RequestPhase::Prefill, step_id: 0 });
            self.ready.push_back(session_id);
            let _ = request.reply.send(PipelineAdmission { session_id, queue_position });
        }
    }

    pub fn next_ready(&mut self) -> Option<SessionId> { self.ready.pop_front() }

    pub fn complete_step(&mut self, session_id: SessionId, finished: bool) {
        if finished {
            self.in_flight.remove(&session_id);
        } else if let Some(state) = self.in_flight.get_mut(&session_id) {
            state.phase = RequestPhase::Decode;
            state.step_id += 1;
            self.ready.push_back(session_id);
        }
        self.admit_ready();
    }

    pub fn max_in_flight(&self) -> usize { self.max_in_flight }
}

pub fn configured_pipeline_depth() -> usize {
    std::env::var("DIAL_PIPELINE_DEPTH").ok().and_then(|v| v.parse().ok()).filter(|v: &usize| *v > 0).unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(id: u64) -> (PipelineRequest, oneshot::Receiver<PipelineAdmission>) {
        let (tx, rx) = oneshot::channel();
        (PipelineRequest { session_id: id, messages: vec![], stream: false, reply: tx }, rx)
    }

    #[tokio::test]
    async fn odd_request_counts_are_admitted_without_pairing() {
        let mut s = PipelineScheduler::new(5);
        let mut replies = Vec::new();
        for id in 1..=5 { let (r, rx) = req(id); s.enqueue(r); replies.push(rx); }
        s.admit_ready();
        assert_eq!(s.in_flight.len(), 5);
        for rx in replies { assert!(rx.await.is_ok()); }
        assert_eq!((0..5).filter_map(|_| s.next_ready()).collect::<Vec<_>>(), vec![1,2,3,4,5]);
    }

    #[tokio::test]
    async fn finished_slot_admits_next_waiter() {
        let mut s = PipelineScheduler::new(3);
        let mut replies = Vec::new();
        for id in 1..=5 { let (r, rx) = req(id); s.enqueue(r); replies.push(rx); }
        s.admit_ready();
        assert_eq!(s.in_flight.len(), 3);
        s.complete_step(2, true);
        assert!(s.in_flight.contains_key(&4));
    }
}

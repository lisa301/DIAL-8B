use std::{collections::HashMap, io::Write};

use crate::models::{chat::Message, Generator, PipelineStageJob, PipelineStageOutput};

use super::{api, reset_distributed_profile, with_remote_session, Context, SessionId};

use anyhow::Result;

/// A master connects to, communicates with and orchestrates the workers.
pub struct Master<G> {
    pub ctx: Context,
    pub model: Box<G>,
    sessions: HashMap<SessionId, Box<dyn std::any::Any + Send>>,
}

//给泛型Master结构体实现方法
impl<G: Generator + Send + Sync + 'static> Master<G> {
    /// 异步创建并初始化Master主节点。
    pub async fn new(ctx: Context) -> Result<Self> {
        let model = G::load(ctx.clone()).await?;
        Ok(Self { ctx, model, sessions: HashMap::new() })
    }
    // 整个程序的入口
    pub async fn run(mut self) -> Result<()> {
        if self.ctx.args.api.is_some() {
            // 如果命令行带 --api 参数，就启动HTTP接口服务。
            api::start(self).await?;
        } else {
            // CLI模式，添加系统提示词+用户提示词。
            self.model
                .add_message(Message::system(self.ctx.args.system_prompt.clone()))?;
            self.model
                .add_message(Message::user(self.ctx.args.prompt.clone()))?;

            // 生成回复并输出到终端
            self.generate_with_session(0, |data| {
                if data.is_empty() {
                    println!();
                } else {
                    print!("{data}")
                }
                std::io::stdout().flush().unwrap();
            })
            .await?;
        }

        Ok(())
    }

    /// 重置整个主节点
    pub fn reset(&mut self) -> Result<()> {
        reset_distributed_profile();
        self.model.reset()
    }

    /// Create an isolated request context while retaining the single shared model weights.
    pub fn create_session(&mut self, session_id: SessionId, messages: Vec<Message>) -> Result<()> {
        self.model.reset()?;
        for message in messages { self.model.add_message(message)?; }
        let state = self.model.save_session()?.ok_or_else(|| anyhow::anyhow!("{} does not support pipeline sessions", G::MODEL_NAME))?;
        self.sessions.insert(session_id, state);
        Ok(())
    }

    pub async fn prepare_session_step(&mut self, session_id: SessionId, index: usize) -> Result<Option<Box<dyn std::any::Any + Send>>> {
        let state = self.sessions.remove(&session_id).ok_or_else(|| anyhow::anyhow!("unknown pipeline session {session_id}"))?;
        self.model.restore_session(state)?;
        let hidden = with_remote_session(session_id, self.model.pipeline_prepare(index)).await?;
        let saved = self.model.save_session()?.ok_or_else(|| anyhow::anyhow!("missing session state"))?;
        self.sessions.insert(session_id, saved);
        Ok(hidden)
    }

    pub fn detach_session_stage(
        &self,
        stage: usize,
        hidden: &mut Box<dyn std::any::Any + Send>,
    ) -> Result<PipelineStageJob> {
        self.model.pipeline_detach_stage(stage, hidden)
    }

    pub fn attach_session_stage(
        &self,
        hidden: &mut Box<dyn std::any::Any + Send>,
        output: PipelineStageOutput,
    ) -> Result<()> {
        self.model.pipeline_attach_stage(hidden, output)
    }


    pub async fn finish_session_step(&mut self, session_id: SessionId, hidden: Box<dyn std::any::Any + Send>) -> Result<crate::models::Token> {
        let state = self.sessions.remove(&session_id).ok_or_else(|| anyhow::anyhow!("unknown pipeline session {session_id}"))?;
        self.model.restore_session(state)?;
        let token = self.model.pipeline_finish(hidden).await;
        let saved = self.model.save_session()?;
        if let Some(saved) = saved { self.sessions.insert(session_id, saved); }
        token
    }

    pub fn pipeline_stage_count(&self) -> usize { self.model.pipeline_stage_count() }
    pub fn pipeline_stage_key(&self, stage: usize) -> Result<String> {
        self.model.pipeline_stage_key(stage)
    }

    /// Execute exactly one autoregressive step for a session. This is the scheduling
    /// primitive used to interleave A/B/C instead of running A to completion first.
    pub async fn step_session(&mut self, session_id: SessionId, index: usize) -> Result<crate::models::Token> {
        let state = self.sessions.remove(&session_id)
            .ok_or_else(|| anyhow::anyhow!("unknown pipeline session {session_id}"))?;
        self.model.restore_session(state)?;
        let result = with_remote_session(session_id, self.model.next_token(index)).await;
        let saved = self.model.save_session()?;
        if let Some(state) = saved { self.sessions.insert(session_id, state); }
        result
    }

    pub fn take_session_release_handles(&mut self, session_id: SessionId) -> Vec<std::sync::Arc<dyn crate::spm::Forwarder>> {
        self.sessions.remove(&session_id);
        self.model.pipeline_release_handles()
    }

    pub async fn release_session(&mut self, session_id: SessionId) {
        self.sessions.remove(&session_id);
        if let Err(e) = self.model.release_pipeline_session(session_id).await {
            log::warn!("failed to release remote pipeline session {}: {}", session_id, e);
        }
    }

    /// 逐一生成token，并通过stream函数实时输出。
    pub async fn generate<S>(&mut self, stream: S) -> Result<()>
    where
        S: FnMut(&str),
    {
        self.generate_with_session(0, stream).await
    }

    /// Generate one request under a stable distributed session id so worker KV caches
    /// remain isolated when multiple requests are interleaved on the same connections.
    pub async fn generate_with_session<S>(&mut self, session_id: SessionId, mut stream: S) -> Result<()>
    where
        S: FnMut(&str),
    {
        /// 打印日志
        log::info!(
            "starting the inference loop (mem={})\n\n",
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        log::debug!("  ctx.args.sample_len = {}", self.ctx.args.sample_len);

        stream(&self.ctx.args.prompt);

        let mut start_gen = std::time::Instant::now();

        for index in 0..self.ctx.args.sample_len {
            if index == 1 {
                // record start time again since the first token is the warmup
                start_gen = std::time::Instant::now()
            }
            /// 生成下一个词/字
            let token = with_remote_session(session_id, self.model.next_token(index)).await?;
            /// 如果生成结束，停止循环；否则把生成的token实时输出。
            if token.is_end_of_stream {
                break;
            } else {
                stream(&token.to_string());
                // Yield to let HTTP streaming tasks flush chunked responses promptly.
                tokio::task::yield_now().await;
            }
        }

        // 输出结束标记
        stream("");

        let dt = start_gen.elapsed();
        let generated = self.model.generated_tokens();
        /// 打印性能统计
        log::info!(
            "{} tokens generated ({} token/s) - mem={}",
            generated,
            (generated - 1) as f64 / dt.as_secs_f64(),
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        Ok(())
    }
}

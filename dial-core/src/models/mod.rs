use std::sync::Arc;
pub mod chat;
pub mod llama3;
pub mod qwen3_8;
pub mod qwen3_vl;

use crate::spm::{Context, Forwarder};

use anyhow::Result;
use async_trait::async_trait;
use chat::Message;

/// Token 结构体.
pub struct Token {
    /// 定义id.
    pub id: u32,
    /// 解析后的文本片段.
    pub text: Option<String>,
    /// Token流结束标记.
    pub is_end_of_stream: bool,
}
/// 为token实现Display trait.
impl std::fmt::Display for Token {
    ///  Display 唯一要求实现的方法
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            if let Some(text) = &self.text {
                text.clone()
            } else {
                // 无文本时显示token ID
                format!("<token {}>", self.id)
            }
        )
    }
}


/// A transformer stage detached from the mutable model/session owner.  It owns the
/// activation and the Master's request-local KV cache, so it can safely await a
/// remote worker while the Engine prepares or executes another request.
pub struct PipelineStageJob {
    pub x: candle_core::Tensor,
    pub cache: llama3::Cache,
    pub executors: Vec<Arc<dyn Forwarder>>,
    pub batch: Vec<(String, usize, usize)>,
    pub remote_batch: bool,
}

pub struct PipelineStageOutput {
    pub x: candle_core::Tensor,
    pub cache: llama3::Cache,
}

impl PipelineStageJob {
    pub async fn execute(mut self) -> Result<PipelineStageOutput> {
        if self.executors.is_empty() || self.batch.is_empty() {
            return Err(anyhow::anyhow!("empty pipeline stage"));
        }
        if self.remote_batch {
            self.x = self.executors[0]
                .forward_batch_shared(&self.x, self.batch, &mut self.cache)
                .await?;
        } else {
            if self.executors.len() != self.batch.len() {
                return Err(anyhow::anyhow!("local pipeline executor/batch length mismatch"));
            }
            for (executor, (_, index_pos, block_idx)) in self.executors.iter().zip(self.batch.iter()) {
                self.x = executor.forward(&self.x, *index_pos, *block_idx, &mut self.cache).await?;
            }
        }
        Ok(PipelineStageOutput { x: self.x, cache: self.cache })
    }
}

/// 一个模型必须实现这个trait,才能被dial使用.
#[async_trait]
/// 定义公共trait.
pub trait Generator {
    /// This associated type determines which part of the model can be sharded.
    type Shardable: Forwarder;

    /// The model name.
    const MODEL_NAME: &'static str;

    /// Load the model from the context.
    async fn load(context: Context) -> Result<Box<Self>>;

    /// Add a message to the chat.
    fn add_message(&mut self, message: Message) -> Result<()>;
    /// Clear chat history.
    fn reset(&mut self) -> Result<()>;

    /// Return the next token.
    async fn next_token(&mut self, index: usize) -> Result<Token>;

    /// Number of independently schedulable transformer stages for this model.
    fn pipeline_stage_count(&self) -> usize { 1 }
    /// Stable execution-resource key for a stage (e.g. "local" or worker address).
    /// Stages with the same key share one scheduler slot.
    fn pipeline_stage_key(&self, stage: usize) -> Result<String> {
        Ok(format!("stage-{stage}"))
    }

    /// Stage-split execution hooks. Models may expose a prepared hidden state and
    /// advance it one owner-contiguous stage at a time. Defaults preserve legacy behavior.
    async fn pipeline_prepare(&mut self, _index: usize) -> Result<Option<Box<dyn std::any::Any + Send>>> { Ok(None) }
    fn pipeline_detach_stage(&self, _stage: usize, _state: &mut Box<dyn std::any::Any + Send>) -> Result<PipelineStageJob> {
        Err(anyhow::anyhow!("{} does not support detached pipeline stages", Self::MODEL_NAME))
    }
    fn pipeline_attach_stage(&self, _state: &mut Box<dyn std::any::Any + Send>, _output: PipelineStageOutput) -> Result<()> {
        Err(anyhow::anyhow!("{} does not support detached pipeline stages", Self::MODEL_NAME))
    }
    async fn pipeline_finish(&mut self, _state: Box<dyn std::any::Any + Send>) -> Result<Token> {
        Err(anyhow::anyhow!("{} does not support stage pipeline", Self::MODEL_NAME))
    }
    async fn release_pipeline_session(&self, _session_id: crate::spm::SessionId) -> Result<()> {
        Ok(())
    }
    /// Return the number of generated tokens so far.
    fn generated_tokens(&self) -> usize;

    /// Move request-local generation state out of the model. Implementations that
    /// support pipeline concurrency override these hooks; legacy models keep the
    /// default and remain single-session.
    fn save_session(&mut self) -> Result<Option<Box<dyn std::any::Any + Send>>> { Ok(None) }
    fn restore_session(&mut self, _state: Box<dyn std::any::Any + Send>) -> Result<()> {
        Err(anyhow::anyhow!("{} does not support request session swapping", Self::MODEL_NAME))
    }
}

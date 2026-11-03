// Copyright 2026 fastokens contributors
// Copyright 2026 Baseten
// SPDX-License-Identifier: Apache-2.0
//
// Derived from Crusoe fastokens and modified by Baseten.

pub mod bpe;

use self::bpe::Bpe;
use crate::json_structs::{ModelConfig, ModelKind};

pub(crate) type Result<T> = std::result::Result<T, String>;

/// A constructed tokenization model ready for encoding.
#[derive(Debug)]
pub enum Model {
    Bpe(Bpe),
}

impl Model {
    /// Build a model from its JSON configuration.
    ///
    /// Takes the config by value to avoid cloning large structures like the BPE
    /// automaton.
    pub fn from_config(config: ModelConfig) -> Result<Self> {
        match config {
            ModelConfig::Bpe(bpe) => Ok(Self::Bpe(*bpe)),
            // Older tokenizer.json files (including roberta-base) omit type.
            ModelConfig::Other(value)
                if value.get("type").is_none()
                    && value.get("vocab").is_some_and(serde_json::Value::is_object)
                    && value.get("merges").is_some_and(serde_json::Value::is_array) =>
            {
                serde_json::from_value(value)
                    .map(Self::Bpe)
                    .map_err(|e| e.to_string())
            }
            other => {
                let kind = ModelKind::from(&other);
                Err(format!("unsupported model type: {kind}"))
            }
        }
    }

    /// Tokenize a pre-tokenized piece of text into token IDs.
    pub fn tokenize(&self, input: &str) -> Result<Vec<u32>> {
        match self {
            Self::Bpe(bpe) => bpe.tokenize(input),
        }
    }

    /// Tokenize directly into an existing buffer, avoiding intermediate
    /// allocation on cache hits.
    #[inline(always)]
    pub fn tokenize_into(&self, input: &str, out: &mut Vec<u32>) -> Result<()> {
        match self {
            Self::Bpe(bpe) => bpe.tokenize_into(input, out),
        }
    }

    /// Fused tokenize: `input` is raw (pre-ByteLevel) text. On cache hit,
    /// ByteLevel encoding is skipped entirely. On cache miss, ByteLevel
    /// encoding is done inline before running the BPE forward DP.
    #[inline(always)]
    pub fn tokenize_into_fused(&self, input: &str, out: &mut Vec<u32>) -> Result<()> {
        match self {
            Self::Bpe(bpe) => bpe.tokenize_into_fused(input, out),
        }
    }

    /// Batch fused tokenize: processes an entire chunk of splits within a
    /// single thread-local cache access, avoiding per-split TLS overhead.
    #[inline(always)]
    pub fn tokenize_batch_fused(
        &self,
        buffer: &str,
        splits: &[crate::pre_tokenized::Split],
        out: &mut Vec<u32>,
    ) -> Result<()> {
        match self {
            Self::Bpe(bpe) => bpe.tokenize_batch_fused(buffer, splits, out),
        }
    }

    #[inline(always)]
    pub(crate) fn tokenize_batch_fused_by_indices(
        &self,
        buffer: &str,
        splits: &[crate::pre_tokenized::Split],
        indices: &[u32],
        out: &mut Vec<u32>,
        ends: &mut Vec<usize>,
    ) -> Result<()> {
        match self {
            Self::Bpe(bpe) => {
                bpe.tokenize_batch_fused_by_indices(buffer, splits, indices, out, ends)
            }
        }
    }

    pub(crate) fn take_content_affinity(&self) -> bool {
        match self {
            Self::Bpe(bpe) => bpe.take_content_affinity(),
        }
    }

    /// Look up the string representation of a token ID.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        match self {
            Self::Bpe(bpe) => bpe.id_to_token(id),
        }
    }

    /// Look up the token ID for a string.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            Self::Bpe(bpe) => bpe.token_to_id(token),
        }
    }

    /// Return the vocabulary size (number of model tokens).
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Bpe(bpe) => bpe.vocab_size(),
        }
    }
}

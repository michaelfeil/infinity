// Copyright 2026 fastokens contributors
// Copyright 2026 Baseten
// SPDX-License-Identifier: Apache-2.0
//
// Derived from Crusoe fastokens and modified by Baseten.

pub(crate) mod byte_level;
pub(crate) mod fast_split;
pub(crate) mod fast_split_o200k;
mod split;

use crate::{
    json_structs::{PreTokenizerConfig, PreTokenizerKind},
    pre_tokenized::PreTokenizedString,
};

pub use self::{
    byte_level::ByteLevel,
    split::{Split, SplitBehavior},
};

pub(crate) use self::byte_level::BYTE_TO_CHAR;

/// Errors from constructing or running a pre-tokenizer.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A JSON value could not be deserialized into the expected type (e.g. an
    /// unrecognized pattern format or behavior string).
    #[error("invalid config value: {0}")]
    Json(#[from] serde_json::Error),

    /// The regex pattern failed to compile or exceeded backtracking limits at
    /// runtime.
    #[error("regex error: {0}")]
    Regex(#[from] fancy_regex::Error),

    /// The pre-tokenizer type is not yet implemented.
    #[error("unsupported pre-tokenizer type: {0}")]
    Unsupported(String),
}

/// A compiled pre-tokenizer ready for use.
#[derive(Clone, Debug)]
pub enum PreTokenizer {
    ByteLevel(ByteLevel),
    Split(Split),
    Sequence(Vec<PreTokenizer>),
}

impl PreTokenizer {
    pub(crate) fn prefers_content_affine_bpe(&self) -> bool {
        matches!(self, Self::Split(split) if split.prefers_content_affine_bpe())
    }

    /// Build a pre-tokenizer from its JSON configuration.
    pub fn from_config(config: PreTokenizerConfig) -> Result<Self, Error> {
        match config {
            PreTokenizerConfig::ByteLevel(bl) => Ok(Self::ByteLevel(bl)),
            PreTokenizerConfig::Split(s) => Ok(Self::Split(s)),
            PreTokenizerConfig::Sequence { pretokenizers } => {
                let steps = pretokenizers
                    .into_iter()
                    .map(Self::from_config)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self::Sequence(steps))
            }
            PreTokenizerConfig::Other(v) => {
                let typ = v.get("type").and_then(|t| t.as_str()).unwrap_or("unknown");
                Err(Error::Unsupported(typ.to_string()))
            }
            other => {
                let kind = PreTokenizerKind::from(&other);
                Err(Error::Unsupported(kind.to_string()))
            }
        }
    }

    /// Refine the splits of `pts` in place.
    pub fn pre_tokenize(&self, pts: &mut PreTokenizedString) -> Result<(), Error> {
        match self {
            Self::ByteLevel(bl) => bl.pre_tokenize(pts),
            Self::Split(s) => s.pre_tokenize(pts),
            Self::Sequence(steps) => {
                for step in steps {
                    step.pre_tokenize(pts)?;
                }
                Ok(())
            }
        }
    }
}

// Copyright 2026 fastokens contributors
// Copyright 2026 Baseten
// SPDX-License-Identifier: Apache-2.0
//
// Derived from Crusoe fastokens and modified by Baseten.

mod nfc;

use std::borrow::Cow;

pub use self::nfc::Nfc;
use crate::json_structs::{NormalizerConfig, NormalizerKind};

/// Errors from constructing a normalizer.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("unsupported normalizer type: {0}")]
    Unsupported(String),
}

/// A compiled normalizer ready for use.
#[derive(Debug)]
pub enum Normalizer {
    Nfc(Nfc),
    Prepend(String),
    Replace(Replace),
    Sequence(Vec<Normalizer>),
}

#[derive(Debug)]
pub struct Replace {
    pattern: String,
    content: String,
}

impl Replace {
    fn from_config(pattern: serde_json::Value, content: String) -> Result<Self, Error> {
        let Some(pattern) = pattern.get("String").and_then(|v| v.as_str()) else {
            let kind = pattern
                .as_object()
                .and_then(|object| object.keys().next())
                .map(String::as_str)
                .unwrap_or("unknown");
            return Err(Error::Unsupported(format!("Replace pattern type {kind}")));
        };
        Ok(Self {
            pattern: pattern.to_owned(),
            content,
        })
    }

    fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        if input.contains(&self.pattern) {
            Cow::Owned(input.replace(&self.pattern, &self.content))
        } else {
            Cow::Borrowed(input)
        }
    }
}

impl Normalizer {
    /// Build a normalizer from its JSON configuration.
    pub fn from_config(config: NormalizerConfig) -> Result<Self, Error> {
        match config {
            NormalizerConfig::Nfc => Ok(Self::Nfc(Nfc)),
            NormalizerConfig::Prepend { prepend } => Ok(Self::Prepend(prepend)),
            NormalizerConfig::Replace { pattern, content } => {
                Ok(Self::Replace(Replace::from_config(pattern, content)?))
            }
            NormalizerConfig::Sequence { normalizers } => {
                let steps = normalizers
                    .into_iter()
                    .map(Self::from_config)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self::Sequence(steps))
            }
            NormalizerConfig::Other(v) => {
                let typ = v.get("type").and_then(|t| t.as_str()).unwrap_or("unknown");
                Err(Error::Unsupported(typ.to_string()))
            }
            other => {
                let kind = NormalizerKind::from(&other);
                Err(Error::Unsupported(kind.to_string()))
            }
        }
    }

    /// Normalize `input`, returning `Cow::Borrowed` when unchanged.
    pub fn normalize<'a>(&self, input: &'a str) -> Cow<'a, str> {
        match self {
            Self::Nfc(nfc) => nfc.normalize(input),
            Self::Prepend(prefix) => {
                if input.is_empty() || prefix.is_empty() {
                    Cow::Borrowed(input)
                } else {
                    Cow::Owned(format!("{prefix}{input}"))
                }
            }
            Self::Replace(replace) => replace.normalize(input),
            Self::Sequence(steps) => {
                let mut current = Cow::Borrowed(input);
                for step in steps {
                    current = match current {
                        Cow::Borrowed(s) => step.normalize(s),
                        Cow::Owned(s) => Cow::Owned(step.normalize(&s).into_owned()),
                    };
                }
                current
            }
        }
    }
}

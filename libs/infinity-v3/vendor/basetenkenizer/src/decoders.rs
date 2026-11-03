// Copyright 2026 fastokens contributors
// Copyright 2026 Baseten
// SPDX-License-Identifier: Apache-2.0
//
// Derived from Crusoe fastokens and modified by Baseten.

mod byte_level;

use crate::json_structs::{DecoderConfig, DecoderKind};

pub use self::byte_level::ByteLevelDecoder;

/// Errors from constructing or running a decoder.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("unsupported decoder type: {0}")]
    Unsupported(String),
}

/// A compiled decoder ready for use.
#[derive(Debug)]
pub enum Decoder {
    ByteLevel(ByteLevelDecoder),
    ByteFallback,
    Fuse,
    Strip {
        content: char,
        start: usize,
        stop: usize,
    },
    Replace {
        pattern: String,
        content: String,
    },
    Sequence(Vec<Decoder>),
}

impl Decoder {
    /// Build a decoder from its JSON configuration.
    pub fn from_config(config: DecoderConfig) -> Result<Self, Error> {
        match config {
            DecoderConfig::ByteLevel => Ok(Self::ByteLevel(ByteLevelDecoder)),
            DecoderConfig::ByteFallback => Ok(Self::ByteFallback),
            DecoderConfig::Replace { pattern, content } => {
                let Some(pattern) = pattern.get("String").and_then(|v| v.as_str()) else {
                    let kind = pattern
                        .as_object()
                        .and_then(|object| object.keys().next())
                        .map(String::as_str)
                        .unwrap_or("unknown");
                    return Err(Error::Unsupported(format!("Replace pattern type {kind}")));
                };
                Ok(Self::Replace {
                    pattern: pattern.to_owned(),
                    content,
                })
            }
            DecoderConfig::Fuse => Ok(Self::Fuse),
            DecoderConfig::Strip {
                content,
                start,
                stop,
            } => Ok(Self::Strip {
                content,
                start,
                stop,
            }),
            DecoderConfig::Sequence { decoders } => {
                let steps = decoders
                    .into_iter()
                    .map(Self::from_config)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self::Sequence(steps))
            }
            DecoderConfig::Other(v) => {
                let typ = v.get("type").and_then(|t| t.as_str()).unwrap_or("unknown");
                Err(Error::Unsupported(typ.to_string()))
            }
            other => {
                let kind = DecoderKind::from(&other);
                Err(Error::Unsupported(kind.to_string()))
            }
        }
    }

    /// Apply this decoder step to a list of token strings, returning the
    /// transformed list.
    ///
    /// Follows HuggingFace's `decode_chain` semantics: each decoder step
    /// transforms the token list, and the final result is joined.
    pub fn decode_chain(&self, tokens: Vec<String>) -> Result<Vec<String>, Error> {
        match self {
            Self::ByteLevel(bl) => Ok(bl.decode_chain(tokens)),
            Self::Fuse => Ok(vec![tokens.concat()]),
            Self::Strip {
                content,
                start,
                stop,
            } => Ok(tokens
                .into_iter()
                .map(|token| {
                    let mut text = token.as_str();
                    for _ in 0..*start {
                        match text.strip_prefix(*content) {
                            Some(rest) => text = rest,
                            None => break,
                        }
                    }
                    for _ in 0..*stop {
                        match text.strip_suffix(*content) {
                            Some(rest) => text = rest,
                            None => break,
                        }
                    }
                    text.to_owned()
                })
                .collect()),
            Self::Replace { pattern, content } => Ok(tokens
                .into_iter()
                .map(|token| {
                    if token.contains(pattern) {
                        token.replace(pattern, content)
                    } else {
                        token
                    }
                })
                .collect()),
            Self::ByteFallback => {
                let mut decoded = Vec::with_capacity(tokens.len());
                let mut bytes = Vec::new();
                for token in tokens {
                    if let Some(byte) = fallback_byte(&token) {
                        bytes.push(byte);
                    } else {
                        flush_fallback_bytes(&mut decoded, &mut bytes);
                        decoded.push(token);
                    }
                }
                flush_fallback_bytes(&mut decoded, &mut bytes);
                Ok(decoded)
            }
            Self::Sequence(steps) => {
                let mut current = tokens;
                for step in steps {
                    current = step.decode_chain(current)?;
                }
                Ok(current)
            }
        }
    }

    /// High-level decode: apply `decode_chain` then join.
    pub fn decode(&self, tokens: Vec<String>) -> Result<String, Error> {
        let result = self.decode_chain(tokens)?;
        Ok(result.join(""))
    }
}

fn fallback_byte(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    (hex.len() == 2)
        .then(|| u8::from_str_radix(hex, 16).ok())
        .flatten()
}

fn flush_fallback_bytes(decoded: &mut Vec<String>, bytes: &mut Vec<u8>) {
    if !bytes.is_empty() {
        match std::str::from_utf8(bytes) {
            Ok(text) => decoded.push(text.to_owned()),
            Err(_) => decoded.extend(bytes.iter().map(|_| "�".to_owned())),
        }
        bytes.clear();
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::Decoder;
    use crate::json_structs::DecoderConfig;

    #[test]
    fn fuse_joins_the_decode_chain() {
        let decoder = Decoder::from_config(DecoderConfig::Fuse).unwrap();
        assert_eq!(
            decoder
                .decode_chain(vec!["one".into(), "two".into(), "three".into()])
                .unwrap(),
            vec!["onetwothree"]
        );
    }

    #[test]
    fn replace_reports_unsupported_pattern_kind() {
        let error = Decoder::from_config(DecoderConfig::Replace {
            pattern: json!({"Regex": "x+"}),
            content: String::new(),
        })
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "unsupported decoder type: Replace pattern type Regex"
        );
    }
}

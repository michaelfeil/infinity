// Copyright 2026 fastokens contributors
// Copyright 2026 Baseten
// SPDX-License-Identifier: Apache-2.0
//
// Derived from Crusoe fastokens and modified by Baseten.

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use crate::json_structs::PostProcessorConfig;

/// Errors from constructing a post-processor.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The post-processor type is not yet implemented.
    #[error("unsupported post-processor type: {0}")]
    Unsupported(String),

    /// A configuration value could not be parsed.
    #[error("invalid post-processor config: {0}")]
    InvalidConfig(String),
}

// ── TemplateProcessing types ─────────────────────────────────────────────

/// Which input sequence a template piece refers to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum SequenceId {
    A,
    B,
}

/// A single piece in a TemplateProcessing template.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub enum TemplatePiece {
    Sequence {
        id: SequenceId,
        #[allow(dead_code)]
        type_id: u32,
    },
    SpecialToken {
        id: String,
        #[allow(dead_code)]
        type_id: u32,
    },
}

/// Special token definition as stored in the tokenizer JSON.
#[derive(Debug, Deserialize)]
struct SpecialTokenDef {
    id: String,
    ids: Vec<u32>,
    // `tokens` is optional — some tokenizer.json files omit it.
    #[serde(default)]
    #[allow(dead_code)]
    tokens: Vec<String>,
}

/// Compiled TemplateProcessing post-processor.
///
/// For single-sequence encoding, iterates over the `single` template:
/// `Sequence` pieces are replaced by the encoded token IDs, `SpecialToken`
/// pieces are replaced by the looked-up IDs from `special_tokens`.
#[derive(Debug)]
pub struct TemplateProcessing {
    single: Vec<TemplatePiece>,
    #[allow(dead_code)]
    pair: Vec<TemplatePiece>,
    special_tokens: HashMap<String, Vec<u32>>,
}

impl TemplateProcessing {
    /// Build from the raw JSON values in `PostProcessorConfig`.
    pub fn from_config(
        single: Value,
        pair: Value,
        special_tokens_val: Value,
    ) -> Result<Self, Error> {
        // Null / absent fields are treated as empty — some tokenizer.json files
        // omit `pair` or `special_tokens` entirely.
        let single: Vec<TemplatePiece> = match single {
            Value::Null => vec![],
            v => serde_json::from_value(v)
                .map_err(|e| Error::InvalidConfig(format!("single template: {e}")))?,
        };
        let pair: Vec<TemplatePiece> = match pair {
            Value::Null => vec![],
            v => serde_json::from_value(v)
                .map_err(|e| Error::InvalidConfig(format!("pair template: {e}")))?,
        };
        let special_tokens_raw: HashMap<String, SpecialTokenDef> = match special_tokens_val {
            Value::Null => HashMap::new(),
            v => serde_json::from_value(v)
                .map_err(|e| Error::InvalidConfig(format!("special_tokens: {e}")))?,
        };

        // Key by the inner `id` field (the string the template pieces reference),
        // not the outer JSON key.  The two are usually identical, but using the
        // inner field is consistent with how the HF tokenizers builder constructs
        // the map and with what `apply_single` looks up.
        let special_tokens: HashMap<String, Vec<u32>> = special_tokens_raw
            .into_values()
            .map(|v| (v.id, v.ids))
            .collect();

        Ok(Self {
            single,
            pair,
            special_tokens,
        })
    }

    /// Apply the single-sequence template, inserting special token IDs
    /// around the encoded sequence.
    pub fn apply_single(&self, encoded: Vec<u32>) -> Vec<u32> {
        let mut result = Vec::with_capacity(encoded.len() + 4);
        for piece in &self.single {
            match piece {
                TemplatePiece::Sequence {
                    id: SequenceId::A, ..
                } => {
                    result.extend_from_slice(&encoded);
                }
                TemplatePiece::SpecialToken { id, .. } => {
                    if let Some(ids) = self.special_tokens.get(id) {
                        result.extend_from_slice(ids);
                    }
                }
                // Sequence B in a single template is ignored.
                _ => {}
            }
        }
        result
    }
}

// ── PostProcessor enum ───────────────────────────────────────────────────

/// A constructed post-processor.
///
/// Since this tokenizer only produces token IDs (not offset information),
/// `ByteLevel` is a no-op. `TemplateProcessing` inserts special tokens
/// (BOS/EOS/CLS/SEP) when `add_special_tokens` is true.
#[derive(Debug)]
pub enum PostProcessor {
    ByteLevel,
    TemplateProcessing(TemplateProcessing),
    Sequence(Vec<PostProcessor>),
}

impl PostProcessor {
    /// Build a post-processor from its JSON configuration.
    pub fn from_config(config: PostProcessorConfig) -> Result<Self, Error> {
        match config {
            PostProcessorConfig::ByteLevel { .. } => Ok(Self::ByteLevel),
            PostProcessorConfig::TemplateProcessing {
                single,
                pair,
                special_tokens,
            } => Ok(Self::TemplateProcessing(TemplateProcessing::from_config(
                single,
                pair,
                special_tokens,
            )?)),
            PostProcessorConfig::Sequence { processors } => {
                let steps = processors
                    .into_iter()
                    .map(Self::from_config)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self::Sequence(steps))
            }
            PostProcessorConfig::BertProcessing { .. } => {
                Err(Error::Unsupported("BertProcessing".to_string()))
            }
            PostProcessorConfig::RobertaProcessing { sep, cls, .. } => {
                Ok(Self::TemplateProcessing(TemplateProcessing {
                    single: vec![
                        TemplatePiece::SpecialToken {
                            id: cls.0.clone(),
                            type_id: 0,
                        },
                        TemplatePiece::Sequence {
                            id: SequenceId::A,
                            type_id: 0,
                        },
                        TemplatePiece::SpecialToken {
                            id: sep.0.clone(),
                            type_id: 0,
                        },
                    ],
                    pair: vec![], // Pair encoding is not exposed by basetenkenizer.
                    special_tokens: HashMap::from([(cls.0, vec![cls.1]), (sep.0, vec![sep.1])]),
                }))
            }
            PostProcessorConfig::Other(v) => {
                let typ = v.get("type").and_then(|t| t.as_str()).unwrap_or("unknown");
                Err(Error::Unsupported(typ.to_string()))
            }
        }
    }

    /// Apply post-processing to a single-sequence encoding.
    ///
    /// Only has an effect when `add_special_tokens` is true and the processor
    /// adds special tokens (e.g. `TemplateProcessing`).
    pub fn post_process_single(&self, encoded: Vec<u32>, add_special_tokens: bool) -> Vec<u32> {
        if !add_special_tokens {
            return encoded;
        }
        match self {
            Self::ByteLevel => encoded,
            Self::TemplateProcessing(tp) => tp.apply_single(encoded),
            Self::Sequence(steps) => steps.iter().fold(encoded, |acc, step| {
                step.post_process_single(acc, add_special_tokens)
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_processing_bos_only() {
        let tp = TemplateProcessing {
            single: vec![
                TemplatePiece::SpecialToken {
                    id: "<s>".into(),
                    type_id: 0,
                },
                TemplatePiece::Sequence {
                    id: SequenceId::A,
                    type_id: 0,
                },
            ],
            pair: vec![],
            special_tokens: HashMap::from([("<s>".to_string(), vec![1])]),
        };
        assert_eq!(tp.apply_single(vec![100, 200, 300]), vec![1, 100, 200, 300]);
    }

    #[test]
    fn template_processing_cls_sep() {
        let tp = TemplateProcessing {
            single: vec![
                TemplatePiece::SpecialToken {
                    id: "[CLS]".into(),
                    type_id: 0,
                },
                TemplatePiece::Sequence {
                    id: SequenceId::A,
                    type_id: 0,
                },
                TemplatePiece::SpecialToken {
                    id: "[SEP]".into(),
                    type_id: 0,
                },
            ],
            pair: vec![],
            special_tokens: HashMap::from([
                ("[CLS]".to_string(), vec![101]),
                ("[SEP]".to_string(), vec![102]),
            ]),
        };
        assert_eq!(tp.apply_single(vec![50, 60]), vec![101, 50, 60, 102]);
    }

    #[test]
    fn template_processing_empty_input() {
        let tp = TemplateProcessing {
            single: vec![
                TemplatePiece::SpecialToken {
                    id: "<s>".into(),
                    type_id: 0,
                },
                TemplatePiece::Sequence {
                    id: SequenceId::A,
                    type_id: 0,
                },
            ],
            pair: vec![],
            special_tokens: HashMap::from([("<s>".to_string(), vec![1])]),
        };
        assert_eq!(tp.apply_single(vec![]), vec![1]);
    }

    #[test]
    fn parse_from_json() {
        let single = serde_json::json!([
            {"SpecialToken": {"id": "<s>", "type_id": 0}},
            {"Sequence": {"id": "A", "type_id": 0}}
        ]);
        let pair = serde_json::json!([]);
        let special_tokens = serde_json::json!({
            "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}
        });
        let tp = TemplateProcessing::from_config(single, pair, special_tokens).unwrap();
        assert_eq!(tp.apply_single(vec![10, 20]), vec![1, 10, 20]);
    }

    #[test]
    fn post_process_single_respects_flag() {
        let pp = PostProcessor::TemplateProcessing(TemplateProcessing {
            single: vec![
                TemplatePiece::SpecialToken {
                    id: "<s>".into(),
                    type_id: 0,
                },
                TemplatePiece::Sequence {
                    id: SequenceId::A,
                    type_id: 0,
                },
            ],
            pair: vec![],
            special_tokens: HashMap::from([("<s>".to_string(), vec![1])]),
        });

        // With special tokens
        assert_eq!(pp.post_process_single(vec![10, 20], true), vec![1, 10, 20]);
        // Without special tokens
        assert_eq!(pp.post_process_single(vec![10, 20], false), vec![10, 20]);
    }

    /// A missing or null `special_tokens` field must not cause a load failure;
    /// it should produce a processor that leaves the sequence unchanged.
    #[test]
    fn template_processing_null_special_tokens_loads_ok() {
        let single = serde_json::json!([
            {"Sequence": {"id": "A", "type_id": 0}}
        ]);
        let pp = PostProcessor::from_config(PostProcessorConfig::TemplateProcessing {
            single,
            pair: Value::Null,           // absent / null
            special_tokens: Value::Null, // absent / null
        })
        .unwrap();
        // No special tokens in the template — should be a pass-through.
        assert_eq!(pp.post_process_single(vec![10, 20], true), vec![10, 20]);
    }

    /// A `SpecialTokenDef` without a `tokens` field must deserialise correctly.
    #[test]
    fn template_processing_special_token_def_without_tokens_field() {
        let single = serde_json::json!([
            {"SpecialToken": {"id": "<bos>", "type_id": 0}},
            {"Sequence":     {"id": "A",    "type_id": 0}},
        ]);
        // `tokens` field deliberately absent from the special token definition.
        let special_tokens = serde_json::json!({
            "<bos>": {"id": "<bos>", "ids": [1]}
        });
        let pp = PostProcessor::from_config(PostProcessorConfig::TemplateProcessing {
            single,
            pair: Value::Null,
            special_tokens,
        })
        .unwrap();
        assert_eq!(pp.post_process_single(vec![10, 20], true), vec![1, 10, 20]);
    }

    /// Verifies that the special-token lookup key is the *inner* `id` field,
    /// not the outer JSON object key.  Some tokenizer.json files use a
    /// different string (or an integer) as the outer key.
    #[test]
    fn template_processing_special_token_keyed_by_inner_id() {
        // Outer key ("bos_alias") differs from the inner id ("<s>") used in
        // the template piece.  The BOS token must still be inserted.
        let single = serde_json::json!([
            {"SpecialToken": {"id": "<s>", "type_id": 0}},
            {"Sequence":     {"id": "A",   "type_id": 0}},
        ]);
        let special_tokens = serde_json::json!({
            "bos_alias": {"id": "<s>", "ids": [1], "tokens": ["<s>"]}
        });
        let pp = PostProcessor::from_config(PostProcessorConfig::TemplateProcessing {
            single,
            pair: serde_json::json!([]),
            special_tokens,
        })
        .unwrap();

        assert_eq!(
            pp.post_process_single(vec![10, 20], true),
            vec![1, 10, 20],
            "BOS must be added even when outer JSON key differs from SpecialToken.id"
        );
    }

    #[test]
    fn bert_processing_returns_unsupported() {
        assert!(
            PostProcessor::from_config(PostProcessorConfig::BertProcessing {
                sep: ("[SEP]".to_string(), 102),
                cls: ("[CLS]".to_string(), 101),
            })
            .is_err()
        );
    }

    #[test]
    fn roberta_processing_adds_bos_eos() {
        let processor = PostProcessor::from_config(PostProcessorConfig::RobertaProcessing {
            sep: ("</s>".to_string(), 2),
            cls: ("<s>".to_string(), 0),
            trim_offsets: true,
            add_prefix_space: true,
        })
        .unwrap();
        assert_eq!(
            processor.post_process_single(vec![42], true),
            vec![0, 42, 2]
        );
        assert_eq!(processor.post_process_single(vec![], true), vec![0, 2]);
        assert_eq!(processor.post_process_single(vec![42], false), vec![42]);
    }

    #[test]
    fn template_processing_suffix_only() {
        let tp = TemplateProcessing {
            single: vec![
                TemplatePiece::Sequence {
                    id: SequenceId::A,
                    type_id: 0,
                },
                TemplatePiece::SpecialToken {
                    id: "</s>".into(),
                    type_id: 0,
                },
            ],
            pair: vec![],
            special_tokens: HashMap::from([("</s>".to_string(), vec![2])]),
        };
        assert_eq!(tp.apply_single(vec![10, 20]), vec![10, 20, 2]);
    }

    #[test]
    fn template_processing_bos_and_eos() {
        let tp = TemplateProcessing {
            single: vec![
                TemplatePiece::SpecialToken {
                    id: "<s>".into(),
                    type_id: 0,
                },
                TemplatePiece::Sequence {
                    id: SequenceId::A,
                    type_id: 0,
                },
                TemplatePiece::SpecialToken {
                    id: "</s>".into(),
                    type_id: 0,
                },
            ],
            pair: vec![],
            special_tokens: HashMap::from([
                ("<s>".to_string(), vec![1]),
                ("</s>".to_string(), vec![2]),
            ]),
        };
        assert_eq!(tp.apply_single(vec![10, 20, 30]), vec![1, 10, 20, 30, 2]);
        // Empty sequence still gets BOS + EOS
        assert_eq!(tp.apply_single(vec![]), vec![1, 2]);
    }

    #[test]
    fn template_processing_multi_id_special_token() {
        // Some tokenizers map a single special token to multiple IDs
        let tp = TemplateProcessing {
            single: vec![
                TemplatePiece::SpecialToken {
                    id: "<prefix>".into(),
                    type_id: 0,
                },
                TemplatePiece::Sequence {
                    id: SequenceId::A,
                    type_id: 0,
                },
            ],
            pair: vec![],
            special_tokens: HashMap::from([("<prefix>".to_string(), vec![100, 101])]),
        };
        assert_eq!(tp.apply_single(vec![10, 20]), vec![100, 101, 10, 20]);
    }

    #[test]
    fn byte_level_post_processor_is_identity() {
        let pp = PostProcessor::ByteLevel;
        assert_eq!(pp.post_process_single(vec![1, 2, 3], true), vec![1, 2, 3]);
        assert_eq!(pp.post_process_single(vec![1, 2, 3], false), vec![1, 2, 3]);
        assert_eq!(
            pp.post_process_single(Vec::<u32>::new(), true),
            Vec::<u32>::new()
        );
    }

    #[test]
    fn num_special_tokens_matches_post_process_delta() {
        // Build a BOS+EOS processor and verify the delta matches.
        let pp = PostProcessor::TemplateProcessing(TemplateProcessing {
            single: vec![
                TemplatePiece::SpecialToken {
                    id: "<s>".into(),
                    type_id: 0,
                },
                TemplatePiece::Sequence {
                    id: SequenceId::A,
                    type_id: 0,
                },
                TemplatePiece::SpecialToken {
                    id: "</s>".into(),
                    type_id: 0,
                },
            ],
            pair: vec![],
            special_tokens: HashMap::from([
                ("<s>".to_string(), vec![1]),
                ("</s>".to_string(), vec![2]),
            ]),
        });
        let payload = vec![10u32, 20, 30];
        let with_special = pp.post_process_single(payload.clone(), true);
        let without_special = pp.post_process_single(payload.clone(), false);
        assert_eq!(with_special.len() - without_special.len(), 2); // BOS + EOS
    }

    #[test]
    fn sequence_post_processor_applies_all() {
        let pp_inner_a = PostProcessor::TemplateProcessing(TemplateProcessing {
            single: vec![
                TemplatePiece::SpecialToken {
                    id: "<a>".into(),
                    type_id: 0,
                },
                TemplatePiece::Sequence {
                    id: SequenceId::A,
                    type_id: 0,
                },
            ],
            pair: vec![],
            special_tokens: HashMap::from([("<a>".to_string(), vec![99])]),
        });
        let pp_inner_b = PostProcessor::ByteLevel; // no-op second stage
        let pp = PostProcessor::Sequence(vec![pp_inner_a, pp_inner_b]);
        assert_eq!(pp.post_process_single(vec![10, 20], true), vec![99, 10, 20]);
        assert_eq!(pp.post_process_single(vec![10, 20], false), vec![10, 20]);
    }
}

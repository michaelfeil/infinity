// Copyright 2026 fastokens contributors
// Copyright 2026 Baseten
// SPDX-License-Identifier: Apache-2.0
//
// Derived from Crusoe fastokens and modified by Baseten.

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;
use strum::EnumDiscriminants;

use crate::{models, pre_tokenizers};

/// An entry in the `added_tokens` array of `tokenizer.json`.
#[derive(Clone, Debug, Deserialize)]
pub struct AddedTokenConfig {
    /// Token ID.
    pub id: u32,
    /// Literal text content that triggers this token.
    pub content: String,
    /// Whether the token should only match as a whole word.
    #[serde(default)]
    pub single_word: bool,
    /// Whether to strip whitespace on the left when matching.
    #[serde(default)]
    pub lstrip: bool,
    /// Whether to strip whitespace on the right when matching.
    #[serde(default)]
    pub rstrip: bool,
    /// Whether the content should be matched against normalized
    /// text.
    #[serde(default)]
    pub normalized: bool,
    /// Whether this is a "special" token (e.g. BOS/EOS). All added
    /// tokens are matched regardless of this flag; it only affects
    /// post-processing.
    #[serde(default)]
    pub special: bool,
}

#[derive(Debug, Deserialize)]
pub struct TokenizerConfig {
    #[serde(default)]
    pub added_tokens_decoder: HashMap<String, Value>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

impl TokenizerConfig {
    pub fn added_token_configs(&self) -> Result<Vec<AddedTokenConfig>, String> {
        let mut entries = Vec::with_capacity(self.added_tokens_decoder.len());
        for (id, value) in &self.added_tokens_decoder {
            let id = id
                .parse::<u32>()
                .map_err(|e| format!("invalid added token id {id:?}: {e}"))?;
            let mut value = value.clone();
            let object = value
                .as_object_mut()
                .ok_or_else(|| format!("added token {id} is not a JSON object"))?;
            object.insert("id".into(), Value::from(id));
            entries.push(
                serde_json::from_value(value)
                    .map_err(|e| format!("failed to parse added token {id}: {e}"))?,
            );
        }
        entries.sort_by_key(|entry: &AddedTokenConfig| entry.id);
        Ok(entries)
    }

    pub fn tokenizer_class(&self) -> Option<&str> {
        self.extra.get("tokenizer_class")?.as_str()
    }

    pub fn auto_map_tokenizers(&self) -> Vec<&str> {
        let Some(value) = self
            .extra
            .get("auto_map")
            .and_then(|auto_map| auto_map.get("AutoTokenizer"))
        else {
            return Vec::new();
        };
        match value {
            Value::String(tokenizer) => vec![tokenizer],
            Value::Array(tokenizers) => tokenizers.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        }
    }

    pub fn declared_pattern(&self) -> Option<&str> {
        self.extra
            .get("pat_str")
            .and_then(Value::as_str)
            .or_else(|| self.extra.get("pattern").and_then(Value::as_str))
    }

    pub fn has_kimi_marker(&self) -> bool {
        const MARKERS: &[&str] = &[
            "<|open|>",
            "<|close|>",
            "<|sep|>",
            "<|end_of_msg|>",
            "<|im_user|>",
            "<|im_assistant|>",
            "<|im_middle|>",
            "<|tool_calls_section_begin|>",
        ];
        self.added_tokens_decoder.values().any(|token| {
            token
                .get("content")
                .and_then(Value::as_str)
                .is_some_and(|content| MARKERS.contains(&content))
        })
    }
}

#[cfg(test)]
mod tokenizer_config_tests {
    use super::TokenizerConfig;

    #[test]
    fn auto_map_tokenizer_accepts_string_and_array_forms() {
        for auto_tokenizer in [
            serde_json::json!("tokenization_kimi.TikTokenTokenizer"),
            serde_json::json!([null, "tokenization_kimi.TikTokenTokenizer"]),
        ] {
            let config: TokenizerConfig = serde_json::from_value(serde_json::json!({
                "tokenizer_class": "TikTokenTokenizer",
                "auto_map": {"AutoTokenizer": auto_tokenizer}
            }))
            .unwrap();
            assert_eq!(
                config.auto_map_tokenizers(),
                ["tokenization_kimi.TikTokenTokenizer"]
            );
        }
    }
}

/// Parsed `tokenizer.json`.
///
/// The five pipeline steps are deserialized into typed enums. Any other
/// top-level keys (e.g. `version`) are captured in [`extra`](Self::extra).
#[derive(Debug, Deserialize)]
pub struct TokenizerJson {
    #[serde(default)]
    pub added_tokens: Vec<AddedTokenConfig>,
    pub normalizer: Option<NormalizerConfig>,
    pub pre_tokenizer: Option<PreTokenizerConfig>,
    pub model: ModelConfig,
    pub post_processor: Option<PostProcessorConfig>,
    pub decoder: Option<DecoderConfig>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// Normalizer pipeline step from `tokenizer.json`.
///
/// Unrecognised types are stored as [`Other`](Self::Other).
#[derive(Clone, Debug, Deserialize, EnumDiscriminants)]
#[strum_discriminants(name(NormalizerKind), derive(strum::Display, Hash, Ord, PartialOrd))]
#[serde(tag = "type")]
pub enum NormalizerConfig {
    Sequence {
        #[serde(default)]
        normalizers: Vec<NormalizerConfig>,
    },
    #[serde(rename = "NFC")]
    #[strum_discriminants(strum(to_string = "NFC"))]
    Nfc,
    #[serde(rename = "NFD")]
    #[strum_discriminants(strum(to_string = "NFD"))]
    Nfd,
    #[serde(rename = "NFKC")]
    #[strum_discriminants(strum(to_string = "NFKC"))]
    Nfkc,
    #[serde(rename = "NFKD")]
    #[strum_discriminants(strum(to_string = "NFKD"))]
    Nfkd,
    Lowercase,
    StripAccents,
    Strip {
        #[serde(default)]
        strip_left: bool,
        #[serde(default)]
        strip_right: bool,
    },
    Prepend {
        #[serde(default)]
        prepend: String,
    },
    Replace {
        pattern: Value,
        #[serde(default)]
        content: String,
    },
    BertNormalizer {
        #[serde(default)]
        clean_text: bool,
        #[serde(default)]
        handle_chinese_chars: bool,
        strip_accents: Option<bool>,
        #[serde(default)]
        lowercase: bool,
    },
    ByteLevel,
    #[serde(untagged)]
    Other(Value),
}

/// Pre-tokenizer pipeline step from `tokenizer.json`.
#[derive(Clone, Debug, Deserialize, EnumDiscriminants)]
#[strum_discriminants(name(PreTokenizerKind), derive(strum::Display, Hash, Ord, PartialOrd))]
#[serde(tag = "type")]
pub enum PreTokenizerConfig {
    Sequence {
        #[serde(default)]
        pretokenizers: Vec<PreTokenizerConfig>,
    },
    ByteLevel(pre_tokenizers::ByteLevel),
    Whitespace,
    WhitespaceSplit,
    Split(pre_tokenizers::Split),
    Punctuation {
        #[serde(default)]
        behavior: String,
    },
    Metaspace {
        #[serde(default = "default_meta_replacement")]
        replacement: char,
        prepend_scheme: Option<String>,
        add_prefix_space: Option<bool>,
    },
    Digits {
        #[serde(default)]
        individual_digits: bool,
    },
    BertPreTokenizer,
    UnicodeScripts,
    #[serde(untagged)]
    Other(Value),
}

fn default_meta_replacement() -> char {
    '\u{2581}' // ▁
}

/// Tokenization model from `tokenizer.json`.
#[derive(Clone, Debug, Deserialize, EnumDiscriminants)]
#[strum_discriminants(name(ModelKind), derive(strum::Display, Hash, Ord, PartialOrd))]
#[serde(tag = "type")]
pub enum ModelConfig {
    #[serde(rename = "BPE")]
    #[strum_discriminants(strum(to_string = "BPE"))]
    Bpe(Box<models::bpe::Bpe>),
    WordPiece {
        #[serde(default)]
        vocab: HashMap<String, u32>,
        #[serde(default)]
        unk_token: String,
        #[serde(default)]
        continuing_subword_prefix: String,
        max_input_chars_per_word: Option<u64>,
    },
    WordLevel {
        #[serde(default)]
        vocab: HashMap<String, u32>,
        #[serde(default)]
        unk_token: String,
    },
    Unigram {
        #[serde(default)]
        vocab: Vec<(String, f64)>,
        unk_id: Option<u32>,
        #[serde(default)]
        byte_fallback: bool,
    },
    #[serde(untagged)]
    Other(Value),
}

/// Post-processor pipeline step from `tokenizer.json`.
#[derive(Clone, Debug, Deserialize, EnumDiscriminants)]
#[strum_discriminants(name(PostProcessorKind), derive(strum::Display, Hash, Ord, PartialOrd))]
#[serde(tag = "type")]
pub enum PostProcessorConfig {
    Sequence {
        #[serde(default)]
        processors: Vec<PostProcessorConfig>,
    },
    ByteLevel {
        #[serde(default)]
        add_prefix_space: bool,
        #[serde(default)]
        trim_offsets: bool,
        #[serde(default)]
        use_regex: bool,
    },
    TemplateProcessing {
        #[serde(default)]
        single: Value,
        #[serde(default)]
        pair: Value,
        #[serde(default)]
        special_tokens: Value,
    },
    BertProcessing {
        sep: (String, u32),
        cls: (String, u32),
    },
    RobertaProcessing {
        sep: (String, u32),
        cls: (String, u32),
        #[serde(default)]
        trim_offsets: bool,
        #[serde(default)]
        add_prefix_space: bool,
    },
    #[serde(untagged)]
    Other(Value),
}

/// Decoder pipeline step from `tokenizer.json`.
#[derive(Clone, Debug, Deserialize, EnumDiscriminants)]
#[strum_discriminants(name(DecoderKind), derive(strum::Display, Hash, Ord, PartialOrd))]
#[serde(tag = "type")]
pub enum DecoderConfig {
    Sequence {
        #[serde(default)]
        decoders: Vec<DecoderConfig>,
    },
    ByteLevel,
    WordPiece {
        #[serde(default = "default_wp_prefix")]
        prefix: String,
        #[serde(default)]
        cleanup: bool,
    },
    Metaspace {
        #[serde(default = "default_meta_replacement")]
        replacement: char,
        prepend_scheme: Option<String>,
        add_prefix_space: Option<bool>,
    },
    #[serde(rename = "BPEDecoder")]
    #[strum_discriminants(strum(to_string = "BPEDecoder"))]
    BpeDecoder {
        #[serde(default)]
        suffix: String,
    },
    #[serde(rename = "CTC")]
    #[strum_discriminants(strum(to_string = "CTC"))]
    Ctc {
        #[serde(default)]
        pad_token: String,
        word_delimiter_token: Option<String>,
        #[serde(default)]
        cleanup: bool,
    },
    Strip {
        #[serde(default = "default_space")]
        content: char,
        #[serde(default)]
        start: usize,
        #[serde(default)]
        stop: usize,
    },
    Replace {
        pattern: Value,
        #[serde(default)]
        content: String,
    },
    Fuse,
    ByteFallback,
    #[serde(untagged)]
    Other(Value),
}

fn default_wp_prefix() -> String {
    "##".to_string()
}

fn default_space() -> char {
    ' '
}

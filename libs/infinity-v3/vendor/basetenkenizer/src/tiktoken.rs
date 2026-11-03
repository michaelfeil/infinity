// Copyright 2026 fastokens contributors
// Copyright 2026 Baseten
// SPDX-License-Identifier: Apache-2.0
// Derived from Crusoe fastokens and modified by Baseten.

//! Native support for the line-oriented `tiktoken.model` format.

use std::collections::HashMap;

use crate::Error;

/// Kimi's pre-tokenization pattern, kept byte-identical to
/// `tokenization_kimi.py` so it selects the specialized Unicode scanner.
pub const KIMI_PATTERN: &str = concat!(
    r"[\p{Han}]+",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

pub const KIMI_RESERVED_SPECIAL_TOKENS: u32 = 256;
pub const KIMI_MAX_ADDED_TOKENS: u32 = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TiktokenFamily {
    Kimi,
}

impl TiktokenFamily {
    pub fn detect<'a>(
        tokenizer_class: Option<&str>,
        auto_map_tokenizers: impl IntoIterator<Item = &'a str>,
        has_kimi_marker: bool,
    ) -> Option<Self> {
        if auto_map_tokenizers
            .into_iter()
            .any(|entry| entry.contains("tokenization_kimi.TikTokenTokenizer"))
            || (tokenizer_class == Some("TikTokenTokenizer") && has_kimi_marker)
        {
            Some(Self::Kimi)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct TiktokenConfig {
    pub pattern: String,
    pub special_tokens: Vec<(String, u32)>,
}

impl TiktokenConfig {
    pub fn new(pattern: impl Into<String>, special_tokens: Vec<(String, u32)>) -> Self {
        Self {
            pattern: pattern.into(),
            special_tokens,
        }
    }

    /// Reproduce Kimi's reserved window, extending the usual 256 slots when a
    /// configuration declares a higher token ID.
    pub fn kimi(
        num_ranks: u32,
        named: impl IntoIterator<Item = (u32, String)>,
    ) -> Result<Self, String> {
        let mut names: HashMap<u32, String> = named
            .into_iter()
            .filter(|(id, _)| *id >= num_ranks)
            .collect();
        let declared_len = names
            .keys()
            .max()
            .map_or(Ok(0), |id| {
                id.checked_sub(num_ranks)
                    .and_then(|offset| offset.checked_add(1))
                    .ok_or_else(|| "Kimi added-token range overflows u32".to_string())
            })?
            .max(KIMI_RESERVED_SPECIAL_TOKENS);
        if declared_len > KIMI_MAX_ADDED_TOKENS {
            return Err(format!(
                "Kimi added-token range has {declared_len} entries; maximum is {KIMI_MAX_ADDED_TOKENS}"
            ));
        }
        let end = num_ranks
            .checked_add(declared_len)
            .ok_or_else(|| "Kimi added-token range overflows u32".to_string())?;
        let special_tokens = (num_ranks..end)
            .map(|id| {
                let content = names
                    .remove(&id)
                    .unwrap_or_else(|| format!("<|reserved_token_{id}|>"));
                (content, id)
            })
            .collect();
        Ok(Self::new(KIMI_PATTERN, special_tokens))
    }
}

/// Parse `<base64(token bytes)> <rank>` lines without the Python tiktoken package.
pub fn parse_tiktoken_model(contents: &str) -> Result<Vec<(Vec<u8>, u32)>, Error> {
    let mut out = Vec::new();
    for (index, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut fields = line.split_whitespace();
        let encoded = fields
            .next()
            .ok_or_else(|| Error::Model(format!("line {}: empty entry", index + 1)))?;
        let rank = fields
            .next()
            .ok_or_else(|| Error::Model(format!("line {}: missing rank", index + 1)))?
            .parse::<u32>()
            .map_err(|_| Error::Model(format!("line {}: invalid rank", index + 1)))?;
        if fields.next().is_some() {
            return Err(Error::Model(format!("line {}: too many fields", index + 1)));
        }
        let bytes = base64_decode(encoded)
            .map_err(|error| Error::Model(format!("line {}: {error}", index + 1)))?;
        out.push((bytes, rank));
    }
    if out.is_empty() {
        return Err(Error::Model("no token entries found".into()));
    }
    Ok(out)
}

fn base64_decode(value: &str) -> Result<Vec<u8>, String> {
    fn sextet(byte: u8) -> Option<u8> {
        match byte {
            b'A'..=b'Z' => Some(byte - b'A'),
            b'a'..=b'z' => Some(byte - b'a' + 26),
            b'0'..=b'9' => Some(byte - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }

    let bytes = value.as_bytes();
    let mut end = bytes.len();
    while end > 0 && bytes[end - 1] == b'=' {
        end -= 1;
    }
    let padding = bytes.len() - end;
    if end == 0
        || end % 4 == 1
        || padding > 2
        || (padding != 0 && (bytes.len() % 4 != 0 || padding != (4 - end % 4) % 4))
    {
        return Err("invalid base64 length or padding".into());
    }
    let mut out = Vec::with_capacity(end * 3 / 4 + 1);
    let mut accumulator = 0u32;
    let mut bits = 0u32;
    for &byte in &bytes[..end] {
        accumulator = (accumulator << 6)
            | u32::from(sextet(byte).ok_or_else(|| "invalid base64 character".to_string())?);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((accumulator >> bits) as u8);
        }
    }
    if bits != 0 && accumulator & ((1 << bits) - 1) != 0 {
        return Err("invalid base64 trailing bits".into());
    }
    Ok(out)
}

use crate::{
    Tokenizer,
    json_structs::{AddedTokenConfig, TokenizerConfig},
    pre_tokenizers::BYTE_TO_CHAR,
};
use serde_json::{Value, json};
use std::{fs, path::Path};

/// Default pattern used for raw rank files without a tokenizer configuration.
/// Rank files do not contain a regex or special-token table; callers can
/// override both with [`TiktokenConfig`].
pub const DEFAULT_PATTERN: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|",
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|",
    r"\s+(?!\S)|\s+",
);

impl Tokenizer {
    /// Construct a tokenizer from a line-oriented tiktoken ranks string.
    pub fn from_tiktoken_str(contents: &str, config: TiktokenConfig) -> Result<Self, Error> {
        Self::from_tiktoken_ranks(&parse_tiktoken_model(contents)?, config)
    }

    /// Load the exact ranks file supplied, with an explicit regex and specials.
    pub fn from_tiktoken_file(path: &Path, config: TiktokenConfig) -> Result<Self, Error> {
        Self::from_tiktoken_str(&fs::read_to_string(path)?, config)
    }

    /// Load `tiktoken.model` and optional `tokenizer_config.json` in a directory.
    pub fn from_tiktoken_dir(path: &Path) -> Result<Self, Error> {
        Self::from_tiktoken_model(&path.join("tiktoken.model"))
    }

    /// Load an exact model path, inferring the pattern and added tokens from
    /// its adjacent `tokenizer_config.json`. Known Kimi configurations receive
    /// their native pattern and reserved tokens; unidentified raw files use
    /// [`DEFAULT_PATTERN`]. Other identified families require a declared regex.
    pub fn from_tiktoken_model(path: &Path) -> Result<Self, Error> {
        let path = if path.is_dir() {
            path.join("tiktoken.model")
        } else {
            path.to_path_buf()
        };
        let ranks = parse_tiktoken_model(&fs::read_to_string(&path)?)?;
        let config_path = path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("tokenizer_config.json");
        let config: TokenizerConfig = match fs::read_to_string(&config_path) {
            Ok(contents) => serde_json::from_str(&contents)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                serde_json::from_value(json!({}))?
            }
            Err(error) => return Err(error.into()),
        };
        let family = TiktokenFamily::detect(
            config.tokenizer_class(),
            config.auto_map_tokenizers(),
            config.has_kimi_marker(),
        );
        let pattern = match config.declared_pattern() {
            Some(pattern) => pattern,
            None if family == Some(TiktokenFamily::Kimi) => KIMI_PATTERN,
            None if config.extra.contains_key("tokenizer_class")
                || config.extra.contains_key("auto_map") =>
            {
                return Err(Error::Model("unrecognized tiktoken family: tokenizer_config.json must declare pat_str or pattern".into()));
            }
            None => DEFAULT_PATTERN,
        };
        let mut added_tokens = config.added_token_configs().map_err(Error::Model)?;
        if family == Some(TiktokenFamily::Kimi) {
            let num_ranks = u32::try_from(ranks.len())
                .map_err(|_| Error::Model("too many tiktoken ranks".into()))?;
            let reserved = TiktokenConfig::kimi(
                num_ranks,
                added_tokens
                    .iter()
                    .map(|token| (token.id, token.content.clone())),
            )
            .map_err(Error::Model)?;
            let declared: HashMap<_, _> = added_tokens
                .iter()
                .map(|token| (token.id, token.clone()))
                .collect();
            added_tokens = reserved
                .special_tokens
                .into_iter()
                .map(|(content, id)| {
                    declared.get(&id).cloned().unwrap_or(AddedTokenConfig {
                        id,
                        content,
                        single_word: false,
                        lstrip: false,
                        rstrip: false,
                        normalized: false,
                        special: true,
                    })
                })
                .collect();
        }
        Self::from_json(ranks_to_json(&ranks, pattern, &added_tokens)?)
    }

    /// Construct directly from decoded byte sequences and their BPE ranks.
    pub fn from_tiktoken_ranks(
        ranks: &[(Vec<u8>, u32)],
        config: TiktokenConfig,
    ) -> Result<Self, Error> {
        let added = config
            .special_tokens
            .into_iter()
            .map(|(content, id)| AddedTokenConfig {
                id,
                content,
                single_word: false,
                lstrip: false,
                rstrip: false,
                normalized: false,
                special: true,
            })
            .collect::<Vec<_>>();
        Self::from_json(ranks_to_json(ranks, &config.pattern, &added)?)
    }
}

fn token_string(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|&byte| BYTE_TO_CHAR[byte as usize])
        .collect()
}

/// Reconstruct all valid ranked merge pairs, matching the Python converter.
/// The existing BPE implementation then builds its optimized lookup tables.
fn ranks_to_json(
    ranks: &[(Vec<u8>, u32)],
    pattern: &str,
    added: &[AddedTokenConfig],
) -> Result<Value, Error> {
    if ranks.is_empty() {
        return Err(Error::Model("no tiktoken ranks supplied".into()));
    }
    let mut by_bytes = HashMap::with_capacity(ranks.len());
    let mut by_id = HashMap::with_capacity(ranks.len());
    let mut vocab = serde_json::Map::new();
    for (bytes, rank) in ranks {
        if bytes.is_empty() {
            return Err(Error::Model("empty tiktoken token".into()));
        }
        if by_bytes.insert(bytes.as_slice(), *rank).is_some() {
            return Err(Error::Model("duplicate tiktoken token bytes".into()));
        }
        if by_id.insert(*rank, bytes.as_slice()).is_some() {
            return Err(Error::Model(format!("duplicate tiktoken rank: {rank}")));
        }
        // The BPE model uses a dense ID lookup table. Reject pathological IDs
        // before its allocation, while allowing ordinary sparse vocabularies.
        if *rank > 16_777_216 {
            return Err(Error::Model(
                "tiktoken rank exceeds supported vocabulary size".into(),
            ));
        }
        vocab.insert(token_string(bytes), json!(rank));
    }
    let mut special_ids = std::collections::HashSet::new();
    let mut special_contents = std::collections::HashSet::new();
    for token in added {
        if !special_ids.insert(token.id) || !special_contents.insert(token.content.as_str()) {
            return Err(Error::Model("duplicate tiktoken added token".into()));
        }
        if let Some(bytes) = by_id.get(&token.id) {
            if String::from_utf8_lossy(bytes) != token.content {
                return Err(Error::Model(format!(
                    "added token {} conflicts with a tiktoken rank",
                    token.id
                )));
            }
        }
    }
    let mut candidates = Vec::new();
    for (bytes, rank) in ranks {
        for index in 1..bytes.len() {
            let (left, right) = bytes.split_at(index);
            if let (Some(left_rank), Some(right_rank)) = (by_bytes.get(left), by_bytes.get(right)) {
                candidates.push((*rank, *left_rank, *right_rank, left, right));
            }
        }
    }
    candidates.sort_unstable_by_key(|&(rank, left, right, _, _)| (rank, left, right));
    let merges: Vec<_> = candidates
        .into_iter()
        .map(|(_, _, _, left, right)| json!([token_string(left), token_string(right)]))
        .collect();
    let added: Vec<_> = added.iter().map(|token| json!({
        "id": token.id, "content": token.content, "single_word": token.single_word,
        "lstrip": token.lstrip, "rstrip": token.rstrip, "normalized": token.normalized, "special": token.special,
    })).collect();
    Ok(json!({
        "added_tokens": added, "normalizer": null,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
        ]},
        "post_processor": null, "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE", "vocab": vocab, "merges": merges, "ignore_merges": true,
            "byte_fallback": false, "fuse_unk": false, "continuing_subword_prefix": "", "end_of_word_suffix": ""}
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_supports_whitespace_and_arbitrary_bytes() {
        assert_eq!(
            parse_tiktoken_model("\r\nIQ== 0\r\n /w==\t1\n").unwrap(),
            vec![(b"!".to_vec(), 0), (vec![255], 1)]
        );
    }

    #[test]
    fn parser_rejects_malformed_model_with_line_number() {
        for model in [
            "YQ==",
            "YQ== x",
            "YQ== 0 extra",
            "Y 0",
            "YQ=== 0",
            "YR== 0",
            "Y=Q= 0",
        ] {
            let error = parse_tiktoken_model(model).unwrap_err().to_string();
            assert!(error.contains("line 1"), "{model}: {error}");
        }
        assert!(parse_tiktoken_model(" \n").is_err());
    }

    #[test]
    fn rank_validation_rejects_duplicate_bytes_ids_and_special_conflicts() {
        for ranks in [
            vec![(b"a".to_vec(), 0), (b"b".to_vec(), 0)],
            vec![(b"a".to_vec(), 0), (b"a".to_vec(), 1)],
            vec![(Vec::new(), 0)],
        ] {
            assert!(
                Tokenizer::from_tiktoken_ranks(&ranks, TiktokenConfig::new(r"\S+", vec![]))
                    .is_err()
            );
        }
        assert!(
            Tokenizer::from_tiktoken_str(
                "YQ== 0\n",
                TiktokenConfig::new(r"\S+", vec![("special".into(), 0)])
            )
            .is_err()
        );
    }

    #[test]
    fn native_ranks_preserve_merges_unicode_and_specials() {
        let mut ranks: Vec<_> = (0u8..=255)
            .map(|byte| (vec![byte], u32::from(byte)))
            .collect();
        ranks.extend([(b"ab".to_vec(), 256), (b"abc".to_vec(), 257)]);
        let tokenizer = Tokenizer::from_tiktoken_ranks(
            &ranks,
            TiktokenConfig::new(r"\S+|\s+", vec![("<special>".into(), 258)]),
        )
        .unwrap();
        assert_eq!(tokenizer.encode("abc").unwrap(), vec![257]);
        assert_eq!(tokenizer.encode("ababc").unwrap(), vec![256, 257]);
        for text in ["abc é 🐈\n", "<special>ab", "\0\n\t"] {
            let ids = tokenizer.encode(text).unwrap();
            assert_eq!(tokenizer.decode(&ids, false).unwrap(), text);
        }
        assert_eq!(tokenizer.encode("<special>ab").unwrap(), vec![258, 256]);
        assert_eq!(tokenizer.decode(&[258, 256], true).unwrap(), "ab");
    }

    #[test]
    fn kimi_reserved_window_is_bounded_and_preserves_named_entries() {
        let config = TiktokenConfig::kimi(256, [(515, "last".into())]).unwrap();
        assert_eq!(config.special_tokens.len(), 260);
        assert_eq!(config.special_tokens[259], ("last".into(), 515));
        assert!(TiktokenConfig::kimi(256, [(u32::MAX - 1, "bad".into())]).is_err());
    }
}

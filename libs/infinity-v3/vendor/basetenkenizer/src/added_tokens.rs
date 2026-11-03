// Copyright 2026 fastokens contributors
// Copyright 2026 Baseten
// SPDX-License-Identifier: Apache-2.0
//
// Derived from Crusoe fastokens and modified by Baseten.

use std::collections::{HashMap, HashSet};
use std::fmt;

use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder};

use crate::json_structs::AddedTokenConfig;

/// A compiled set of added tokens that can be matched against input text.
///
/// The HuggingFace `tokenizer.json` format includes an `added_tokens` array of
/// literal patterns that are matched *before* the normal tokenization pipeline.
/// Matched spans are assigned their token IDs directly; unmatched spans pass
/// through normalization, pre-tokenization and the model as usual.
pub struct AddedTokens {
    all: AddedTokenMatcher,
    has_special_tokens: bool,
    has_non_special_tokens: bool,
    /// Mapping from token ID to token content string.
    id_to_content: HashMap<u32, String>,
    /// Reverse mapping: token content string → token ID.
    content_to_id: HashMap<String, u32>,
    /// Set of token IDs marked as special (e.g. BOS/EOS).
    special_ids: HashSet<u32>,
    /// Only tokenizers with lstrip tokens need the whitespace adjustment pass.
    lstrip_ids: HashSet<u32>,
}

/// A segment of the input after added-token splitting.
#[derive(Debug, PartialEq, Eq)]
pub enum Segment<'a> {
    /// A span that matched an added token. The `u32` is the token ID to emit
    /// directly.
    Token(u32),
    /// A span that did not match any added token. The `&str` should be run
    /// through the normal pipeline.
    Text(&'a str),
}

/// Public view of one added-token entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddedTokenInfo<'a> {
    pub id: u32,
    pub content: &'a str,
    pub special: bool,
}

struct AddedTokenMatcher {
    daac: DoubleArrayAhoCorasick<u32>,
    /// Distinct first bytes of all added token strings. Used to quickly skip
    /// positions that cannot start any token via SIMD memchr.
    start_bytes: Vec<u8>,
    /// Longest added token in bytes. Limits the DAAC scan window.
    max_token_len: usize,
}

impl AddedTokenMatcher {
    fn new(patterns: Vec<(&str, u32)>) -> Result<Option<Self>, String> {
        if patterns.is_empty() {
            return Ok(None);
        }

        // Collect distinct first bytes for memchr prefilter.
        let mut start_set = [false; 256];
        let mut max_token_len = 0;
        for (content, _) in &patterns {
            if let Some(&b) = content.as_bytes().first() {
                start_set[b as usize] = true;
            }
            max_token_len = max_token_len.max(content.len());
        }
        let start_bytes: Vec<u8> = start_set
            .iter()
            .enumerate()
            .filter(|&(_, v)| *v)
            .map(|(i, _)| i as u8)
            .collect();

        let daac = DoubleArrayAhoCorasickBuilder::new()
            .match_kind(daachorse::MatchKind::LeftmostLongest)
            .build_with_values(patterns)
            .map_err(|e| format!("error building added-tokens DAAC: {e}"))?;

        Ok(Some(Self {
            daac,
            start_bytes,
            max_token_len,
        }))
    }

    /// Split `input` into segments: spans matching added tokens and spans of
    /// regular text.
    ///
    /// Added tokens are matched leftmost-longest. Non-overlapping matches are
    /// emitted as [`Segment::Token`]; the gaps between them as
    /// [`Segment::Text`].
    fn split<'a>(&self, input: &'a str) -> Vec<Segment<'a>> {
        // When there are few distinct start bytes, use SIMD memchr to skip
        // positions that cannot start any added token. This avoids scanning
        // the full input through the Aho-Corasick automaton.
        match self.start_bytes.len() {
            1 => self.split_prefilter(
                input,
                memchr::memchr_iter(self.start_bytes[0], input.as_bytes()),
            ),
            2 => self.split_prefilter(
                input,
                memchr::memchr2_iter(self.start_bytes[0], self.start_bytes[1], input.as_bytes()),
            ),
            3 => self.split_prefilter(
                input,
                memchr::memchr3_iter(
                    self.start_bytes[0],
                    self.start_bytes[1],
                    self.start_bytes[2],
                    input.as_bytes(),
                ),
            ),
            _ => self.split_full_scan(input),
        }
    }

    /// Prefiltered split: only check positions identified by memchr.
    fn split_prefilter<'a>(
        &self,
        input: &'a str,
        candidates: impl Iterator<Item = usize>,
    ) -> Vec<Segment<'a>> {
        let mut segments = Vec::new();
        let mut prev_end = 0;

        for pos in candidates {
            if pos < prev_end {
                continue;
            }
            // Run the DAAC on a short window starting at this position.
            let mut window_end = (pos + self.max_token_len).min(input.len());
            // Ensure window_end is at a UTF-8 char boundary.
            while window_end < input.len() && !input.is_char_boundary(window_end) {
                window_end += 1;
            }
            let window = &input[pos..window_end];
            if let Some(m) = self.daac.leftmost_find_iter(window).next()
                && m.start() == 0
            {
                if pos > prev_end {
                    segments.push(Segment::Text(&input[prev_end..pos]));
                }
                segments.push(Segment::Token(m.value()));
                prev_end = pos + m.end();
            }
        }

        if prev_end < input.len() {
            segments.push(Segment::Text(&input[prev_end..]));
        }
        if segments.is_empty() && !input.is_empty() {
            segments.push(Segment::Text(input));
        }

        segments
    }

    /// Full-scan fallback for >3 distinct start bytes.
    fn split_full_scan<'a>(&self, input: &'a str) -> Vec<Segment<'a>> {
        let mut segments = Vec::new();
        let mut prev_end = 0;

        for m in self.daac.leftmost_find_iter(input) {
            if m.start() > prev_end {
                segments.push(Segment::Text(&input[prev_end..m.start()]));
            }
            segments.push(Segment::Token(m.value()));
            prev_end = m.end();
        }

        if prev_end < input.len() {
            segments.push(Segment::Text(&input[prev_end..]));
        }

        segments
    }
}

impl AddedTokens {
    /// Build from the `added_tokens` array in `tokenizer.json`.
    ///
    /// Returns `None` if there are no added tokens.
    pub fn from_configs(configs: &[AddedTokenConfig]) -> Result<Option<Self>, String> {
        if configs.is_empty() {
            return Ok(None);
        }

        let mut id_to_content = HashMap::with_capacity(configs.len());
        let mut special_ids = HashSet::new();

        let mut content_to_id = HashMap::with_capacity(configs.len());

        let patterns: Vec<(&str, u32)> = configs
            .iter()
            .map(|c| {
                id_to_content.insert(c.id, c.content.clone());
                content_to_id.insert(c.content.clone(), c.id);
                if c.special {
                    special_ids.insert(c.id);
                }
                (c.content.as_str(), c.id)
            })
            .collect();

        let Some(all) = AddedTokenMatcher::new(patterns)? else {
            debug_assert!(false, "non-empty configs should produce token patterns");
            return Ok(None);
        };
        let has_special_tokens = !special_ids.is_empty();
        let has_non_special_tokens = special_ids.len() != configs.len();

        Ok(Some(Self {
            all,
            has_special_tokens,
            has_non_special_tokens,
            id_to_content,
            content_to_id,
            special_ids,
            lstrip_ids: configs.iter().filter(|c| c.lstrip).map(|c| c.id).collect(),
        }))
    }

    /// Look up the string content of an added token by ID.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.id_to_content.get(&id).map(String::as_str)
    }

    /// Look up the token ID for a content string.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.content_to_id.get(token).copied()
    }

    /// Check if a token ID is a special added token.
    pub fn is_special(&self, id: u32) -> bool {
        self.special_ids.contains(&id)
    }

    /// Return the number of added tokens.
    pub fn len(&self) -> usize {
        self.id_to_content.len()
    }

    /// Return whether there are no added tokens.
    pub fn is_empty(&self) -> bool {
        self.id_to_content.is_empty()
    }

    /// Iterate over added-token entries.
    ///
    /// The iteration order is unspecified. Callers that need a stable order
    /// should sort by `id` themselves.
    pub fn iter(&self) -> impl Iterator<Item = AddedTokenInfo<'_>> {
        self.id_to_content
            .iter()
            .map(|(&id, content)| AddedTokenInfo {
                id,
                content: content.as_str(),
                special: self.special_ids.contains(&id),
            })
    }

    /// Split `input` into segments: spans matching added tokens and spans of
    /// regular text.
    ///
    /// Added tokens are matched leftmost-longest. Non-overlapping matches are
    /// emitted as [`Segment::Token`]; the gaps between them as
    /// [`Segment::Text`].
    pub fn split<'a>(&self, input: &'a str) -> Vec<Segment<'a>> {
        let mut segments = self.all.split(input);
        self.apply_lstrip(&mut segments);
        segments
    }

    fn apply_lstrip(&self, segments: &mut [Segment<'_>]) {
        if self.lstrip_ids.is_empty() {
            return;
        }
        for index in 1..segments.len() {
            if let Segment::Token(id) = segments[index]
                && self.lstrip_ids.contains(&id)
                && let Segment::Text(text) = &mut segments[index - 1]
            {
                *text = text.trim_end_matches(char::is_whitespace);
            }
        }
    }

    /// Split `input` into segments while leaving special added tokens as text.
    ///
    /// This mirrors HuggingFace's `split_special_tokens=True` encode option:
    /// special-token strings are tokenized through the normal model pipeline,
    /// while non-special added tokens can still match as added tokens.
    pub fn split_special_as_text<'a>(&self, input: &'a str) -> Vec<Segment<'a>> {
        if input.is_empty() {
            return Vec::new();
        }

        if !self.has_special_tokens {
            return self.split(input);
        }
        if !self.has_non_special_tokens {
            return vec![Segment::Text(input)];
        }

        let mut segments = Vec::new();
        let mut cursor = 0;
        let mut pending_text_start: Option<usize> = None;

        for segment in self.all.split(input) {
            match segment {
                Segment::Text(text) => {
                    if !text.is_empty() && pending_text_start.is_none() {
                        pending_text_start = Some(cursor);
                    }
                    cursor += text.len();
                }
                Segment::Token(id) => {
                    let content = self
                        .id_to_content
                        .get(&id)
                        .expect("matcher id must exist in added-token content map");
                    let start = cursor;
                    let end = start + content.len();

                    if self.special_ids.contains(&id) {
                        if pending_text_start.is_none() {
                            pending_text_start = Some(start);
                        }
                    } else {
                        if let Some(text_start) = pending_text_start.take()
                            && text_start < start
                        {
                            segments.push(Segment::Text(&input[text_start..start]));
                        }
                        segments.push(Segment::Token(id));
                    }

                    cursor = end;
                }
            }
        }

        if let Some(text_start) = pending_text_start
            && text_start < cursor
        {
            segments.push(Segment::Text(&input[text_start..cursor]));
        }

        self.apply_lstrip(&mut segments);
        segments
    }
}

impl fmt::Debug for AddedTokens {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AddedTokens")
            .field("count", &self.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(id: u32, content: &str) -> AddedTokenConfig {
        AddedTokenConfig {
            id,
            content: content.to_string(),
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: false,
            special: false,
        }
    }

    #[test]
    fn empty_configs() {
        let result = AddedTokens::from_configs(&[]).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn no_match() {
        let configs = vec![make_config(100, "<special>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("hello world");
        assert_eq!(segs, vec![Segment::Text("hello world")]);
    }

    #[test]
    fn single_match_at_start() {
        let configs = vec![make_config(100, "<s>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("<s>hello");
        assert_eq!(segs, vec![Segment::Token(100), Segment::Text("hello")]);
    }

    #[test]
    fn single_match_at_end() {
        let configs = vec![make_config(100, "</s>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("hello</s>");
        assert_eq!(segs, vec![Segment::Text("hello"), Segment::Token(100)]);
    }

    #[test]
    fn match_in_middle() {
        let configs = vec![make_config(42, "<sep>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("hello<sep>world");
        assert_eq!(
            segs,
            vec![
                Segment::Text("hello"),
                Segment::Token(42),
                Segment::Text("world"),
            ]
        );
    }

    #[test]
    fn multiple_matches() {
        let configs = vec![make_config(1, "<a>"), make_config(2, "<b>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("x<a>y<b>z");
        assert_eq!(
            segs,
            vec![
                Segment::Text("x"),
                Segment::Token(1),
                Segment::Text("y"),
                Segment::Token(2),
                Segment::Text("z"),
            ]
        );
    }

    #[test]
    fn adjacent_matches() {
        let configs = vec![make_config(1, "<a>"), make_config(2, "<b>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("<a><b>");
        assert_eq!(segs, vec![Segment::Token(1), Segment::Token(2)]);
    }

    #[test]
    fn longest_match_wins() {
        let configs = vec![make_config(1, "<file>"), make_config(2, "<filename>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("a<filename>b");
        assert_eq!(
            segs,
            vec![Segment::Text("a"), Segment::Token(2), Segment::Text("b"),]
        );
    }

    #[test]
    fn entire_input_is_added_token() {
        let configs = vec![make_config(99, "hello")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("hello");
        assert_eq!(segs, vec![Segment::Token(99)]);
    }

    #[test]
    fn empty_input() {
        let configs = vec![make_config(1, "<s>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("");
        assert!(segs.is_empty());
    }

    // ── token_to_id (content → id reverse lookup) ───────────────────────

    #[test]
    fn token_to_id_finds_added_token() {
        let configs = vec![make_config(42, "<special>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        assert_eq!(at.token_to_id("<special>"), Some(42));
    }

    #[test]
    fn token_to_id_returns_none_for_unknown() {
        let configs = vec![make_config(1, "<known>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        assert_eq!(at.token_to_id("<unknown>"), None);
    }

    #[test]
    fn token_to_id_and_id_to_token_are_inverses() {
        let configs = vec![
            make_config(10, "<bos>"),
            make_config(11, "<eos>"),
            make_config(12, "<pad>"),
        ];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        for cfg in &configs {
            let id = at.token_to_id(&cfg.content).unwrap();
            assert_eq!(id, cfg.id);
            assert_eq!(at.id_to_token(id), Some(cfg.content.as_str()));
        }
    }

    // ── Unicode and multi-byte token content ────────────────────────────

    #[test]
    fn unicode_token_content() {
        let configs = vec![
            make_config(1, "▁"), // U+2581  (SentencePiece metaspace)
            make_config(2, "Ġ"), // U+0120  (GPT-2 space marker)
            make_config(3, "日本語"),
        ];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        assert_eq!(
            at.split("▁hello"),
            vec![Segment::Token(1), Segment::Text("hello")]
        );
        assert_eq!(
            at.split("Ġworld"),
            vec![Segment::Token(2), Segment::Text("world")]
        );
        assert_eq!(
            at.split("日本語text"),
            vec![Segment::Token(3), Segment::Text("text")]
        );
        assert_eq!(at.token_to_id("▁"), Some(1));
        assert_eq!(at.token_to_id("Ġ"), Some(2));
        assert_eq!(at.token_to_id("日本語"), Some(3));
    }

    #[test]
    fn emoji_token_content() {
        let configs = vec![make_config(7, "🌍")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        assert_eq!(
            at.split("hello 🌍 world"),
            vec![
                Segment::Text("hello "),
                Segment::Token(7),
                Segment::Text(" world"),
            ]
        );
    }

    // ── is_special ──────────────────────────────────────────────────────

    #[test]
    fn is_special_only_for_marked_tokens() {
        let mut special = make_config(1, "<bos>");
        special.special = true;
        let non_special = make_config(2, "<extra>");
        let at = AddedTokens::from_configs(&[special, non_special])
            .unwrap()
            .unwrap();
        assert!(at.is_special(1));
        assert!(!at.is_special(2));
        assert!(!at.is_special(99)); // unknown id
    }

    #[test]
    fn split_special_as_text_keeps_non_special_matches() {
        let mut special = make_config(1, "<think>");
        special.special = true;
        let non_special = make_config(2, "<|tool_calls_section_begin|>");
        let at = AddedTokens::from_configs(&[special, non_special])
            .unwrap()
            .unwrap();

        assert_eq!(
            at.split_special_as_text("a<think>b<|tool_calls_section_begin|>c"),
            vec![
                Segment::Text("a<think>b"),
                Segment::Token(2),
                Segment::Text("c"),
            ]
        );
    }

    #[test]
    fn split_special_as_text_without_non_special_tokens_returns_text() {
        let mut special = make_config(1, "<think>");
        special.special = true;
        let mut end_special = make_config(2, "</think>");
        end_special.special = true;
        let at = AddedTokens::from_configs(&[special, end_special])
            .unwrap()
            .unwrap();

        assert_eq!(
            at.split_special_as_text("a<think>b</think>c"),
            vec![Segment::Text("a<think>b</think>c")]
        );
        assert!(at.split_special_as_text("").is_empty());
    }

    #[test]
    fn split_special_as_text_with_no_special_tokens_uses_all_matches() {
        let plain = make_config(1, "<extra>");
        let at = AddedTokens::from_configs(&[plain]).unwrap().unwrap();

        assert_eq!(
            at.split_special_as_text("a<extra>b"),
            vec![Segment::Text("a"), Segment::Token(1), Segment::Text("b")]
        );
    }

    #[test]
    fn split_special_as_text_blocks_non_special_match_inside_special_text() {
        let mut special = make_config(1, "<mask>");
        special.special = true;
        let non_special = make_config(2, "ask>");
        let at = AddedTokens::from_configs(&[special, non_special])
            .unwrap()
            .unwrap();

        assert_eq!(
            at.split_special_as_text("<mask>"),
            vec![Segment::Text("<mask>")]
        );
    }

    #[test]
    fn iter_exposes_id_content_and_special_flag() {
        let mut special = make_config(1, "<bos>");
        special.special = true;
        let plain = make_config(2, "<extra>");
        let at = AddedTokens::from_configs(&[special, plain])
            .unwrap()
            .unwrap();

        let mut entries: Vec<_> = at.iter().collect();
        entries.sort_by_key(|entry| entry.id);

        assert_eq!(
            entries,
            vec![
                AddedTokenInfo {
                    id: 1,
                    content: "<bos>",
                    special: true,
                },
                AddedTokenInfo {
                    id: 2,
                    content: "<extra>",
                    special: false,
                },
            ]
        );
    }

    // ── len / is_empty ───────────────────────────────────────────────────

    #[test]
    fn len_returns_token_count() {
        let configs = vec![
            make_config(1, "<a>"),
            make_config(2, "<b>"),
            make_config(3, "<c>"),
        ];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        assert_eq!(at.len(), 3);
        assert!(!at.is_empty());
    }

    #[test]
    fn three_tokens_with_shared_start_byte() {
        // <, <s>, <sep> all start with '<'  — exercises the memchr prefilter
        // (≤3 distinct first bytes → SIMD path).
        let configs = vec![
            make_config(1, "<"),
            make_config(2, "<s>"),
            make_config(3, "<sep>"),
        ];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        // Longest match: <sep> wins over <s> or <
        let segs = at.split("x<sep>y<s>z<");
        assert_eq!(
            segs,
            vec![
                Segment::Text("x"),
                Segment::Token(3),
                Segment::Text("y"),
                Segment::Token(2),
                Segment::Text("z"),
                Segment::Token(1),
            ]
        );
    }

    #[test]
    fn four_distinct_start_bytes_uses_full_scan() {
        // >3 distinct first bytes → full-scan path (no memchr prefilter).
        let configs = vec![
            make_config(1, "<bos>"),
            make_config(2, "[SEP]"),
            make_config(3, "{pad}"),
            make_config(4, "|mask|"),
        ];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("<bos>[SEP]{pad}|mask|");
        assert_eq!(
            segs,
            vec![
                Segment::Token(1),
                Segment::Token(2),
                Segment::Token(3),
                Segment::Token(4),
            ]
        );
    }

    #[test]
    fn token_surrounded_by_text() {
        let configs = vec![make_config(5, "<mid>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("prefix <mid> suffix");
        assert_eq!(
            segs,
            vec![
                Segment::Text("prefix "),
                Segment::Token(5),
                Segment::Text(" suffix"),
            ]
        );
    }

    #[test]
    fn repeated_same_token() {
        let configs = vec![make_config(9, "<r>")];
        let at = AddedTokens::from_configs(&configs).unwrap().unwrap();
        let segs = at.split("<r><r><r>");
        assert_eq!(
            segs,
            vec![Segment::Token(9), Segment::Token(9), Segment::Token(9)]
        );
    }
}

//! Request-local reuse at the Qwen tokenizer's state/question split.
use tokenizers::{
    models::ModelWrapper, normalizers::NormalizerWrapper, processors::PostProcessorWrapper,
    Tokenizer,
};

const QWEN_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub(super) fn supported(tokenizer: &Tokenizer) -> bool {
    // NFC cannot compose across the newline before Question/Rate. The exact
    // pre-tokenizer below also ends a piece there, so BPE cannot merge across it.
    // Keep arbitrary normalizers, regexes and post-processors on full encoding.
    matches!(tokenizer.get_model(), ModelWrapper::BPE(model) if model.dropout.is_none_or(|p| p == 0.0))
        && matches!(tokenizer.get_normalizer(), None | Some(NormalizerWrapper::NFC(_)))
        && tokenizer.get_post_processor().is_none_or(|processor| matches!(processor, PostProcessorWrapper::ByteLevel(level) if !level.trim_offsets))
        && tokenizer.get_truncation().is_none()
        && tokenizer.get_padding().is_none()
        && tokenizer.get_added_vocabulary().get_added_tokens_decoder().values().all(|token| {
            !token.content.contains('\n') && !token.lstrip && !token.rstrip && !token.single_word
        })
        && serde_json::to_value(tokenizer.get_pre_tokenizer()).is_ok_and(|value| value == serde_json::json!({
            "type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": QWEN_SPLIT}, "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}
            ]
        }))
}

pub(super) struct PromptTokenizer<'a> {
    tokenizer: &'a Tokenizer,
    reuse_prefix: bool,
    prefix: Option<(String, Vec<u32>)>,
}

impl<'a> PromptTokenizer<'a> {
    pub(super) fn new(tokenizer: &'a Tokenizer, reuse_prefix: bool) -> Self {
        Self {
            tokenizer,
            reuse_prefix,
            prefix: None,
        }
    }

    pub(super) fn encode(
        &mut self,
        prompt: &str,
        boundary: Option<usize>,
    ) -> Result<(Vec<u32>, usize), String> {
        // Render the complete prompt for every question: templates may inspect
        // or transform content. Compare actual bytes, never a state hash alone.
        let split = boundary
            .and_then(|boundary| Some((prompt.get(..boundary)?, prompt.get(boundary..)?)))
            .filter(|(prefix, suffix)| {
                self.reuse_prefix
                    && prefix.ends_with("\n</state>\n\n")
                    && (suffix.starts_with("Question: ") || suffix.starts_with("Rate the state: "))
            });
        if let (Some((prefix, suffix)), Some((cached, ids))) = (split, &self.prefix) {
            if prefix == cached {
                let tail = self
                    .tokenizer
                    .encode(suffix, false)
                    .map_err(|e| e.to_string())?;
                let mut output = Vec::with_capacity(ids.len() + tail.len());
                output.extend_from_slice(ids);
                output.extend_from_slice(tail.get_ids());
                return Ok((output, ids.len()));
            }
        }
        let encoded = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| e.to_string())?;
        let prefix_length = boundary.map_or(0, |boundary| {
            encoded
                .get_offsets()
                .iter()
                .take_while(|(_, end)| *end <= boundary)
                .count()
        });
        if let Some((prefix, _)) = split {
            // Seed from a full encoding, including its exact byte boundary.
            if prefix_length > 0
                && prefix_length < encoded.len()
                && encoded
                    .get_offsets()
                    .get(prefix_length - 1)
                    .map(|(_, end)| *end)
                    == boundary
            {
                if let Some(ids) = encoded.get_ids().get(..prefix_length) {
                    self.prefix = Some((prefix.to_owned(), ids.to_vec()));
                }
            }
        }
        Ok((encoded.get_ids().to_vec(), prefix_length))
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use serde_json::json;
    use tokenizers::AddedToken;

    pub(in super::super) fn tokenizer() -> Tokenizer {
        let mut alphabet: Vec<_> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
            .into_iter()
            .collect();
        alphabet.sort_unstable();
        let mut vocab: serde_json::Map<String, serde_json::Value> = alphabet
            .into_iter()
            .enumerate()
            .map(|(id, c)| (c.to_string(), json!(id)))
            .collect();
        // Include a merge that WOULD cross the block boundary without the split.
        for token in ["ĊĊ", "ĊĊQ", "Qu"] {
            vocab.insert(token.into(), json!(vocab.len()));
        }
        let mut tokenizer = Tokenizer::from_bytes(serde_json::to_vec(&json!({
            "normalizer": {"type":"NFC"},
            "pre_tokenizer": {"type":"Sequence", "pretokenizers":[
                {"type":"Split", "pattern":{"Regex":QWEN_SPLIT}, "behavior":"Isolated", "invert":false},
                {"type":"ByteLevel", "add_prefix_space":false, "trim_offsets":true, "use_regex":false}
            ]},
            "post_processor": {"type":"ByteLevel", "add_prefix_space":false, "trim_offsets":false, "use_regex":false},
            "model":{"type":"BPE", "vocab":vocab, "merges":[["Ċ","Ċ"],["ĊĊ","Q"],["Q","u"]]}
        })).unwrap()).unwrap();
        tokenizer.add_special_tokens(
            &[
                "<|im_start|>",
                "<|im_end|>",
                "<|vision_start|>",
                "<|image_pad|>",
                "<|vision_end|>",
            ]
            .map(|s| AddedToken::from(s, true)),
        );
        tokenizer
    }

    #[test]
    fn shared_prefix_matches_full_encoding_with_unicode_images_and_changed_templates() {
        let tokenizer = tokenizer();
        assert!(supported(&tokenizer));
        let mut cached = PromptTokenizer::new(&tokenizer, true);
        for state in [
            "cafe\u{301} 東京 👩🏽‍💻\r\n\n  ",
            "<|vision_start|><|image_pad|><|vision_end|>",
            "",
            "Question: duplicated suffix",
        ] {
            for template_prefix in ["<|im_start|>user\n", "Changed template: "] {
                for suffix in [
                    "Question: yes?",
                    "Rate the state: 0–5?",
                    "Question: <|im_end|> trailing",
                ] {
                    let prefix = format!("{template_prefix}<state>\n{state}\n</state>\n\n");
                    let prompt = format!("{prefix}{suffix}<|im_end|>\n");
                    let expected = tokenizer.encode(prompt.as_str(), false).unwrap();
                    let (ids, boundary) = cached.encode(&prompt, Some(prefix.len())).unwrap();
                    assert_eq!(ids, expected.get_ids());
                    assert_eq!(
                        boundary,
                        expected
                            .get_offsets()
                            .iter()
                            .take_while(|(_, end)| *end <= prefix.len())
                            .count()
                    );
                    assert!(
                        cached.prefix.is_some(),
                        "exercise reuse after the first question"
                    );
                }
            }
        }
    }

    #[test]
    fn unsupported_tokenizers_keep_cross_boundary_tokens_and_offsets() {
        let base = serde_json::to_value(tokenizer()).unwrap();
        let mut variants = Vec::new();
        for field in ["normalizer", "pre_tokenizer", "post_processor"] {
            let mut value = base.clone();
            value[field] = match field {
                "normalizer" => json!({"type":"Lowercase"}),
                "pre_tokenizer" => {
                    json!({"type":"ByteLevel", "add_prefix_space":false, "trim_offsets":true, "use_regex":false})
                }
                _ => {
                    json!({"type":"ByteLevel", "add_prefix_space":false, "trim_offsets":true, "use_regex":false})
                }
            };
            variants.push(Tokenizer::from_bytes(serde_json::to_vec(&value).unwrap()).unwrap());
        }
        for token in [
            AddedToken::from("\n\nQuestion", true),
            AddedToken::from("Question", true).lstrip(true),
            AddedToken::from("state>", true).rstrip(true),
        ] {
            let mut tokenizer = tokenizer();
            tokenizer.add_special_tokens(&[token]);
            variants.push(tokenizer);
        }
        let prefix = "<state>\ncafe\u{301}\n</state>\n\n";
        let mut crosses_boundary = false;
        for tokenizer in variants {
            assert!(!supported(&tokenizer));
            let mut cached = PromptTokenizer::new(&tokenizer, supported(&tokenizer));
            for suffix in ["Question: one?", "Question: two?", "Rate the state: three?"] {
                let prompt = format!("{prefix}{suffix}");
                let expected = tokenizer.encode(prompt.as_str(), false).unwrap();
                crosses_boundary |= expected
                    .get_offsets()
                    .iter()
                    .any(|(start, end)| *start < prefix.len() && *end > prefix.len());
                let (ids, boundary) = cached.encode(&prompt, Some(prefix.len())).unwrap();
                assert_eq!(ids, expected.get_ids());
                assert_eq!(
                    boundary,
                    expected
                        .get_offsets()
                        .iter()
                        .take_while(|(_, end)| *end <= prefix.len())
                        .count()
                );
                assert!(cached.prefix.is_none());
            }
        }
        assert!(
            crosses_boundary,
            "cover a real token spanning state and question"
        );
    }
}

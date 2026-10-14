//! BPE ID encoding for embeddings. HF remains responsible for request-specific
//! truncation and post-processing, and for endpoints requiring source alignment.
use tokenizers::{Encoding, Tokenizer};

pub(crate) struct FastTokenizer {
    tokenizer: basetenkenizer::Tokenizer,
}

pub(crate) fn load(tokenizer: &Tokenizer) -> Option<FastTokenizer> {
    if !matches!(
        tokenizer.get_model(),
        tokenizers::models::ModelWrapper::BPE(_)
    ) {
        return None;
    }
    // Serialize the configured tokenizer, including any server-added processors.
    // Unsupported models/normalizers simply retain the existing implementation.
    let result = serde_json::to_value(tokenizer)
        .map_err(|e| e.to_string())
        .and_then(|mut json| {
            // Keep configurations with specialized BPE/added-token semantics on
            // HF until their ID parity has been verified for this backend.
            if ["continuing_subword_prefix", "end_of_word_suffix"]
                .iter()
                .any(|key| json["model"][key].as_str().is_some_and(|s| !s.is_empty()))
                || json["model"]["dropout"].as_f64().is_some_and(|p| p > 0.0)
                || json["added_tokens"].as_array().is_some_and(|tokens| {
                    tokens.iter().any(|token| {
                        token["single_word"].as_bool() == Some(true)
                            || token["rstrip"].as_bool() == Some(true)
                    })
                })
            {
                return Err("Tokenizer semantics require Hugging Face encoding".into());
            }
            // HF applies the configured processor after request-specific truncation.
            // Disable it here so native encoding returns only content IDs.
            json["post_processor"] = serde_json::Value::Null;
            let tokenizer =
                basetenkenizer::Tokenizer::from_json(json).map_err(|e| e.to_string())?;
            Ok(FastTokenizer { tokenizer })
        });
    match result {
        Ok(tokenizer) => {
            tracing::info!("Using Basetenkenizer BPE encoding for embedding text inputs");
            Some(tokenizer)
        }
        Err(err) => {
            tracing::debug!("Using Hugging Face tokenizer: {err}");
            None
        }
    }
}

pub(crate) fn encode(
    fast: &FastTokenizer,
    tokenizer: &Tokenizer,
    text: &str,
    add_special_tokens: bool,
) -> tokenizers::Result<Encoding> {
    // Do not truncate final IDs: HF reserves space for special tokens before
    // truncating content, and applies configured type IDs and post-processors.
    // Basetenkenizer 0.2.9 owns a bounded BPE pool for parallel work. Calling
    // directly keeps short inputs on this tokenizer worker and avoids a second hop.
    let ids = fast.tokenizer.encode(text)?;
    let len = ids.len();
    let raw = Encoding::new(
        ids,
        vec![0; len],
        vec![String::new(); len],
        vec![None; len],
        vec![(0, 0); len],
        vec![0; len],
        vec![1; len],
        Vec::new(),
        Default::default(),
    );
    // Alignment placeholders only support HF's Encoding transformations. The
    // caller discards these fields; NER and /tokenize never enter this path.
    tokenizer.post_process(raw, None, add_special_tokens)
}

#[cfg(test)]
pub(crate) fn test_tokenizer() -> Tokenizer {
    let mut alphabet: Vec<_> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
        .into_iter()
        .collect();
    alphabet.sort_unstable();
    let vocab: serde_json::Map<String, serde_json::Value> = alphabet
        .into_iter()
        .enumerate()
        .map(|(i, c)| (c.to_string(), serde_json::json!(i)))
        .collect();
    let json = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null,
        "added_tokens": [], "normalizer": null,
        "pre_tokenizer": {"type":"ByteLevel", "add_prefix_space":false, "trim_offsets":true, "use_regex":true},
        "post_processor": null, "decoder": {"type":"ByteLevel", "add_prefix_space":false, "trim_offsets":true, "use_regex":true},
        "model": {"type":"BPE", "vocab":vocab, "merges":[], "dropout":null,
            "unk_token":null, "continuing_subword_prefix":null, "end_of_word_suffix":null,
            "fuse_unk":false, "byte_fallback":false, "ignore_merges":false}
    });
    Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::processors::template::TemplateProcessing;
    use tokenizers::{AddedToken, TruncationDirection, TruncationParams};

    #[test]
    fn preserves_postprocessing_and_truncation() {
        let mut tokenizer = test_tokenizer();
        tokenizer.add_special_tokens(&[
            AddedToken::from("<s>", true),
            AddedToken::from("</s>", true),
        ]);
        let fast = load(&tokenizer).unwrap();
        // A processor installed after Basetenkenizer initialization must still apply.
        tokenizer.with_post_processor(Some(
            TemplateProcessing::builder()
                .try_single("<s>:2 $A:3 </s>:2")
                .unwrap()
                .special_tokens(vec![("<s>", 256), ("</s>", 257)])
                .build()
                .unwrap(),
        ));
        // Loading with an existing processor must also avoid applying it twice.
        let fast_with_processor = load(&tokenizer).unwrap();
        for direction in [TruncationDirection::Left, TruncationDirection::Right] {
            for limit in [2, 3, 8, 128] {
                tokenizer
                    .with_truncation(Some(TruncationParams {
                        max_length: limit,
                        direction,
                        ..Default::default()
                    }))
                    .unwrap();
                for text in ["hello world", " Café 東京 👩🏽‍💻", "<s> hello </s>", ""] {
                    for special in [true, false] {
                        let expected = tokenizer.encode(text, special).unwrap();
                        let actual = encode(&fast, &tokenizer, text, special).unwrap();
                        assert_eq!(
                            actual.get_ids(),
                            expected.get_ids(),
                            "{text:?}, {direction:?}, {limit}"
                        );
                        let preconfigured =
                            encode(&fast_with_processor, &tokenizer, text, special).unwrap();
                        assert_eq!(preconfigured.get_ids(), expected.get_ids());
                        assert_eq!(preconfigured.get_type_ids(), expected.get_type_ids());
                        assert_eq!(actual.get_type_ids(), expected.get_type_ids());
                        assert_eq!(actual.get_attention_mask(), expected.get_attention_mask());
                        assert_eq!(
                            actual.get_special_tokens_mask(),
                            expected.get_special_tokens_mask()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn unsupported_models_and_dropout_use_hf() {
        let wordpiece = Tokenizer::new(tokenizers::models::wordpiece::WordPiece::default());
        assert!(load(&wordpiece).is_none());
        let mut json = serde_json::to_value(test_tokenizer()).unwrap();
        json["model"]["dropout"] = serde_json::json!(0.1);
        let tokenizer = Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap();
        assert!(load(&tokenizer).is_none());
    }
    #[test]
    fn specialized_semantics_use_hf() {
        let base = serde_json::json!({
            "version":"1.0", "truncation":null, "padding":null,
            "added_tokens":[], "normalizer":null, "pre_tokenizer":null,
            "post_processor":null, "decoder":null,
            "model":{"type":"BPE", "vocab":{"a":0,"b":1,"##b":2,"b</w>":3,"a</w>":4,"x":5," ":6},"merges":[]}
        });
        for (kind, input) in [
            ("single_word", "xab"),
            ("rstrip", "ab b"),
            ("continuing_subword_prefix", "ab"),
            ("end_of_word_suffix", "ab"),
        ] {
            let mut json = base.clone();
            if kind == "single_word" || kind == "rstrip" {
                json["added_tokens"] = serde_json::json!([{
                    "id":7,"content":"ab","single_word":kind == "single_word",
                    "lstrip":false,"rstrip":kind == "rstrip","normalized":false,"special":false
                }]);
            } else {
                json["model"][kind] = serde_json::json!(if kind == "continuing_subword_prefix" {
                    "##"
                } else {
                    "</w>"
                });
            }
            let hf = Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap();
            hf.encode(input, false).unwrap();
            assert!(load(&hf).is_none(), "{kind}");
        }
    }

    #[test]
    fn normalized_added_tokens_remain_accelerated() {
        let json = serde_json::json!({
            "version":"1.0", "truncation":null, "padding":null,
            "added_tokens":[{"id":1,"content":"café","single_word":false,"lstrip":false,
                             "rstrip":false,"normalized":true,"special":false}],
            "normalizer":{"type":"NFC"}, "pre_tokenizer":null,
            "post_processor":null, "decoder":null,
            "model":{"type":"BPE", "vocab":{"a":0},"merges":[]}
        });
        let hf = Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap();
        let input = "cafe\u{301}";
        let expected = hf.encode(input, false).unwrap();
        let fast = load(&hf).unwrap();
        let actual = encode(&fast, &hf, input, false).unwrap();
        assert_eq!(actual.get_ids(), expected.get_ids());
        assert_eq!(actual.get_ids(), &[1]);
    }
}

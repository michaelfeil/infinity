// Copyright 2026 fastokens contributors
// Copyright 2026 Baseten
// SPDX-License-Identifier: Apache-2.0
//
// Derived from Crusoe fastokens and modified by Baseten.

use crate::pre_tokenizers::byte_level::BYTE_TO_CHAR;

/// Reverse mapping: Unicode char → original byte value.
///
/// All chars in `BYTE_TO_CHAR` are in the range U+0000..U+0143 (max codepoint
/// 323), so a flat 324-element array gives O(1) lookup.
const CHAR_TO_BYTE: [Option<u8>; 324] = build_char_to_byte();

const fn build_char_to_byte() -> [Option<u8>; 324] {
    let mut table = [None; 324];
    let mut i = 0u16;
    while i < 256 {
        let ch = BYTE_TO_CHAR[i as usize];
        table[ch as usize] = Some(i as u8);
        i += 1;
    }
    table
}

/// ByteLevel decoder: reverses the GPT-2 byte-to-unicode mapping.
#[derive(Debug)]
pub struct ByteLevelDecoder;

impl ByteLevelDecoder {
    /// Apply byte-level decoding to a list of token strings.
    ///
    /// Maps each token through the reverse GPT-2 table, preserving tokens
    /// containing an unmapped character verbatim, then interprets the bytes as UTF-8.
    pub fn decode_chain(&self, tokens: Vec<String>) -> Vec<String> {
        let mut bytes = Vec::with_capacity(tokens.iter().map(String::len).sum());
        for token in tokens {
            let start = bytes.len();
            for c in token.chars() {
                match CHAR_TO_BYTE.get(c as usize).copied().flatten() {
                    Some(byte) => bytes.push(byte),
                    None => {
                        // Added tokens can contain literal spaces or other chars
                        // outside the byte alphabet. HF preserves the entire token
                        // in that case, including any mapped chars before this one.
                        bytes.truncate(start);
                        bytes.extend_from_slice(token.as_bytes());
                        break;
                    }
                }
            }
        }
        vec![String::from_utf8_lossy(&bytes).into_owned()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_ascii() {
        let dec = ByteLevelDecoder;
        let result = dec.decode_chain(vec!["Hello".to_string()]);
        assert_eq!(result, vec!["Hello"]);
    }

    #[test]
    fn roundtrip_space() {
        let dec = ByteLevelDecoder;
        // GPT-2 maps space (0x20) to Ġ (U+0120)
        let result = dec.decode_chain(vec!["\u{120}Hello".to_string()]);
        assert_eq!(result, vec![" Hello"]);
    }

    #[test]
    fn roundtrip_multibyte() {
        let dec = ByteLevelDecoder;
        // Euro sign €: UTF-8 bytes [0xE2, 0x82, 0xAC]
        // BYTE_TO_CHAR maps these to specific unicode chars
        let encoded: String = [0xE2u8, 0x82, 0xAC]
            .iter()
            .map(|&b| BYTE_TO_CHAR[b as usize])
            .collect();
        let result = dec.decode_chain(vec![encoded]);
        assert_eq!(result, vec!["€"]);
    }

    #[test]
    fn non_gpt2_chars_preserved() {
        let dec = ByteLevelDecoder;
        // DeepSeek uses ｜ (U+FF5C) and ▁ (U+2581) in token strings
        // like <｜begin▁of▁sentence｜>. These are outside the GPT-2
        // byte-to-char table and must pass through unchanged.
        let result = dec.decode_chain(vec![
            "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>".to_string(),
        ]);
        assert_eq!(result, vec!["<｜begin▁of▁sentence｜>"]);
    }
}

#[cfg(test)]
mod parity_tests {
    use super::*;
    use tokenizers::Decoder;

    #[test]
    fn literal_added_tokens_match_huggingface() {
        let cases = [
            vec!["  ", "spaces"],
            vec!["def", "Ċ", "    ", "return"],
            vec!["Ġ literal", "Ġmapped"],
            vec!["Ġ👩🏽‍💻", "Ġnext"],
            vec!["\0", "\t", "\n", "\u{7f}"],
            vec!["Ā"], // Mapped zero byte must remain distinct from an unmapped char.
        ];
        for case in cases {
            let tokens: Vec<String> = case.into_iter().map(str::to_owned).collect();
            let expected = tokenizers::pre_tokenizers::byte_level::ByteLevel::default()
                .decode_chain(tokens.clone())
                .unwrap();
            assert_eq!(ByteLevelDecoder.decode_chain(tokens), expected);
        }
    }

    #[test]
    fn complete_byte_alphabet_matches_huggingface() {
        let tokens: Vec<String> = BYTE_TO_CHAR.iter().map(char::to_string).collect();
        let expected = tokenizers::pre_tokenizers::byte_level::ByteLevel::default()
            .decode_chain(tokens.clone())
            .unwrap();
        assert_eq!(ByteLevelDecoder.decode_chain(tokens), expected);
    }
}

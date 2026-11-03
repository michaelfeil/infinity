// Copyright 2026 Baseten
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Map, Value};

/// Return baseline special-token variables for standalone template rendering.
///
/// Python callers can override these with tokenizer-specific values. Tokens
/// default to null and are omitted from the template context so absent tokenizer
/// metadata behaves like Transformers' undefined variables.
pub fn default_special_tokens() -> Map<String, Value> {
    Map::from_iter([
        ("bos_token".to_string(), Value::Null),
        ("eos_token".to_string(), Value::Null),
        ("unk_token".to_string(), Value::Null),
        ("pad_token".to_string(), Value::Null),
    ])
}

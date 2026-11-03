// Copyright 2026 Baseten
// SPDX-License-Identifier: Apache-2.0

mod runtime;
mod special_tokens;

use minijinja::Error as MiniJinjaError;
use serde_json::{Map, Value};

pub use runtime::ChatTemplateRenderer;
pub use special_tokens::default_special_tokens;

// Re-exported so bindings can deserialize directly into the same
// `minijinja::Value` the renderer consumes (see `ChatTemplateRenderer::render_value`)
// without depending on a possibly-mismatched minijinja version themselves.
pub use minijinja;

/// Options for rendering a HuggingFace-style chat template.
#[derive(Clone, Debug, Default)]
pub struct ChatTemplateOptions {
    pub add_generation_prompt: bool,
    pub continue_final_message: bool,
    pub tools: Option<Value>,
    pub documents: Option<Value>,
    pub special_tokens: Map<String, Value>,
    pub extra_context: Map<String, Value>,
}

/// Render a HuggingFace-style Jinja chat template.
///
/// This covers the hot serving path for `apply_chat_template(..., tokenize=False)`.
/// Python bindings can tokenize the rendered string directly afterward, avoiding
/// the Python-side template loop while keeping the tokenizer API surface small.
pub fn apply_chat_template(
    chat_template: &str,
    messages: Value,
    options: ChatTemplateOptions,
) -> Result<String, MiniJinjaError> {
    ChatTemplateRenderer::new(chat_template)?.render(messages, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    #[test]
    fn renders_common_hf_style_template() {
        let template = r#"
{%- for message in messages -%}
{%- if message['role'] == 'user' -%}
{{ bos_token }}user: {{ message['content'] }}{{ eos_token }}
{%- elif message['role'] == 'assistant' -%}
assistant: {{ message.get('content') }}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}assistant:{%- endif -%}
"#;
        let messages = json!([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"}
        ]);
        let special_tokens = Map::from_iter([
            ("bos_token".to_string(), json!("[BOS]")),
            ("eos_token".to_string(), json!("[EOS]")),
        ]);
        let options = ChatTemplateOptions {
            add_generation_prompt: true,
            special_tokens,
            ..Default::default()
        };

        let rendered = apply_chat_template(template, messages, options).unwrap();

        assert_eq!(rendered, "[BOS]user: hello[EOS]assistant: hiassistant:");
    }

    #[test]
    fn omits_null_special_tokens_from_context() {
        let mut special_tokens = default_special_tokens();
        special_tokens.insert("custom_token".to_string(), Value::Null);

        let rendered = apply_chat_template(
            "{% if pad_token is defined %}defined{% else %}undefined{% endif %}{{ pad_token }}",
            json!([]),
            ChatTemplateOptions {
                special_tokens,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(rendered, "undefined");
    }

    #[test]
    fn supports_tools_documents_tojson_and_kwargs() {
        let template = r#"{{ tools[0].name }} {{ documents[0].title }} {{ custom | tojson }}"#;
        let mut extra = Map::new();
        extra.insert("custom".to_string(), json!({"x": 1}));
        let options = ChatTemplateOptions {
            tools: Some(json!([{"name": "search"}])),
            documents: Some(json!([{"title": "doc"}])),
            extra_context: extra,
            ..Default::default()
        };

        let rendered = apply_chat_template(template, json!([]), options).unwrap();

        assert_eq!(rendered, r#"search doc {"x": 1}"#);
    }

    #[test]
    fn supports_python_compat_methods() {
        let template = r#"{{ message.get('missing', 'fallback') }} {{ 'abc'.upper() }} {{ [1, 1, 2].count(1) }}"#;
        let mut extra = Map::new();
        extra.insert("message".to_string(), json!({"content": "hello"}));

        let rendered = apply_chat_template(
            template,
            json!([]),
            ChatTemplateOptions {
                extra_context: extra,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(rendered, "fallback ABC 2");
    }

    #[test]
    fn tojson_matches_transformers_defaults() {
        let template = r#"{{ value | tojson }}"#;
        let mut extra = Map::new();
        extra.insert("value".to_string(), json!({"text": "<tag>&'"}));

        let rendered = apply_chat_template(
            template,
            json!([]),
            ChatTemplateOptions {
                extra_context: extra,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(rendered, r#"{"text": "<tag>&'"}"#);
    }

    #[test]
    fn tojson_supports_compact_separators_and_ascii_escape() {
        let template = r#"{{ value | tojson(separators=(',', ':'), ensure_ascii=True) }}"#;
        let mut extra = Map::new();
        extra.insert("value".to_string(), json!({"text": "é"}));

        let rendered = apply_chat_template(
            template,
            json!([]),
            ChatTemplateOptions {
                extra_context: extra,
                ..Default::default()
            },
        )
        .unwrap();

        assert_eq!(rendered, r#"{"text":"\u00e9"}"#);
    }

    #[test]
    fn strips_generation_markers() {
        let rendered = apply_chat_template(
            "a{% generation %}b{% endgeneration %}c",
            json!([]),
            ChatTemplateOptions::default(),
        )
        .unwrap();

        assert_eq!(rendered, "abc");
    }

    #[test]
    fn rewrites_generation_markers_with_whitespace_control() {
        let rendered = apply_chat_template(
            "a\n{%- generation -%}\nb\n{%- endgeneration -%}\nc",
            json!([]),
            ChatTemplateOptions::default(),
        )
        .unwrap();

        assert_eq!(rendered, "abc");
    }

    #[test]
    fn tojson_indent_matches_json_dumps() {
        let mut extra = Map::new();
        extra.insert("value".to_string(), json!({"b": [1, 2], "a": {}, "c": "x"}));

        let rendered = apply_chat_template(
            "{{ value | tojson(indent=2) }}",
            json!([]),
            ChatTemplateOptions {
                extra_context: extra,
                ..Default::default()
            },
        )
        .unwrap();

        // json.dumps(value, indent=2)
        assert_eq!(
            rendered,
            "{\n  \"b\": [\n    1,\n    2\n  ],\n  \"a\": {},\n  \"c\": \"x\"\n}"
        );
    }

    #[test]
    fn tojson_indent_honors_separators() {
        let mut extra = Map::new();
        extra.insert("value".to_string(), json!({"b": [1, 2], "a": {}, "c": "x"}));

        let rendered = apply_chat_template(
            "{{ value | tojson(indent=2, separators=(',', ':')) }}",
            json!([]),
            ChatTemplateOptions {
                extra_context: extra,
                ..Default::default()
            },
        )
        .unwrap();

        // json.dumps(value, indent=2, separators=(",", ":"))
        assert_eq!(
            rendered,
            "{\n  \"b\":[\n    1,\n    2\n  ],\n  \"a\":{},\n  \"c\":\"x\"\n}"
        );
    }

    #[test]
    fn tojson_indent_keeps_separator_whitespace_like_json_dumps() {
        let mut extra = Map::new();
        extra.insert("value".to_string(), json!({"b": [1, 2], "a": {}, "c": "x"}));

        let rendered = apply_chat_template(
            "{{ value | tojson(indent=2, separators=(', ', ': '), sort_keys=True) }}",
            json!([]),
            ChatTemplateOptions {
                extra_context: extra,
                ..Default::default()
            },
        )
        .unwrap();

        // json.dumps keeps the item separator verbatim before the newline,
        // trailing space included.
        assert_eq!(
            rendered,
            "{\n  \"a\": {}, \n  \"b\": [\n    1, \n    2\n  ], \n  \"c\": \"x\"\n}"
        );
    }

    #[test]
    fn exposes_undeclared_template_variables() {
        let renderer = ChatTemplateRenderer::new(
            "{% for m in messages %}{{ m.content }}{% endfor %}{% if enable_thinking %}t{% endif %}",
        )
        .unwrap();

        let variables = renderer.undeclared_variables();

        assert!(variables.contains("messages"));
        assert!(variables.contains("enable_thinking"));
        assert!(!variables.contains("m"));
    }

    /// `loop.previtem` / `loop.nextitem` require minijinja's
    /// `adjacent_loop_items` feature, which `default-features = false` drops.
    /// Chat templates use them to group adjacent same-role messages (notably
    /// consecutive tool responses under one header), so without the feature
    /// `loop.nextitem.role` raises "undefined value" and a `loop.previtem`
    /// guard silently takes the wrong branch instead.
    #[test]
    fn adjacent_loop_items_are_available() {
        let template = concat!(
            "{%- for message in messages -%}",
            "{%- if not loop.previtem or loop.previtem.role != message.role -%}",
            "[open {{ message.role }}]",
            "{%- endif -%}",
            "{{ message.content }}",
            "{%- if loop.last or loop.nextitem.role != message.role -%}",
            "[close]",
            "{%- endif -%}",
            "{%- endfor -%}",
        );
        let messages = json!([
            {"role": "user", "content": "q"},
            {"role": "tool", "content": "a"},
            {"role": "tool", "content": "b"},
            {"role": "user", "content": "z"},
        ]);

        let rendered =
            apply_chat_template(template, messages, ChatTemplateOptions::default()).unwrap();

        // The two adjacent tool messages share one [open tool]...[close] group.
        assert_eq!(
            rendered,
            "[open user]q[close][open tool]ab[close][open user]z[close]"
        );
    }

    /// Jinja2 ships `urlencode` as a builtin, so without minijinja's
    /// `urlencode` feature a template using it fails here with "unknown filter"
    /// while rendering fine under Transformers. For strings — the case a chat
    /// template plausibly hits — minijinja matches Jinja2's
    /// `quote(s, safe="/")` byte for byte, including UTF-8 percent-encoding.
    #[test]
    fn urlencode_filter_matches_jinja2_for_strings() {
        for (input, expected) in [
            ("a b/c?d", "a%20b/c%3Fd"),
            ("hello world&x=1", "hello%20world%26x%3D1"),
            ("path/to/file.txt", "path/to/file.txt"),
            ("héllo 中文", "h%C3%A9llo%20%E4%B8%AD%E6%96%87"),
            ("a+b=c%d", "a%2Bb%3Dc%25d"),
        ] {
            let mut extra = Map::new();
            extra.insert("value".to_string(), json!(input));

            let rendered = apply_chat_template(
                "{{ value | urlencode }}",
                json!([]),
                ChatTemplateOptions {
                    extra_context: extra,
                    ..Default::default()
                },
            )
            .unwrap();

            assert_eq!(rendered, expected, "urlencode({input:?})");
        }
    }

    /// Known divergence, pinned deliberately: for a **mapping** minijinja
    /// percent-encodes spaces as `%20`, while Jinja2 builds query strings with
    /// `quote_plus` (`+`). Chat templates do not build query strings, so this is
    /// accepted rather than worked around — pinned here so the difference stays
    /// a documented fact rather than a future surprise.
    #[test]
    fn urlencode_filter_diverges_from_jinja2_for_mappings() {
        let mut extra = Map::new();
        extra.insert("value".to_string(), json!({"q": "a b", "n": 1}));

        let rendered = apply_chat_template(
            "{{ value | urlencode }}",
            json!([]),
            ChatTemplateOptions {
                extra_context: extra,
                ..Default::default()
            },
        )
        .unwrap();

        // Jinja2 would render `q=a+b&n=1`.
        assert_eq!(rendered, "q=a%20b&n=1");
    }

    #[test]
    fn raise_exception_returns_error() {
        let err = apply_chat_template(
            "{{ raise_exception('bad role') }}",
            json!([]),
            ChatTemplateOptions::default(),
        )
        .unwrap_err();

        assert!(err.to_string().contains("bad role"));
    }

    #[test]
    fn strftime_now_invalid_format_returns_error() {
        let err = apply_chat_template(
            "{{ strftime_now('%Q %#z') }}",
            json!([]),
            ChatTemplateOptions::default(),
        )
        .unwrap_err();

        assert!(err.to_string().contains("invalid strftime_now format"));
    }
}

// Copyright 2026 Baseten
// SPDX-License-Identifier: Apache-2.0

use chrono::{DateTime, Local};
use minijinja::{Environment, Error, ErrorKind, Value as MiniJinjaValue, value::Kwargs};
use serde::Serialize;
use serde_json::Value;
use std::collections::HashSet;
use std::{fmt::Write as _, io};

use super::ChatTemplateOptions;

/// Compiled chat-template renderer.
///
/// Reuse this for repeated `apply_chat_template` calls with the same template;
/// constructing a MiniJinja environment and parsing the template is avoidable
/// work on the hot serving path.
pub struct ChatTemplateRenderer {
    env: Environment<'static>,
    undeclared_variables: HashSet<String>,
}

impl ChatTemplateRenderer {
    pub fn new(chat_template: &str) -> Result<Self, Error> {
        let mut env = Environment::new();
        env.set_lstrip_blocks(true);
        env.set_trim_blocks(true);
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);
        env.add_template_owned("chat", strip_known_non_jinja_tags(chat_template))?;
        let undeclared_variables = env.get_template("chat")?.undeclared_variables(false);

        Ok(Self {
            env,
            undeclared_variables,
        })
    }

    /// Top-level variables the template reads from the render context.
    ///
    /// Computed statically from the template AST at compile time, so dynamic
    /// lookups (e.g. `context[name]`) are invisible. Use this to reject
    /// keyword arguments the template can never see, not to require ones it
    /// might.
    pub fn undeclared_variables(&self) -> &HashSet<String> {
        &self.undeclared_variables
    }

    pub fn render(&self, messages: Value, options: ChatTemplateOptions) -> Result<String, Error> {
        self.render_value(json_to_minijinja_value(messages), options)
    }

    /// Render with messages already converted to a MiniJinja value.
    ///
    /// Bindings that can deserialize straight into [`minijinja::Value`] should
    /// prefer this over [`render`](Self::render): it skips the intermediate
    /// `serde_json::Value` tree, which dominates per-call cost on large
    /// conversations.
    pub fn render_value(
        &self,
        messages: MiniJinjaValue,
        options: ChatTemplateOptions,
    ) -> Result<String, Error> {
        self.render_context(build_context(messages, options))
    }

    pub fn render_context(&self, context: MiniJinjaValue) -> Result<String, Error> {
        self.env.get_template("chat")?.render(context)
    }
}

fn build_context(messages: MiniJinjaValue, options: ChatTemplateOptions) -> MiniJinjaValue {
    let mut context_entries =
        Vec::with_capacity(6 + options.special_tokens.len() + options.extra_context.len());
    context_entries.push(("messages".to_string(), messages));
    context_entries.push((
        "add_generation_prompt".to_string(),
        MiniJinjaValue::from(options.add_generation_prompt),
    ));
    context_entries.push((
        "continue_final_message".to_string(),
        MiniJinjaValue::from(options.continue_final_message),
    ));
    context_entries.push((
        "tools".to_string(),
        options
            .tools
            .map(json_to_minijinja_value)
            .unwrap_or_else(|| MiniJinjaValue::from(())),
    ));
    context_entries.push((
        "documents".to_string(),
        options
            .documents
            .map(json_to_minijinja_value)
            .unwrap_or_else(|| MiniJinjaValue::from(())),
    ));
    context_entries.extend(
        options
            .special_tokens
            .into_iter()
            .filter(|(_, value)| !value.is_null())
            .map(|(key, value)| (key, json_to_minijinja_value(value))),
    );
    context_entries.extend(
        options
            .extra_context
            .into_iter()
            .map(|(key, value)| (key, json_to_minijinja_value(value))),
    );
    context_entries.into_iter().collect()
}

fn json_to_minijinja_value(value: Value) -> MiniJinjaValue {
    match value {
        Value::Null => MiniJinjaValue::from(()),
        Value::Bool(value) => MiniJinjaValue::from(value),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                MiniJinjaValue::from(value)
            } else if let Some(value) = value.as_u64() {
                MiniJinjaValue::from(value)
            } else if let Some(value) = value.as_f64() {
                MiniJinjaValue::from(value)
            } else {
                MiniJinjaValue::from(())
            }
        }
        Value::String(value) => MiniJinjaValue::from(value),
        Value::Array(values) => MiniJinjaValue::from(
            values
                .into_iter()
                .map(json_to_minijinja_value)
                .collect::<Vec<_>>(),
        ),
        Value::Object(values) => values
            .into_iter()
            .map(|(key, value)| (key, json_to_minijinja_value(value)))
            .collect(),
    }
}

fn strip_known_non_jinja_tags(template: &str) -> String {
    let mut stripped = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("{%") {
        stripped.push_str(&rest[..start]);
        rest = &rest[start..];
        let Some(end) = rest.find("%}") else {
            stripped.push_str(rest);
            return stripped;
        };
        let tag = &rest[..end + 2];
        let inner = tag
            .strip_prefix("{%")
            .and_then(|tag| tag.strip_suffix("%}"))
            .unwrap_or_default();
        let inner = inner.strip_prefix('-').unwrap_or(inner);
        let inner = inner.strip_suffix('-').unwrap_or(inner).trim();

        match inner {
            "generation" => stripped.push_str(&rewrite_block_tag(tag, "if true")),
            "endgeneration" => stripped.push_str(&rewrite_block_tag(tag, "endif")),
            _ => stripped.push_str(tag),
        }
        rest = &rest[end + 2..];
    }
    stripped.push_str(rest);
    stripped
}

fn rewrite_block_tag(tag: &str, replacement: &str) -> String {
    let left_trim = tag.starts_with("{%-");
    let right_trim = tag.ends_with("-%}");
    format!(
        "{{%{} {replacement} {}%}}",
        if left_trim { "-" } else { "" },
        if right_trim { "-" } else { "" }
    )
}

fn raise_exception(message: String) -> Result<String, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message))
}

fn strftime_now(format: &str) -> Result<MiniJinjaValue, Error> {
    let now: DateTime<Local> = Local::now();
    let mut rendered = String::new();
    write!(&mut rendered, "{}", now.format(format)).map_err(|err| {
        Error::new(ErrorKind::InvalidOperation, "invalid strftime_now format").with_source(err)
    })?;
    Ok(MiniJinjaValue::from_safe_string(rendered))
}

fn tojson(value: MiniJinjaValue, kwargs: Kwargs) -> Result<MiniJinjaValue, Error> {
    let ensure_ascii: Option<bool> = kwargs.get("ensure_ascii")?;
    let indent: Option<usize> = kwargs.get("indent")?;
    let separators: Option<Vec<String>> = kwargs.get("separators")?;
    let sort_keys: Option<bool> = kwargs.get("sort_keys")?;
    kwargs.assert_all_used()?;

    let serialized = if sort_keys.unwrap_or(false) {
        let mut value = serde_json::to_value(&value).map_err(json_error)?;
        sort_json_value(&mut value);
        serialize_json(&value, indent, separators)?
    } else {
        serialize_json(&value, indent, separators)?
    };

    let serialized = if ensure_ascii.unwrap_or(false) {
        escape_non_ascii(&serialized)
    } else {
        serialized
    };

    Ok(MiniJinjaValue::from_safe_string(serialized))
}

fn serialize_json<T: Serialize>(
    value: &T,
    indent: Option<usize>,
    separators: Option<Vec<String>>,
) -> Result<String, Error> {
    let (item_separator, key_separator) = json_separators(separators, indent.is_some())?;
    let formatter = HfJsonFormatter {
        item_separator,
        key_separator,
        indent: indent.map(|width| b" ".repeat(width)),
        current_indent: 0,
        has_value: false,
    };
    let mut buf = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut buf, formatter);
    value.serialize(&mut serializer).map_err(json_error)?;
    String::from_utf8(buf).map_err(|err| {
        Error::new(ErrorKind::BadSerialization, "cannot serialize to JSON").with_source(err)
    })
}

/// Serializer matching Python's `json.dumps` output, including its indented
/// form: separators apply verbatim even when `indent` is set (so a custom
/// item separator keeps its trailing whitespace before the newline, exactly
/// like Python), and empty containers stay `[]`/`{}`.
struct HfJsonFormatter {
    item_separator: Vec<u8>,
    key_separator: Vec<u8>,
    indent: Option<Vec<u8>>,
    current_indent: usize,
    has_value: bool,
}

impl HfJsonFormatter {
    fn write_newline_indent<W>(&self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if let Some(indent) = &self.indent {
            writer.write_all(b"\n")?;
            for _ in 0..self.current_indent {
                writer.write_all(indent)?;
            }
        }
        Ok(())
    }

    fn begin_container<W>(&mut self, writer: &mut W, open: &[u8]) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if self.indent.is_some() {
            self.current_indent += 1;
            self.has_value = false;
        }
        writer.write_all(open)
    }

    fn end_container<W>(&mut self, writer: &mut W, close: &[u8]) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if self.indent.is_some() {
            self.current_indent -= 1;
            if self.has_value {
                self.write_newline_indent(writer)?;
            }
        }
        writer.write_all(close)
    }
}

impl serde_json::ser::Formatter for HfJsonFormatter {
    fn begin_array<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.begin_container(writer, b"[")
    }

    fn end_array<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.end_container(writer, b"]")
    }

    fn begin_object<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.begin_container(writer, b"{")
    }

    fn end_object<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.end_container(writer, b"}")
    }

    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if !first {
            writer.write_all(&self.item_separator)?;
        }
        self.write_newline_indent(writer)
    }

    fn end_array_value<W>(&mut self, _writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.has_value = true;
        Ok(())
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        if !first {
            writer.write_all(&self.item_separator)?;
        }
        self.write_newline_indent(writer)
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        writer.write_all(&self.key_separator)
    }

    fn end_object_value<W>(&mut self, _writer: &mut W) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        self.has_value = true;
        Ok(())
    }
}

fn json_separators(
    separators: Option<Vec<String>>,
    indented: bool,
) -> Result<(Vec<u8>, Vec<u8>), Error> {
    let Some(separators) = separators else {
        // json.dumps defaults: (", ", ": ") normally, (",", ": ") when
        // indent is given (the newline replaces the space after commas).
        let item_separator = if indented {
            b",".to_vec()
        } else {
            b", ".to_vec()
        };
        return Ok((item_separator, b": ".to_vec()));
    };
    if separators.len() != 2 {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "tojson separators must contain exactly two strings",
        ));
    }
    Ok((
        separators[0].as_bytes().to_vec(),
        separators[1].as_bytes().to_vec(),
    ))
}

fn sort_json_value(value: &mut Value) {
    match value {
        Value::Array(values) => {
            for value in values {
                sort_json_value(value);
            }
        }
        Value::Object(map) => {
            let mut entries: Vec<_> = std::mem::take(map).into_iter().collect();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            for (key, mut value) in entries {
                sort_json_value(&mut value);
                map.insert(key, value);
            }
        }
        _ => {}
    }
}

fn json_error(err: serde_json::Error) -> Error {
    Error::new(ErrorKind::BadSerialization, "cannot serialize to JSON").with_source(err)
}

fn escape_non_ascii(serialized: &str) -> String {
    let mut escaped = String::with_capacity(serialized.len());
    for c in serialized.chars() {
        if c.is_ascii() {
            escaped.push(c);
        } else {
            for unit in c.encode_utf16(&mut [0; 2]) {
                write!(&mut escaped, "\\u{unit:04x}").expect("writing to string cannot fail");
            }
        }
    }
    escaped
}

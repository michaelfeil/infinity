// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.

use crate::model::read_json;
use infinity_v3_backend_core::BackendError;
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide qwen3.5: {error}"))
}

const DEFAULT_RMS_NORM_EPS: f64 = 1e-6;
const DEFAULT_ROPE_THETA: f64 = 10_000_000.0;
const DEFAULT_MAX_POSITION_EMBEDDINGS: usize = 32_768;
const DELTA_HEAD_DIM: usize = 128;
const DEFAULT_LINEAR_CONV_KERNEL_DIM: usize = 4;
const CANDIDATE_MAX_POSITION_EMBEDDINGS: usize = 262_144;
const CANDIDATE_RMS_NORM_EPS: f32 = 1e-6;
const SUPPORTED_ARCHITECTURES: [&str; 4] = [
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5ForCausalLM",
    "Qwen3_5TextModel",
    "Qwen3_5Model",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Qwen35LayerType {
    Full,
    Linear,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Qwen35Config {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) rotary_dim: usize,
    pub(crate) vocab_size: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) linear_num_key_heads: usize,
    pub(crate) linear_num_value_heads: usize,
    pub(crate) linear_key_head_dim: usize,
    pub(crate) linear_value_head_dim: usize,
    pub(crate) linear_conv_kernel_dim: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta: f32,
    pub(crate) layer_types: Vec<Qwen35LayerType>,
    pub(crate) attn_output_gate: bool,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct RawRopeParameters {
    rope_type: Option<String>,
    rope_theta: Option<f64>,
    partial_rotary_factor: Option<f64>,
    mrope_interleaved: Option<bool>,
    mrope_section: Option<Vec<usize>>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct RawQwen35Config {
    model_type: Option<String>,
    architectures: Option<Vec<String>>,
    hidden_size: Option<usize>,
    intermediate_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    vocab_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    rms_norm_eps: Option<f64>,
    rope_theta: Option<f64>,
    rope_type: Option<String>,
    partial_rotary_factor: Option<f64>,
    rope_parameters: Option<RawRopeParameters>,
    rope_scaling: Option<Value>,
    layer_types: Option<Vec<String>>,
    attn_output_gate: Option<bool>,
    attention_bias: Option<bool>,
    hidden_act: Option<String>,
    linear_num_key_heads: Option<usize>,
    linear_num_value_heads: Option<usize>,
    linear_key_head_dim: Option<usize>,
    linear_value_head_dim: Option<usize>,
    linear_conv_kernel_dim: Option<usize>,
    num_experts: Option<usize>,
    quantization_config: Option<Value>,
    text_config: Option<Box<RawQwen35Config>>,
}

fn resolve_field<T: PartialEq>(
    nested: Option<&RawQwen35Config>,
    root: &RawQwen35Config,
    field: fn(&RawQwen35Config) -> Option<T>,
    name: &str,
) -> Result<Option<T>, BackendError> {
    match (nested.and_then(field), field(root)) {
        (Some(nested), Some(root)) if nested != root => Err(start(format!(
            "conflicting {name} values across text_config and root"
        ))),
        (Some(nested), _) => Ok(Some(nested)),
        (None, root) => Ok(root),
    }
}

macro_rules! required {
    ($text:expr, $raw:expr, $field:ident) => {
        resolve_field($text, $raw, |config| config.$field, stringify!($field))?
            .ok_or_else(|| start(format!("missing config field {}", stringify!($field))))?
    };
}

macro_rules! optional {
    ($text:expr, $raw:expr, $field:ident, $default:expr) => {
        resolve_field($text, $raw, |config| config.$field, stringify!($field))?.unwrap_or($default)
    };
}

fn merge_values<T, I>(values: I, name: &str) -> Result<Option<T>, BackendError>
where
    T: PartialEq,
    I: IntoIterator<Item = Option<T>>,
{
    let mut selected = None;
    for value in values.into_iter().flatten() {
        if let Some(previous) = selected.as_ref() {
            if previous != &value {
                return Err(start(format!("conflicting {name} selectors")));
            }
        } else {
            selected = Some(value);
        }
    }
    Ok(selected)
}

fn positive_f32(value: f64, name: &str) -> Result<f32, BackendError> {
    let narrowed = value as f32;
    if !value.is_finite() || value <= 0.0 || !narrowed.is_finite() || narrowed <= 0.0 {
        return Err(start(format!(
            "{name} must be finite, positive and representable as f32"
        )));
    }
    Ok(narrowed)
}

fn rotary_width(head_dim: usize, factor: f64) -> Result<usize, BackendError> {
    if !factor.is_finite() || factor <= 0.0 || factor > 1.0 {
        return Err(start("partial_rotary_factor must be within (0, 1]"));
    }
    let scaled = head_dim as f64 * factor;
    let rounded = scaled.round();
    if !rounded.is_finite()
        || (scaled - rounded).abs() > 1e-9
        || rounded <= 0.0
        || rounded > head_dim as f64
    {
        return Err(start(
            "partial_rotary_factor must produce an integral rotary width",
        ));
    }
    let rotary = rounded as usize;
    if !rotary.is_multiple_of(2) {
        return Err(start("rotary_dim must be positive and even"));
    }
    Ok(rotary)
}

fn validate_mrope_sections(sections: &[usize], rotary_dim: usize) -> Result<(), BackendError> {
    if sections.len() != 3 || sections.contains(&0) {
        return Err(start(
            "mrope_section must contain three positive text sections",
        ));
    }
    let total = sections
        .iter()
        .try_fold(0usize, |total, section| total.checked_add(*section))
        .ok_or_else(|| start("mrope_section width overflow"))?;
    if total != rotary_dim / 2 {
        return Err(start(format!(
            "mrope_section sums to {total}, expected {}",
            rotary_dim / 2
        )));
    }
    Ok(())
}

fn active(value: &Option<Value>) -> bool {
    value.as_ref().is_some_and(|value| !value.is_null())
}

impl Qwen35Config {
    pub(crate) fn load(path: &Path) -> Result<Self, BackendError> {
        let raw: RawQwen35Config = read_json(&path.join("config.json"))?;
        Self::from_raw(raw)
    }

    pub(crate) fn q_width(&self) -> usize {
        self.num_attention_heads.saturating_mul(self.head_dim)
    }

    pub(crate) fn kv_width(&self) -> usize {
        self.num_key_value_heads.saturating_mul(self.head_dim)
    }

    pub(crate) fn validate_candidate(&self) -> Result<(), BackendError> {
        self.validate_geometry()?;
        let geometry = (
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.rotary_dim,
            self.linear_num_key_heads,
            self.linear_num_value_heads,
        );
        let elephant = (4096, 12288, 32, 16, 4, 256, 64, 16, 32);
        let mammoth = (5120, 17408, 64, 24, 4, 256, 64, 16, 48);
        if geometry != elephant && geometry != mammoth {
            return Err(start(
                "config is not an initial dense Elephant/Mammoth text candidate",
            ));
        }
        if self.max_position_embeddings != CANDIDATE_MAX_POSITION_EMBEDDINGS
            || self.linear_key_head_dim != DELTA_HEAD_DIM
            || self.linear_value_head_dim != DELTA_HEAD_DIM
            || self.linear_conv_kernel_dim != DEFAULT_LINEAR_CONV_KERNEL_DIM
            || (self.rms_norm_eps - CANDIDATE_RMS_NORM_EPS).abs() > 1e-9
            || !self.attn_output_gate
        {
            return Err(start(
                "initial candidate geometry requires max 262144, 128-wide heads, conv 4, eps 1e-6 and query gate",
            ));
        }
        Ok(())
    }

    fn from_raw(raw: RawQwen35Config) -> Result<Self, BackendError> {
        let text = raw.text_config.as_deref();
        let root_type = raw
            .model_type
            .as_deref()
            .ok_or_else(|| start("missing model_type"))?;
        if root_type.contains("moe") {
            return Err(start(format!(
                "unsupported MoE model_type {root_type}; dense text only"
            )));
        }
        match root_type {
            "qwen3_5" => {
                let nested = text.ok_or_else(|| start("qwen3_5 root requires text_config"))?;
                if nested.text_config.is_some() {
                    return Err(start("nested text_config containers are unsupported"));
                }
            }
            "qwen3_5_text" if text.is_some() => {
                return Err(start("qwen3_5_text root must not also define text_config"))
            }
            "qwen3_5_text" => {}
            other => {
                return Err(start(format!(
                    "unsupported model_type {other}; expected qwen3_5 or qwen3_5_text"
                )))
            }
        }
        let selected_type = text
            .and_then(|config| config.model_type.as_deref())
            .unwrap_or(root_type);
        if selected_type.contains("moe") {
            return Err(start(format!(
                "unsupported selected text MoE model_type {selected_type}"
            )));
        }
        if !matches!(selected_type, "qwen3_5" | "qwen3_5_text") {
            return Err(start(format!(
                "unsupported selected text model_type {selected_type}; dense only"
            )));
        }

        for config in std::iter::once(&raw).chain(text) {
            if config
                .model_type
                .as_deref()
                .is_some_and(|kind| kind.contains("moe"))
            {
                return Err(start("MoE Qwen3.5 text models are unsupported"));
            }
            if config.num_experts.is_some_and(|experts| experts != 0) {
                return Err(start("nonzero expert routing is unsupported"));
            }
            if active(&config.quantization_config) {
                return Err(start("quantized Qwen3.5 checkpoints are unsupported"));
            }
            if config.attention_bias == Some(true) {
                return Err(start("attention_bias checkpoints are unsupported"));
            }
            if config
                .rope_scaling
                .as_ref()
                .is_some_and(|value| !value.is_null())
            {
                return Err(start("scaled RoPE selectors are unsupported"));
            }
            if let Some(architectures) = &config.architectures {
                for architecture in architectures {
                    if !SUPPORTED_ARCHITECTURES.contains(&architecture.as_str()) {
                        return Err(start(format!(
                            "unsupported architecture {architecture}; Qwen3.5 text tasks only"
                        )));
                    }
                }
            }
        }

        let hidden_size = required!(text, &raw, hidden_size);
        let intermediate_size = required!(text, &raw, intermediate_size);
        let num_hidden_layers = required!(text, &raw, num_hidden_layers);
        let num_attention_heads = required!(text, &raw, num_attention_heads);
        let num_key_value_heads = optional!(text, &raw, num_key_value_heads, num_attention_heads);
        let vocab_size = required!(text, &raw, vocab_size);
        let max_position_embeddings = optional!(
            text,
            &raw,
            max_position_embeddings,
            DEFAULT_MAX_POSITION_EMBEDDINGS
        );
        for (name, value) in [
            ("hidden_size", hidden_size),
            ("intermediate_size", intermediate_size),
            ("num_hidden_layers", num_hidden_layers),
            ("num_attention_heads", num_attention_heads),
            ("num_key_value_heads", num_key_value_heads),
            ("vocab_size", vocab_size),
            ("max_position_embeddings", max_position_embeddings),
        ] {
            if value == 0 {
                return Err(start(format!("config {name} must be nonzero")));
            }
        }
        let head_dim = match resolve_field(text, &raw, |config| config.head_dim, "head_dim")? {
            Some(0) => return Err(start("config head_dim must be nonzero")),
            Some(head_dim) => head_dim,
            None if hidden_size.is_multiple_of(num_attention_heads) => {
                hidden_size / num_attention_heads
            }
            None => return Err(start(
                "hidden_size must divide evenly by num_attention_heads when head_dim is omitted",
            )),
        };

        let raw_layer_types = resolve_field(
            text,
            &raw,
            |config| config.layer_types.clone(),
            "layer_types",
        )?
        .ok_or_else(|| start("missing config field layer_types"))?;
        if raw_layer_types.len() != num_hidden_layers {
            return Err(start(format!(
                "layer_types has {} entries for {num_hidden_layers} layers",
                raw_layer_types.len()
            )));
        }
        let mut layer_types = Vec::new();
        layer_types
            .try_reserve_exact(raw_layer_types.len())
            .map_err(start)?;
        for (index, kind) in raw_layer_types.iter().enumerate() {
            layer_types.push(match kind.as_str() {
                "full_attention" => Qwen35LayerType::Full,
                "linear_attention" => Qwen35LayerType::Linear,
                other => {
                    return Err(start(format!(
                        "unsupported layer_types[{index}] {other}; expected full_attention or linear_attention"
                    )))
                }
            });
        }

        let attn_output_gate = optional!(text, &raw, attn_output_gate, true);
        let hidden_act =
            resolve_field(text, &raw, |config| config.hidden_act.clone(), "hidden_act")?;
        if hidden_act.as_deref().is_some_and(|act| act != "silu") {
            return Err(start("only dense SiLU text activation is supported"));
        }

        let rope_nested = text.and_then(|config| config.rope_parameters.as_ref());
        let rope_root = raw.rope_parameters.as_ref();
        let rope_type = merge_values(
            [
                rope_nested.and_then(|rope| rope.rope_type.clone()),
                rope_root.and_then(|rope| rope.rope_type.clone()),
                text.and_then(|config| config.rope_type.clone()),
                raw.rope_type.clone(),
            ],
            "rope_type",
        )?;
        if rope_type.as_deref().is_some_and(|kind| kind != "default") {
            return Err(start("only default rope_type is supported"));
        }
        let rope_theta = merge_values(
            [
                rope_nested.and_then(|rope| rope.rope_theta),
                rope_root.and_then(|rope| rope.rope_theta),
                text.and_then(|config| config.rope_theta),
                raw.rope_theta,
            ],
            "rope_theta",
        )?
        .unwrap_or(DEFAULT_ROPE_THETA);
        let rope_theta = positive_f32(rope_theta, "rope_theta")?;
        let rms_norm_eps = positive_f32(
            resolve_field(text, &raw, |config| config.rms_norm_eps, "rms_norm_eps")?
                .unwrap_or(DEFAULT_RMS_NORM_EPS),
            "rms_norm_eps",
        )?;
        let partial_rotary_factor = merge_values(
            [
                rope_nested.and_then(|rope| rope.partial_rotary_factor),
                rope_root.and_then(|rope| rope.partial_rotary_factor),
                text.and_then(|config| config.partial_rotary_factor),
                raw.partial_rotary_factor,
            ],
            "partial_rotary_factor",
        )?
        .unwrap_or(1.0);
        let rotary_dim = rotary_width(head_dim, partial_rotary_factor)?;
        let mrope_interleaved = merge_values(
            [
                rope_nested.and_then(|rope| rope.mrope_interleaved),
                rope_root.and_then(|rope| rope.mrope_interleaved),
            ],
            "mrope_interleaved",
        )?
        .unwrap_or(false);
        let mrope_section = merge_values(
            [
                rope_nested.and_then(|rope| rope.mrope_section.clone()),
                rope_root.and_then(|rope| rope.mrope_section.clone()),
            ],
            "mrope_section",
        )?;
        if let Some(sections) = mrope_section.as_deref() {
            validate_mrope_sections(sections, rotary_dim)?;
        } else if mrope_interleaved {
            return Err(start("interleaved mRoPE requires mrope_section"));
        }

        let linear_num_key_heads = optional!(text, &raw, linear_num_key_heads, 0);
        let linear_num_value_heads = optional!(text, &raw, linear_num_value_heads, 0);
        let linear_key_head_dim = optional!(text, &raw, linear_key_head_dim, DELTA_HEAD_DIM);
        let linear_value_head_dim = optional!(text, &raw, linear_value_head_dim, DELTA_HEAD_DIM);
        let linear_conv_kernel_dim = optional!(
            text,
            &raw,
            linear_conv_kernel_dim,
            DEFAULT_LINEAR_CONV_KERNEL_DIM
        );

        let config = Self {
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rotary_dim,
            vocab_size,
            max_position_embeddings,
            linear_num_key_heads,
            linear_num_value_heads,
            linear_key_head_dim,
            linear_value_head_dim,
            linear_conv_kernel_dim,
            rms_norm_eps,
            rope_theta,
            layer_types,
            attn_output_gate,
        };
        config.validate_geometry()?;
        Ok(config)
    }

    fn validate_geometry(&self) -> Result<(), BackendError> {
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("rotary_dim", self.rotary_dim),
            ("vocab_size", self.vocab_size),
            ("max_position_embeddings", self.max_position_embeddings),
        ] {
            if value == 0 {
                return Err(start(format!("config {name} must be nonzero")));
            }
        }
        if u32::try_from(self.max_position_embeddings).is_err() {
            return Err(start("max_position_embeddings must fit CUDA position IDs"));
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(start(format!(
                "layer_types has {} entries for {} layers",
                self.layer_types.len(),
                self.num_hidden_layers
            )));
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(start(
                "num_attention_heads must divide evenly by num_key_value_heads",
            ));
        }
        if self.rotary_dim > self.head_dim || !self.rotary_dim.is_multiple_of(2) {
            return Err(start(
                "rotary_dim must be even, positive and no wider than head_dim",
            ));
        }
        if !self.rms_norm_eps.is_finite() || self.rms_norm_eps <= 0.0 {
            return Err(start("rms_norm_eps must be finite and positive"));
        }
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(start("rope_theta must be finite and positive"));
        }
        self.num_attention_heads
            .checked_mul(self.head_dim)
            .and_then(|width| width.checked_mul(2))
            .ok_or_else(|| start("query-gate width overflow"))?;
        self.num_key_value_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| start("key/value width overflow"))?;
        self.vocab_size
            .checked_mul(self.hidden_size)
            .ok_or_else(|| start("embedding width overflow"))?;
        self.hidden_size
            .checked_mul(self.intermediate_size)
            .ok_or_else(|| start("MLP width overflow"))?;

        let has_full = self.layer_types.contains(&Qwen35LayerType::Full);
        let has_linear = self.layer_types.contains(&Qwen35LayerType::Linear);
        if has_full && !self.attn_output_gate {
            return Err(start(
                "full attention requires the packed query/output gate",
            ));
        }
        if has_linear {
            if self.linear_key_head_dim != DELTA_HEAD_DIM
                || self.linear_value_head_dim != DELTA_HEAD_DIM
            {
                return Err(start(
                    "native Qwen3.5 GDN requires 128-wide key/value heads",
                ));
            }
            for (name, value) in [
                ("linear_num_key_heads", self.linear_num_key_heads),
                ("linear_num_value_heads", self.linear_num_value_heads),
                ("linear_key_head_dim", self.linear_key_head_dim),
                ("linear_value_head_dim", self.linear_value_head_dim),
                ("linear_conv_kernel_dim", self.linear_conv_kernel_dim),
            ] {
                if value == 0 {
                    return Err(start(format!("config {name} must be nonzero")));
                }
            }
            if !self
                .linear_num_value_heads
                .is_multiple_of(self.linear_num_key_heads)
            {
                return Err(start(
                    "linear_num_value_heads must divide evenly by linear_num_key_heads",
                ));
            }
            let key_width = self
                .linear_num_key_heads
                .checked_mul(self.linear_key_head_dim)
                .ok_or_else(|| start("linear key width overflow"))?;
            let value_width = self
                .linear_num_value_heads
                .checked_mul(self.linear_value_head_dim)
                .ok_or_else(|| start("linear value width overflow"))?;
            key_width
                .checked_mul(2)
                .and_then(|width| width.checked_add(value_width))
                .ok_or_else(|| start("linear QKV width overflow"))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    const ELEPHANT: &str = "/workspace/model-performance/michaelfeil/elephant-1";
    const MAMMOTH: &str = "/workspace/model-performance/michaelfeil/onejev-perf/models/mammoth-1";

    struct Fixture(PathBuf);
    impl Fixture {
        fn new(value: Value) -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "infinity-qwen35-config-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            fs::write(
                path.join("config.json"),
                serde_json::to_vec(&value).unwrap(),
            )
            .unwrap();
            Self(path)
        }

        fn load(&self) -> Result<Qwen35Config, BackendError> {
            Qwen35Config::load(&self.0)
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn text_config() -> Value {
        json!({
            "model_type": "qwen3_5_text",
            "architectures": ["Qwen3_5TextModel"],
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 4,
            "num_attention_heads": 5,
            "num_key_value_heads": 1,
            "head_dim": 12,
            "vocab_size": 100,
            "max_position_embeddings": 128,
            "rms_norm_eps": 1e-6,
            "partial_rotary_factor": 0.5,
            "layer_types": [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention"
            ],
            "attn_output_gate": true,
            "attention_bias": false,
            "hidden_act": "silu",
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10_000_000.0,
                "partial_rotary_factor": 0.5,
                "mrope_interleaved": true,
                "mrope_section": [1, 1, 1]
            },
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false
        })
    }

    fn nested_config() -> Value {
        json!({
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "dtype": "float32",
            "text_config": text_config(),
            "tie_word_embeddings": false,
            "vision_config": {"depth": 1, "hidden_size": 8},
            "modeling_overrides": {"head_dim": 999}
        })
    }

    fn candidate_config(
        hidden_size: usize,
        intermediate_size: usize,
        heads: usize,
        kv_heads: usize,
        layers: usize,
        value_heads: usize,
    ) -> Value {
        let layer_types = (0..layers)
            .map(|index| {
                if index % 4 == 3 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect::<Vec<_>>();
        json!({
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "num_hidden_layers": layers,
                "num_attention_heads": heads,
                "num_key_value_heads": kv_heads,
                "head_dim": 256,
                "vocab_size": 248_320,
                "max_position_embeddings": 262_144,
                "rms_norm_eps": 1e-6,
                "partial_rotary_factor": 0.25,
                "layer_types": layer_types,
                "attn_output_gate": true,
                "attention_bias": false,
                "hidden_act": "silu",
                "linear_num_key_heads": 16,
                "linear_num_value_heads": value_heads,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10_000_000.0,
                    "partial_rotary_factor": 0.25,
                    "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10]
                }
            }
        })
    }

    fn elephant_config() -> Value {
        candidate_config(4096, 12288, 16, 4, 32, 32)
    }

    fn mammoth_config() -> Value {
        candidate_config(5120, 17408, 24, 4, 64, 48)
    }

    fn elephant_with(key: &str, value: Value) -> Value {
        let mut config = elephant_config();
        config["text_config"][key] = value;
        config
    }

    fn assert_rejected(value: Value, needle: &str) {
        let fixture = Fixture::new(value);
        let error = match fixture.load() {
            Ok(_) => panic!("invalid qwen3.5 configuration was accepted"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains(needle), "{error}");
    }

    #[test]
    fn loads_nested_dense_text_geometry_and_ignores_vision_and_mtp_metadata() {
        let fixture = Fixture::new(nested_config());
        let config = fixture.load().unwrap();
        assert_eq!(config.hidden_size, 64);
        assert_eq!(config.intermediate_size, 96);
        assert_eq!(config.num_hidden_layers, 4);
        assert_eq!(config.num_attention_heads, 5);
        assert_eq!(config.num_key_value_heads, 1);
        assert_eq!(config.head_dim, 12);
        assert_eq!(config.rotary_dim, 6);
        assert_eq!(config.q_width(), 60);
        assert_eq!(config.kv_width(), 12);
        assert_ne!(config.q_width(), config.hidden_size);
        assert_eq!(config.vocab_size, 100);
        assert_eq!(config.max_position_embeddings, 128);
        assert_eq!(config.linear_num_key_heads, 2);
        assert_eq!(config.linear_num_value_heads, 4);
        assert_eq!(config.linear_key_head_dim, 128);
        assert_eq!(config.linear_value_head_dim, 128);
        assert_eq!(config.linear_conv_kernel_dim, 4);
        assert_eq!(config.rms_norm_eps, 1e-6);
        assert_eq!(config.rope_theta, 10_000_000.0);
        assert!(config.attn_output_gate);
        assert_eq!(
            config.layer_types,
            vec![
                Qwen35LayerType::Linear,
                Qwen35LayerType::Linear,
                Qwen35LayerType::Linear,
                Qwen35LayerType::Full
            ]
        );
    }

    #[test]
    fn accepts_flat_qwen3_5_text_root_and_defaults_kv_heads() {
        let mut flat = text_config();
        {
            let object = flat.as_object_mut().unwrap();
            object.remove("num_key_value_heads");
            object.remove("head_dim");
            object.insert("hidden_size".into(), json!(60));
        }
        let fixture = Fixture::new(flat);
        let config = fixture.load().unwrap();
        assert_eq!(config.num_key_value_heads, 5);
        assert_eq!(config.head_dim, 12);
        assert_eq!(config.q_width(), 60);
        assert_eq!(config.kv_width(), 60);
    }

    #[test]
    fn candidate_gate_accepts_only_the_two_initial_dense_geometries() {
        for (value, hidden, q, kv, layers) in [
            (elephant_config(), 4096, 4096, 1024, 32),
            (mammoth_config(), 5120, 6144, 1024, 64),
        ] {
            let fixture = Fixture::new(value);
            let config = fixture.load().unwrap();
            assert_eq!(config.hidden_size, hidden);
            assert_eq!(config.q_width(), q);
            assert_eq!(config.kv_width(), kv);
            assert_eq!(config.rotary_dim, 64);
            assert_eq!(config.num_hidden_layers, layers);
            config.validate_candidate().unwrap();
        }

        for (value, needle) in [
            (candidate_config(4097, 12288, 16, 4, 32, 32), "candidate"),
            (candidate_config(4096, 12289, 16, 4, 32, 32), "candidate"),
            (candidate_config(4096, 12288, 16, 4, 31, 32), "candidate"),
            (candidate_config(4096, 12288, 16, 4, 32, 64), "candidate"),
            (candidate_config(4096, 12288, 32, 4, 32, 32), "candidate"),
            (
                elephant_with("max_position_embeddings", json!(131_072)),
                "candidate geometry",
            ),
            (elephant_with("linear_key_head_dim", json!(64)), "128"),
            (
                elephant_with("linear_conv_kernel_dim", json!(3)),
                "candidate geometry",
            ),
            (
                elephant_with("rms_norm_eps", json!(1e-5)),
                "candidate geometry",
            ),
        ] {
            let fixture = Fixture::new(value);
            let error = fixture
                .load()
                .and_then(|config| config.validate_candidate())
                .unwrap_err()
                .to_string();
            assert!(error.contains(needle), "{error}");
        }
        assert_rejected(
            elephant_with("attn_output_gate", json!(false)),
            "query/output gate",
        );
    }

    #[test]
    fn actual_elephant_and_mammoth_config_metadata_are_initial_candidates() {
        let mut checked = 0;
        for (path, hidden, q, kv, layers, value_heads) in [
            (ELEPHANT, 4096, 4096, 1024, 32, 32),
            (MAMMOTH, 5120, 6144, 1024, 64, 48),
        ] {
            let directory = Path::new(path);
            if !directory.join("config.json").is_file() {
                eprintln!("actual Qwen3.5 metadata unavailable at {path}");
                continue;
            }
            let config = Qwen35Config::load(directory).unwrap();
            assert_eq!(config.hidden_size, hidden);
            assert_eq!(config.head_dim, 256);
            assert_eq!(config.rotary_dim, 64);
            assert_eq!(config.q_width(), q);
            assert_eq!(config.kv_width(), kv);
            assert_eq!(config.num_hidden_layers, layers);
            assert_eq!(config.layer_types.len(), layers);
            assert_eq!(config.linear_key_head_dim, 128);
            assert_eq!(config.linear_value_head_dim, 128);
            assert_eq!(config.linear_num_key_heads, 16);
            assert_eq!(config.linear_num_value_heads, value_heads);
            assert_eq!(config.linear_conv_kernel_dim, 4);
            assert_eq!(config.rms_norm_eps, 1e-6);
            assert_eq!(config.max_position_embeddings, 262_144);
            assert!(config.attn_output_gate);
            config.validate_candidate().unwrap();
            checked += 1;
        }
        if checked == 0 {
            eprintln!("actual Elephant and Mammoth config metadata were not present");
        }
    }

    #[test]
    fn rejects_moe_quantization_bias_activation_and_task_mismatches() {
        let mut value = nested_config();
        value["model_type"] = json!("qwen3_5_moe");
        assert_rejected(value, "MoE");

        let mut value = nested_config();
        value["text_config"]["model_type"] = json!("qwen3_5_moe_text");
        assert_rejected(value, "MoE");

        let mut value = nested_config();
        value["text_config"]["model_type"] = json!("qwen2");
        assert_rejected(value, "model_type");

        let mut value = nested_config();
        value["architectures"] = json!(["Qwen3_5ForSequenceClassification"]);
        assert_rejected(value, "architecture");

        let mut value = nested_config();
        value["num_experts"] = json!(8);
        assert_rejected(value, "expert");

        let mut value = nested_config();
        value["text_config"]["num_experts"] = json!(8);
        assert_rejected(value, "expert");

        let mut value = nested_config();
        value["quantization_config"] = json!({"quant_method": "fp8"});
        assert_rejected(value, "quantized");

        let mut value = nested_config();
        value["text_config"]["quantization_config"] = json!({"quant_method": "fp8"});
        assert_rejected(value, "quantized");

        let mut value = nested_config();
        value["text_config"]["attention_bias"] = json!(true);
        assert_rejected(value, "attention_bias");

        let mut value = nested_config();
        value["text_config"]["hidden_act"] = json!("relu");
        assert_rejected(value, "SiLU");

        let mut value = nested_config();
        value["text_config"]["rope_parameters"]["rope_type"] = json!("linear");
        assert_rejected(value, "rope_type");

        let mut value = nested_config();
        value["rope_scaling"] = json!({"rope_type": "linear", "factor": 8.0});
        assert_rejected(value, "scaled RoPE");

        assert_rejected(
            {
                let mut value = nested_config();
                value.as_object_mut().unwrap().remove("text_config");
                value
            },
            "text_config",
        );

        let mut flat = text_config();
        flat["text_config"] = text_config();
        assert_rejected(flat, "text_config");

        let mut value = nested_config();
        value["text_config"]["text_config"] = json!({});
        assert_rejected(value, "nested text_config");
    }

    #[test]
    fn rejects_bad_geometry_layers_rope_and_overflow_metadata() {
        for (key, replacement, needle) in [
            ("hidden_size", json!(0), "hidden_size"),
            ("num_hidden_layers", json!(0), "num_hidden_layers"),
            ("num_attention_heads", json!(0), "num_attention_heads"),
            ("head_dim", json!(0), "head_dim"),
            ("vocab_size", json!(0), "vocab_size"),
            (
                "max_position_embeddings",
                json!(0),
                "max_position_embeddings",
            ),
            ("num_key_value_heads", json!(2), "divide evenly"),
            (
                "max_position_embeddings",
                json!((u32::MAX as u64) + 1),
                "CUDA position IDs",
            ),
            (
                "layer_types",
                json!(["linear_attention", "full_attention", "linear_attention"]),
                "entries",
            ),
            (
                "layer_types",
                json!([
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "sliding_attention"
                ]),
                "sliding_attention",
            ),
        ] {
            let mut value = nested_config();
            value["text_config"][key] = replacement;
            assert_rejected(value, needle);
        }

        for (factor, needle) in [(json!(0.25), "even"), (json!(0.3), "integral")] {
            let mut value = nested_config();
            value["text_config"]["partial_rotary_factor"] = factor.clone();
            value["text_config"]["rope_parameters"]["partial_rotary_factor"] = factor;
            assert_rejected(value, needle);
        }

        let mut value = nested_config();
        value["text_config"]["rope_parameters"]["mrope_section"] = json!([1, 1, 2]);
        assert_rejected(value, "mrope_section");

        let mut value = nested_config();
        value["text_config"]["rope_parameters"]["rope_theta"] = json!(0.0);
        assert_rejected(value, "rope_theta");

        let mut value = nested_config();
        value["text_config"]["rms_norm_eps"] = json!(0.0);
        assert_rejected(value, "rms_norm_eps");

        let huge_head_dim = 1usize << 62;
        let mut value = nested_config();
        value["text_config"]["head_dim"] = json!(huge_head_dim);
        value["text_config"]["partial_rotary_factor"] = json!(1.0);
        value["text_config"]["rope_parameters"]["partial_rotary_factor"] = json!(1.0);
        value["text_config"]["rope_parameters"]["mrope_interleaved"] = json!(false);
        value["text_config"]["rope_parameters"]
            .as_object_mut()
            .unwrap()
            .remove("mrope_section");
        assert_rejected(value, "overflow");
    }

    #[test]
    fn rejects_conflicting_nested_rope_and_top_level_selectors() {
        let mut value = nested_config();
        value["rope_theta"] = json!(1_000.0);
        assert_rejected(value, "conflicting rope_theta");

        let mut value = nested_config();
        value["partial_rotary_factor"] = json!(0.25);
        assert_rejected(value, "conflicting partial_rotary_factor");

        let mut value = nested_config();
        value["rope_parameters"] = json!({"rope_theta": 1_000.0});
        assert_rejected(value, "conflicting rope_theta");

        let mut value = nested_config();
        value["text_config"]["rope_theta"] = json!(1_000.0);
        assert_rejected(value, "conflicting rope_theta");

        let mut value = nested_config();
        value["text_config"]["rope_type"] = json!("neox");
        assert_rejected(value, "rope_type");
    }

    #[test]
    fn rejects_nonfinite_scalars_from_typed_raw_configuration() {
        let mut raw: RawQwen35Config = serde_json::from_value(nested_config()).unwrap();
        raw.text_config.as_deref_mut().unwrap().rms_norm_eps = Some(f64::NAN);
        let error = Qwen35Config::from_raw(raw).unwrap_err().to_string();
        assert!(error.contains("rms_norm_eps"), "{error}");

        let mut raw: RawQwen35Config = serde_json::from_value(nested_config()).unwrap();
        raw.text_config
            .as_deref_mut()
            .and_then(|config| config.rope_parameters.as_mut())
            .unwrap()
            .rope_theta = Some(f64::INFINITY);
        let error = Qwen35Config::from_raw(raw).unwrap_err().to_string();
        assert!(error.contains("rope_theta"), "{error}");
    }
}

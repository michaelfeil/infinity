// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//!
//! Typed cold-staging for dense Qwen3.5 text checkpoints.
//!
//! Geometry and tensor names follow the pinned native Qwen3.5 path in
//! `models/qwen35_models.cpp`. This module validates configuration and lazily
//! exposes checkpoint metadata through the shared [`Checkpoint`] /
//! [`WeightSource`] reader. It never reads weight payloads, uploads device
//! weights, builds rotary tables, selects kernels, or claims executor support.
//!
//! The text backbone contains two attention kinds. Full-attention layers pack
//! query and output gate per head as `q_proj` rows `[q | gate]`; the source is
//! intentionally left packed because the executor interprets that layout.
//! Gated-delta layers keep their `a` and `b` projections separate, exactly as
//! the native loader consumes them, so the existing bounded uploader can
//! concatenate them once at upload time. The dense MLP keeps `gate_proj` and
//! `up_proj` separate for the same reason.
//!
//! RMSNorm affine weights stored in Qwen3.5 checkpoints are offsets: the
//! executor applies `1 + weight` in F32. The gated-delta `linear_attn.norm`
//! is an ordinary multiplicative weight with no offset.

use crate::model::{Checkpoint, WeightSource};
use crate::qwen35_config::{Qwen35Config, Qwen35LayerType};
use infinity_v3_backend_core::BackendError;
use std::path::Path;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide qwen3.5: {error}"))
}

const NATIVE_ROOTS: [&str; 4] = ["model.language_model.", "language_model.", "model.", ""];
const DELTA_HEAD_DIM: usize = 128;

#[derive(Clone, Debug)]
// Cold metadata only: keeping the nine lazy views inline avoids individual boxes.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Qwen35AttentionWeights {
    Full {
        query_gate: WeightSource,
        key: WeightSource,
        value: WeightSource,
        output: WeightSource,
        q_norm: WeightSource,
        k_norm: WeightSource,
    },
    Linear {
        qkv: WeightSource,
        z: WeightSource,
        a: WeightSource,
        b: WeightSource,
        output: WeightSource,
        conv: WeightSource,
        a_log: WeightSource,
        dt_bias: WeightSource,
        norm: WeightSource,
    },
}

/// Lazily viewed sources for one Qwen3.5 text layer.
///
/// `input_norm` and `post_attention_norm` are offset RMSNorm weights: the
/// executor applies `1 + weight` in F32. `mlp_gate` and `mlp_up` are kept
/// separate so the existing bounded uploader can concatenate them exactly once.
#[derive(Clone, Debug)]
pub(crate) struct Qwen35LayerWeights {
    pub(crate) input_norm: WeightSource,
    pub(crate) post_attention_norm: WeightSource,
    pub(crate) mlp_gate: WeightSource,
    pub(crate) mlp_up: WeightSource,
    pub(crate) mlp_down: WeightSource,
    pub(crate) attention: Qwen35AttentionWeights,
}

/// Metadata-only Qwen3.5 text checkpoint. Payload bytes stay in the shards and
/// are read later by the executor's bounded uploader.
#[derive(Clone, Debug)]
pub(crate) struct Qwen35Weights {
    pub(crate) config: Qwen35Config,
    pub(crate) embeddings: WeightSource,
    /// Final offset RMSNorm (`1 + weight` applied later in F32).
    pub(crate) norm: WeightSource,
    pub(crate) layers: Vec<Qwen35LayerWeights>,
    /// Separate, top-level `lm_head.weight` view `[V, H]` when the checkpoint
    /// has an untied head. The loader never reads or uploads it.
    pub(crate) lm_head: Option<WeightSource>,
    pub(crate) root: String,
}

impl Qwen35Weights {
    /// Load configuration once, open the checkpoint once, and lazily bind every
    /// dense text tensor. Vision, MTP and other extra tensors are ignored, as
    /// the native loader does; unsupported MoE configurations are rejected.
    pub(crate) fn load(path: &Path) -> Result<Self, BackendError> {
        let config = Qwen35Config::load(path)?;
        let checkpoint = Checkpoint::open(path)?;
        let root = select_root(&checkpoint)?;

        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let q = config.q_width();
        let kv = config.kv_width();
        let head_dim = config.head_dim;
        let query_gate = q
            .checked_mul(2)
            .ok_or_else(|| start("qwen3.5 query-gate width overflow"))?;
        let linear_key_width = config
            .linear_num_key_heads
            .checked_mul(DELTA_HEAD_DIM)
            .ok_or_else(|| start("qwen3.5 linear key width overflow"))?;
        let linear_value_width = config
            .linear_num_value_heads
            .checked_mul(DELTA_HEAD_DIM)
            .ok_or_else(|| start("qwen3.5 linear value width overflow"))?;
        let linear_qkv_width = linear_key_width
            .checked_mul(2)
            .and_then(|width| width.checked_add(linear_value_width))
            .ok_or_else(|| start("qwen3.5 linear QKV width overflow"))?;

        let embeddings = checkpoint.source(
            &format!("{root}embed_tokens.weight"),
            &[config.vocab_size, hidden],
        )?;
        let norm = checkpoint.source(&format!("{root}norm.weight"), &[hidden])?;

        let mut layers = Vec::new();
        layers
            .try_reserve_exact(config.num_hidden_layers)
            .map_err(start)?;
        for (index, kind) in config.layer_types.iter().enumerate() {
            let prefix = format!("{root}layers.{index}");
            let input_norm =
                checkpoint.source(&format!("{prefix}.input_layernorm.weight"), &[hidden])?;
            let post_attention_norm = checkpoint.source(
                &format!("{prefix}.post_attention_layernorm.weight"),
                &[hidden],
            )?;
            let mlp_gate = checkpoint.source(
                &format!("{prefix}.mlp.gate_proj.weight"),
                &[intermediate, hidden],
            )?;
            let mlp_up = checkpoint.source(
                &format!("{prefix}.mlp.up_proj.weight"),
                &[intermediate, hidden],
            )?;
            let mlp_down = checkpoint.source(
                &format!("{prefix}.mlp.down_proj.weight"),
                &[hidden, intermediate],
            )?;
            let attention = match *kind {
                Qwen35LayerType::Full => Qwen35AttentionWeights::Full {
                    query_gate: checkpoint.source(
                        &format!("{prefix}.self_attn.q_proj.weight"),
                        &[query_gate, hidden],
                    )?,
                    key: checkpoint
                        .source(&format!("{prefix}.self_attn.k_proj.weight"), &[kv, hidden])?,
                    value: checkpoint
                        .source(&format!("{prefix}.self_attn.v_proj.weight"), &[kv, hidden])?,
                    output: checkpoint
                        .source(&format!("{prefix}.self_attn.o_proj.weight"), &[hidden, q])?,
                    q_norm: checkpoint
                        .source(&format!("{prefix}.self_attn.q_norm.weight"), &[head_dim])?,
                    k_norm: checkpoint
                        .source(&format!("{prefix}.self_attn.k_norm.weight"), &[head_dim])?,
                },
                Qwen35LayerType::Linear => Qwen35AttentionWeights::Linear {
                    qkv: checkpoint.source(
                        &format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                        &[linear_qkv_width, hidden],
                    )?,
                    z: checkpoint.source(
                        &format!("{prefix}.linear_attn.in_proj_z.weight"),
                        &[linear_value_width, hidden],
                    )?,
                    a: checkpoint.source(
                        &format!("{prefix}.linear_attn.in_proj_a.weight"),
                        &[config.linear_num_value_heads, hidden],
                    )?,
                    b: checkpoint.source(
                        &format!("{prefix}.linear_attn.in_proj_b.weight"),
                        &[config.linear_num_value_heads, hidden],
                    )?,
                    output: checkpoint.source(
                        &format!("{prefix}.linear_attn.out_proj.weight"),
                        &[hidden, linear_value_width],
                    )?,
                    conv: checkpoint.source(
                        &format!("{prefix}.linear_attn.conv1d.weight"),
                        &[linear_qkv_width, 1, config.linear_conv_kernel_dim],
                    )?,
                    a_log: checkpoint.source(
                        &format!("{prefix}.linear_attn.A_log"),
                        &[config.linear_num_value_heads],
                    )?,
                    dt_bias: checkpoint.source(
                        &format!("{prefix}.linear_attn.dt_bias"),
                        &[config.linear_num_value_heads],
                    )?,
                    norm: checkpoint.source(
                        &format!("{prefix}.linear_attn.norm.weight"),
                        &[DELTA_HEAD_DIM],
                    )?,
                },
            };
            layers.push(Qwen35LayerWeights {
                input_norm,
                post_attention_norm,
                mlp_gate,
                mlp_up,
                mlp_down,
                attention,
            });
        }

        let lm_head = checkpoint.optional_source("lm_head.weight", &[config.vocab_size, hidden])?;
        Ok(Self {
            config,
            embeddings,
            norm,
            layers,
            lm_head,
            root: root.to_owned(),
        })
    }
}

fn is_backbone_suffix(suffix: &str) -> bool {
    suffix.starts_with("embed_tokens.")
        || suffix.starts_with("norm.")
        || suffix.starts_with("layers.")
}

/// Select exactly one complete native text root. A second embedding anchor is
/// ambiguous; text backbone tensors under a different native root make the
/// namespace mixed; an anchor without the final norm and first layer is only a
/// partial root. Vision, MTP and `lm_head` tensors never count as text
/// backbone tensors.
fn select_root(checkpoint: &Checkpoint) -> Result<&'static str, BackendError> {
    let mut selected = None;
    let mut anchors = 0usize;
    for name in checkpoint.names() {
        for root in NATIVE_ROOTS {
            if name.strip_prefix(root) == Some("embed_tokens.weight") {
                anchors = anchors
                    .checked_add(1)
                    .ok_or_else(|| start("qwen3.5 checkpoint anchor overflow"))?;
                selected = Some(root);
            }
        }
    }
    if anchors == 0 {
        return Err(start("missing weight embed_tokens.weight"));
    }
    if anchors != 1 {
        return Err(start(
            "ambiguous qwen3.5 checkpoint namespace: multiple embed_tokens.weight roots",
        ));
    }
    let root = selected.ok_or_else(|| start("missing qwen3.5 checkpoint root"))?;
    for name in checkpoint.names() {
        for other in NATIVE_ROOTS {
            if other == root {
                continue;
            }
            if name.strip_prefix(other).is_some_and(is_backbone_suffix) {
                return Err(start(format!(
                    "mixed qwen3.5 checkpoint roots: found {name} alongside the {root:?} root"
                )));
            }
        }
    }
    for suffix in ["norm.weight", "layers.0.input_layernorm.weight"] {
        let name = format!("{root}{suffix}");
        if !checkpoint.contains(&name) {
            return Err(start(format!(
                "partial Qwen3.5 text root {root:?}: missing weight {name}"
            )));
        }
    }
    Ok(root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;
    use serde_json::Value;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "infinity-qwen35-weights-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            let fixture = Self(path);
            fixture.raw_config(qwen35_config());
            fixture
        }

        fn raw_config(&self, value: Value) {
            fs::write(
                self.0.join("config.json"),
                serde_json::to_vec(&value).unwrap(),
            )
            .unwrap();
        }

        fn set_text(&self, overrides: Value) {
            let mut value = qwen35_config();
            {
                let text = value
                    .get_mut("text_config")
                    .and_then(|value| value.as_object_mut())
                    .unwrap();
                for (key, field) in overrides.as_object().unwrap() {
                    text.insert(key.clone(), field.clone());
                }
            }
            self.raw_config(value);
        }

        fn config(&self) -> Qwen35Config {
            Qwen35Config::load(&self.0).unwrap()
        }

        fn write_entries(&self, entries: &[(String, Vec<usize>)], sharded: bool) {
            let data = entries
                .iter()
                .map(|(_, shape)| {
                    let count = shape
                        .iter()
                        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
                        .expect("test fixture shape overflow");
                    vec![0u8; count.checked_mul(2).expect("test fixture byte overflow")]
                })
                .collect::<Vec<_>>();
            let views = entries
                .iter()
                .zip(&data)
                .map(|((name, shape), bytes)| {
                    (
                        name.clone(),
                        TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
                    )
                })
                .collect::<Vec<_>>();
            if !sharded {
                fs::write(
                    self.0.join("model.safetensors"),
                    serialize(
                        views
                            .iter()
                            .map(|(name, view)| (name.clone(), view.clone())),
                        None,
                    )
                    .unwrap(),
                )
                .unwrap();
                return;
            }
            let mut map = BTreeMap::new();
            for (index, chunk) in views.chunks(views.len().div_ceil(2).max(1)).enumerate() {
                let file = format!("model-{index}.safetensors");
                fs::write(
                    self.0.join(&file),
                    serialize(
                        chunk
                            .iter()
                            .map(|(name, view)| (name.clone(), view.clone())),
                        None,
                    )
                    .unwrap(),
                )
                .unwrap();
                for (name, _) in chunk {
                    map.insert(name.clone(), file.clone());
                }
            }
            fs::write(
                self.0.join("model.safetensors.index.json"),
                serde_json::to_vec(&serde_json::json!({"weight_map": map})).unwrap(),
            )
            .unwrap();
        }

        fn error(&self) -> String {
            match Qwen35Weights::load(&self.0) {
                Ok(_) => panic!("invalid qwen3.5 fixture was accepted"),
                Err(error) => error.to_string(),
            }
        }

        fn config_error(&self) -> String {
            match Qwen35Config::load(&self.0) {
                Ok(_) => panic!("invalid qwen3.5 config was accepted"),
                Err(error) => error.to_string(),
            }
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn qwen35_config() -> Value {
        serde_json::json!({
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "dtype": "bfloat16",
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 10,
                "intermediate_size": 14,
                "num_hidden_layers": 2,
                "num_attention_heads": 3,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "vocab_size": 10,
                "max_position_embeddings": 32,
                "rms_norm_eps": 1e-6,
                "rope_parameters": {
                    "rope_theta": 10_000_000.0,
                    "partial_rotary_factor": 0.5,
                    "rope_type": "default",
                    "mrope_interleaved": false
                },
                "partial_rotary_factor": 0.5,
                "layer_types": ["full_attention", "linear_attention"],
                "attn_output_gate": true,
                "linear_num_key_heads": 1,
                "linear_num_value_heads": 2,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "mtp_num_hidden_layers": 1,
                "tie_word_embeddings": false
            },
            "vision_config": {"depth": 1},
            "tie_word_embeddings": false
        })
    }

    fn qwen35_shapes(
        config: &Qwen35Config,
        root: &str,
        with_lm_head: bool,
    ) -> Vec<(String, Vec<usize>)> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let q = config.q_width();
        let kv = config.kv_width();
        let head_dim = config.head_dim;
        let key_width = config.linear_num_key_heads * DELTA_HEAD_DIM;
        let value_width = config.linear_num_value_heads * DELTA_HEAD_DIM;
        let qkv_width = key_width * 2 + value_width;
        let mut out = vec![
            (
                format!("{root}embed_tokens.weight"),
                vec![config.vocab_size, hidden],
            ),
            (format!("{root}norm.weight"), vec![hidden]),
        ];
        for (index, kind) in config.layer_types.iter().enumerate() {
            let prefix = format!("{root}layers.{index}");
            out.extend([
                (format!("{prefix}.input_layernorm.weight"), vec![hidden]),
                (
                    format!("{prefix}.post_attention_layernorm.weight"),
                    vec![hidden],
                ),
                (
                    format!("{prefix}.mlp.gate_proj.weight"),
                    vec![intermediate, hidden],
                ),
                (
                    format!("{prefix}.mlp.up_proj.weight"),
                    vec![intermediate, hidden],
                ),
                (
                    format!("{prefix}.mlp.down_proj.weight"),
                    vec![hidden, intermediate],
                ),
            ]);
            match *kind {
                Qwen35LayerType::Full => out.extend([
                    (
                        format!("{prefix}.self_attn.q_proj.weight"),
                        vec![2 * q, hidden],
                    ),
                    (
                        format!("{prefix}.self_attn.k_proj.weight"),
                        vec![kv, hidden],
                    ),
                    (
                        format!("{prefix}.self_attn.v_proj.weight"),
                        vec![kv, hidden],
                    ),
                    (format!("{prefix}.self_attn.o_proj.weight"), vec![hidden, q]),
                    (format!("{prefix}.self_attn.q_norm.weight"), vec![head_dim]),
                    (format!("{prefix}.self_attn.k_norm.weight"), vec![head_dim]),
                ]),
                Qwen35LayerType::Linear => out.extend([
                    (
                        format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                        vec![qkv_width, hidden],
                    ),
                    (
                        format!("{prefix}.linear_attn.in_proj_z.weight"),
                        vec![value_width, hidden],
                    ),
                    (
                        format!("{prefix}.linear_attn.in_proj_a.weight"),
                        vec![config.linear_num_value_heads, hidden],
                    ),
                    (
                        format!("{prefix}.linear_attn.in_proj_b.weight"),
                        vec![config.linear_num_value_heads, hidden],
                    ),
                    (
                        format!("{prefix}.linear_attn.out_proj.weight"),
                        vec![hidden, value_width],
                    ),
                    (
                        format!("{prefix}.linear_attn.conv1d.weight"),
                        vec![qkv_width, 1, config.linear_conv_kernel_dim],
                    ),
                    (
                        format!("{prefix}.linear_attn.A_log"),
                        vec![config.linear_num_value_heads],
                    ),
                    (
                        format!("{prefix}.linear_attn.dt_bias"),
                        vec![config.linear_num_value_heads],
                    ),
                    (
                        format!("{prefix}.linear_attn.norm.weight"),
                        vec![DELTA_HEAD_DIM],
                    ),
                ]),
            }
        }
        if with_lm_head {
            out.push((
                "lm_head.weight".to_string(),
                vec![config.vocab_size, hidden],
            ));
        }
        out
    }

    #[test]
    fn loads_sharded_mixed_layers_with_query_width_wider_than_hidden() {
        let fixture = Fixture::new();
        let config = fixture.config();
        assert_eq!(config.hidden_size, 10);
        assert_eq!(config.q_width(), 12);
        assert_ne!(config.q_width(), config.hidden_size);
        assert_eq!(config.kv_width(), 4);
        assert_eq!(config.rotary_dim, 2);
        assert_eq!(config.linear_num_key_heads, 1);
        assert_eq!(config.linear_num_value_heads, 2);

        let entries = qwen35_shapes(&config, "model.language_model.", true);
        fixture.write_entries(&entries, true);
        let loaded = Qwen35Weights::load(&fixture.0).unwrap();
        assert_eq!(loaded.root, "model.language_model.");
        assert_eq!(loaded.layers.len(), 2);
        assert_eq!(loaded.embeddings.shape(), &[10, 10]);
        assert_eq!(loaded.norm.shape(), &[10]);
        assert_eq!(loaded.lm_head.as_ref().unwrap().shape(), &[10, 10]);

        let full = &loaded.layers[0];
        assert_eq!(full.input_norm.shape(), &[10]);
        assert_eq!(full.post_attention_norm.shape(), &[10]);
        assert_eq!(full.mlp_gate.shape(), &[14, 10]);
        assert_eq!(full.mlp_up.shape(), &[14, 10]);
        assert_eq!(full.mlp_down.shape(), &[10, 14]);
        match &full.attention {
            Qwen35AttentionWeights::Full {
                query_gate,
                key,
                value,
                output,
                q_norm,
                k_norm,
            } => {
                assert_eq!(query_gate.shape(), &[24, 10]);
                assert_eq!(key.shape(), &[4, 10]);
                assert_eq!(value.shape(), &[4, 10]);
                assert_eq!(output.shape(), &[10, 12]);
                assert_eq!(q_norm.shape(), &[4]);
                assert_eq!(k_norm.shape(), &[4]);
            }
            Qwen35AttentionWeights::Linear { .. } => {
                panic!("configured full layer loaded linear sources")
            }
        }

        match &loaded.layers[1].attention {
            Qwen35AttentionWeights::Linear {
                qkv,
                z,
                a,
                b,
                output,
                conv,
                a_log,
                dt_bias,
                norm,
            } => {
                assert_eq!(qkv.shape(), &[512, 10]);
                assert_eq!(z.shape(), &[256, 10]);
                assert_eq!(a.shape(), &[2, 10]);
                assert_eq!(b.shape(), &[2, 10]);
                assert_eq!(output.shape(), &[10, 256]);
                assert_eq!(conv.shape(), &[512, 1, 4]);
                assert_eq!(a_log.shape(), &[2]);
                assert_eq!(dt_bias.shape(), &[2]);
                assert_eq!(norm.shape(), &[128]);
            }
            Qwen35AttentionWeights::Full { .. } => {
                panic!("configured linear layer loaded full-attention sources")
            }
        }
    }

    #[test]
    fn omits_lm_head_for_tied_embeddings() {
        let fixture = Fixture::new();
        let config = fixture.config();
        let entries = qwen35_shapes(&config, "model.language_model.", false);
        fixture.write_entries(&entries, false);
        let loaded = Qwen35Weights::load(&fixture.0).unwrap();
        assert!(loaded.lm_head.is_none());
    }

    #[test]
    fn rejects_missing_and_wrong_mixed_layer_tensors() {
        let fixture = Fixture::new();
        let config = fixture.config();
        let mut missing = qwen35_shapes(&config, "model.language_model.", false);
        missing.retain(|(name, _)| {
            name != "model.language_model.layers.1.linear_attn.in_proj_qkv.weight"
        });
        fixture.write_entries(&missing, false);
        let error = fixture.error();
        assert!(
            error.contains(
                "missing weight model.language_model.layers.1.linear_attn.in_proj_qkv.weight"
            ),
            "{error}"
        );

        let mut wrong = qwen35_shapes(&config, "model.language_model.", false);
        for (name, shape) in &mut wrong {
            if name == "model.language_model.layers.0.self_attn.q_proj.weight" {
                *shape = vec![config.q_width(), config.hidden_size];
            }
        }
        fixture.write_entries(&wrong, false);
        let error = fixture.error();
        assert!(
            error.contains("wrong shape for model.language_model.layers.0.self_attn.q_proj.weight"),
            "{error}"
        );
    }

    #[test]
    fn rejects_ambiguous_mixed_and_partial_text_roots() {
        let fixture = Fixture::new();
        let config = fixture.config();
        let entries = qwen35_shapes(&config, "model.language_model.", false);

        let mut ambiguous = entries.clone();
        ambiguous.push((
            "language_model.embed_tokens.weight".to_string(),
            vec![config.vocab_size, config.hidden_size],
        ));
        fixture.write_entries(&ambiguous, false);
        let error = fixture.error();
        assert!(error.contains("ambiguous"), "{error}");

        let mut mixed = entries.clone();
        mixed.push((
            "language_model.layers.0.input_layernorm.weight".to_string(),
            vec![config.hidden_size],
        ));
        fixture.write_entries(&mixed, false);
        let error = fixture.error();
        assert!(error.contains("mixed"), "{error}");

        let partial = vec![
            (
                "model.language_model.embed_tokens.weight".to_string(),
                vec![config.vocab_size, config.hidden_size],
            ),
            (
                "model.language_model.norm.weight".to_string(),
                vec![config.hidden_size],
            ),
        ];
        fixture.write_entries(&partial, false);
        let error = fixture.error();
        assert!(error.contains("partial"), "{error}");
    }

    #[test]
    fn rejects_non_native_linear_heads_and_moe() {
        let fixture = Fixture::new();
        fixture.set_text(serde_json::json!({"linear_value_head_dim": 64}));
        let error = fixture.config_error();
        assert!(error.contains("128"), "{error}");

        fixture.set_text(serde_json::json!({"attn_output_gate": false}));
        let error = fixture.config_error();
        assert!(error.contains("output gate"), "{error}");

        fixture.set_text(serde_json::json!({"num_experts": 8}));
        let error = fixture.config_error();
        assert!(error.contains("expert"), "{error}");

        fixture.set_text(serde_json::json!({
            "num_hidden_layers": 2,
            "layer_types": ["full_attention"]
        }));
        let error = fixture.config_error();
        assert!(error.contains("layer_types"), "{error}");
    }

    #[test]
    #[ignore = "CPU-only actual Elephant metadata/header inventory; set INFINITY_OXIDE_TEST_QWEN35_ELEPHANT"]
    fn actual_elephant_metadata_inventory() {
        let path = std::env::var_os("INFINITY_OXIDE_TEST_QWEN35_ELEPHANT")
            .expect("set INFINITY_OXIDE_TEST_QWEN35_ELEPHANT to the Elephant checkpoint directory");
        let weights = Qwen35Weights::load(Path::new(&path)).unwrap();
        assert_eq!(weights.root, "model.language_model.");
        assert_eq!(weights.config.hidden_size, 4096);
        assert_eq!(weights.config.q_width(), 4096);
        assert_eq!(weights.config.kv_width(), 1024);
        assert_eq!(weights.config.rotary_dim, 64);
        assert_eq!(weights.config.vocab_size, 248_320);
        assert_eq!(weights.layers.len(), 32);
        assert!(matches!(
            &weights.layers[0].attention,
            Qwen35AttentionWeights::Linear { .. }
        ));
        assert!(matches!(
            &weights.layers[3].attention,
            Qwen35AttentionWeights::Full { .. }
        ));
        match &weights.layers[0].attention {
            Qwen35AttentionWeights::Linear { qkv, z, conv, .. } => {
                assert_eq!(qkv.shape(), &[8192, 4096]);
                assert_eq!(z.shape(), &[4096, 4096]);
                assert_eq!(conv.shape(), &[8192, 1, 4]);
            }
            Qwen35AttentionWeights::Full { .. } => panic!("Elephant layer 0 is not linear"),
        }
        assert_eq!(weights.lm_head.as_ref().unwrap().shape(), &[248_320, 4096]);
    }

    #[test]
    #[ignore = "CPU-only actual Mammoth metadata/header inventory; set INFINITY_OXIDE_TEST_QWEN35_MAMMOTH"]
    fn actual_mammoth_metadata_inventory() {
        let path = std::env::var_os("INFINITY_OXIDE_TEST_QWEN35_MAMMOTH")
            .expect("set INFINITY_OXIDE_TEST_QWEN35_MAMMOTH to the Mammoth checkpoint directory");
        let weights = Qwen35Weights::load(Path::new(&path)).unwrap();
        assert_eq!(weights.root, "model.language_model.");
        assert_eq!(weights.config.hidden_size, 5120);
        assert_eq!(weights.config.q_width(), 6144);
        assert_ne!(weights.config.q_width(), weights.config.hidden_size);
        assert_eq!(weights.config.kv_width(), 1024);
        assert_eq!(weights.config.rotary_dim, 64);
        assert_eq!(weights.config.intermediate_size, 17_408);
        assert_eq!(weights.config.vocab_size, 248_320);
        assert_eq!(weights.layers.len(), 64);
        match &weights.layers[3].attention {
            Qwen35AttentionWeights::Full {
                query_gate, output, ..
            } => {
                assert_eq!(query_gate.shape(), &[12_288, 5120]);
                assert_eq!(output.shape(), &[5120, 6144]);
            }
            Qwen35AttentionWeights::Linear { .. } => panic!("Mammoth layer 3 is not full"),
        }
        match &weights.layers[0].attention {
            Qwen35AttentionWeights::Linear {
                qkv,
                z,
                a,
                b,
                output,
                ..
            } => {
                assert_eq!(qkv.shape(), &[10_240, 5120]);
                assert_eq!(z.shape(), &[6144, 5120]);
                assert_eq!(a.shape(), &[48, 5120]);
                assert_eq!(b.shape(), &[48, 5120]);
                assert_eq!(output.shape(), &[5120, 6144]);
            }
            Qwen35AttentionWeights::Full { .. } => panic!("Mammoth layer 0 is not linear"),
        }
        assert_eq!(weights.lm_head.as_ref().unwrap().shape(), &[248_320, 5120]);
    }
}

// Qwen3-VL geometry coverage and central FA4 gap report.
//
// Native accepts any dense Qwen3-VL text geometry whose decoder/attention
// providers exist, plus the two aliases `colqwen3` and
// `qwen3_vl_nemotron_embed`. This module reports the actual cached config
// tuples, the derived public-family tuples that are not locally cached, and
// exactly which FA4/text-decoder capability is missing. It never runs a GPU or
// invents a profile.

use crate::decoder::validate_recipe;
use crate::decoder_model::DecoderConfig;
use crate::fa4::{AttentionSpec, DType, Fa4, Scale, Window};
use crate::qwen3_vl_config::{Qwen3VlConfig, QwenVlAlias, ReadoutKind};
use infinity_v3_backend_core::BackendError;
use serde_json::json;
use std::path::Path;

#[derive(Clone, Debug)]
pub(crate) struct GeometryTuple {
    pub(crate) name: &'static str,
    pub(crate) source: &'static str,
    pub(crate) config_path: Option<&'static str>,
    pub(crate) alias: QwenVlAlias,
    pub(crate) dense_text: bool,
    pub(crate) text_hidden: usize,
    pub(crate) text_heads: usize,
    pub(crate) text_kv: usize,
    pub(crate) text_head_dim: usize,
    pub(crate) text_layers: usize,
    pub(crate) text_intermediate: usize,
    pub(crate) text_vocab: usize,
    pub(crate) text_theta: f64,
    pub(crate) mrope: [usize; 3],
    pub(crate) vision_hidden: usize,
    pub(crate) vision_heads: usize,
    pub(crate) vision_depth: usize,
    pub(crate) vision_intermediate: usize,
    pub(crate) vision_out: usize,
    pub(crate) vision_positions: usize,
    pub(crate) deepstack: &'static [usize],
}

#[derive(Clone, Debug)]
pub(crate) struct GeometrySupport {
    pub(crate) structural: bool,
    pub(crate) dense_decoder: bool,
    pub(crate) vision_fa4: bool,
    pub(crate) text_fa4: bool,
    pub(crate) status: &'static str,
    pub(crate) text_profile: AttentionSpec,
    pub(crate) vision_profile: AttentionSpec,
    pub(crate) missing: Vec<serde_json::Value>,
}

pub(crate) fn cached_tuples() -> Vec<GeometryTuple> {
    vec![
        GeometryTuple {
            name: "Qwen3-VL-8B-Embedding",
            source: "actual cached checkpoint",
            config_path: Some(
                "/workspace/model-performance/michaelfeil/benchmark-overview/models/Qwen--Qwen3-VL-Embedding-8B/config.json",
            ),
            alias: QwenVlAlias::Qwen3Vl,
            dense_text: true,
            text_hidden: 4096,
            text_heads: 32,
            text_kv: 8,
            text_head_dim: 128,
            text_layers: 36,
            text_intermediate: 12288,
            text_vocab: 151936,
            text_theta: 5_000_000.0,
            mrope: [24, 20, 20],
            vision_hidden: 1152,
            vision_heads: 16,
            vision_depth: 27,
            vision_intermediate: 4304,
            vision_out: 4096,
            vision_positions: 2304,
            deepstack: &[8, 16, 24],
        },
        GeometryTuple {
            name: "TomoroAI-ColQwen3-4B",
            source: "actual cached checkpoint",
            config_path: Some(
                "/workspace/model-performance/michaelfeil/torch215-comparison-20261029/models/TomoroAI--tomoro-colqwen3-embed-4b/config.json",
            ),
            alias: QwenVlAlias::ColQwen3,
            dense_text: true,
            text_hidden: 2560,
            text_heads: 32,
            text_kv: 8,
            text_head_dim: 128,
            text_layers: 36,
            text_intermediate: 9728,
            text_vocab: 151936,
            text_theta: 5_000_000.0,
            mrope: [24, 20, 20],
            vision_hidden: 1024,
            vision_heads: 16,
            vision_depth: 24,
            vision_intermediate: 4096,
            vision_out: 2560,
            vision_positions: 2304,
            deepstack: &[5, 11, 17],
        },
        GeometryTuple {
            name: "Nemotron-ColEmbed-VL-4B-v2",
            source: "actual cached checkpoint",
            config_path: Some(
                "/workspace/model-performance/michaelfeil/torch215-comparison-20261029/models/nvidia--nemotron-colembed-vl-4b-v2/config.json",
            ),
            alias: QwenVlAlias::NemotronColEmbed,
            dense_text: true,
            text_hidden: 2560,
            text_heads: 32,
            text_kv: 8,
            text_head_dim: 128,
            text_layers: 36,
            text_intermediate: 9728,
            text_vocab: 151936,
            text_theta: 5_000_000.0,
            mrope: [24, 20, 20],
            vision_hidden: 1024,
            vision_heads: 16,
            vision_depth: 24,
            vision_intermediate: 4096,
            vision_out: 2560,
            vision_positions: 2304,
            deepstack: &[5, 11, 17],
        },
        GeometryTuple {
            name: "Nemotron-ColEmbed-VL-8B-v2",
            source: "actual cached checkpoint",
            config_path: Some(
                "/tmp/perplexity-embedding-audit/multivector-support/nvidia--nemotron-colembed-vl-8b-v2/config.json",
            ),
            alias: QwenVlAlias::NemotronColEmbed,
            dense_text: true,
            text_hidden: 4096,
            text_heads: 32,
            text_kv: 8,
            text_head_dim: 128,
            text_layers: 36,
            text_intermediate: 12288,
            text_vocab: 151936,
            text_theta: 5_000_000.0,
            mrope: [24, 20, 20],
            vision_hidden: 1152,
            vision_heads: 16,
            vision_depth: 27,
            vision_intermediate: 4304,
            vision_out: 4096,
            vision_positions: 2304,
            deepstack: &[8, 16, 24],
        },
    ]
}

/// Public-family tuples that are not locally cached. They are reported so Root
/// can verify the exact config before admission; the structural classification
/// is still useful because the binding constraints are the FA4 profiles.
pub(crate) fn derived_tuples() -> Vec<GeometryTuple> {
    vec![
        GeometryTuple {
            name: "Qwen3-VL-2B",
            source: "derived public family tuple; verify config before admission",
            config_path: None,
            alias: QwenVlAlias::Qwen3Vl,
            dense_text: true,
            text_hidden: 2048,
            text_heads: 16,
            text_kv: 8,
            text_head_dim: 128,
            text_layers: 28,
            text_intermediate: 6144,
            text_vocab: 151936,
            text_theta: 5_000_000.0,
            mrope: [24, 20, 20],
            vision_hidden: 1024,
            vision_heads: 16,
            vision_depth: 24,
            vision_intermediate: 4096,
            vision_out: 2048,
            vision_positions: 2304,
            deepstack: &[5, 11, 17],
        },
        GeometryTuple {
            name: "Qwen3-VL-32B",
            source: "derived public family tuple; verify config before admission",
            config_path: None,
            alias: QwenVlAlias::Qwen3Vl,
            dense_text: true,
            text_hidden: 5120,
            text_heads: 64,
            text_kv: 8,
            text_head_dim: 128,
            text_layers: 64,
            text_intermediate: 25600,
            text_vocab: 151936,
            text_theta: 5_000_000.0,
            mrope: [24, 20, 20],
            vision_hidden: 1152,
            vision_heads: 16,
            vision_depth: 27,
            vision_intermediate: 4304,
            vision_out: 5120,
            vision_positions: 2304,
            deepstack: &[8, 16, 24],
        },
        GeometryTuple {
            name: "Qwen3-VL-30B-A3B",
            source: "derived public family tuple; MoE text is outside the dense VL foundation",
            config_path: None,
            alias: QwenVlAlias::Qwen3Vl,
            dense_text: false,
            text_hidden: 2048,
            text_heads: 32,
            text_kv: 4,
            text_head_dim: 128,
            text_layers: 48,
            text_intermediate: 0,
            text_vocab: 151936,
            text_theta: 5_000_000.0,
            mrope: [24, 20, 20],
            vision_hidden: 1152,
            vision_heads: 16,
            vision_depth: 27,
            vision_intermediate: 4304,
            vision_out: 2048,
            vision_positions: 2304,
            deepstack: &[8, 16, 24],
        },
    ]
}

impl GeometryTuple {
    fn config_json(&self) -> serde_json::Value {
        let mut text = json!({
            "dtype": "bfloat16",
            "hidden_size": self.text_hidden,
            "num_attention_heads": self.text_heads,
            "num_key_value_heads": self.text_kv,
            "head_dim": self.text_head_dim,
            "num_hidden_layers": self.text_layers,
            "intermediate_size": self.text_intermediate.max(1),
            "vocab_size": self.text_vocab,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "model_type": if self.dense_text { "qwen3_vl_text" } else { "qwen3_moe" },
            "rope_scaling": {
                "mrope_interleaved": true,
                "mrope_section": self.mrope,
                "rope_theta": self.text_theta,
                "rope_type": "default",
            },
            "rope_theta": self.text_theta,
        });
        if !self.dense_text {
            text["num_experts"] = 128.into();
        }
        json!({
            "architectures": [match self.alias {
                QwenVlAlias::Qwen3Vl => "Qwen3VLForConditionalGeneration",
                QwenVlAlias::ColQwen3 => "ColQwen3",
                QwenVlAlias::NemotronColEmbed => "Qwen3VLNemotronEmbedModel",
            }],
            "dtype": "bfloat16",
            "model_type": self.alias.model_type(),
            "image_token_id": 151655,
            "video_token_id": 151656,
            "vision_start_token_id": 151652,
            "vision_end_token_id": 151653,
            "text_config": text,
            "vision_config": {
                "dtype": "bfloat16",
                "hidden_size": self.vision_hidden,
                "num_heads": self.vision_heads,
                "depth": self.vision_depth,
                "intermediate_size": self.vision_intermediate,
                "patch_size": 16,
                "temporal_patch_size": 2,
                "in_channels": 3,
                "spatial_merge_size": 2,
                "num_position_embeddings": self.vision_positions,
                "out_hidden_size": self.vision_out,
                "hidden_act": "gelu_pytorch_tanh",
                "deepstack_visual_indexes": self.deepstack,
                "model_type": "qwen3_vl",
            },
        })
    }

    pub(crate) fn config(&self) -> Result<Qwen3VlConfig, BackendError> {
        Qwen3VlConfig::from_value(self.config_json())
    }

    pub(crate) fn classify(&self) -> Result<GeometrySupport, BackendError> {
        let config = self.config()?;
        let mut missing = Vec::new();
        let text_causal = config.readout.kind != ReadoutKind::NemotronColEmbed;
        let text_profile = AttentionSpec {
            heads: self.text_heads,
            kv_heads: self.text_kv,
            head_dim: self.text_head_dim,
            causal: text_causal,
            window: Window::Global,
            dtype: DType::Bf16,
            scale: Scale::RsqrtHeadDim,
        };
        let vision_profile = AttentionSpec {
            heads: self.vision_heads,
            kv_heads: self.vision_heads,
            head_dim: self.vision_hidden / self.vision_heads,
            causal: false,
            window: Window::Global,
            dtype: DType::Bf16,
            scale: Scale::RsqrtHeadDim,
        };
        let mut decoder_error = None;
        let dense_decoder = if self.dense_text {
            let text = config.raw.get("text_config").ok_or_else(|| {
                infinity_v3_backend_core::BackendError::Start("missing text_config".into())
            })?;
            match DecoderConfig::from_text_config(text)
                .and_then(|decoder| validate_recipe(&decoder))
            {
                Ok(()) => true,
                Err(error) => {
                    decoder_error = Some(format!("{error:?}"));
                    false
                }
            }
        } else {
            false
        };
        let vision_fa4 = Fa4::supports(&vision_profile);
        let text_fa4 = Fa4::supports(&text_profile);
        if !vision_fa4 {
            missing.push(profile_json(
                "vision",
                &vision_profile,
                "noncausal vision tower",
            ));
        }
        if !text_fa4 {
            missing.push(profile_json(
                "text",
                &text_profile,
                if text_causal {
                    "causal dense text"
                } else {
                    "noncausal ColEmbed text"
                },
            ));
        }
        if !dense_decoder {
            missing.push(json!({
                "kind": "text_decoder",
                "reason": decoder_error.unwrap_or_else(|| {
                    "MoE text is outside the dense Qwen3-VL foundation".to_owned()
                }),
            }));
        }
        let status = if !dense_decoder {
            "missing_decoder"
        } else if !vision_fa4 || !text_fa4 {
            "missing_fa4"
        } else {
            "ready"
        };
        Ok(GeometrySupport {
            structural: true,
            dense_decoder,
            vision_fa4,
            text_fa4,
            status,
            text_profile,
            vision_profile,
            missing,
        })
    }
}

fn profile_json(kind: &str, spec: &AttentionSpec, role: &str) -> serde_json::Value {
    json!({
        "kind": kind,
        "role": role,
        "heads": spec.heads,
        "kv_heads": spec.kv_heads,
        "head_dim": spec.head_dim,
        "causal": spec.causal,
        "window": "global",
        "dtype": "bf16",
        "scale": "rsqrt_head_dim",
    })
}

fn sha256(path: &Path) -> Option<String> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).ok()?;
    Some(format!("{:x}", Sha256::digest(bytes)))
}

/// Profile requests that are not tied to a fully verified model tuple yet.
/// The audit proves the FA4 ABI bakes the grouping ratio, not absolute heads.
pub fn additional_profile_requests() -> Vec<serde_json::Value> {
    let causal = AttentionSpec {
        heads: 40,
        kv_heads: 8,
        head_dim: 128,
        causal: true,
        window: Window::Global,
        dtype: DType::Bf16,
        scale: Scale::RsqrtHeadDim,
    };
    let noncausal = AttentionSpec {
        causal: false,
        ..causal
    };
    let mut requests = Vec::new();
    for (role, spec) in [
        ("Qwen3-VL-14B/other Q40/KV8 d128 causal global", causal),
        (
            "Qwen3-VL-14B/other Q40/KV8 d128 noncausal global",
            noncausal,
        ),
    ] {
        if !Fa4::supports(&spec) {
            requests.push(json!({
                "tuple": "Qwen3-VL-14B (derived profile-only request)",
                "missing": {
                    "kind": "text",
                    "role": role,
                    "heads": spec.heads,
                    "kv_heads": spec.kv_heads,
                    "head_dim": spec.head_dim,
                    "causal": spec.causal,
                    "window": "global",
                    "dtype": "bf16",
                    "scale": "rsqrt_head_dim",
                    "reason": "grouping ratio 5 is absent from canonical26",
                }
            }));
        }
    }
    requests
}

pub fn report() -> serde_json::Value {
    let mut tuples = Vec::new();
    let mut required = Vec::new();
    for tuple in cached_tuples().into_iter().chain(derived_tuples()) {
        let support = tuple.classify().ok();
        if let Some(support) = support.as_ref() {
            for item in &support.missing {
                if item.get("kind").and_then(|value| value.as_str()) == Some("text")
                    || item.get("kind").and_then(|value| value.as_str()) == Some("vision")
                {
                    required.push(json!({
                        "tuple": tuple.name,
                        "missing": item,
                    }));
                }
            }
        }
        tuples.push(json!({
            "name": tuple.name,
            "source": tuple.source,
            "config_path": tuple.config_path,
            "config_sha256": tuple.config_path.and_then(|path| sha256(Path::new(path))),
            "alias": tuple.alias.model_type(),
            "dense_text": tuple.dense_text,
            "text": {
                "hidden_size": tuple.text_hidden,
                "heads": tuple.text_heads,
                "kv_heads": tuple.text_kv,
                "head_dim": tuple.text_head_dim,
                "layers": tuple.text_layers,
                "intermediate_size": tuple.text_intermediate,
                "vocab": tuple.text_vocab,
                "theta": tuple.text_theta,
                "mrope": tuple.mrope,
            },
            "vision": {
                "hidden_size": tuple.vision_hidden,
                "heads": tuple.vision_heads,
                "head_dim": tuple.vision_hidden / tuple.vision_heads,
                "depth": tuple.vision_depth,
                "intermediate_size": tuple.vision_intermediate,
                "out_hidden_size": tuple.vision_out,
                "positions": tuple.vision_positions,
                "deepstack": tuple.deepstack,
            },
            "support": support.map(|support| json!({
                "structural": support.structural,
                "dense_decoder": support.dense_decoder,
                "vision_fa4": support.vision_fa4,
                "text_fa4": support.text_fa4,
                "status": support.status,
                "text_profile": Fa4::profile_name(&support.text_profile),
                "vision_profile": Fa4::profile_name(&support.vision_profile),
                "shell_only": false,
            })),
        }));
    }
    required.extend(additional_profile_requests());
    json!({
        "schema": "oxide-qwen3-vl-geometry/v1",
        "base_commit": "e582ce90eb1544fcff081a495cb8fd4f3c17b112",
        "structural_envelope": "Native accepts dense Qwen3-VL text with head_dim 128, square vision position tables, temporal_patch_size 2, in_channels 3 and text hidden equal to vision out_hidden_size. The Oxide generic Qwen3 recipe covers arbitrary dense head_dim-128 hidden widths with the existing generic norm/RoPE/QK exports. FA4 selection uses the grouping ratio, head_dim, dtype, causal bit and mask family, not absolute Q/KV heads: Q64/KV8 d128 reuses mode 22/5, Q12/KV12 d64 global reuses mode 0, and only the ratio-5 Q40/KV8 d128 combination is genuinely new.",
        "cached_tuples": 4,
        "derived_tuples": 3,
        "tuples": tuples,
        "dtypes": {
            "bf16": {
                "config": "admitted; effective dtype is the top-level checkpoint dtype",
                "device": "concrete Qwen3-VL tower and dense decoder use BF16",
                "readout": "BF16 weights are widened exactly on the host",
                "cached_tuples": ["Qwen3-VL-8B-Embedding", "TomoroAI-ColQwen3-4B", "Nemotron-ColEmbed-VL-4B-v2", "Nemotron-ColEmbed-VL-8B-v2"],
            },
            "f16": {
                "config": "admitted structurally by the config/contract",
                "device": "not implemented: the additive Qwen3-VL kernels are BF16; F16 needs its own shader/exports",
                "readout": "F16 weights are widened exactly on the host",
                "action": "report to central shader owner; do not admit F16 at the backend",
            },
            "f32": {
                "config": "rejected for the GPU adapter; Native CPU reference only",
                "device": "not supported",
                "readout": "F32 host projection supported for tests/CPU readouts",
            },
        },
        "central_fa4_owner_packet": {
            "purpose": "Profiles required for Native-accepted Qwen3-VL tuples that are not in canonical26.",
            "profiles": required,
            "note": "Absent profiles are reported; no existing profile is borrowed or relabelled."
        },
        "non_claims": [
            "Derived 2B/32B tuples must be verified against the real config before admission.",
            "No GPU execution or profile qualification is claimed.",
            "30B-A3B MoE text is outside the dense Qwen3-VL foundation."
        ]
    })
}

// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Cloudflare Clef / Clef-flash typed decision metadata.
//!
//! Native `ClefHead` in `backends/libtorch/cpp/models/decision_models.cpp`
//! reads `joint_head_config.json` plus `joint_head.safetensors`. The head is
//! *not* an option-token readout and is not interchangeable with OneJev:
//!
//! * the backbone is dense Qwen3.5 (BF16 public checkpoints);
//! * the joint head keeps its Native tensors in the backbone model dtype
//!   (`bridge.cpp::weight_dtype` has no Clef override);
//! * requests carry ordered fields, each with a question span and ordered
//!   option spans relative to that request's sequence;
//! * `Warmup` synthesizes one field `{kind:0, question:(0,1), options:[(0,1)]}`;
//! * Clef always returns action probability `1.0`; there is no learned action
//!   classifier in this head.
//!
//! This leaf owns cold sidecar validation and pure request planning. It creates
//! no CUDA state, does not read tensor payloads while validating geometry, and
//! contains no CPU inference fallback.

use crate::model::{path_present, read_json, Checkpoint};
use infinity_v3_backend_core::{BackendError, DecisionInput};
use serde::Deserialize;
use std::{
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

pub(crate) const CLEF_CONFIG: &str = "joint_head_config.json";
pub(crate) const CLEF_WEIGHTS: &str = "joint_head.safetensors";
/// The public Cloudflare Clef checkpoints declare `dtype: bfloat16`.
pub(crate) const CLEF_RUNTIME_DTYPE: &str = "bfloat16";
pub(crate) const CLEF_FIELDS: usize = 3;
pub(crate) const MAX_CLEF_FIELDS: usize = 64;
pub(crate) const MAX_CLEF_ROUTING_LAYERS: usize = 16;
pub(crate) const MAX_CLEF_DECODER_LAYERS: usize = 32;
pub(crate) const MAX_CLEF_OPTIONS: usize = 512;
pub(crate) const MAX_CLEF_OPTION_TOKENS: usize = 8192;
pub(crate) const CLEF_NORM_EPS: f32 = 1e-5;
pub(crate) const CLEF_SCALE_CLAMP: f32 = 100.0;

static NEXT_HEAD_ID: AtomicU64 = AtomicU64::new(0);

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(error.to_string())
}

pub(crate) fn infer(error: impl std::fmt::Display) -> BackendError {
    BackendError::Inference(error.to_string())
}

#[derive(Clone, Debug, Deserialize)]
struct RawClefConfig {
    hidden_size: Option<usize>,
    width: Option<usize>,
    heads: Option<usize>,
    feedforward: Option<usize>,
    routing_layers: Option<usize>,
    layers: Option<usize>,
}

/// Validated `joint_head_config.json`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ClefConfig {
    pub(crate) hidden: usize,
    pub(crate) width: usize,
    pub(crate) heads: usize,
    pub(crate) feedforward: usize,
    pub(crate) routing_layers: usize,
    pub(crate) layers: usize,
}

impl ClefConfig {
    pub(crate) fn load(path: &Path, backbone_hidden: usize) -> Result<Self, BackendError> {
        let raw: RawClefConfig = read_json(&path.join(CLEF_CONFIG))?;
        let width = raw
            .width
            .ok_or_else(|| start("Clef joint_head_config.json is missing width"))?;
        let heads = raw
            .heads
            .ok_or_else(|| start("Clef joint_head_config.json is missing heads"))?;
        let feedforward = raw
            .feedforward
            .ok_or_else(|| start("Clef joint_head_config.json is missing feedforward"))?;
        let routing_layers = raw
            .routing_layers
            .ok_or_else(|| start("Clef joint_head_config.json is missing routing_layers"))?;
        let layers = raw
            .layers
            .ok_or_else(|| start("Clef joint_head_config.json is missing layers"))?;
        let hidden = raw.hidden_size.unwrap_or(backbone_hidden);
        if hidden != backbone_hidden {
            return Err(start(format!(
                "Clef joint head hidden_size {hidden} differs from backbone hidden_size {backbone_hidden}"
            )));
        }
        if width == 0 || heads == 0 || !width.is_multiple_of(heads) || feedforward == 0 {
            return Err(start(
                "Clef width/heads/feedforward must be positive and width divisible by heads",
            ));
        }
        if routing_layers == 0 || routing_layers > MAX_CLEF_ROUTING_LAYERS {
            return Err(start(format!(
                "Clef routing_layers {routing_layers} is outside 1..={MAX_CLEF_ROUTING_LAYERS}"
            )));
        }
        if layers == 0 || layers > MAX_CLEF_DECODER_LAYERS {
            return Err(start(format!(
                "Clef layers {layers} is outside 1..={MAX_CLEF_DECODER_LAYERS}"
            )));
        }
        let head_dim = width / heads;
        if head_dim == 0 || head_dim > 256 || !head_dim.is_multiple_of(8) {
            return Err(start(format!(
                "Clef attention head_dim {head_dim} is unsupported; Native requires a multiple of 8 up to 256"
            )));
        }
        Ok(Self {
            hidden,
            width,
            heads,
            feedforward,
            routing_layers,
            layers,
        })
    }

    pub(crate) fn head_dim(&self) -> usize {
        self.width / self.heads
    }
}

pub(crate) fn next_head_id() -> u64 {
    NEXT_HEAD_ID.fetch_add(1, Ordering::Relaxed)
}

/// Reject a dtype the current Qwen3.5 Clef route does not implement. The
/// public backbone and joint-head contract is BF16; F16 keeps an explicit
/// pending capability instead of silently running through BF16 math.
pub(crate) fn validate_runtime_dtype(dtype: &str) -> Result<(), BackendError> {
    if dtype == CLEF_RUNTIME_DTYPE {
        Ok(())
    } else {
        Err(start(format!(
            "Oxide Clef dtype {dtype} is unsupported: the public Clef route implements BF16 prepared storage, weight uploads and FA4 only; float16 Clef requires a separate F16 route"
        )))
    }
}

/// Validate sidecar placement and configuration before a CUDA context exists.
pub(crate) fn validate_checkpoint(
    path: &Path,
    dtype: &str,
    backbone_hidden: usize,
) -> Result<ClefConfig, BackendError> {
    validate_runtime_dtype(dtype)?;
    if !path_present(&path.join(CLEF_CONFIG))
        .map_err(|error| start(format!("Clef sidecar check failed: {error}")))?
    {
        return Err(start("Clef checkpoint requires joint_head_config.json"));
    }
    if !path_present(&path.join(CLEF_WEIGHTS))
        .map_err(|error| start(format!("Clef weight check failed: {error}")))?
    {
        return Err(start("Clef checkpoint requires joint_head.safetensors"));
    }
    ClefConfig::load(path, backbone_hidden)
}

/// Open the joint-head shard only. Backbone lexical weights come from the
/// caller's already validated Qwen3.5 checkpoint.
pub(crate) fn open_head_checkpoint(path: &Path) -> Result<Checkpoint, BackendError> {
    Checkpoint::open(&path.join(CLEF_WEIGHTS))
}

/// Number of flattened option logits for one request. This mirrors
/// `ClefHead::output_count`; Warmup always contributes one row.
pub(crate) fn request_output_count(request: &DecisionInput) -> Result<usize, BackendError> {
    match request {
        DecisionInput::Warmup => Ok(1),
        DecisionInput::Clef { fields } if !fields.is_empty() => {
            let mut total = 0usize;
            for field in fields {
                if field.kind >= CLEF_FIELDS || field.options.is_empty() {
                    return Err(infer("Clef field kind must be 0..3 with nonempty options"));
                }
                total = total
                    .checked_add(field.options.len())
                    .ok_or_else(|| infer("Clef option count overflow"))?;
            }
            Ok(total)
        }
        DecisionInput::Clef { .. } => Err(infer("Clef requires at least one field")),
        _ => Err(infer("Only Clef and Warmup requests reach the Clef head")),
    }
}

/// Flattened F32 option logits plus one action probability per request.
pub(crate) fn output_elements(requests: &[DecisionInput]) -> Result<usize, BackendError> {
    let mut logits = 0usize;
    for request in requests {
        logits = logits
            .checked_add(request_output_count(request)?)
            .ok_or_else(|| infer("Clef output count overflow"))?;
    }
    logits
        .checked_add(requests.len())
        .ok_or_else(|| infer("Clef action/output count overflow"))
}

pub(crate) fn result_bytes(output_elements: usize) -> Result<usize, BackendError> {
    output_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| infer("Clef result byte size overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use infinity_v3_backend_core::ClefField;
    use serde_json::json;
    use std::{
        fs,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering},
    };

    static NEXT: AtomicU64 = AtomicU64::new(0);

    fn fixture(config: serde_json::Value) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "infinity-clef-config-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        fs::write(path.join(CLEF_CONFIG), serde_json::to_vec(&config).unwrap()).unwrap();
        path
    }

    #[test]
    fn loads_native_joint_config_and_rejects_dtype_or_geometry() {
        let path = fixture(json!({
            "hidden_size": 16, "width": 16, "heads": 2, "feedforward": 32,
            "routing_layers": 1, "layers": 1
        }));
        let config = ClefConfig::load(&path, 16).unwrap();
        assert_eq!(config.head_dim(), 8);
        assert!(validate_runtime_dtype("bfloat16").is_ok());
        assert!(validate_runtime_dtype("float16").is_err());
        assert!(ClefConfig::load(&path, 32).is_err());
        fs::write(
            path.join(CLEF_CONFIG),
            serde_json::to_vec(&json!({
                "hidden_size": 16, "width": 16, "heads": 0, "feedforward": 32,
                "routing_layers": 1, "layers": 1
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(ClefConfig::load(&path, 16).is_err());
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn output_counts_preserve_field_and_option_order() {
        let requests = [
            DecisionInput::Clef {
                fields: vec![ClefField {
                    kind: 0,
                    question: (0, 1),
                    options: vec![(1, 2), (2, 3)],
                }],
            },
            DecisionInput::Warmup,
        ];
        assert_eq!(output_elements(&requests).unwrap(), 5);
        assert_eq!(result_bytes(5).unwrap(), 20);
        assert!(request_output_count(&DecisionInput::Clef { fields: vec![] }).is_err());
        assert!(request_output_count(&DecisionInput::OptionTokens { token_ids: vec![1] }).is_err());
    }
}

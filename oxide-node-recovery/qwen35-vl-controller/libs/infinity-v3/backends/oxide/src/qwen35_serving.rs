// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Cold serving admission for dense BF16 Qwen3.5 checkpoints.
//!
//! This module only reads model metadata and environment resource selectors. It
//! does not open weight payloads, bind CUDA state, or duplicate the explicit
//! Qwen3.5 constructors. The constructors remain authoritative for the actual
//! host config and device validation.

use crate::model::read_json;
use crate::qwen35_config::Qwen35Config;
use crate::qwen35_runtime::Qwen35TextOptions;
use infinity_v3_backend_core::{BackendError, ModelType, Pool};
use serde::Deserialize;
use std::path::{Path, PathBuf};

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide qwen3.5 serving: {error}"))
}

/// Block history currently retains exactly `kernel - 1 == 3` rows. Validate
/// the same checked layout used by Work before allocating or uploading a
/// checkpoint; the standalone fresh GDN operator accepts a wider geometry.
pub(crate) fn validate_block_config(config: &Qwen35Config) -> Result<(), BackendError> {
    if config.linear_conv_kernel_dim != 4 {
        return Err(start("block convolution history requires kernel width 4"));
    }
    let channels = config
        .linear_num_key_heads
        .checked_mul(2)
        .and_then(|keys| keys.checked_add(config.linear_num_value_heads))
        .and_then(|heads| heads.checked_mul(128))
        .ok_or_else(|| start("block GDN projection width overflow"))?;
    crate::qwen35_block_layout::Qwen35BlockLayout::new(
        1,
        1,
        config.linear_num_value_heads,
        channels,
    )
    .map_err(start)?;
    Ok(())
}

pub(crate) const FA4_ROOT_ENV: &str = infinity_v3_config::infinity_env!("FA4_ROOT");
pub(crate) const GDN_RAW_ENV: &str = infinity_v3_config::infinity_env!("OXIDE_GDN_RAW_KERNELS");
pub(crate) const GDN_PROVIDER_ENV: &str = infinity_v3_config::infinity_env!("OXIDE_GDN_PROVIDER");
pub(crate) const GDN_AUX_ENV: &str = infinity_v3_config::infinity_env!("OXIDE_GDN_AUX_KERNELS");

#[derive(Debug, Deserialize)]
struct Qwen35Marker {
    model_type: Option<String>,
    text_config: Option<Qwen35TextMarker>,
}

#[derive(Debug, Deserialize)]
struct Qwen35TextMarker {
    model_type: Option<String>,
}

fn is_qwen35_type(value: &str) -> bool {
    value == "qwen3_5" || value.starts_with("qwen3_5_")
}

/// Whether `config.json` declares the Qwen3.5 family at the root or in
/// `text_config`. Geometry and tensor validation stay in the explicit
/// constructors, which reload the actual checkpoint metadata.
pub(crate) fn is_qwen35_checkpoint(path: &Path) -> Result<bool, BackendError> {
    let marker: Qwen35Marker = read_json(&path.join("config.json"))?;
    let root = marker.model_type.as_deref().is_some_and(is_qwen35_type);
    let text = marker
        .text_config
        .as_ref()
        .and_then(|config| config.model_type.as_deref())
        .is_some_and(is_qwen35_type);
    Ok(root || text)
}

/// Backbone attention mode selected by the actual OneJev modeling overrides.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecisionAttention {
    Causal,
    BlockBidirectional,
}

impl DecisionAttention {
    const fn causal(self) -> bool {
        matches!(self, Self::Causal)
    }
}

/// Read model-owned attention metadata without trusting a stale config copy.
pub(crate) fn decision_attention(path: &Path) -> Result<DecisionAttention, BackendError> {
    let block =
        infinity_v3_backend_core::block_attention_config::block_attention(path).map_err(start)?;
    Ok(if block {
        DecisionAttention::BlockBidirectional
    } else {
        DecisionAttention::Causal
    })
}

/// Native's ordinary full-attention mask is causal unless the checkpoint
/// explicitly requests bidirectional attention. A block mask is admitted only
/// through the Decision route, whose prefix comes from the actual request.
pub(crate) fn text_causal(path: &Path) -> Result<bool, BackendError> {
    let config: serde_json::Value = read_json(&path.join("config.json"))?;
    let selector = |name: &str| -> Result<Option<&serde_json::Value>, BackendError> {
        let root = config.get(name).filter(|value| !value.is_null());
        let text = config
            .get("text_config")
            .and_then(|text| text.get(name))
            .filter(|value| !value.is_null());
        match (root, text) {
            (Some(root), Some(text)) if root != text => Err(start(format!(
                "conflicting {name} across root and text_config"
            ))),
            (_, Some(text)) => Ok(Some(text)),
            (root, None) => Ok(root),
        }
    };
    let bidirectional = selector("use_bidirectional_attention")?
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| start("use_bidirectional_attention must be boolean"))
        })
        .transpose()?
        .unwrap_or(false);
    let mode = selector("decision_attention_mode")?
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| start("decision_attention_mode must be a string"))
        })
        .transpose()?
        .unwrap_or("causal");
    match mode {
        "causal" => Ok(!bidirectional),
        "noncausal_full_attention" => Ok(false),
        _ => Err(start(format!(
            "unsupported ordinary full-attention mode {mode}"
        ))),
    }
}

/// Cold resource paths required by the explicit Qwen3.5 text/decision routes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ColdResources {
    pub(crate) fa4_root: PathBuf,
    pub(crate) gdn_raw_kernels: PathBuf,
    pub(crate) gdn_provider: PathBuf,
    pub(crate) gdn_aux_kernels: PathBuf,
}

impl ColdResources {
    pub(crate) fn into_options(self, causal: bool) -> Qwen35TextOptions {
        Qwen35TextOptions {
            causal,
            fa4_root: self.fa4_root,
            gdn_raw_kernels: self.gdn_raw_kernels,
            gdn_provider: self.gdn_provider,
            gdn_aux_kernels: self.gdn_aux_kernels,
        }
    }
}

fn required_path(name: &str, value: Option<PathBuf>) -> Result<PathBuf, BackendError> {
    value
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or_else(|| start(format!("{name} is required for Qwen3.5 serving")))
}

pub(crate) fn resolve_cold_resources(
    fa4_root: Option<PathBuf>,
    gdn_raw_kernels: Option<PathBuf>,
    gdn_provider: Option<PathBuf>,
    gdn_aux_kernels: Option<PathBuf>,
) -> Result<ColdResources, BackendError> {
    Ok(ColdResources {
        fa4_root: required_path(FA4_ROOT_ENV, fa4_root)?,
        gdn_raw_kernels: required_path(GDN_RAW_ENV, gdn_raw_kernels)?,
        gdn_provider: required_path(GDN_PROVIDER_ENV, gdn_provider)?,
        gdn_aux_kernels: required_path(GDN_AUX_ENV, gdn_aux_kernels)?,
    })
}

/// Resolve every explicit cold resource before any CUDA context or weight
/// upload. Internal qualification controls keep their separate selectors.
pub(crate) fn cold_resources() -> Result<ColdResources, BackendError> {
    resolve_cold_resources(
        std::env::var_os(FA4_ROOT_ENV).map(PathBuf::from),
        std::env::var_os(GDN_RAW_ENV).map(PathBuf::from),
        std::env::var_os(GDN_PROVIDER_ENV).map(PathBuf::from),
        std::env::var_os(GDN_AUX_ENV).map(PathBuf::from),
    )
}

/// One admitted serving route. Decision checkpoints select the block route only
/// when the actual OneJev metadata says `block_bidirectional`; causal
/// `OptionTokens` checkpoints keep the ordinary text-decision route.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ServingRoute {
    Text { pool: Pool, causal: bool },
    TextDecision,
    DynamicBlockDecision,
}

pub(crate) fn route(
    dtype: &str,
    model_type: &ModelType,
    dense_paths: Option<&[String]>,
    attention: DecisionAttention,
) -> Result<ServingRoute, BackendError> {
    if dtype != "bfloat16" {
        return Err(start(
            "Qwen3.5 serving requires --dtype bfloat16; FP16 and FP32 are unsupported",
        ));
    }
    if dense_paths.is_some_and(|paths| !paths.is_empty()) {
        return Err(start(
            "Qwen3.5 Dense projection modules are not yet admitted",
        ));
    }
    match model_type {
        ModelType::Embedding(pool) => {
            if attention == DecisionAttention::BlockBidirectional {
                return Err(start(
                    "Block attention requires a Qwen3.5 Decision model; ordinary embeddings do not carry a prefix",
                ));
            }
            if !matches!(pool, Pool::Cls | Pool::Mean | Pool::LastToken) {
                return Err(start(
                    "Qwen3.5 serving accepts only CLS, mean and last-token embeddings",
                ));
            }
            Ok(ServingRoute::Text {
                pool: Pool::clone(pool),
                causal: attention.causal(),
            })
        }
        ModelType::Decision => Ok(match attention {
            DecisionAttention::BlockBidirectional => ServingRoute::DynamicBlockDecision,
            DecisionAttention::Causal => ServingRoute::TextDecision,
        }),
        ModelType::Classifier => Err(start("Qwen3.5 classifier serving is unsupported")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn temp_dir() -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "oxide-qwen35-serving-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).expect("create fixture directory");
        path
    }

    fn marker(path: &Path, value: &[u8]) {
        fs::write(path.join("config.json"), value).expect("write config fixture");
    }

    #[test]
    fn detects_root_and_text_config_families() {
        let path = temp_dir();
        marker(
            &path,
            br#"{"model_type":"qwen3_5","text_config":{"model_type":"qwen3_5_text"}}"#,
        );
        assert!(is_qwen35_checkpoint(&path).unwrap());
        marker(
            &path,
            br#"{"model_type":"conditional_generation","text_config":{"model_type":"qwen3_5_text"}}"#,
        );
        assert!(is_qwen35_checkpoint(&path).unwrap());
        marker(&path, br#"{"model_type":"modernbert"}"#);
        assert!(!is_qwen35_checkpoint(&path).unwrap());
        marker(&path, b"{");
        assert!(is_qwen35_checkpoint(&path).is_err());
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn decision_attention_follows_actual_modeling_overrides() {
        let path = temp_dir();
        marker(
            &path,
            br#"{"model_type":"qwen3_5","modeling_overrides":{"readout":"slot_lm","attention":{"full_attention":"block_bidirectional","linear_attention":"causal"}}}"#,
        );
        assert_eq!(
            decision_attention(&path).unwrap(),
            DecisionAttention::BlockBidirectional
        );
        marker(
            &path,
            br#"{"model_type":"qwen3_5","modeling_overrides":{"readout":"slot_lm","attention":{"full_attention":"causal","linear_attention":"causal"}}}"#,
        );
        assert_eq!(
            decision_attention(&path).unwrap(),
            DecisionAttention::Causal
        );
        marker(
            &path,
            br#"{"model_type":"qwen3_5","modeling_overrides":{"readout":"slot_lm","attention":{"full_attention":"bidirectional","linear_attention":"causal"}}}"#,
        );
        assert!(decision_attention(&path).is_err());
        marker(
            &path,
            br#"{"model_type":"qwen3_5","modeling_overrides":{"readout":"pointer","attention":{"full_attention":"block_bidirectional","linear_attention":"causal"}}}"#,
        );
        assert!(decision_attention(&path).is_err());
        let _ = fs::remove_dir_all(path);
    }

    #[test]
    fn ordinary_attention_respects_explicit_native_mask_and_rejects_conflicts() {
        let path = temp_dir();
        for (value, expected) in [
            (br#"{"model_type":"qwen3_5_text"}"#.as_slice(), true),
            (br#"{"use_bidirectional_attention":true}"#.as_slice(), false),
            (
                br#"{"decision_attention_mode":"noncausal_full_attention"}"#.as_slice(),
                false,
            ),
            (
                br#"{"text_config":{"use_bidirectional_attention":true}}"#.as_slice(),
                false,
            ),
        ] {
            marker(&path, value);
            assert_eq!(text_causal(&path).unwrap(), expected);
        }
        for value in [
            br#"{"use_bidirectional_attention":"true"}"#.as_slice(),
            br#"{"use_bidirectional_attention":true,"text_config":{"use_bidirectional_attention":false}}"#.as_slice(),
            br#"{"decision_attention_mode":"block_bidirectional"}"#.as_slice(),
        ] {
            marker(&path, value);
            assert!(text_causal(&path).is_err());
        }
        fs::remove_dir_all(path).unwrap();
    }

    fn p(name: &str) -> PathBuf {
        PathBuf::from(name)
    }

    #[test]
    fn cold_resources_require_each_explicit_selector() {
        let valid = resolve_cold_resources(
            Some(p("fa4")),
            Some(p("raw")),
            Some(p("provider")),
            Some(p("aux")),
        )
        .unwrap();
        assert_eq!(valid.fa4_root, p("fa4"));
        assert_eq!(valid.gdn_aux_kernels, p("aux"));

        let cases = [
            (
                (None, Some(p("raw")), Some(p("provider")), Some(p("aux"))),
                FA4_ROOT_ENV,
            ),
            (
                (Some(p("fa4")), None, Some(p("provider")), Some(p("aux"))),
                GDN_RAW_ENV,
            ),
            (
                (Some(p("fa4")), Some(p("raw")), None, Some(p("aux"))),
                GDN_PROVIDER_ENV,
            ),
            (
                (Some(p("fa4")), Some(p("raw")), Some(p("provider")), None),
                GDN_AUX_ENV,
            ),
            (
                (
                    Some(p("fa4")),
                    Some(p("raw")),
                    Some(p("provider")),
                    Some(PathBuf::new()),
                ),
                GDN_AUX_ENV,
            ),
        ];
        for ((fa4, raw, provider, aux), expected) in cases {
            let error = resolve_cold_resources(fa4, raw, provider, aux)
                .expect_err("missing resource must reject")
                .to_string();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn block_admission_matches_actual_history_and_projection_layout() {
        use crate::qwen35_config::Qwen35LayerType;
        let mut config = Qwen35Config {
            hidden_size: 4096,
            intermediate_size: 12288,
            num_hidden_layers: 1,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 256,
            rotary_dim: 64,
            vocab_size: 256,
            max_position_embeddings: 32768,
            linear_num_key_heads: 16,
            linear_num_value_heads: 32,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 10000000.0,
            layer_types: vec![Qwen35LayerType::Linear],
            attn_output_gate: true,
        };
        for value_heads in [32, 48] {
            config.linear_num_value_heads = value_heads;
            validate_block_config(&config).unwrap();
        }
        for kernel in [2, 3, 5, 8] {
            config.linear_conv_kernel_dim = kernel;
            assert!(validate_block_config(&config)
                .unwrap_err()
                .to_string()
                .contains("kernel width 4"));
        }
        config.linear_conv_kernel_dim = 4;
        config.linear_num_value_heads = 16;
        assert!(validate_block_config(&config).is_err());
        config.linear_num_value_heads = 32;
        config.linear_num_key_heads = 8;
        assert!(validate_block_config(&config).is_err());
    }

    #[test]
    fn dispatch_preserves_text_and_selects_decision_route_from_metadata() {
        let dense = ["2_Dense".to_string()];
        assert_eq!(
            route(
                "bfloat16",
                &ModelType::Embedding(Pool::Mean),
                None,
                DecisionAttention::Causal,
            )
            .unwrap(),
            ServingRoute::Text {
                pool: Pool::Mean,
                causal: true,
            }
        );
        assert!(route(
            "bfloat16",
            &ModelType::Embedding(Pool::LastToken),
            None,
            DecisionAttention::BlockBidirectional,
        )
        .unwrap_err()
        .to_string()
        .contains("do not carry a prefix"));
        assert_eq!(
            route(
                "bfloat16",
                &ModelType::Decision,
                None,
                DecisionAttention::BlockBidirectional,
            )
            .unwrap(),
            ServingRoute::DynamicBlockDecision
        );
        assert_eq!(
            route(
                "bfloat16",
                &ModelType::Decision,
                None,
                DecisionAttention::Causal,
            )
            .unwrap(),
            ServingRoute::TextDecision
        );

        for (dtype, model_type, paths, diagnostic) in [
            (
                "float16",
                ModelType::Embedding(Pool::Mean),
                None,
                "bfloat16",
            ),
            (
                "bfloat16",
                ModelType::Embedding(Pool::Splade),
                None,
                "CLS, mean and last-token",
            ),
            (
                "bfloat16",
                ModelType::Embedding(Pool::Mean),
                Some(&dense[..]),
                "Dense projection",
            ),
            ("bfloat16", ModelType::Classifier, None, "classifier"),
        ] {
            let error = route(dtype, &model_type, paths, DecisionAttention::Causal)
                .expect_err("unsupported route must reject")
                .to_string();
            assert!(error.contains(diagnostic), "{error}");
        }
    }
}

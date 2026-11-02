// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Cold kernel-entry loading for the Clef joint head.

use crate::{infer, start};
use cuda_core::{Context, Function};
use infinity_v3_backend_core::BackendError;
use std::sync::Arc;

/// Functions loaded from the existing shared oxide artifact. The `clef_*`
/// entries are the small source-only device leaf; until Root composes them the
/// head fails at cold load instead of substituting a CPU or generic readout.
pub(crate) struct ClefKernels {
    pub(crate) layer_norm: Function,
    pub(crate) scalar_prep: Function,
    pub(crate) span_mean: Function,
    pub(crate) embedding_gather: Function,
    pub(crate) bias_seed: Function,
    pub(crate) gelu_exact: Function,
    pub(crate) residual_add: Function,
    pub(crate) query_compose: Function,
    pub(crate) route_scores: Function,
    pub(crate) softmax_sum: Function,
    pub(crate) field_compose: Function,
    pub(crate) anchor: Function,
    pub(crate) prior: Function,
    pub(crate) features: Function,
    pub(crate) logits: Function,
}

impl ClefKernels {
    pub(crate) fn load(context: &Arc<Context>) -> Result<Self, BackendError> {
        let path = std::env::var_os(infinity_v3_config::infinity_env!("OXIDE_KERNEL_MODULE"))
            .ok_or_else(|| {
                start("Set INFINITY_OXIDE_KERNEL_MODULE to the artifact compiled by cargo oxide")
            })?;
        let path = std::path::PathBuf::from(path);
        let path = if path.is_dir() {
            let (major, minor) = context.compute_capability().map_err(infer)?;
            path.join(format!("sm_{major}{minor}"))
                .join("kernels.cubin")
        } else {
            path
        };
        let data =
            std::fs::read(&path).map_err(|error| start(format!("{}: {error}", path.display())))?;
        let module = context.load_module_from_image(&data).map_err(infer)?;
        let load = |name: &str| module.load_function(name).map_err(infer);
        Ok(Self {
            layer_norm: load("infinity_layer_norm")?,
            scalar_prep: load("infinity_clef_scalar_prep_bf16")?,
            span_mean: load("infinity_clef_span_mean_bf16")?,
            embedding_gather: load("infinity_embedding_gather")?,
            bias_seed: load("infinity_classifier_bias_bf16")?,
            gelu_exact: load("infinity_gelu_exact")?,
            residual_add: load("infinity_clef_residual_add_bf16")?,
            query_compose: load("infinity_clef_query_compose_bf16")?,
            route_scores: load("infinity_clef_route_scores_bf16")?,
            softmax_sum: load("infinity_clef_softmax_sum_bf16")?,
            field_compose: load("infinity_clef_field_compose_bf16")?,
            anchor: load("infinity_clef_anchor_bf16")?,
            prior: load("infinity_clef_prior_bf16")?,
            features: load("infinity_clef_features_bf16")?,
            logits: load("infinity_clef_logits_f32")?,
        })
    }
}

// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! CPU-only routed Qwen3 MLP metadata and retained checkpoint views.
//! This foundation does not admit a model, load CUDA, or select GPU operators.
#[path = "moe_config.rs"]
mod config;
#[path = "moe_weights.rs"]
mod weights;
pub(crate) use config::RoutedMlpConfig;
pub(crate) use weights::{load_routed_layer, GateUpSources, RoutedLayerWeights};

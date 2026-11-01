// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
//! Prepared GTE Native RoPE and gated feed-forward policy.
//!
//! Tables and CUDA functions are bound once during model construction. Requests
//! only launch the prepared functions against retained buffers; no recipe string
//! matching or table allocation occurs on the inference path.

use crate::{
    model::{BertConfig, RotaryRecipe},
    rotary::BaseRopeTables,
    start, u32_size,
};
use cuda_core::{DeviceBuffer, Function, Stream};
use infinity_v3_backend_core::BackendError;
use std::sync::Arc;

const ROTARY_TABLE_BYTE_BUDGET: usize = 64 * 1024 * 1024;
const GTE_HEAD_DIM: usize = 64;

/// Kernels loaded from the encoder's existing CUDA module. Keeping the module
/// owner in `Kernels` avoids loading a second copy of the same artifact.
pub(crate) struct RotaryFunctions {
    pub(crate) rope_hfma: Function,
    pub(crate) gated_gelu: Function,
}

/// Load-time GTE policy. The F32 device tables are expanded from the BF16
/// phases produced by `BaseRopeTables`, matching the native HFMA ABI.
pub(crate) struct RotaryEncoder {
    pub(crate) rope_hfma: Function,
    pub(crate) gated_gelu: Function,
    pub(crate) cosine: DeviceBuffer<f32>,
    pub(crate) sine: DeviceBuffer<f32>,
    pub(crate) head_dim: u32,
    pub(crate) table_stride: u32,
}
impl RotaryEncoder {
    pub(crate) fn new(
        recipe: RotaryRecipe,
        config: &BertConfig,
        stream: &Arc<Stream>,
        functions: RotaryFunctions,
    ) -> Result<Self, BackendError> {
        let head_dim = config
            .hidden_size
            .checked_div(config.num_attention_heads)
            .ok_or_else(|| start("Invalid GTE rotary head dimension"))?;
        if head_dim != GTE_HEAD_DIM {
            return Err(start(format!(
                "GTE rotary policy requires a {GTE_HEAD_DIM}-wide attention head, got {head_dim}"
            )));
        }
        let capacity = config.max_position_embeddings;
        let tables = match recipe.ntk_factor {
            Some(factor) => BaseRopeTables::new_ntk(
                head_dim,
                recipe.theta,
                factor,
                capacity,
                ROTARY_TABLE_BYTE_BUDGET,
            ),
            None => BaseRopeTables::new(head_dim, recipe.theta, capacity, ROTARY_TABLE_BYTE_BUDGET),
        }
        .map_err(|error| start(format!("GTE rotary table construction failed: {error}")))?;
        let head_dim = u32_size(tables.head_dim())?;
        let table_stride = u32_size(tables.table_stride())?;
        let cosine = DeviceBuffer::from_host(stream, tables.cosine()).map_err(start)?;
        let sine = DeviceBuffer::from_host(stream, tables.sine()).map_err(start)?;
        Ok(Self {
            rope_hfma: functions.rope_hfma,
            gated_gelu: functions.gated_gelu,
            cosine,
            sine,
            head_dim,
            table_stride,
        })
    }
}

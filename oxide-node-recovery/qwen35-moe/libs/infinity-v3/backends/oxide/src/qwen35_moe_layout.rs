// Native Qwen3.5-MoE routed workspace ABI.
//
// This layout mirrors the portable non-Hopper provider selected by
// `qwen35_models.cpp`: `routed_moe_cuda` -> `qwen35_moe_forward_bf16` in the
// candle `grouped_gemm.cu` / `gemma4_moe.cu` kernels. Hopper builds use
// `grouped_gemm_hopper.cu` with a different ABI (BF16 expert output,
// sorted/row weights, 2 MiB GEMM workspace) and are out of scope.
// The portable provider allocates one scratch arena and carves it with
// 256-byte alignment, in this exact order:
//
//   int counts[E], int offsets[E+1], int cursors[E],
//   u32 ids[slots], u32 mapping[slots], float weights[slots],
//   bf16 packed[slots*H], bf16 gate_up[slots*2I], bf16 activated[slots*I],
//   float expert_output[slots*H],
//   GemmCoord problems[E] (12 bytes each),
//   bf16* a[E], bf16* b[E], bf16* c[E], float* c_f32[E],
//   int64 lda[E], ldb[E], ldc[E]
//
// Every `take` aligns to 256 bytes, and the final size is aligned to 256.
// The down projection writes FP32 `expert_output`; the combine then rounds
// each weighted term to BF16 and sums in FP32. Omitting that FP32 accumulator
// would understate the real Native workspace (review F3).
//
// This is a cold ABI layout only. It allocates nothing and executes nothing.

use crate::qwen35_layout::AttentionSlabLayout;
use crate::qwen35_moe_config::Qwen35MoeConfig;
use infinity_v3_backend_core::BackendError;

const NATIVE_ALIGNMENT: usize = 256;
const INT_MAX: usize = i32::MAX as usize;
const GEMM_COORD_BYTES: usize = 12;
const POINTER_BYTES: usize = 8;
const I64_BYTES: usize = 8;
const BF16_BYTES: usize = 2;
const F32_BYTES: usize = 4;
const U32_BYTES: usize = 4;
const I32_BYTES: usize = 4;
const WORKSPACE_RESERVE: usize = 4 * 1024 * 1024;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide qwen3.5-moe layout: {error}"))
}

fn product(left: usize, right: usize, name: &str) -> Result<usize, BackendError> {
    left.checked_mul(right)
        .ok_or_else(|| start(format!("{name} overflow")))
}

fn add(left: usize, right: usize, name: &str) -> Result<usize, BackendError> {
    left.checked_add(right)
        .ok_or_else(|| start(format!("{name} overflow")))
}

fn align_up(value: usize, alignment: usize) -> Result<usize, BackendError> {
    if alignment == 0 {
        return Err(start("alignment must be positive"));
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| start("alignment overflow"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MoeSpan {
    offset_bytes: usize,
    bytes: usize,
    row_width: usize,
    element_bytes: usize,
}

impl MoeSpan {
    fn new(
        offset_bytes: usize,
        elements: usize,
        row_width: usize,
        element_bytes: usize,
        name: &str,
    ) -> Result<Self, BackendError> {
        if element_bytes == 0 {
            return Err(start("span element width must be nonzero"));
        }
        if elements != 0 && row_width == 0 {
            return Err(start(format!("{name} has elements but no row width")));
        }
        if row_width != 0 && !elements.is_multiple_of(row_width) {
            return Err(start(format!("{name} elements are not whole rows")));
        }
        if !offset_bytes.is_multiple_of(NATIVE_ALIGNMENT) {
            return Err(start(format!("{name} offset is not 256-byte aligned")));
        }
        let bytes = product(elements, element_bytes, name)?;
        add(offset_bytes, bytes, name)?;
        Ok(Self {
            offset_bytes,
            bytes,
            row_width,
            element_bytes,
        })
    }

    pub(crate) const fn offset_bytes(&self) -> usize {
        self.offset_bytes
    }

    pub(crate) const fn bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) const fn row_width(&self) -> usize {
        self.row_width
    }

    pub(crate) const fn element_bytes(&self) -> usize {
        self.element_bytes
    }

    pub(crate) fn elements(&self) -> usize {
        if self.element_bytes == 0 {
            0
        } else {
            self.bytes / self.element_bytes
        }
    }

    pub(crate) fn rows(&self) -> usize {
        if self.row_width == 0 {
            0
        } else {
            self.elements() / self.row_width
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Qwen35MoeNativeScratchLayout {
    pub(crate) capacity_tokens: usize,
    pub(crate) experts: usize,
    pub(crate) hidden: usize,
    pub(crate) intermediate: usize,
    pub(crate) slots: usize,
    pub(crate) counts_i32: MoeSpan,
    pub(crate) offsets_i32: MoeSpan,
    pub(crate) cursors_i32: MoeSpan,
    pub(crate) ids_u32: MoeSpan,
    pub(crate) mapping_u32: MoeSpan,
    pub(crate) route_weights_f32: MoeSpan,
    pub(crate) packed_bf16: MoeSpan,
    pub(crate) gate_up_bf16: MoeSpan,
    pub(crate) activated_bf16: MoeSpan,
    pub(crate) expert_output_f32: MoeSpan,
    pub(crate) problems_gemm_coord: MoeSpan,
    pub(crate) a_ptrs: MoeSpan,
    pub(crate) b_ptrs: MoeSpan,
    pub(crate) c_ptrs: MoeSpan,
    pub(crate) c_f32_ptrs: MoeSpan,
    pub(crate) lda_i64: MoeSpan,
    pub(crate) ldb_i64: MoeSpan,
    pub(crate) ldc_i64: MoeSpan,
    pub(crate) bytes: usize,
}

struct Cursor {
    offset: usize,
}

impl Cursor {
    fn new() -> Self {
        Self { offset: 0 }
    }

    fn take(
        &mut self,
        elements: usize,
        row_width: usize,
        element_bytes: usize,
        name: &str,
    ) -> Result<MoeSpan, BackendError> {
        self.offset = align_up(self.offset, NATIVE_ALIGNMENT)?;
        let span = MoeSpan::new(self.offset, elements, row_width, element_bytes, name)?;
        self.offset = add(self.offset, span.bytes(), name)?;
        Ok(span)
    }
}

/// Exact integer mirror of `routed_workspace_bytes` from `grouped_gemm.cu`.
pub(crate) fn native_workspace_bytes(
    tokens: usize,
    hidden: usize,
    intermediate: usize,
    experts: usize,
) -> Result<usize, BackendError> {
    if tokens == 0 || tokens > INT_MAX / 8 {
        return Err(start("routed tokens must be in 1..=INT_MAX/8"));
    }
    if hidden == 0 || hidden > INT_MAX || !hidden.is_multiple_of(8) {
        return Err(start("routed hidden must be a positive multiple of 8"));
    }
    if intermediate == 0 || intermediate > INT_MAX / 2 || !intermediate.is_multiple_of(8) {
        return Err(start(
            "routed intermediate must be a positive multiple of 8 no larger than INT_MAX/2",
        ));
    }
    if experts == 0 {
        return Err(start("routed expert count must be positive"));
    }
    let slots = product(tokens, 8, "routed slot count")?;
    let slots64 = slots as u64;
    let hidden64 = hidden as u64;
    let intermediate64 = intermediate as u64;
    let denominator = hidden64
        .checked_mul(6)
        .and_then(|value| value.checked_add(intermediate64 * 6))
        .and_then(|value| value.checked_add(64))
        .ok_or_else(|| start("routed workspace guard overflow"))?;
    let limit = u64::MAX
        .checked_sub(WORKSPACE_RESERVE as u64)
        .ok_or_else(|| start("routed workspace reserve overflow"))?
        / denominator;
    if slots64 > limit {
        return Err(start("routed workspace dimensions exceed the Native guard"));
    }
    let layout = Qwen35MoeNativeScratchLayout::build(tokens, experts, hidden, intermediate, slots)?;
    Ok(layout.bytes)
}

impl Qwen35MoeNativeScratchLayout {
    pub(crate) fn new(
        capacity_tokens: usize,
        hidden: usize,
        intermediate: usize,
        experts: usize,
    ) -> Result<Self, BackendError> {
        let bytes = native_workspace_bytes(capacity_tokens, hidden, intermediate, experts)?;
        let slots = product(capacity_tokens, 8, "routed slot count")?;
        let mut layout = Self::build(capacity_tokens, experts, hidden, intermediate, slots)?;
        if layout.bytes != bytes {
            return Err(start(
                "routed workspace byte count disagrees with its spans",
            ));
        }
        layout.bytes = bytes;
        Ok(layout)
    }

    pub(crate) fn from_config(
        config: &Qwen35MoeConfig,
        capacity_tokens: usize,
    ) -> Result<Self, BackendError> {
        if config.moe.num_experts_per_tok != 8 {
            return Err(start(
                "the Native Qwen3.5 routed workspace contract is fixed at top-8",
            ));
        }
        Self::new(
            capacity_tokens,
            config.text.hidden_size,
            config.moe.moe_intermediate_size,
            config.moe.num_experts,
        )
    }

    fn build(
        capacity_tokens: usize,
        experts: usize,
        hidden: usize,
        intermediate: usize,
        slots: usize,
    ) -> Result<Self, BackendError> {
        let two_i = product(intermediate, 2, "gate/up width")?;
        let mut cursor = Cursor::new();
        let counts_i32 = cursor.take(experts, 1, I32_BYTES, "counts")?;
        let offsets_i32 = cursor.take(add(experts, 1, "offset count")?, 1, I32_BYTES, "offsets")?;
        let cursors_i32 = cursor.take(experts, 1, I32_BYTES, "cursors")?;
        let ids_u32 = cursor.take(slots, 1, U32_BYTES, "ids")?;
        let mapping_u32 = cursor.take(slots, 1, U32_BYTES, "mapping")?;
        let route_weights_f32 = cursor.take(slots, 1, F32_BYTES, "route weights")?;
        let packed_bf16 = cursor.take(
            product(slots, hidden, "packed elements")?,
            hidden,
            BF16_BYTES,
            "packed rows",
        )?;
        let gate_up_bf16 = cursor.take(
            product(slots, two_i, "gate/up elements")?,
            two_i,
            BF16_BYTES,
            "gate/up rows",
        )?;
        let activated_bf16 = cursor.take(
            product(slots, intermediate, "activation elements")?,
            intermediate,
            BF16_BYTES,
            "activation rows",
        )?;
        let expert_output_f32 = cursor.take(
            product(slots, hidden, "expert output elements")?,
            hidden,
            F32_BYTES,
            "expert output rows",
        )?;
        let problems_gemm_coord = cursor.take(experts, 1, GEMM_COORD_BYTES, "grouped problems")?;
        let a_ptrs = cursor.take(experts, 1, POINTER_BYTES, "a pointers")?;
        let b_ptrs = cursor.take(experts, 1, POINTER_BYTES, "b pointers")?;
        let c_ptrs = cursor.take(experts, 1, POINTER_BYTES, "c pointers")?;
        let c_f32_ptrs = cursor.take(experts, 1, POINTER_BYTES, "c f32 pointers")?;
        let lda_i64 = cursor.take(experts, 1, I64_BYTES, "lda")?;
        let ldb_i64 = cursor.take(experts, 1, I64_BYTES, "ldb")?;
        let ldc_i64 = cursor.take(experts, 1, I64_BYTES, "ldc")?;
        let bytes = align_up(cursor.offset, NATIVE_ALIGNMENT)?;
        Ok(Self {
            capacity_tokens,
            experts,
            hidden,
            intermediate,
            slots,
            counts_i32,
            offsets_i32,
            cursors_i32,
            ids_u32,
            mapping_u32,
            route_weights_f32,
            packed_bf16,
            gate_up_bf16,
            activated_bf16,
            expert_output_f32,
            problems_gemm_coord,
            a_ptrs,
            b_ptrs,
            c_ptrs,
            c_f32_ptrs,
            lda_i64,
            ldb_i64,
            ldc_i64,
            bytes,
        })
    }

    pub(crate) const fn total_bytes(&self) -> usize {
        self.bytes
    }
}

/// Attention-only allocation geometry for the hybrid MoE backbone.
///
/// This is intentionally not a dense `Qwen35Layout`: there is no
/// `intermediate_size`, no dense gate/up/down weight, and no dense FFN span.
/// The three BF16 phase slabs cover only embedding/attention/GDN work; routed
/// and shared FFN scratch is allocated separately by `Qwen35MoeFullScratch`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35MoeAttentionLayout {
    pub(crate) capacity_tokens: usize,
    pub(crate) hidden_elements: usize,
    pub(crate) projection_elements: usize,
    pub(crate) activation_elements: usize,
    pub(crate) slab_bytes: usize,
}

impl Qwen35MoeAttentionLayout {
    pub fn new(config: &Qwen35MoeConfig, capacity_tokens: usize) -> Result<Self, BackendError> {
        if capacity_tokens == 0 {
            return Err(start("capacity_tokens must be nonzero"));
        }
        let text = &config.text;
        if text.hidden_size == 0
            || text.head_dim == 0
            || text.num_attention_heads == 0
            || text.num_key_value_heads == 0
        {
            return Err(start("hidden and attention dimensions must be nonzero"));
        }
        let q_width = product(text.num_attention_heads, text.head_dim, "query width")?;
        let kv_width = product(text.num_key_value_heads, text.head_dim, "key/value width")?;
        let query_gate_width = product(q_width, 2, "query gate width")?;
        let full_projection_width = add(
            query_gate_width,
            product(kv_width, 2, "KV width")?,
            "full projection width",
        )?;
        let full_activation_width = add(
            product(q_width, 2, "full activation width")?,
            kv_width,
            "full activation width",
        )?;

        let linear = text
            .layer_types
            .contains(&crate::qwen35_config::Qwen35LayerType::Linear);
        let (linear_qkv_width, linear_activation_width) = if linear {
            if text.linear_num_key_heads == 0 || text.linear_num_value_heads == 0 {
                return Err(start(
                    "linear attention requires nonzero key and value heads",
                ));
            }
            let key_width = product(
                text.linear_num_key_heads,
                text.linear_key_head_dim,
                "linear key width",
            )?;
            let value_width = product(
                text.linear_num_value_heads,
                text.linear_value_head_dim,
                "linear value width",
            )?;
            let qkv = add(
                product(key_width, 2, "linear QKV width")?,
                value_width,
                "linear QKV width",
            )?;
            let ab_width = product(text.linear_num_value_heads, 2, "linear A/B width")?;
            let activation = add(
                add(text.hidden_size, value_width, "linear activation width")?,
                ab_width,
                "linear activation width",
            )?;
            (qkv, activation)
        } else {
            (0, 0)
        };

        let projection_width = full_projection_width
            .max(linear_qkv_width)
            .max(text.hidden_size);
        let activation_width = full_activation_width
            .max(linear_activation_width)
            .max(text.hidden_size);
        let hidden_elements = product(capacity_tokens, text.hidden_size, "hidden slab")?;
        let projection_elements = product(capacity_tokens, projection_width, "projection slab")?;
        let activation_elements = product(capacity_tokens, activation_width, "activation slab")?;
        let slab_bytes = product(
            add(
                add(hidden_elements, projection_elements, "slab elements")?,
                activation_elements,
                "slab elements",
            )?,
            BF16_BYTES,
            "slab bytes",
        )?;
        Ok(Self {
            capacity_tokens,
            hidden_elements,
            projection_elements,
            activation_elements,
            slab_bytes,
        })
    }
}

impl Qwen35MoeAttentionLayout {
    pub const fn capacity_tokens_value(&self) -> usize {
        self.capacity_tokens
    }
    pub const fn hidden_elements_value(&self) -> usize {
        self.hidden_elements
    }
    pub const fn projection_elements_value(&self) -> usize {
        self.projection_elements
    }
    pub const fn activation_elements_value(&self) -> usize {
        self.activation_elements
    }
    pub const fn slab_bytes(&self) -> usize {
        self.slab_bytes
    }
    pub const fn has_dense_ffn_span(&self) -> bool {
        false
    }
}

impl AttentionSlabLayout for Qwen35MoeAttentionLayout {
    fn capacity_tokens(&self) -> usize {
        self.capacity_tokens
    }
    fn hidden_elements(&self) -> usize {
        self.hidden_elements
    }
    fn projection_elements(&self) -> usize {
        self.projection_elements
    }
    fn activation_elements(&self) -> usize {
        self.activation_elements
    }
    fn slab_bytes(&self) -> usize {
        self.slab_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config() -> Qwen35MoeConfig {
        Qwen35MoeConfig::from_json(&json!({
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "layer_types": ["full_attention"],
            "attn_output_gate": true,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8
        }))
        .unwrap()
    }

    #[test]
    fn native_abi_has_f32_accumulator_and_256_alignment() {
        let config = config();
        let layout = Qwen35MoeNativeScratchLayout::from_config(&config, 3).unwrap();
        assert_eq!(layout.slots, 24);
        assert_eq!(layout.expert_output_f32.element_bytes(), 4);
        assert_eq!(layout.expert_output_f32.elements(), 24 * 16);
        assert_eq!(layout.route_weights_f32.element_bytes(), 4);
        assert_eq!(layout.route_weights_f32.elements(), 24);
        for span in [
            layout.counts_i32,
            layout.offsets_i32,
            layout.cursors_i32,
            layout.ids_u32,
            layout.mapping_u32,
            layout.route_weights_f32,
            layout.packed_bf16,
            layout.gate_up_bf16,
            layout.activated_bf16,
            layout.expert_output_f32,
            layout.problems_gemm_coord,
            layout.a_ptrs,
            layout.b_ptrs,
            layout.c_ptrs,
            layout.c_f32_ptrs,
            layout.lda_i64,
            layout.ldb_i64,
            layout.ldc_i64,
        ] {
            assert_eq!(span.offset_bytes() % 256, 0);
        }
        assert_eq!(layout.total_bytes() % 256, 0);
        assert!(layout.total_bytes() > layout.expert_output_f32.offset_bytes());
    }

    #[test]
    fn exact_native_bytes_for_tiny_and_actual_geometry() {
        // Independently computed from the 256-byte `Workspace::take` order.
        assert_eq!(
            native_workspace_bytes(1, 16, 8, 256).unwrap(),
            expected_tiny_bytes()
        );
        assert_eq!(
            native_workspace_bytes(1, 2048, 512, 256).unwrap(),
            expected_35b_bytes()
        );
        assert_eq!(
            native_workspace_bytes(2, 2048, 512, 256).unwrap()
                - native_workspace_bytes(1, 2048, 512, 256).unwrap(),
            expected_35b_one_token_step()
        );
    }

    #[test]
    fn rejects_native_dimension_guards_and_overflow() {
        assert!(native_workspace_bytes(0, 16, 8, 256).is_err());
        assert!(native_workspace_bytes(1, 15, 8, 256).is_err());
        assert!(native_workspace_bytes(1, 16, 9, 256).is_err());
        assert!(native_workspace_bytes(1, 16, 8, 0).is_err());
        assert!(native_workspace_bytes(usize::MAX / 2, 16, 8, 256).is_err());
        assert!(native_workspace_bytes(1, 16, 8, 256)
            .unwrap()
            .is_multiple_of(256));
    }

    fn expected_tiny_bytes() -> usize {
        22_784
    }

    fn expected_35b_bytes() -> usize {
        144_384
    }

    fn expected_35b_one_token_step() -> usize {
        122_880
    }
}

// Native execution contracts for the two Qwen3.5-MoE FFN paths.
//
// This module exists because the two Native paths are numerically different:
//
// * `qwen35_models.cpp::feed_forward` selects `routed_moe_cuda(...)` when the
//   input is CUDA BF16 with top_k == 8, 128/256 experts, hidden % 8 == 0 and
//   intermediate % 8 == 0. That is the real accelerated Qwen3.5 path.
// * `decoder_moe_forward` / `decoder_moe_dispatch` in `decoder_models.cpp` is
//   the non-native fallback used when `routed_moe_cuda` returns undefined.
//
// The contract below is the portable non-Hopper candle provider selected by
// `routed_moe_cuda` (`grouped_gemm.cu` / `gemma4_moe.cu`). Hopper builds with
// `TEI_TORCH_MOE_HOPPER` dispatch to `grouped_gemm_hopper.cu`, which has a
// different workspace ABI (BF16 expert output, sorted/row weights, 2 MiB GEMM
// workspace) and different combine rounding. Hopper is deliberately out of
// scope here and must get its own contract/layout before H100 provider work.
//
// * router logits are FP32 (the model casts the BF16 router linear output);
// * routing weights stay FP32 (`qwen35_moe_route_256_8_f32`);
// * the routed gate/up activation is SiLU in FP32, rounded to BF16, then
//   multiplied by BF16 up and rounded to BF16 (`qwen3_moe_silu_mul`);
// * the down projection accumulates in FP32 (`grouped_gemm<float>`);
// * the combine multiplies each FP32 expert output by its FP32 routing weight,
//   rounds every term to BF16, sums the terms in FP32, then rounds the final
//   sum to BF16 once (`gemma4_moe_combine_vec8` / `gemma4_moe_unpermute_combine`).
//
// The fallback contract instead casts routing to the input dtype (BF16) before
// dispatch. It is intentionally not implemented as a numerical oracle here.
//
// IMPORTANT: the scalar helpers below model the *order and rounding points* of
// the Native CUDA kernels. They are not an exact numerical oracle: CUDA uses
// tree reductions for softmax and hardware/CUTLASS GEMM accumulation. Exact
// router weights and expert outputs require paired Root Native gold captured
// separately. Top-k ids are exact only for clearly separated probabilities.

use crate::qwen35_moe_config::{Qwen35MoeConfig, Qwen35MoeDtype};
use infinity_v3_backend_core::BackendError;
use std::path::Path;

fn start(error: impl std::fmt::Display) -> BackendError {
    BackendError::Start(format!("Oxide qwen3.5-moe contract: {error}"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Qwen35MoeNativeContract {
    /// Real accelerated `routed_moe_cuda` path.
    RoutedMoeCuda,
    /// Non-native `decoder_moe_forward` fallback.
    DecoderMoeForwardFallback,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RoutedMoeCudaContract {
    pub(crate) logits: &'static str,
    pub(crate) routing_weights: &'static str,
    pub(crate) activation: &'static str,
    pub(crate) down_accumulator: &'static str,
    pub(crate) combine: &'static str,
}

pub(crate) const ROUTED_MOE_CUDA: RoutedMoeCudaContract = RoutedMoeCudaContract {
    logits: "FP32",
    routing_weights: "FP32",
    activation: "FP32 SiLU -> BF16 round -> BF16 multiply -> BF16 round",
    down_accumulator: "FP32 grouped GEMM",
    combine: "per-term FP32*FP32 -> BF16 round; BF16 terms summed in FP32; one final BF16 round",
};

/// Select the path Native would take without touching a device. `cuda_bf16`
/// must be supplied by a caller that already knows the execution context; this
/// foundation never queries a device.
pub(crate) fn select_native_contract(
    cuda_bf16: bool,
    config: &Qwen35MoeConfig,
) -> Qwen35MoeNativeContract {
    let native = config.dtype == Qwen35MoeDtype::Bf16
        && config.moe.num_experts_per_tok == 8
        && matches!(config.moe.num_experts, 128 | 256)
        && config.text.hidden_size.is_multiple_of(8)
        && config.moe.moe_intermediate_size.is_multiple_of(8);
    if cuda_bf16 && native {
        Qwen35MoeNativeContract::RoutedMoeCuda
    } else {
        Qwen35MoeNativeContract::DecoderMoeForwardFallback
    }
}

/// Scalar semantic model of `qwen35_moe_route_256_8_f32` and
/// `qwen3_moe_route_128_8_f32`.
///
/// Returns FP32 routing weights, exactly like the CUDA route kernels. The
/// scalar exp/sum order differs from the CUDA tree reduction, so callers must
/// use tolerances for weights and only rely on ids for separated scores.
pub(crate) fn route_topk_f32(
    logits: &[f32],
    top_k: usize,
    renormalize: bool,
) -> Result<(Vec<u32>, Vec<f32>), BackendError> {
    if logits.is_empty() {
        return Err(start("router logits must not be empty"));
    }
    if top_k == 0 || top_k > logits.len() {
        return Err(start("top-k must be in 1..=num_experts"));
    }
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(start("router logits must be finite"));
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = Vec::new();
    probabilities
        .try_reserve_exact(logits.len())
        .map_err(|_| start("router probability allocation failed"))?;
    let mut total = 0.0f32;
    for value in logits {
        let probability = (value - max).exp();
        total += probability;
        probabilities.push(probability);
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(start("softmax denominator is not finite and positive"));
    }
    for probability in &mut probabilities {
        *probability /= total;
    }
    let mut order = Vec::new();
    order
        .try_reserve_exact(logits.len())
        .map_err(|_| start("router order allocation failed"))?;
    order.extend(0..logits.len());
    order.sort_unstable_by(|left, right| {
        probabilities[*right]
            .total_cmp(&probabilities[*left])
            .then_with(|| left.cmp(right))
    });
    let mut ids = Vec::new();
    ids.try_reserve_exact(top_k)
        .map_err(|_| start("router id allocation failed"))?;
    let mut weights = Vec::new();
    weights
        .try_reserve_exact(top_k)
        .map_err(|_| start("router weight allocation failed"))?;
    let mut selected_sum = 0.0f32;
    for expert in order.into_iter().take(top_k) {
        ids.push(u32::try_from(expert).map_err(|_| start("expert id does not fit u32"))?);
        selected_sum += probabilities[expert];
        weights.push(probabilities[expert]);
    }
    if renormalize {
        if !selected_sum.is_finite() || selected_sum <= 0.0 {
            return Err(start(
                "selected probability mass is not finite and positive",
            ));
        }
        for weight in &mut weights {
            *weight /= selected_sum;
        }
    }
    if weights.iter().any(|value| !value.is_finite()) {
        return Err(start("routing weights are not finite"));
    }
    Ok((ids, weights))
}

/// Scalar semantic model of `qwen3_moe_silu_mul`: FP32 SiLU, BF16 rounding of
/// the SiLU result, then BF16 multiply by up and BF16 rounding.
pub(crate) fn cuda_silu_mul_bf16(gate: u16, up: u16) -> u16 {
    let gate = half::bf16::from_bits(gate);
    let up = half::bf16::from_bits(up);
    recall_silu_mul(gate, up).to_bits()
}

pub(crate) fn recall_silu_mul(gate: half::bf16, up: half::bf16) -> half::bf16 {
    let gate = gate.to_f32();
    let up = up.to_f32();
    let silu = gate / (1.0 + (-gate).exp());
    let rounded_silu = half::bf16::from_f32(silu).to_f32();
    half::bf16::from_f32(rounded_silu * up)
}

/// Scalar semantic model of the Native BF16 `sigmoid` expression used in the
/// Qwen3.5 CUDA `feed_forward` path:
/// `reciprocal(ones_like(logits) + exp(-logits))`, with every BF16 op rounded.
pub(crate) fn cuda_shared_gate_bf16(logit: u16) -> u16 {
    let logit = half::bf16::from_bits(logit);
    let exp = half::bf16::from_f32((-logit).to_f32().exp());
    let denominator = half::bf16::from_f32(1.0 + exp.to_f32());
    half::bf16::from_f32(1.0 / denominator.to_f32()).to_bits()
}

/// Scalar semantic model of the CUDA combine order. `expert_outputs` is
/// `[tokens, top_k, hidden]` FP32 and `weights` is `[tokens, top_k]` FP32.
/// Every weighted term is rounded to BF16, the BF16-valued terms are summed in
/// FP32, and the final sum is rounded to BF16 once.
pub(crate) fn cuda_combine_bf16(
    expert_outputs: &[f32],
    weights: &[f32],
    tokens: usize,
    top_k: usize,
    hidden: usize,
) -> Result<Vec<u16>, BackendError> {
    let slots = tokens
        .checked_mul(top_k)
        .ok_or_else(|| start("combine slot count overflow"))?;
    let expected = slots
        .checked_mul(hidden)
        .ok_or_else(|| start("combine output element count overflow"))?;
    if expert_outputs.len() != expected || weights.len() != slots {
        return Err(start(
            "combine input lengths disagree with tokens/top-k/hidden",
        ));
    }
    let output_elements = tokens
        .checked_mul(hidden)
        .ok_or_else(|| start("combine output count overflow"))?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(output_elements)
        .map_err(|_| start("combine output allocation failed"))?;
    for index in 0..output_elements {
        let token = index / hidden;
        let column = index % hidden;
        let mut sum = 0.0f32;
        for rank in 0..top_k {
            let slot = token * top_k + rank;
            let weighted = expert_outputs[slot * hidden + column] * weights[slot];
            sum += half::bf16::from_f32(weighted).to_f32();
        }
        output.push(half::bf16::from_f32(sum).to_bits());
    }
    Ok(output)
}

/// Explicit format admission for the cold foundation. The donor crate accepts
/// PyTorch archives through its shared `Checkpoint`, but this MoE foundation
/// deliberately admits safetensors only so standalone and donor behavior match.
pub(crate) fn reject_pytorch_checkpoint(path: &Path) -> Result<(), BackendError> {
    for name in ["pytorch_model.bin", "pytorch_model.bin.index.json"] {
        if path.join(name).exists() {
            return Err(start(
                "PyTorch checkpoint payloads are not admitted by the Qwen3.5-MoE foundation",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_selection_matches_native_guard() {
        let config = tiny_config();
        assert_eq!(
            select_native_contract(true, &config),
            Qwen35MoeNativeContract::RoutedMoeCuda
        );
        assert_eq!(
            select_native_contract(false, &config),
            Qwen35MoeNativeContract::DecoderMoeForwardFallback
        );
        let mut top_four = config.clone();
        top_four.moe.num_experts_per_tok = 4;
        assert_eq!(
            select_native_contract(true, &top_four),
            Qwen35MoeNativeContract::DecoderMoeForwardFallback
        );
    }

    #[test]
    fn cuda_activation_rounds_silu_before_multiply() {
        let gate = half::bf16::from_f32(1.0).to_bits();
        let up = half::bf16::from_f32(2.0).to_bits();
        assert_eq!(cuda_silu_mul_bf16(gate, up), expected_silu_mul_bits());
    }

    #[test]
    fn cuda_shared_gate_rounds_bf16_ops() {
        assert_eq!(
            cuda_shared_gate_bf16(half::bf16::from_f32(0.0).to_bits()),
            half::bf16::from_f32(0.5).to_bits()
        );
        assert_eq!(
            cuda_shared_gate_bf16(half::bf16::from_f32(2.0).to_bits()),
            expected_gate_bits()
        );
    }

    #[test]
    fn combine_rounds_each_term_before_f32_sum() {
        let tokens = 1;
        let top_k = 2;
        let hidden = 2;
        let outputs = [1.0f32, 2.0, 1.0, 2.0];
        let weights = [0.333_333_34f32, 0.333_333_34];
        let combined = cuda_combine_bf16(&outputs, &weights, tokens, top_k, hidden).unwrap();
        assert_eq!(combined, expected_combine_bits());
    }

    fn tiny_config() -> Qwen35MoeConfig {
        Qwen35MoeConfig::from_json(&serde_json::json!({
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

    // Hard-coded independent expectations produced with a separate BF16
    // round-to-nearest helper, not by calling the functions under test.
    fn expected_silu_mul_bits() -> u16 {
        0x3FBB // bf16(bf16(silu(1.0)) * 2.0)
    }

    fn expected_gate_bits() -> u16 {
        0x3F62 // bf16(1 / bf16(1 + bf16(exp(-2.0))))
    }

    fn expected_combine_bits() -> Vec<u16> {
        vec![0x3F2B, 0x3FAB] // bf16(bf16(w)+bf16(w)) and bf16(bf16(2w)+bf16(2w))
    }
}

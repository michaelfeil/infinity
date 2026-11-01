# SPDX-License-Identifier: MIT
# Adapted from candle-flash-attn-v4; see provenance.json and LICENSE-MIT.
"""Jina-small ALiBi model-bias score hook for the FA4 AOT bundle.

The ALiBi export uses the ordinary non-causal self-attention path with
external ``softmax_scale=1.0``.  This hook applies the model's native
``1/sqrt(64) == 0.125`` QK scale and adds a per-head ALiBi bias that is
computed in F32, rounded ONCE to the model dtype, and then widened back to
F32 before the add.  The QK score and the final sum are never rounded to the
model dtype.

At most one F32[8] slope vector is borrowed for the launch; no O(T*T) bias
tensor is ever materialized.  The caller retains ownership of the slope,
model, and context storage until the stream drains.
"""
import struct

try:
    import cutlass
    import cutlass.cute as cute
except ImportError:  # CPU-only tooling (tests, metadata) must not need CuTe.
    cutlass = None
    cute = None

ALIBI_HEADS = 8
ALIBI_HEAD_DIM = 64
INT32_MAX = 2**31 - 1
INT64_MAX = 2**63 - 1


def f32(value):
    """Round a Python float to IEEE binary32, the hook's working type."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def bits_f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def round_f32_to_bf16(value):
    """Round F32 to BF16 (round-to-nearest, ties-to-even), returned as F32."""
    bits = f32_bits(value)
    if (bits & 0x7F800000) == 0x7F800000:
        return bits_f32(bits & 0xFFFF0000)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return bits_f32(rounded & 0xFFFF0000)


def round_f32_to_fp16(value):
    """Round F32 to F16 (round-to-nearest, ties-to-even), returned as F32."""
    return struct.unpack("<e", struct.pack("<e", float(f32(value))))[0]


def round_to_model_dtype(value, dtype):
    if dtype == "bf16":
        return round_f32_to_bf16(value)
    if dtype == "fp16":
        return round_f32_to_fp16(value)
    raise ValueError(f"unsupported ALiBi model dtype: {dtype!r}")


def alibi_bias_f32(slope_f32, distance, dtype):
    """CPU reference for ``F32(round_to_model_dtype(-slopes[h]*abs(q-k)))``."""
    distance = int(distance)
    if distance < 0:
        raise ValueError("ALiBi distance must be non-negative")
    product = f32(f32(slope_f32) * f32(distance))
    return round_to_model_dtype(-product, dtype)


def alibi_score_f32(qk_f32, slope_f32, distance, dtype):
    """CPU reference for ``qk*0.125 + F32(round_to_model_dtype(-slope*d))``."""
    return f32(f32(f32(qk_f32) * f32(0.125)) + alibi_bias_f32(slope_f32, distance, dtype))


def validate_alibi_shape(dtype, total, batch, heads, max_q, strides):
    """Validate the host-side geometry mirrored by bridge.cpp.

    Returns True for a usable packed non-causal self-attention call.  Raises
    ``ValueError`` with a short reason otherwise.  Pointer non-nullness is
    necessarily a C++-side check because this helper is CPU-only.
    """
    if dtype not in (0, 1):
        raise ValueError("dtype must be 0 (fp16) or 1 (bf16)")
    if total <= 0 or total > INT32_MAX // (ALIBI_HEADS * ALIBI_HEAD_DIM):
        raise ValueError("total is outside the packed int32 geometry")
    if batch <= 0 or batch >= INT32_MAX:
        raise ValueError("batch is outside the int32 geometry")
    if heads != ALIBI_HEADS:
        raise ValueError("ALiBi export is fixed at 8 self-attention heads")
    if max_q <= 0 or max_q > INT32_MAX or max_q > total:
        raise ValueError("max_q must be positive and no larger than total")
    if len(strides) != 12:
        raise ValueError("expected 12 Q/K/V/output strides")
    if any(not isinstance(s, int) or s <= 0 for s in strides):
        raise ValueError("strides must be positive integers")
    for base in (0, 3, 6, 9):
        extent = ((total - 1) * strides[base]
                  + (heads - 1) * strides[base + 1]
                  + (ALIBI_HEAD_DIM - 1) * strides[base + 2])
        if extent > INT64_MAX:
            raise ValueError("stride/geometry product overflows int64")
    return True


def native_link_defines(kernels):
    """Return the additive link defines selected by a manifest kernel list."""
    symbols = [str(k.get("symbol", "")) for k in kernels]
    defines = []
    if any(s.startswith("fa4_deberta_") for s in symbols):
        defines.append("-DFA4_DEBERTA")
    if any(s.startswith("fa4_alibi_model_bias_") for s in symbols):
        defines.append("-DFA4_ALIBI")
    return defines


# Both hooks use the local packed q/k positions.  This export is packed
# self-attention, so Q and K share the same per-batch packed offsets and those
# offsets cancel; the ALiBi distance is abs(q_idx - kv_idx).
if cute is not None:
    @cute.jit
    def alibi_model_bias_fp16(score, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        slopes = aux_tensors[0]
        hi = cute.make_rmem_tensor(1, cutlass.Int32)
        hi.store(h_idx)
        qi = cute.make_rmem_tensor(1, cutlass.Int32)
        qi.store(q_idx)
        ki = cute.make_rmem_tensor(1, cutlass.Int32)
        ki.store(kv_idx)
        q_local = qi[0]
        k_local = ki[0]
        dist = cute.abs(q_local - k_local)
        slope = slopes[hi[0]].to(cutlass.Float32)
        bias = (-slope * dist.to(cutlass.Float32)).to(cutlass.Float16).to(cutlass.Float32)
        return score.to(cutlass.Float32) * cutlass.Float32(0.125) + bias

    @cute.jit
    def alibi_model_bias_bf16(score, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        slopes = aux_tensors[0]
        hi = cute.make_rmem_tensor(1, cutlass.Int32)
        hi.store(h_idx)
        qi = cute.make_rmem_tensor(1, cutlass.Int32)
        qi.store(q_idx)
        ki = cute.make_rmem_tensor(1, cutlass.Int32)
        ki.store(kv_idx)
        q_local = qi[0]
        k_local = ki[0]
        dist = cute.abs(q_local - k_local)
        slope = slopes[hi[0]].to(cutlass.Float32)
        bias = (-slope * dist.to(cutlass.Float32)).to(cutlass.BFloat16).to(cutlass.Float32)
        return score.to(cutlass.Float32) * cutlass.Float32(0.125) + bias

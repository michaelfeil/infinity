# SPDX-License-Identifier: MIT
# Additive score hook; original hook/objects remain unchanged. See provenance.json.
"""Exact fixed-divisor BF16 result using scalar-qualified RN multiplication."""
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def materialized_bf16_multiplication(value: cutlass.Float32, reciprocal: cutlass.Float32, *, loc=None, ip=None) -> cutlass.Float32:
    # For raw BF16 inputs only, these two fixed FP32 reciprocals produce the
    # same BF16 result as the existing RN division. Preserve explicit RN mul
    # and the BF16 RNE intermediate; do not use FTZ or contract arithmetic.
    result = llvm.inline_asm(
        T.f32(),
        [cutlass.Float32(value).ir_value(loc=loc, ip=ip), cutlass.Float32(reciprocal).ir_value(loc=loc, ip=ip)],
        "{ .reg .f32 divided; .reg .b32 packed, widened; "
        "mul.rn.f32 divided, $1, $2; "
        "cvt.rn.bf16x2.f32 packed, divided, divided; "
        "shl.b32 widened, packed, 16; mov.b32 $0, widened; }",
        "=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(result)


@cute.jit
def exact_mul_relative_score(score, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors, reciprocal: cutlass.Constexpr):
    c2p, p2c, buckets = aux_tensors
    qi = cute.make_rmem_tensor(1, cutlass.Int32); qi.store(q_idx)
    ki = cute.make_rmem_tensor(1, cutlass.Int32); ki.store(kv_idx)
    hi = cute.make_rmem_tensor(1, cutlass.Int32); hi.store(h_idx)
    idx = buckets[qi[0] - ki[0] + (buckets.shape[0] // 2)]
    first = materialized_bf16_multiplication(c2p[hi[0], qi[0] + seqlen_info.offset_q, idx].to(cutlass.Float32), cutlass.Float32(reciprocal))
    second = materialized_bf16_multiplication(p2c[hi[0], ki[0] + seqlen_info.offset_k, idx].to(cutlass.Float32), cutlass.Float32(reciprocal))
    # Preserve the original Native grouping: BF16(c2p + p2c), then
    # BF16(BF16(QK) + that BF16 relative bias), widened for softmax.
    relative = cute.make_rmem_tensor(1, c2p.element_type)
    relative[0] = (first + second).to(c2p.element_type)
    return (score.to(c2p.element_type).to(cutlass.Float32) + relative.load().to(cutlass.Float32)).to(c2p.element_type).to(cutlass.Float32)


@cute.jit
def deberta_exact_mul_both(score, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    return exact_mul_relative_score(score, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors, 0.07207207381725311)


@cute.jit
def deberta_exact_mul_single(score, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    return exact_mul_relative_score(score, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors, 0.0883977934718132)

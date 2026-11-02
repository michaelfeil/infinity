# SPDX-License-Identifier: MIT
"""Actual JinaCode TorchFlashALiBi64 score recipe, before softmax scaling.

The packed self-attention positions are local. Native Flash subtracts the F32
slope divided by .125 from the unscaled QK accumulator with FMA enabled; its
softmax then applies .125. FA4 calls modifiers after its external
scale, so this wrapper requests external1 and applies .125 after native FMA. No bias is rounded to the model storage type.
"""
import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def native_score(score, slope, distance, *, loc=None, ip=None):
    # Power-of-two rescaling is exact for all twelve positive normal slopes.
    # Explicit PTX keeps the one-rounding native FMA and prevents moving the
    # external softmax factor into the score hook.
    values = [cutlass.Float32(x).ir_value(loc=loc, ip=ip)
              for x in (score, slope, distance)]
    return cutlass.Float32(llvm.inline_asm(
        cutlass.Float32.mlir_type, values,
        "{ .reg .f32 scaled; mul.rn.f32 scaled, $2, 0f41000000; "
        "neg.f32 scaled, scaled; fma.rn.f32 $0, scaled, $3, $1; }",
        "=f,f,f,f", has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def alibi12_native_flash_fp16(score, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    hi = cute.make_rmem_tensor(1, cutlass.Int32)
    hi.store(h_idx)
    qi = cute.make_rmem_tensor(1, cutlass.Int32)
    qi.store(q_idx)
    ki = cute.make_rmem_tensor(1, cutlass.Int32)
    ki.store(kv_idx)
    distance = cute.abs(qi[0] - ki[0]).to(cutlass.Float32)
    result = cute.make_rmem_tensor(1, cutlass.Float32)
    result[0] = native_score(score[0], aux_tensors[0][hi[0]], distance) * cutlass.Float32(0.125)
    return result.load()


@cute.jit
def alibi12_native_flash_bf16(score, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    hi = cute.make_rmem_tensor(1, cutlass.Int32)
    hi.store(h_idx)
    qi = cute.make_rmem_tensor(1, cutlass.Int32)
    qi.store(q_idx)
    ki = cute.make_rmem_tensor(1, cutlass.Int32)
    ki.store(kv_idx)
    distance = cute.abs(qi[0] - ki[0]).to(cutlass.Float32)
    result = cute.make_rmem_tensor(1, cutlass.Float32)
    result[0] = native_score(score[0], aux_tensors[0][hi[0]], distance) * cutlass.Float32(0.125)
    return result.load()

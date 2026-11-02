# SPDX-License-Identifier: MIT
# Adapted from candle-flash-attn-v4; see provenance.json and LICENSE-MIT.
"""DeBERTa-v2/v3 self-attention score hook for the FA4 AOT bundle."""
import cutlass
import cutlass.cute as cute

@cute.jit
def deberta_score(score, b_idx, h_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
    c2p, p2c, buckets = aux_tensors
    qi = cute.make_rmem_tensor(1, cutlass.Int32); qi.store(q_idx)
    ki = cute.make_rmem_tensor(1, cutlass.Int32); ki.store(kv_idx)
    hi = cute.make_rmem_tensor(1, cutlass.Int32); hi.store(h_idx)
    idx = buckets[qi[0] - ki[0] + (buckets.shape[0] // 2)]
    val = cute.make_rmem_tensor(1, c2p.element_type)
    # HF disentangled_attention_bias first adds c2p + p2c in model dtype.
    # Its caller then adds that rounded bias to QK (also model dtype).
    # Keep this grouping: (QK + c2p) + p2c would change the rounding.
    # QK has already used scaled K.
    rel = (c2p[hi[0], qi[0] + seqlen_info.offset_q, idx].to(cutlass.Float32)
           + p2c[hi[0], ki[0] + seqlen_info.offset_k, idx].to(cutlass.Float32)).to(c2p.element_type)
    val[0] = rel
    return (score.to(c2p.element_type).to(cutlass.Float32) + val.load().to(cutlass.Float32)).to(c2p.element_type).to(cutlass.Float32)

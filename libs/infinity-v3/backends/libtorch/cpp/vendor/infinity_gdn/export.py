# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
"""AOT host adapters around the unmodified FlashInfer 0.7.2 GDN kernels."""
import hashlib
import json
import os
import pathlib
import sys

output, family = pathlib.Path(sys.argv[1]), int(sys.argv[2])
arch = {9: "sm_90a", 10: "sm_100f", 12: "sm_120f"}[family]
os.environ["CUTE_DSL_ARCH"] = arch
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor as tensor, make_fake_stream
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "infinity_fa4"))
from portable_host import export_portable

def export(dtype, label, initial_state):
    T, H, V, B = (cute.sym_int() for _ in range(4))
    def packed(typ, shape):
        return tensor(typ, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=16)
    q, k = (packed(dtype, (T, H, 128)) for _ in range(2))
    v, o = (packed(dtype, (T, V, 128)) for _ in range(2))
    alpha, beta = (packed(cutlass.Float32, (cute.sym_int(), V)) for _ in range(2))
    cu = tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4)
    state = packed(cutlass.Float32, (B, V, 128, 128))
    initial = packed(cutlass.Float32, (B, V, 128, 128))
    workspace = tensor(cutlass.Int8, (cute.sym_int(),), assumed_align=128)
    if family == 10:
        from flashinfer.gdn_kernels.blackwell.gated_delta_net_chunked import GatedDeltaNetChunkedKernel
        kernel = GatedDeltaNetChunkedKernel(
            io_dtype=dtype, inverse_dtype=dtype, acc_dtype=cutlass.Float32, state_dtype=cutlass.Float32,
            mma_tiler_qk=(64, 64, 128), mma_tiler_qs=(128, 64, 128),
            mma_tiler_qkv=(128, 64, 64), mma_tiler_kv=(128, 128, 64),
            max_active_clusters=1, num_sm=1, is_GQA=False, use_initial_state=initial_state,
            store_final_state=True, enable_checkpoints=False, is_persistent=True)
        @cute.jit
        def launch(q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, alpha: cute.Tensor,
                   beta: cute.Tensor, o: cute.Tensor, cu: cute.Tensor, state: cute.Tensor,
                   initial: cute.Tensor, workspace: cute.Tensor, sm_count: cutlass.Int32, stream):
            kernel.max_active_clusters = sm_count
            kernel(q, k, v, alpha, beta, o, cu, initial if initial_state else None, state, None, None, None,
                   cutlass.Int32(0), cutlass.Float32(1), workspace, stream)
    else:
        if family == 9:
            from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm90 import _FullyFusedDeltaRuleSm90 as Kernel
        else:
            from flashinfer.gdn_kernels.delta_rule_dsl.delta_rule_sm120 import _FullyFusedDeltaRuleSm120 as Kernel
        kernel = Kernel(True, True, initial_state, False, dtype=dtype, cu_seqlens_dtype=torch.int32)
        @cute.jit
        def launch(q: cute.Tensor, k: cute.Tensor, v: cute.Tensor, alpha: cute.Tensor,
                   beta: cute.Tensor, o: cute.Tensor, cu: cute.Tensor, state: cute.Tensor,
                   initial: cute.Tensor, workspace: cute.Tensor, sm_count: cutlass.Int32, stream):
            qt = cute.make_tensor(q.iterator, cute.make_layout((q.shape[0], 128, q.shape[1]), stride=(q.stride[0], 1, 128)))
            kt = cute.make_tensor(k.iterator, cute.make_layout((128, k.shape[0], k.shape[1]), stride=(1, k.stride[0], 128)))
            vt = cute.make_tensor(v.iterator, cute.make_layout((128, v.shape[0], v.shape[1]), stride=(1, v.stride[0], 128)))
            ot = cute.make_tensor(o.iterator, cute.make_layout((128, o.shape[0], o.shape[1]), stride=(1, o.stride[0], 128)))
            a = cute.make_tensor(alpha.iterator, cute.make_layout(cute.size(alpha.shape), stride=1))
            b = cute.make_tensor(beta.iterator, cute.make_layout(cute.size(beta.shape), stride=1))
            batch = cu.shape[0] - 1
            kernel(qt, kt, vt, ot, a, b, state, initial if initial_state else None, None, None, None, workspace, cu,
                   cutlass.Float32(1), cutlass.Int32(q.shape[1]), cutlass.Int32(k.shape[1]),
                   cutlass.Int32(v.shape[1]), cutlass.Int32(v.shape[1]), cutlass.Int32(batch),
                   cutlass.Int32(1), cutlass.Int32(0), batch * v.shape[1], stream)
    compiled = cute.compile(launch, q, k, v, alpha, beta, o, cu, state, initial, workspace,
                            cutlass.Int32(1), make_fake_stream(use_tvm_ffi_env_stream=True),
                            options="--enable-tvm-ffi")
    symbol = f"gdn_{family}_{label}" + ("_initial" if initial_state else "")
    path = output / f"{symbol}.o"
    export_portable(compiled, path, symbol)
    return dict(symbol=path.stem, initial_state=initial_state, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
exports = [export(dtype, label, initial)
           for initial in (False, True)
           for dtype, label in ((cutlass.Float16, "fp16"), (cutlass.BFloat16, "bf16"))]
(output / f"sm_{family}.json").write_text(json.dumps(dict(architecture=arch, exports=exports), indent=2))

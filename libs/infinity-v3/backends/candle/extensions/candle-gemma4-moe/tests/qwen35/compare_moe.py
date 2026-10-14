"""Qwen3.5-MoE real-weight primitive diagnostic against vLLM; not a model-quality gate."""

import argparse
import ctypes
import json
from pathlib import Path

import torch
from safetensors import safe_open
from vllm.model_executor.layers.fused_moe.fused_moe import MoEActivation, fused_experts
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--library", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
lib = ctypes.CDLL(str(args.library.resolve()))
P, I, S = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
stream = torch.cuda.current_stream().cuda_stream
lib.qwen35_test_route.argtypes = [P, P, P, I, I, P]
torch.manual_seed(24)
for renormalize in [False, True]:
    logits = torch.randn(257, 256, device="cuda")
    logits[0] = 0
    logits[1] = torch.arange(256, device="cuda") * -100.0
    logits[2, 8:] = -10000
    ids = torch.empty(257, 8, device="cuda", dtype=torch.int32)
    weights = torch.empty(257, 8, device="cuda")
    assert (
        lib.qwen35_test_route(
            logits.data_ptr(),
            ids.data_ptr(),
            weights.data_ptr(),
            257,
            int(renormalize),
            stream,
        )
        == 0
    )
    expected_w, expected_i, _ = fused_topk(logits, logits, 8, renormalize)
    assert torch.equal(ids, expected_i), (
        "routing IDs",
        renormalize,
        (ids != expected_i).sum().item(),
    )
    torch.testing.assert_close(weights, expected_w, rtol=2e-6, atol=2e-7)
print("Routing agrees with vLLM, with and without renormalization", flush=True)
index = json.loads((args.model / "model.safetensors.index.json").read_text())[
    "weight_map"
]
handles = {}


def weight(name):
    shard = index[name]
    if shard not in handles:
        handles[shard] = safe_open(args.model / shard, framework="pt", device="cpu")
    return handles[shard].get_tensor(name).cuda()


prefix = "model.language_model.layers.0.mlp.experts."
w1 = weight(prefix + "gate_up_proj")
w2 = weight(prefix + "down_proj")
results = []
for hopper in [False, True]:
    stem = "hopper_" if hopper else ""
    workspace = getattr(lib, stem + "qwen35_moe_workspace_bytes")
    workspace.argtypes = [I, I, I]
    workspace.restype = S
    launch = getattr(lib, stem + "qwen35_moe_forward_bf16")
    launch.argtypes = [P] * 5 + [I] * 4 + [P, S, P]
    for tokens in [1, 17, 257, 2048]:
        x = torch.randn(tokens, 2048, device="cuda", dtype=torch.bfloat16)
        logits = torch.randn(tokens, 256, device="cuda")
        scratch = torch.empty(
            workspace(tokens, 2048, 512), device="cuda", dtype=torch.uint8
        )
        out = torch.empty_like(x)
        for renormalize in [False, True]:
            assert (
                launch(
                    logits.data_ptr(),
                    x.data_ptr(),
                    w1.data_ptr(),
                    w2.data_ptr(),
                    out.data_ptr(),
                    tokens,
                    2048,
                    512,
                    int(renormalize),
                    scratch.data_ptr(),
                    scratch.numel(),
                    stream,
                )
                == 0
            )
            weights, ids, _ = fused_topk(x, logits, 8, renormalize)
            expected = fused_experts(
                x, w1, w2, weights, ids, activation=MoEActivation.SILU
            )
            torch.cuda.synchronize()
            err = out.float() - expected.float()
            rms = (err.square().mean() / expected.float().square().mean()).sqrt().item()
            cosine = torch.nn.functional.cosine_similarity(
                out.float().flatten(), expected.float().flatten(), dim=0
            ).item()
            row = {
                "hopper": hopper,
                "tokens": tokens,
                "renormalize": renormalize,
                "relative_rmse": rms,
                "cosine": cosine,
                "max_abs_error": err.abs().max().item(),
            }
            print(row, flush=True)
            results.append(row)
            assert rms < 0.001 and cosine > 0.999999, row
args.output.write_text(json.dumps(results, indent=2))

"""Real-weight expert-block diagnostic; this does not qualify full-model accuracy."""

import argparse
import ctypes
import json
from pathlib import Path

import torch
from safetensors import safe_open
from vllm.model_executor.layers.fused_moe.fused_moe import MoEActivation, fused_experts
from vllm.model_executor.models.gemma4 import gemma4_fused_routing_kernel_triton

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--library", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--tokens", type=int, nargs="+", default=[1, 17, 257, 1024])
parser.add_argument(
    "--hopper", action="store_true", help="Test the SM90a expert entry point"
)
parser.add_argument(
    "--concentrated",
    action="store_true",
    help="Route every token to the same eight experts",
)
parser.add_argument(
    "--unaligned-io",
    action="store_true",
    help="Exercise contiguous input/output views with a one-element storage offset",
)
args = parser.parse_args()
lib = ctypes.CDLL(str(args.library.resolve()))
P = ctypes.c_void_p
I = ctypes.c_int
S = ctypes.c_size_t
symbol_prefix = "hopper_" if args.hopper else ""
workspace_fn = getattr(lib, symbol_prefix + "gemma4_moe_workspace_bytes")
workspace_fn.argtypes = [I, I, I]
workspace_fn.restype = S
fn = getattr(lib, symbol_prefix + "gemma4_moe_forward_bf16")
fn.argtypes = [P] * 6 + [I] * 3 + [P, S, P]


# BF16 GELU must round before multiplication, as in vLLM and PyTorch.
gelufn = lib.gemma4_test_gelu
gelufn.argtypes = [P, P, I, I, P]
torch.manual_seed(421)
for scale in [0.01, 0.1, 1, 3, 10, 100]:
    activations = (torch.randn(1024, 1408, device="cuda") * scale).bfloat16()
    actual = torch.empty(1024, 704, device="cuda", dtype=torch.bfloat16)
    expected = torch.empty_like(actual)
    assert (
        gelufn(
            activations.data_ptr(),
            actual.data_ptr(),
            1024,
            704,
            torch.cuda.current_stream().cuda_stream,
        )
        == 0
    )
    torch.ops._C.gelu_tanh_and_mul(expected, activations)
    assert torch.equal(actual, expected), ("GELU rounding differs", scale)
print("GELU rounding: exact agreement across six input scales", flush=True)

# Cover every BF16 gate bit pattern, vector/scalar paths and storage offsets.
reference_gelu = lib.gemma4_test_gelu_reference
reference_gelu.argtypes = [P, P, I, I, P]
for width in [13, 704]:
    tokens = (65536 + width - 1) // width
    gate = torch.arange(tokens * width, device="cuda", dtype=torch.int32)
    gate = gate.to(torch.int16).view(torch.bfloat16).view(tokens, width)
    for offset in [0, 1]:
        for unit_up in [True, False]:
            up = torch.ones_like(gate) if unit_up else torch.randn_like(gate)
            value = torch.cat((gate, up), dim=1)
            if offset:
                value = torch.cat((value.new_zeros(1), value.flatten()))[1:]
            value = value.view(tokens, 2 * width)
            outputs = []
            for launch in [gelufn, reference_gelu]:
                output = torch.empty(
                    tokens * width + offset, device="cuda", dtype=torch.bfloat16
                )
                output = output[offset:].view(tokens, width)
                assert (
                    launch(
                        value.data_ptr(),
                        output.data_ptr(),
                        tokens,
                        width,
                        torch.cuda.current_stream().cuda_stream,
                    )
                    == 0
                )
                outputs.append(output)
            equal = (outputs[0].view(torch.int16) == outputs[1].view(torch.int16)) | (
                outputs[0].isnan() & outputs[1].isnan()
            )
            assert equal.all().item(), (
                "GELU vectorization differs",
                width,
                offset,
                unit_up,
            )
print(
    "GELU vectorization: all BF16 gate patterns agree with scalar reference", flush=True
)


def weight(name):
    for f in args.model.glob("*.safetensors"):
        with safe_open(f, framework="pt", device="cpu") as s:
            if name in s.keys():  # noqa: SIM118 - safetensors handle is not a dict
                return s.get_tensor(name).cuda()
    raise KeyError(name)


prefix = "model.language_model.layers.0."
w1 = weight(prefix + "experts.gate_up_proj")
w2 = weight(prefix + "experts.down_proj")
scales = weight(prefix + "router.per_expert_scale").float()
torch.manual_seed(20)
results = []
for t in args.tokens:
    assert t > 0
    x = torch.randn(t, 2816, device="cuda", dtype=torch.bfloat16)
    logits = torch.randn(t, 128, device="cuda")
    if args.concentrated:
        logits[:, 8:] = -10000.0
    out = torch.empty_like(x)
    if args.unaligned_io:
        x = torch.cat((x.new_zeros(1), x.flatten()))[1:].view(t, 2816)
        out = torch.empty(t * 2816 + 1, device="cuda", dtype=torch.bfloat16)[1:].view(
            t, 2816
        )
    scratch = torch.empty(workspace_fn(t, 2816, 704), device="cuda", dtype=torch.uint8)
    status = fn(
        logits.data_ptr(),
        scales.data_ptr(),
        x.data_ptr(),
        w1.data_ptr(),
        w2.data_ptr(),
        out.data_ptr(),
        t,
        2816,
        704,
        scratch.data_ptr(),
        scratch.numel(),
        torch.cuda.current_stream().cuda_stream,
    )
    assert status == 0, status
    tw, ti = gemma4_fused_routing_kernel_triton(logits, 8, scales)
    ref = fused_experts(x, w1, w2, tw, ti, activation=MoEActivation.GELU_TANH)
    torch.cuda.synchronize()
    err = out.float() - ref.float()
    rms = (err.square().mean() / ref.float().square().mean()).sqrt().item()
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0
    ).item()

    def ours(logits=logits, x=x, out=out, t=t, scratch=scratch):
        status = fn(
            logits.data_ptr(),
            scales.data_ptr(),
            x.data_ptr(),
            w1.data_ptr(),
            w2.data_ptr(),
            out.data_ptr(),
            t,
            2816,
            704,
            scratch.data_ptr(),
            scratch.numel(),
            torch.cuda.current_stream().cuda_stream,
        )
        assert status == 0, status

    def baseline(logits=logits, x=x):
        tw, ti = gemma4_fused_routing_kernel_triton(logits, 8, scales)
        return fused_experts(x, w1, w2, tw, ti, activation=MoEActivation.GELU_TANH)

    def bench(call):
        for _ in range(5):
            call()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(30):
            call()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 30

    item = {
        "tokens": t,
        "max_abs_error": err.abs().max().item(),
        "relative_rmse": rms,
        "cosine": cos,
        "scratch_bytes": scratch.numel(),
        "cuda_ms": bench(ours),
        "vllm_ms": bench(baseline),
    }
    print(item, flush=True)
    results.append(item)
    assert rms < 0.001 and cos > 0.999999, item
args.output.write_text(json.dumps(results, indent=2))

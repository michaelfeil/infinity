import argparse
import ctypes
import json
from pathlib import Path

import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    torch_recurrent_gated_delta_rule,
)

p = argparse.ArgumentParser()
p.add_argument("--library", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
lib = ctypes.CDLL(str(args.library.resolve()))
P = ctypes.c_void_p
I = ctypes.c_int
lib.gdn_forward.argtypes = [P] * 13 + [I] * 5 + [ctypes.c_float, P]
torch.manual_seed(31)
results = []
for lengths in [[1], [17], [65], [257], [3, 17, 1, 67]]:
    t = sum(lengths)
    kh, vh = 16, 32
    ch = (2 * kh + vh) * 128
    dt = torch.randn(vh, device="cuda")
    alog = torch.randn(vh, device="cuda")
    norm = torch.randn(128, device="cuda").bfloat16()
    w = (torch.randn(ch, 4, device="cuda") * 0.1).bfloat16()
    x = torch.randn(t, ch, device="cuda").bfloat16()
    z = torch.randn(t, vh * 128, device="cuda").bfloat16()
    ab = torch.randn(t, 2 * vh, device="cuda").bfloat16()
    cu = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
        device="cuda",
        dtype=torch.int32,
    )
    mixed = torch.empty_like(x)
    qk = torch.empty(t, kh, 256, device="cuda")
    g = torch.empty(t, 2 * vh, device="cuda")
    rec = torch.empty_like(z)
    out = torch.empty_like(z)
    args = [x, z, ab, w, alog, dt, norm, cu, mixed, qk, g, rec, out]
    assert (
        lib.gdn_forward(
            *[a.data_ptr() for a in args],
            t,
            kh,
            vh,
            4,
            len(lengths),
            1e-6,
            torch.cuda.current_stream().cuda_stream,
        )
        == 0
    )
    expected = []
    off = 0
    convs = []
    recs = []
    for length in lengths:
        xx = x[off : off + length].T[None]
        m = torch.nn.functional.conv1d(xx, w[:, None, :], padding=3, groups=ch)[
            ..., :length
        ]
        m = torch.nn.functional.silu(m).transpose(1, 2)
        convs.append(m[0])
        q, k, v = torch.split(m, [kh * 128, kh * 128, vh * 128], -1)
        q = q.view(1, length, kh, 128).repeat_interleave(vh // kh, 2)
        k = k.view(1, length, kh, 128).repeat_interleave(vh // kh, 2)
        v = v.view(1, length, vh, 128)
        a, b = ab[off : off + length].chunk(2, -1)
        decay = -alog.exp() * torch.nn.functional.softplus(a.float() + dt)
        beta = b.sigmoid()
        rr, _ = torch_recurrent_gated_delta_rule(
            q, k, v, decay[None], beta[None], use_qk_l2norm_in_kernel=True
        )
        recs.append(rr.reshape(length, -1))
        f = rr.float()
        f = f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6)
        f = norm * f.bfloat16()
        f = f * torch.nn.functional.silu(
            z[off : off + length].view(1, length, vh, 128).float()
        )
        expected.append(f.bfloat16().reshape(length, -1))
        off += length
    expected = torch.cat(expected)
    refrec = torch.cat(recs)
    err = out.float() - expected.float()
    row = {
        "lengths": lengths,
        "conv_exact": torch.equal(mixed, torch.cat(convs)),
        "relative_rmse": (err.square().mean() / expected.float().square().mean())
        .sqrt()
        .item(),
        "max_abs": err.abs().max().item(),
        "cosine": torch.nn.functional.cosine_similarity(
            out.float().flatten(), expected.float().flatten(), dim=0
        ).item(),
        "recurrent_max_abs": (rec.float() - refrec.float()).abs().max().item(),
    }
    print(row, flush=True)
    results.append(row)
    assert row["relative_rmse"] < 0.005 and row["cosine"] > 0.99998
args.output.write_text(json.dumps(results, indent=2))

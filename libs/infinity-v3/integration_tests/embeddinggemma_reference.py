"""Generate a tiny independent Gemma3 BF16 fixture (Transformers 5.17).

CUDA_VISIBLE_DEVICES=0 python integration_tests/embeddinggemma_reference.py /tmp/gemma-fixture
EMBEDDINGGEMMA_FIXTURE_DIR=/tmp/gemma-fixture cargo test ... -- --ignored
This validates architecture semantics; it does not replace checkpoint task-quality tests.
"""

import argparse
import json
import pathlib

import torch
from safetensors.torch import save_file
from transformers import Gemma3TextConfig, Gemma3TextModel

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=pathlib.Path)
parser.add_argument(
    "--causal",
    action="store_true",
    help="Test ordinary causal Gemma3 instead of bidirectional EmbeddingGemma",
)
args = parser.parse_args()
root = args.output
root.mkdir(parents=True, exist_ok=True)
torch.manual_seed(731)
torch.backends.cuda.matmul.allow_tf32 = False
config = Gemma3TextConfig(
    vocab_size=67,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=32,
    max_position_embeddings=64,
    sliding_window=4,
    layer_types=["sliding_attention", "full_attention"],
    use_bidirectional_attention=not args.causal,
    query_pre_attn_scalar=32,
    attention_bias=False,
    pad_token_id=0,
    hidden_activation="gelu_pytorch_tanh",
    rope_parameters={
        "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "sliding_attention": {"rope_type": "default", "rope_theta": 1000.0},
    },
)
config._attn_implementation = "eager"
model = Gemma3TextModel(config).to(device="cuda", dtype=torch.bfloat16).eval()
# Nonzero norm scales exercise Gemma's FP32 (1 + weight) arithmetic.
with torch.no_grad():
    for name, p in model.named_parameters():
        if "norm" in name and name.endswith("weight"):
            p.uniform_(-0.25, 0.25)
save_file(
    {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
    str(root / "model.safetensors"),
)
native = {
    "model_type": "gemma3_text",
    "architectures": ["Gemma3TextModel"],
    "attention_bias": False,
    "use_bidirectional_attention": not args.causal,
    "pad_token_id": 0,
    "head_dim": 32,
    "hidden_activation": "gelu_pytorch_tanh",
    "hidden_size": 64,
    "intermediate_size": 128,
    "max_position_embeddings": 64,
    "num_attention_heads": 2,
    "num_hidden_layers": 2,
    "num_key_value_heads": 1,
    "query_pre_attn_scalar": 32,
    "rms_norm_eps": 1e-6,
    "rope_local_base_freq": 1000.0,
    "rope_theta": 10000.0,
    "sliding_window": 4,
    "_sliding_window_pattern": 2,
    "vocab_size": 67,
    "torch_dtype": "bfloat16",
}
(root / "config.json").write_text(json.dumps(native, indent=2))
sequences = [[2, 3, 4, 5, 6, 7, 8], [12, 13, 14, 15, 16, 17, 18], [22, 23, 24], [25]]
# Two identity-activation projection heads, as used by EmbeddingGemma.
heads = (
    torch.nn.Sequential(
        torch.nn.Linear(64, 96, bias=False), torch.nn.Linear(96, 32, bias=False)
    )
    .to(device="cuda", dtype=torch.bfloat16)
    .eval()
)
for name, head in zip(["2_Dense", "3_Dense"], heads):
    folder = root / name
    folder.mkdir(exist_ok=True)
    save_file(
        {"linear.weight": head.weight.detach().cpu().contiguous()},
        str(folder / "model.safetensors"),
    )
    (folder / "config.json").write_text(
        json.dumps(
            {
                "in_features": head.in_features,
                "out_features": head.out_features,
                "bias": False,
                "activation_function": "torch.nn.modules.linear.Identity",
            }
        )
    )
cases = []
for indices in [[0], [0, 1], [0, 2, 3], [3, 0, 2]]:
    seqs = [sequences[i] for i in indices]
    width = max(map(len, seqs))
    ids = torch.tensor([s + [0] * (width - len(s)) for s in seqs], device="cuda")
    mask = torch.tensor(
        [[1] * len(s) + [0] * (width - len(s)) for s in seqs], device="cuda"
    )
    with torch.no_grad():
        out = model(ids, attention_mask=mask).last_hidden_state
    means = [out[i, : len(s)].sum(0) / len(s) for i, s in enumerate(seqs)]
    with torch.no_grad():
        projected = heads(torch.stack(means))
    cases.append(
        {
            "sequences": seqs,
            "pooled": torch.stack(means).float().cpu().tolist(),
            "projected": projected.float().cpu().tolist(),
            "raw": [
                out[i, : len(s)].float().cpu().tolist() for i, s in enumerate(seqs)
            ],
        }
    )
(root / "reference.json").write_text(json.dumps(cases))
print("Generated independent Transformers BF16 reference:", root)

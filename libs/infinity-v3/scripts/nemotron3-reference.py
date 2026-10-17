#!/usr/bin/env python3
"""Generate independent Nemotron 3 Embed 1B mean-pooling references with Transformers.

Requires PyTorch and Transformers >=5.2.0. The default checkpoint revision is pinned.
"""
import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

MODEL = "nvidia/Nemotron-3-Embed-1B-BF16"
REVISION = "c0c9fea93ea424587517f2c59e20db9f1d6bf615"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    args = parser.parse_args()
    root = Path(args.model)
    revision = args.revision or (REVISION if args.model == MODEL else None)
    if not root.is_dir():
        if not revision:
            parser.error("Remote checkpoints require --revision")
        root = Path(snapshot_download(args.model, revision=revision,
                                     allow_patterns=["*.json", "*.safetensors"]))
    model = AutoModel.from_pretrained(root, dtype=getattr(torch, args.dtype),
                                     attn_implementation="sdpa").to(args.device).eval()
    assert model.config.is_causal is False
    tokenizer = AutoTokenizer.from_pretrained(root)
    inputs = [("query", tokenizer.encode("query: What is the capital of France?")),
              ("passage", tokenizer.encode("passage: Paris is the capital of France.")),
              ("multilingual", tokenizer.encode("query: Quelle est la capitale de la France ?"))]
    long_ids = tokenizer.encode("passage: The quick brown fox jumps over the lazy dog. " * 500)
    inputs.extend((f"length_{n}", long_ids[:n]) for n in [1, 31, 129, 513, 2049])
    cases = []
    with torch.inference_mode():
        for name, ids in inputs:
            hidden = model(input_ids=torch.tensor([ids], device=args.device),
                           use_cache=False).last_hidden_state[0].float()
            sampled = sorted({0, len(ids) // 2, len(ids) - 1})
            cases.append(dict(name=name, input_ids=ids, mean=hidden.mean(0).cpu().tolist(),
                              raw_indices=sampled, raw=hidden[sampled].cpu().tolist()))
            print(name, len(ids), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(model=str(root.resolve()), revision=revision,
                                          dtype=args.dtype, cases=cases)) + "\n")


if __name__ == "__main__":
    main()

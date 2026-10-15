#!/usr/bin/env python3
"""Generate independent HF UltraBERT embedding or classification parity fixtures."""
import argparse
import json
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoTokenizer

PUBLIC_MODEL = "dmis-lab/UltraBERT-Embed-150M"
PUBLIC_REVISION = "367d5343b5a0164d57b810cedf5de4b8983d5984"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=PUBLIC_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--head", choices=["embedding", "sequence", "token"], default="embedding")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(args.model)
    if not root.is_dir():
        revision = args.revision or (PUBLIC_REVISION if args.model == PUBLIC_MODEL else None)
        if not revision:
            parser.error("Remote models require --revision for reproducible fixtures")
        root = Path(snapshot_download(args.model, revision=revision,
            allow_patterns=["*.json", "*.py", "*.safetensors"]))
    # UltraBERT's published custom implementation is the independent reference.
    cls = {"embedding": AutoModel, "sequence": AutoModelForSequenceClassification,
           "token": AutoModelForTokenClassification}[args.head]
    model = cls.from_pretrained(root, trust_remote_code=True, dtype=torch.float32,
        attn_implementation="sdpa").to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=True)
    cases = [("short", tokenizer.encode("What is deep learning?")),
             ("document", tokenizer.encode("Neural networks learn representations from data.")),
             ("query", tokenizer.encode("Instruct: Given a question, retrieve passages that answer the question\nQuery: What is deep learning?"))]
    long_ids = tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 500)
    cases.extend((f"length_{n}", long_ids[:n]) for n in [129, 130, 257, 513, 1025, 4097])
    if args.head == "sequence":
        cases.extend((f"pair_{i}", tokenizer.encode("What is the capital of France?", text))
                     for i, text in enumerate(["Paris is the capital of France.", "Bananas grow in warm climates."]))
    references = []
    with torch.inference_mode():
        for name, ids in cases:
            output = model(input_ids=torch.tensor([ids], device=args.device))
            if args.head == "embedding":
                hidden = output.last_hidden_state[0].float().cpu()
                sampled = sorted({0, len(ids)-1, min(128, len(ids)-1), min(129, len(ids)-1)})
                references.append(dict(name=name, input_ids=ids, pooled=hidden[0].tolist(),
                    mean=hidden.mean(0).tolist(), last=hidden[-1].tolist(),
                    raw_indices=sampled, raw=[hidden[i].tolist() for i in sampled]))
            else:
                logits = output.logits[0].float().cpu()
                references.append(dict(input_ids=ids,
                    logits=(logits[0] if args.head == "token" else logits).tolist(),
                    token_logits=logits.tolist() if args.head == "token" else None))
            print(name, len(ids), flush=True)
    if args.head != "embedding":
        references = [dict(model=str(root.resolve()), token=args.head == "token",
            labels=model.config.num_labels, cases=references)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(references) + "\n")

if __name__ == "__main__":
    main()

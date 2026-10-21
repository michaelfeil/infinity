#!/usr/bin/env python3
"""Generate and check Perplexity embedding references against the native HTTP server.

Reference environment: PyTorch 2.14.1, transformers==5.4.0,
sentence-transformers==6.0.0, pillow, torchvision, safetensors.

python scripts/perplexity-reference.py --output /tmp/pplx --tiny
python scripts/perplexity-reference.py --output /tmp/pplx --check-only --server target/debug/infinity_v3

Without --tiny, downloads trained weights. Use --model to select exports and
--device cuda --dtype bfloat16 for an accelerator comparison. Tiny fixtures test
all export contracts; they do not qualify trained retrieval accuracy.
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from safetensors.torch import save_file
from sentence_transformers import MultiVectorEncoder, SentenceTransformer
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

REVISIONS = {
    "pplx-embed-v1-0.6b": "2c4d510dd4a732063c31a0f70193e35067b51fd8",
    "pplx-embed-v1-4b": "06456497a00540a582918fe8dcd3a5eabb207772",
    "pplx-embed-context-v1-0.6b": "b42df969d4d78d1840769e45c68c4e4ce763768b",
    "pplx-embed-context-v1-4b": "679845b2236c653177fff8fa3ffc1b22f3090d30",
    "pplx-embed-v1-late-0.6b": "4d28cf627d225552cfc29fb7df6cb0705ea0f1b3",
    "pplx-embed-v2-context-9b-preview": "e2c06a779d45e8db53066735be1b98ffe3412ca3",
    "pplx-embed-v2-late-0.6b": "8fc2de24534aa3610d85fa59c463313a5f096455",
    "pplx-embed-v2-late-9b": "0f49a9977fe06b83377d598094c5c0204ce18ad9",
}


def export(args, name):
    root = args.output / name
    root.mkdir(parents=True, exist_ok=True)
    if args.metadata_root:
        source = args.metadata_root / name
        for path in source.rglob("*"):
            if path.is_file() and path.suffix in (".json", ".py", ".jinja"):
                dest = root / path.relative_to(source)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
    else:
        patterns = ["*.json", "*.py", "*.jinja"]
        if not args.tiny:
            patterns += ["*.safetensors", "*.bin"]
        snapshot_download(
            "perplexity-ai/" + name, revision=REVISIONS[name], local_dir=root,
            allow_patterns=patterns, ignore_patterns=["onnx/*", ".cache/*"],
        )
    if args.tiny:
        config = AutoConfig.from_pretrained(root, trust_remote_code=True)
        text = getattr(config, "text_config", config)
        text.hidden_size, text.intermediate_size, text.num_hidden_layers = 32, 64, 2
        text.num_attention_heads, text.num_key_value_heads, text.head_dim = 4, 2, 16
        text.max_position_embeddings, text.use_cache = 2048, False
        if "qwen3_5" in config.model_type:
            text.head_dim = 32
            text.layer_types = ["linear_attention", "full_attention"]
            text.rope_parameters["mrope_section"] = [1, 1, 2]
            text.linear_num_key_heads, text.linear_num_value_heads = 1, 2
            vision = config.vision_config
            vision.depth, vision.hidden_size, vision.intermediate_size = 1, 32, 64
            vision.num_heads, vision.out_hidden_size, vision.num_position_embeddings = 4, 32, 16
            vision.deepstack_visual_indexes = []
            if hasattr(config, "embedding_dim"):
                config.embedding_dim = 16
        else:
            text.layer_types = ["full_attention"] * 2
        config._name_or_path, config._attn_implementation = str(root), "sdpa"
        torch.manual_seed(731)
        if hasattr(config, "auto_map") and "AutoModel" in config.auto_map:
            cls = get_class_from_dynamic_module(config.auto_map["AutoModel"], str(root))
            # Use the exact config class imported by the remote model module.
            config = cls.config_class.from_dict(config.to_dict())
            AutoConfig.register(config.model_type,cls.config_class,exist_ok=True)
            config._name_or_path = str(root)
            model = cls(config).eval()
        else:
            model = AutoModel.from_config(config).eval()
        save_file({key: value.clone().contiguous() for key, value in model.state_dict().items()}, root / "model.safetensors")
        config.save_pretrained(root)
        if "late" in name:
            dense_path = root / "1_Dense/config.json"
            dense = json.loads(dense_path.read_text())
            dense["in_features"] = 32
            dense_path.write_text(json.dumps(dense))
            save_file({"linear.weight": torch.randn(128, 32) * .1}, root / "1_Dense/model.safetensors")
        del model
    return root


def references(args, name, root):
    dtype = getattr(torch, args.dtype)
    records = []
    if "context" in name:
        config = AutoConfig.from_pretrained(root, trust_remote_code=True)
        cls = get_class_from_dynamic_module(config.auto_map["AutoModel"], str(root))
        config = cls.config_class.from_dict(config.to_dict())
        AutoConfig.register(config.model_type,cls.config_class,exist_ok=True)
        config._name_or_path = str(root)
        model = cls.from_pretrained(root, config=config, dtype=dtype).to(args.device).eval()
        model.tokenizer = AutoTokenizer.from_pretrained(root, trust_remote_code=True)
        cases = [([["Alpha.", "Beta!"], ["中文 café", "", "Gamma"]], "document"), ([["What is blue?"]], "query")]
        for chunks, task in cases:
            kwargs = {"task": task} if "v2" in name else {}
            vectors = model.encode(chunks, show_progress_bar=False, **kwargs)
            records.append({"request": {"input": chunks, "task": task}, "expected": [v.tolist() for v in vectors]})
    else:
        cls = MultiVectorEncoder if "late" in name else SentenceTransformer
        model = cls(str(root), device=args.device, trust_remote_code=True,
                    model_kwargs={"dtype": dtype, "attn_implementation": "sdpa"}).to(dtype=dtype)
        if "late" in name:
            for texts, task in [(["Hello, world!", "中文 café"], "document"), (["What is blue?"], "query")]:
                vectors = getattr(model, "encode_" + task)(texts, show_progress_bar=False)
                records.append({"request": {"input": texts, "task": task}, "expected": [v.tolist() for v in vectors]})
            if "v2" in name:
                pixels = np.random.default_rng(731).integers(0, 256, (64, 96, 3), dtype=np.uint8)
                image = Image.fromarray(pixels)
                image.save(root / "reference-image.png")
                source = "data:image/png;base64," + base64.b64encode((root / "reference-image.png").read_bytes()).decode()
                vectors = model.encode_document([image], show_progress_bar=False)
                records.append({"request": {"input": {"image_url": source}}, "expected": [v.tolist() for v in vectors]})
        else:
            for quantization in ["int8", "binary", "ubinary"]:
                texts = ["Hello, world!", "中文 café"]
                vectors = model.encode(texts, show_progress_bar=False, quantization=quantization)
                records.append({"request": {"input": texts, "quantization": quantization}, "expected": [v.tolist() for v in vectors]})
    (root / "reference.json").write_text(json.dumps(records))
    print("Generated", name, flush=True)
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()


def check(args, name):
    root = args.output / name
    env = {**os.environ, "OMP_NUM_THREADS": "4", "INFINITY_ANONYMOUS_USAGE_STATS": "0"}
    url = f"http://127.0.0.1:{args.port}"
    with (root / "server.log").open("w") as log:
        command = [str(args.server.resolve()), "--model-id", str(root.resolve()), "--backend", "libtorch",
                   "--torch-device", args.device, "--dtype", args.dtype, "--hostname", "127.0.0.1",
                   "--port", str(args.port), "--tokenization-workers", "2", "--max-batch-tokens", str(args.max_batch_tokens)]
        server = subprocess.Popen(command, stdout=log, stderr=log, env=env)
        try:
            for _ in range(1200):
                if server.poll() is not None:
                    raise RuntimeError((root / "server.log").read_text()[-6000:])
                try:
                    urllib.request.urlopen(url + "/health", timeout=1).close()
                    break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(.1)
            else:
                raise TimeoutError("Server did not start")
            for record in json.loads((root / "reference.json").read_text()):
                route = "/v1/contextualizedembeddings" if "context" in name else "/v1/embeddings"
                req = urllib.request.Request(url + route, data=json.dumps(record["request"]).encode(), headers={"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(req, timeout=60) as response:
                        actual = json.load(response)
                except urllib.error.HTTPError as error:
                    raise RuntimeError(error.read().decode()) from error
                arrays = ([[x["embedding"] for x in doc["data"]] for doc in actual["data"]]
                          if "context" in name else [doc["embedding"] for doc in actual["data"]])
                for a, b in zip(arrays, record["expected"], strict=True):
                    a, b = np.asarray(a), np.asarray(b)
                    assert a.shape == b.shape, (name, a.shape, b.shape)
                    delta = float(np.max(np.abs(a - b))) if a.size else 0.
                    assert delta <= args.atol, (name, "max absolute error", delta)
                    if args.min_cosine and a.size:
                        aa, bb = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
                        norm = np.linalg.norm(aa, axis=-1) * np.linalg.norm(bb, axis=-1)
                        cos = np.sum(aa * bb, axis=-1) / norm.clip(1e-12)
                        assert cos.min() >= args.min_cosine, (name, "minimum cosine", cos.min())
                print("PASS", name, record["request"].get("task", record["request"].get("quantization", "image")), flush=True)
        finally:
            server.terminate()
            try:
                server.wait(15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=REVISIONS, action="append")
    parser.add_argument("--metadata-root", type=Path, help="Use previously downloaded export metadata")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--server", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--port", type=int, default=18797)
    parser.add_argument("--max-batch-tokens", type=int, default=2048)
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--min-cosine", type=float, default=0.)
    args = parser.parse_args()
    if args.check_only and not args.server:
        parser.error("--check-only requires --server")
    args.output = args.output.resolve()
    torch.set_num_threads(4)
    for name in args.model or REVISIONS:
        if not args.check_only:
            references(args, name, export(args, name))
        if args.server:
            check(args, name)


if __name__ == "__main__":
    main()

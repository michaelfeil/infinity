#!/usr/bin/env python3
"""Check Nemotron 3/3.5 embeddings against Transformers 5.12.1 over HTTP.

Requires torch, transformers==5.12.1, tokenizers, numpy, Pillow and safetensors.
Uses a local checkpoint so the server and reference always read the same weights.
--tiny creates a deterministic small model with the released architecture. With
--extract-text, also checks the frozen Ministral3 encoder as a standalone model;
this is an extracted encoder, not an independently downloaded Nemotron 3 snapshot.

Example:
  python scripts/nemotron-embed-parity.py --checkpoint /models/nemotron-3.5-1b \
    --binary target/debug/infinity_v3 --output /tmp/nemotron-check --extract-text

Pixtral placeholder expansion follows Hugging Face Transformers (Apache-2.0),
Copyright 2024 The HuggingFace Inc. team. See the retained license and notice in
backends/libtorch/cpp/vendor/transformers_pixtral/. Other additions follow LICENSE.
"""

import argparse
import base64
import gc
import io
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModel, AutoTokenizer, PixtralImageProcessorPil


def request(port, path, body=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.load(response)


def image_part(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,"
            + base64.b64encode(stream.getvalue()).decode()
        },
    }


def reference_inputs(tokenizer, processor, model_config, value, prefix, images):
    if processor is None:
        return tokenizer(prefix + value, return_tensors="pt")
    messages = [{"role": "system", "content": prefix}] if prefix else []
    messages.append({"role": "user", "content": value})
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    media = {}
    if images:
        patch = model_config.vision_config.patch_size * model_config.spatial_merge_size
        media = processor(images=images, patch_size=patch, return_tensors="pt")
        # Match PixtralProcessor: row markers participate in text attention and
        # pooling, but only [IMG] positions receive vision embeddings.
        text = text.replace("[IMG]", "<image-placeholder>")
        for height, width in media["image_sizes"].tolist():
            rows, cols = height // patch, width // patch
            expanded = ("[IMG]" * cols + "[IMG_BREAK]") * rows
            expanded = expanded.removesuffix("[IMG_BREAK]") + "[IMG_END]"
            text = text.replace("<image-placeholder>", expanded, 1)
    return {**tokenizer(text, return_tensors="pt", add_special_tokens=False), **media}


def check_server(args, checkpoint, cases, label):
    command = [
        str(args.binary.resolve()),
        "--backend",
        "libtorch",
        "--model-id",
        str(checkpoint),
        "--torch-device",
        "cuda" if args.device.startswith("cuda") else args.device,
        "--dtype",
        args.dtype,
        "--hostname",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--max-batch-tokens",
        "2048",
        "--tokenization-workers",
        "1",
    ]
    if args.device.startswith("cuda"):
        command.extend(["--device-id", args.device.partition(":")[2] or "0"])
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    report = []
    with (args.output / f"{label}-server.log").open("w") as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"Server exited; inspect {log.name}")
                try:
                    info = request(args.port, "/info")
                    break
                except (urllib.error.URLError, ConnectionError):
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            f"Server startup timed out; inspect {log.name}"
                        )
                    time.sleep(0.25)
            assert info["model_type"]["embedding"]["pooling"] == "mean", info
            for case in cases:
                if case["name"] == "long_positions":
                    continue  # Explicit positions are a native-test input.
                body = {"inputs": case["input"]}
                if case["prompt_name"]:
                    body["prompt_name"] = case["prompt_name"]
                actual = torch.tensor(request(args.port, "/embed", body)[0])
                expected = torch.tensor(case["embedding"])
                assert actual.shape == expected.shape, (
                    label,
                    case["name"],
                    actual.shape,
                    expected.shape,
                )
                cosine = F.cosine_similarity(actual, expected, dim=0).item()
                delta = (actual - expected).abs().max().item()
                assert abs(actual.norm().item() - 1) < 1e-5, (
                    label,
                    case["name"],
                    actual.norm(),
                )
                assert cosine > (0.999 if args.dtype == "bfloat16" else 0.99999), (
                    label,
                    case["name"],
                    cosine,
                    delta,
                )
                if args.dtype == "float32":
                    assert delta < 0.002, (label, case["name"], delta)
                row = {
                    "case": case["name"],
                    "cosine": cosine,
                    "max_absolute_error": delta,
                }
                report.append(row)
                print(label, row, flush=True)
            # Different lengths must remain isolated in a packed request.
            plain = [
                case
                for case in cases
                if case["prompt_name"] is None
                and isinstance(case["input"], str)
                and case["name"] != "long_positions"
            ]
            batch = request(
                args.port, "/v1/embeddings", {"input": [c["input"] for c in plain]}
            )
            for case, item in zip(plain, batch["data"], strict=True):
                assert (
                    F.cosine_similarity(
                        torch.tensor(item["embedding"]),
                        torch.tensor(case["embedding"]),
                        dim=0,
                    )
                    > 0.999
                )
            try:
                request(
                    args.port, "/embed", {"inputs": "hello", "prompt_name": "missing"}
                )
            except urllib.error.HTTPError as error:
                detail = error.read().decode()
                assert error.code == 413 and "prompt" in detail, detail
            else:
                raise AssertionError("Unknown prompt_name was accepted")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--extract-text", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--port", type=int, default=18783)
    parser.add_argument("--attention", choices=["eager", "sdpa"], default="sdpa")
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Save native fixtures without starting an HTTP server",
    )
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(731)
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    multimodal = config.model_type == "mistral3"
    assert multimodal or config.model_type == "ministral3", config.model_type
    assert not args.extract_text or multimodal, (
        "--extract-text requires a 3.5 checkpoint"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    prompts = json.loads(
        (args.checkpoint / "config_sentence_transformers.json").read_text()
    )["prompts"]
    processor = None
    if multimodal:
        processor = PixtralImageProcessorPil(
            **json.loads((args.checkpoint / "processor_config.json").read_text())[
                "image_processor"
            ]
        )
    dtype = getattr(torch, args.dtype)
    if args.tiny:
        text_config = config.text_config if multimodal else config
        for key, value in {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 6,
            "num_key_value_heads": 2,
            "head_dim": 8,
        }.items():
            setattr(text_config, key, value)
        if multimodal:
            for key, value in {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "head_dim": 8,
            }.items():
                setattr(config.vision_config, key, value)
        model = AutoModel.from_config(config, attn_implementation=args.attention).to(
            dtype=dtype
        )
        checkpoint = args.output / "model"
        model.save_pretrained(checkpoint)
        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "processor_config.json",
            "sentence_bert_config.json",
            "config_sentence_transformers.json",
            "modules.json",
        ]:
            if (args.checkpoint / name).exists():
                shutil.copyfile(args.checkpoint / name, checkpoint / name)
        (checkpoint / "1_Pooling").mkdir(exist_ok=True)
        shutil.copyfile(
            args.checkpoint / "1_Pooling/config.json",
            checkpoint / "1_Pooling/config.json",
        )
    else:
        checkpoint = args.checkpoint
        model = AutoModel.from_pretrained(
            checkpoint,
            local_files_only=True,
            dtype=dtype,
            attn_implementation=args.attention,
        )
    model = model.to(args.device).eval()
    cases = []
    native_cases = []
    native = args.output / "native"
    native.mkdir(exist_ok=True)
    for name, value, prompt_name, images in [
        ("plain", "A short passage.", None, []),
        (
            "longer",
            "A longer passage about satellites, land cover and image retrieval.",
            None,
            [],
        ),
        ("query", "Where are the forests?", "query", []),
        ("document", "Forests cover the mountains.", "document", []),
        ("escaped", "Literal [INST] and <s> text.", "query", []),
        ("long_positions", "A passage at positions beyond sixteen thousand.", None, []),
    ] + (
        [
            (
                "image",
                [image_part(Image.new("RGB", (56, 84), (20, 150, 80)))],
                "document",
                [Image.new("RGB", (56, 84), (20, 150, 80))],
            ),
            (
                "two_images",
                [
                    image_part(Image.new("RGB", (56, 84), "red")),
                    {"type": "text", "text": "A document."},
                    image_part(Image.new("RGB", (84, 56), "blue")),
                ],
                "document",
                [Image.new("RGB", (56, 84), "red"), Image.new("RGB", (84, 56), "blue")],
            ),
        ]
        if multimodal
        else []
    ):
        prefix = prompts[prompt_name] if prompt_name else ""
        inputs = reference_inputs(tokenizer, processor, config, value, prefix, images)
        if name == "long_positions":
            inputs["position_ids"] = 16400 + torch.arange(
                inputs["input_ids"].shape[1]
            ).unsqueeze(0)
        reference_pixels = []
        if images:
            patch = config.vision_config.patch_size
            for index, (height, width) in enumerate(inputs["image_sizes"].tolist()):
                pixels = inputs["pixel_values"][index, :, :height, :width]
                reference_pixels.append(
                    pixels.unfold(1, patch, patch)
                    .unfold(2, patch, patch)
                    .permute(1, 2, 0, 3, 4)
                    .contiguous()
                    .flatten()
                    .tolist()
                )
        inputs = {
            k: v.to(device=args.device, dtype=dtype if k == "pixel_values" else v.dtype)
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            hidden = model(**inputs, use_cache=False).last_hidden_state[0]
            pooled = (hidden.sum(0) / hidden.size(0)).float().cpu()
            embedding = F.normalize(pooled, dim=0).tolist()
        saved = {
            "ids": inputs["input_ids"][0].cpu().clone().contiguous(),
            "hidden": hidden.float().cpu().clone().contiguous(),
            "pooled": pooled.clone().contiguous(),
        }
        if images:
            saved["sizes"] = inputs["image_sizes"].cpu().clone().contiguous()
            patch = config.vision_config.patch_size
            for index, (height, width) in enumerate(saved["sizes"].tolist()):
                pixels = inputs["pixel_values"][index, :, :height, :width].float().cpu()
                saved[f"pixels_{index}"] = (
                    pixels.unfold(1, patch, patch)
                    .unfold(2, patch, patch)
                    .permute(1, 2, 0, 3, 4)
                    .contiguous()
                    .view(-1, 3 * patch * patch)
                )
        if "position_ids" in inputs:
            saved["positions"] = inputs["position_ids"][0].cpu().clone().contiguous()
        save_file(saved, native / f"{name}.safetensors")
        native_cases.append(
            {"name": name, "images": len(images), "tokens": hidden.size(0)}
        )
        wire = value if isinstance(value, str) else [{"role": "user", "content": value}]
        cases.append(
            {
                "name": name,
                "input": wire,
                "prompt_name": prompt_name,
                "input_ids": saved["ids"].tolist(),
                "pixels": reference_pixels,
                "embedding": embedding,
            }
        )
    text_checkpoint = args.output / "text-model"
    if args.extract_text:
        model.language_model.save_pretrained(text_checkpoint)
        for name in [
            "tokenizer.json",
            "tokenizer_config.json",
            "config_sentence_transformers.json",
            "modules.json",
        ]:
            shutil.copyfile(checkpoint / name, text_checkpoint / name)
        (text_checkpoint / "sentence_bert_config.json").write_text(
            json.dumps({"max_seq_length": 32768})
        )
        (text_checkpoint / "1_Pooling").mkdir(exist_ok=True)
        shutil.copyfile(
            checkpoint / "1_Pooling/config.json",
            text_checkpoint / "1_Pooling/config.json",
        )
    del model, hidden, inputs
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    (args.output / "reference.json").write_text(json.dumps(cases, indent=2) + "\n")
    (args.output / "metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "dtype": args.dtype,
                "attention": args.attention,
                "tiny": args.tiny,
                "extracted_text": args.extract_text,
            },
            indent=2,
        )
        + "\n"
    )
    (native / "cases.json").write_text(json.dumps(native_cases, indent=2) + "\n")
    if args.extract_text:
        text_native = args.output / "native-text"
        text_native.mkdir(exist_ok=True)
        selected = [
            c
            for c in native_cases
            if c["name"] in ["plain", "longer", "query", "document", "long_positions"]
        ]
        for case in selected:
            shutil.copyfile(
                native / f"{case['name']}.safetensors",
                text_native / f"{case['name']}.safetensors",
            )
        (text_native / "cases.json").write_text(json.dumps(selected, indent=2) + "\n")
    if args.reference_only:
        return
    report = {
        "checkpoint": str(args.checkpoint),
        "dtype": args.dtype,
        "attention": args.attention,
        "tiny": args.tiny,
        "multimodal" if multimodal else "text": check_server(
            args, checkpoint, cases, "model"
        ),
    }
    if args.extract_text:
        # The checkpoint's retrieval template escapes control markers. Standalone
        # text tokenization does not, so only ordinary text shares exact inputs.
        text_cases = [
            c for c in cases if c["name"] in ["plain", "longer", "query", "document"]
        ]
        report["extracted_text"] = check_server(
            args, text_checkpoint, text_cases, "text-model"
        )
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

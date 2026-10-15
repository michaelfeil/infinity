#!/usr/bin/env python3
"""Public Mistral3 reference fixtures. Requires Transformers 5.12.1 and PyTorch.

Example: python scripts/mistral3-reference.py --output /tmp/mistral3-reference --tiny
For trained BF16 weights omit --tiny and use --device cuda.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from safetensors.torch import save_file
from transformers import AutoConfig, AutoProcessor, Mistral3Model

PUBLIC_MODEL = "mistralai/Ministral-3-3B-Instruct-2512-BF16"
PUBLIC_REVISION = "b6d637bef2393152b3da2b2fde72eecdee30557e"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=PUBLIC_MODEL)
    parser.add_argument("--revision",default=PUBLIC_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--bidirectional", action="store_true", help="Synthetic noncausal decoder with independent attention width")
    parser.add_argument("--pooling", choices=["last-token", "mean"], default="last-token")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--attention", choices=["eager","sdpa"], default="eager")
    parser.add_argument("--dtype", choices=["float32","bfloat16"],default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(731)
    dtype = getattr(torch,args.dtype) if args.dtype else (torch.float32 if args.tiny else torch.bfloat16)
    config = AutoConfig.from_pretrained(args.model,revision=args.revision)
    processor = AutoProcessor.from_pretrained(args.model,revision=args.revision)
    if args.tiny:
        config.text_config.hidden_size = 32
        config.text_config.intermediate_size = 64
        config.text_config.num_hidden_layers = 2
        config.text_config.num_attention_heads = 6 if args.bidirectional else 4
        config.text_config.is_causal = not args.bidirectional
        config.text_config.num_key_value_heads = 2
        config.text_config.head_dim = 8
        config.text_config.max_position_embeddings = 2048
        config.text_config.rope_parameters = {"rope_type":"yarn", "rope_theta":10000.0, "factor":1.0, "original_max_position_embeddings":128, "beta_fast":32.0, "beta_slow":1.0, "mscale":1.0, "mscale_all_dim":1.0, "llama_4_scaling_beta":0.1}
        config.vision_config.hidden_size = 32
        config.vision_config.intermediate_size = 64
        config.vision_config.num_hidden_layers = 2
        config.vision_config.num_attention_heads = 4
        config.vision_config.head_dim = 8
        processor.image_processor.size = {"longest_edge": 112}
        config._attn_implementation = args.attention
        model = Mistral3Model(config).eval()
        model.save_pretrained(args.output / "model")
        processor.save_pretrained(args.output / "model")
        pooling_dir = args.output / "model" / "1_Pooling"
        pooling_dir.mkdir(exist_ok=True)
        (pooling_dir / "config.json").write_text(json.dumps({"pooling_mode": args.pooling.replace("-", "")}))
    else:
        model = Mistral3Model.from_pretrained(args.model,revision=args.revision, dtype=dtype, attn_implementation=args.attention).eval()
    model = model.to(args.device, dtype=dtype)
    # Non-uniform pixels expose channel/patch/merge-order mistakes.
    def pattern(w, h):
        y, x = np.indices((h, w))
        return Image.fromarray(np.stack(((x*13+y*7)%256, (x*3+y*19)%256, (x*29+y*5)%256), -1).astype(np.uint8))
    exif_image=pattern(56,84)
    exif_image.getexif()[274]=6
    edge = int(processor.image_processor.size["longest_edge"])
    cases = [("text", []), ("square", [pattern(56, 56)]),
             ("portrait", [pattern(56, 84)]), ("resize", [pattern(37, 65)]),
             ("two_images", [pattern(56, 84), pattern(84, 56)]), ("exif", [exif_image]), ("downscale", [pattern(edge+19,edge+85)])]
    records = []
    for name, images in cases:
        # Orient the visible image explicitly: the pinned fast processor does not
        # transpose PIL EXIF itself, while the serving API honors orientation.
        content = [{"type": "image", "image": ImageOps.exif_transpose(image)} for image in images]
        content.append({"type": "text", "text": "Describe the shapes and colors."})
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": content}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                                return_tensors="pt", return_dict=True)
        saved = {"ids": inputs["input_ids"][0].contiguous()}
        if images:
            for i, (h, w) in enumerate(inputs["image_sizes"]):
                pixels = inputs["pixel_values"][i, :, :h, :w].clone()
                patch = config.vision_config.patch_size
                saved[f"pixels_{i}"] = pixels.unfold(1, patch, patch).unfold(2, patch, patch).permute(1,2,0,3,4).contiguous().view(-1,3*patch*patch)
            saved["sizes"] = torch.as_tensor(inputs["image_sizes"], dtype=torch.int64).clone()
        inputs = {key: value.to(args.device) if isinstance(value, torch.Tensor) else value for key,value in inputs.items()}
        if images:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        with torch.inference_mode():
            native_hidden = model(**inputs, use_cache=False).last_hidden_state[0]
            pooled = native_hidden.sum(0) / native_hidden.size(0) if args.pooling == "mean" else native_hidden[-1]
            hidden = native_hidden.float().cpu()
        saved["pooled"] = pooled.float().cpu().contiguous().clone()
        saved["hidden"] = hidden.contiguous()
        save_file(saved, args.output / f"{name}.safetensors")
        for i,image in enumerate(images):
            image.save(args.output / f"{name}-{i}.png",exif=image.getexif())
        records.append({"name": name, "images": len(images), "tokens": hidden.size(0)})
        print(name, hidden.shape, flush=True)
    (args.output / "cases.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()

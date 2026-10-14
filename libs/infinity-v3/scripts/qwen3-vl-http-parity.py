"""Check an already running FP16 Qwen3-VL embedding server against Transformers.

Reference dependencies: transformers==5.17.0, torch==2.11.0,
torchvision==0.26.0, Pillow==12.3.0. Use the same local checkpoint/revision
for the server and this script. Tested checkpoint revision:
Qwen/Qwen3-VL-Embedding-2B@9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda.

CUDA_VISIBLE_DEVICES=REFERENCE_GPU python scripts/qwen3-vl-http-parity.py \
  --checkpoint /path/to/checkpoint --url http://127.0.0.1:8080
"""
import argparse
import base64
import concurrent.futures
import io
import json
import urllib.error
import urllib.request

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLModel

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--url", default="http://127.0.0.1:8080")
args = parser.parse_args()


def post(value):
    request = urllib.request.Request(
        args.url + "/v1/embeddings", data=json.dumps({"input": value}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return torch.tensor([item["embedding"] for item in json.load(response)["data"]])


def encode(image, fmt="PNG"):
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


def image_part(raw, mime="png"):
    return {"type": "image_url", "image_url": {
        "url": "data:image/" + mime + ";base64," + base64.b64encode(raw).decode()
    }}


image = Image.new("RGB", (512, 512))
image.putdata([((x*17+y*3)%256, (x*5+y*29)%256, ((x^y)*23)%256)
               for y in range(512) for x in range(512)])
other = Image.new("RGB", image.size, (20, 180, 60))
small = other.resize((96, 64))
text = {"type": "text", "text": "A colorful image."}
png, other_png, webp, jpeg = encode(image), encode(other), encode(image, "WEBP"), encode(image, "JPEG")
cases = [
    ("text", [text], [text]),
    ("image", [image_part(png)], [{"type": "image", "image": image}]),
    ("image_text", [image_part(png), text], [{"type": "image", "image": image}, text]),
    ("different_image", [image_part(other_png), text], [{"type": "image", "image": other}, text]),
    ("two_images", [image_part(png), text, image_part(encode(small))],
     [{"type": "image", "image": image}, text, {"type": "image", "image": small}]),
    ("webp", [image_part(webp, "webp"), text],
     [{"type": "image", "image": Image.open(io.BytesIO(webp)).convert("RGB")}, text]),
    ("jpeg", [image_part(jpeg, "jpeg"), text],
     [{"type": "image", "image": Image.open(io.BytesIO(jpeg)).convert("RGB")}, text]),
]
processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
model = Qwen3VLModel.from_pretrained(
    args.checkpoint, local_files_only=True, dtype=torch.float16, attn_implementation="sdpa"
).cuda().eval()
report = {}
actuals = []
for name, wire, reference in cases:
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": reference}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt",
    )
    inputs = {k: (v.cuda().half() if k == "pixel_values" else v.cuda())
              for k, v in inputs.items() if torch.is_tensor(v)}
    with torch.inference_mode():
        expected = torch.nn.functional.normalize(
            model(**inputs).last_hidden_state[0, -1].float(), dim=-1
        ).cpu()
    actual = post([{"role": "user", "content": wire}])[0]
    actuals.append(actual)
    cosine = float((actual * expected).sum())
    report[name] = {"cosine": cosine, "max_abs": float((actual - expected).abs().max())}
    # JPEG samples are decoded differently by Rust and Pillow; report the
    # difference without treating different input pixels as backend parity.
    if name != "jpeg":
        assert cosine > 0.9995, (name, report[name])

wire_inputs = [[{"role": "user", "content": wire}] for _, wire, _ in cases[:4]]
with concurrent.futures.ThreadPoolExecutor(4) as pool:
    parallel_outputs = torch.cat(list(pool.map(post, wire_inputs)))
with concurrent.futures.ThreadPoolExecutor(4) as pool:
    reordered = torch.cat(list(pool.map(post, wire_inputs[::-1]))).flip(0)
expected = torch.stack(actuals[:4])
report["concurrency_min_cosine"] = float((parallel_outputs * expected).sum(-1).min())
report["reordering_min_cosine"] = float((reordered * expected).sum(-1).min())
assert report["concurrency_min_cosine"] > 0.9995
assert report["reordering_min_cosine"] > 0.9995
report["different_image_cosine"] = float((actuals[2] * actuals[3]).sum())
assert report["different_image_cosine"] < 0.95
# A normal text batch may exceed the CPU image worker count.
assert post(["A colorful image."] * 8).shape == (8, 2048)
for invalid in ["", [{"role": "user", "content": [image_part(b"invalid image")]}],
                [{"role": "user", "content": [{"type": "image_url", "image_url": {
                    "url": "https://127.0.0.1/blocked?secret=must-not-echo"}}]}]]:
    try:
        post(invalid)
        raise AssertionError("Invalid input was accepted")
    except urllib.error.HTTPError as error:
        assert error.code == (400 if invalid == "" else 422), error.code
        assert "must-not-echo" not in error.read().decode()
print(json.dumps(report, indent=2))

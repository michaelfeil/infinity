#!/usr/bin/env python3
"""Generate D3 requests and upstream answers for the HTTP smoke test.

Requires the downloaded vllm-sr/d3 repository and its published Python dependencies
(including accelerate and opencv-python-headless). Generated files stay outside
the source tree.
"""
import argparse
import base64
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text-only", action="store_true", help="Skip checkpoint-specific image/video assets")
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(checkpoint))
    from d3_runtime import D3

    route = {"type": "choice", "instructions": "Which team should handle this request?",
             "criteria": {"returns": "Refunds, replacements and damaged deliveries",
                          "billing": "Payments, invoices and charges", "technical": "Product setup and faults"}}
    direction = {"type": "choice", "instructions": "Which way does the square move?",
                 "criteria": {"right": "From left to right", "left": "From right to left", "still": "It does not move"}}
    receipt = {"type": "noul", "instructions": "Does the receipt list a blender?"}
    image = str(checkpoint / "assets/example-receipt.png")
    video = str(checkpoint / "assets/example-video.mp4")
    cases = {
        "text": {"state": "The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
                 "questions": {"route": route, "receipt": {"type": "noul", "instructions": "Does the customer have a receipt?"},
                               "urgency": {"type": "score", "instructions": "How urgent is this request?", "criteria": ["Routine", "Soon", "Today"]}}},
        "image": {"state": "The customer says the blender arrived cracked and attached the receipt.", "images": [image],
                  "questions": {"receipt": receipt, "payment": {"type": "choice", "instructions": "How was the order paid?",
                                                              "criteria": {"card": None, "cash": None, "gift card": None}}}},
        "video": {"state": "A short clip from a test camera.", "videos": [video],
                  "questions": {"direction": direction, "color": {"type": "noul", "instructions": "Does the square change color?"}}},
        "mixed": {"state": "The attached receipt and a short clip from a test camera.", "images": [image], "videos": [video],
                  "questions": {"receipt": receipt, "direction": direction}},
    }
    if args.text_only:
        cases = {"text": cases["text"]}
    model = D3(checkpoint, device=args.device, batch_size=1)
    for name, request in cases.items():
        prepared = model.prepare(**request)
        result = model.respond(prepared, *model.run(prepared))
        wire = dict(request)
        for field, mime in [("images", "image/png"), ("videos", "video/mp4")]:
            if field in wire:
                wire[field] = [f"data:{mime};base64," + base64.b64encode(Path(p).read_bytes()).decode() for p in wire[field]]
        (output / f"{name}-request.json").write_text(json.dumps(wire))
        (output / f"{name}-response.json").write_text(json.dumps(result, indent=2))
        if prepared.sequences:
            (output / f"{name}-tokens.json").write_text(json.dumps({
                "sequences": prepared.sequences, "states": getattr(prepared, "states", {})
            }, indent=2))
        print(name, json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

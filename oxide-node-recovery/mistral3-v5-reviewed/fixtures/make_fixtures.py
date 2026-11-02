#!/usr/bin/env python3
"""Emit CPU-only Mistral3/Pixtral processor fixtures from the real cached config.

Only config.json, processor_config.json, and tokenizer_config.json are read.
Tensors, tokenizer.json, images, CUDA, torch, and the network are not touched.

The image pixel payloads are *synthetic geometry*: every value is
    ((row * 1009 + col * 1013) % 997) / 997.0
where row is the patch row and col is the patch element index. They are not a
decode of any real image and carry no visual content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = Path(
    "/workspace/model-performance/michaelfeil/benchmark-overview/models/"
    "nvidia--Nemotron-3.5-Rerank-8B-BF16"
)
READ_FILES = ("config.json", "processor_config.json", "tokenizer_config.json")
FILLER_TOKEN_ID = 21


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.read_bytes()
    return json.loads(raw), {"sha256": sha256_bytes(raw), "bytes": len(raw)}


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def patch_rows(grid: tuple[int, int, int]) -> int:
    return product(list(grid))


def token_rows(grid: tuple[int, int, int], merge: int) -> int:
    t, h, w = grid
    assert h % merge == 0 and w % merge == 0, (grid, merge)
    assert t == 1, "the cached Pixtral processor is single-frame"
    return (h // merge) * (w // merge)


def merge_source_rows(grid: tuple[int, int, int], merge: int) -> list[list[int]]:
    """Mistral3 SpatialRowMajor merge order: merge_row * merge + merge_col."""
    _, height, width = grid
    blocks_wide = width // merge
    blocks_high = height // merge
    rows: list[list[int]] = []
    for token_row in range(token_rows(grid, merge)):
        block_row = token_row // blocks_wide
        block_column = token_row % blocks_wide
        token_sources: list[int] = []
        for merge_row in range(merge):
            row = block_row * merge + merge_row
            for merge_column in range(merge):
                column = block_column * merge + merge_column
                token_sources.append(row * width + column)
        assert len(token_sources) == merge * merge
        rows.append(token_sources)
    assert len(rows) == blocks_high * blocks_wide
    return rows


def marker_indices(grid: tuple[int, int, int], merge: int, token_start: int) -> list[int]:
    nw = grid[2] // merge
    count = token_rows(grid, merge)
    return [
        token_start + (image_token // nw) * (nw + 1) + image_token % nw
        for image_token in range(count)
    ]


def pixel_values(grid: tuple[int, int, int], patch_dim: int) -> list[float]:
    values: list[float] = []
    for row in range(patch_rows(grid)):
        for col in range(patch_dim):
            values.append(((row * 1009 + col * 1013) % 997) / 997.0)
    return values


def pixel_sha256(values: list[float]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<f", value))
    return digest.hexdigest()


def geometry(grid: tuple[int, int, int], merge: int, patch_dim: int) -> dict[str, Any]:
    return {
        "grid_thw": list(grid),
        "merge_size": merge,
        "patch_dim": patch_dim,
        "patch_rows": patch_rows(grid),
        "token_rows": token_rows(grid, merge),
        "merge_source_row_order": "[merge_row * merge + merge_col]",
        "marker_formula": (
            "token_start + floor(image_token / (grid_w / merge)) * "
            "(grid_w / merge + 1) + image_token % (grid_w / merge)"
        ),
    }


def image_case(
    name: str,
    cumulative: list[int],
    descriptors: list[dict[str, Any]],
    merge: int,
    patch_dim: int,
    assets: dict[str, Any],
) -> dict[str, Any]:
    total = cumulative[-1]
    ids = [FILLER_TOKEN_ID] * total
    images: list[dict[str, Any]] = []
    for descriptor in descriptors:
        grid = tuple(descriptor["grid_thw"])
        token_start = descriptor["token_start"]
        count = token_rows(grid, merge)
        markers = marker_indices(grid, merge, token_start)
        for index in markers:
            assert 0 <= index < total, (name, index, total)
            ids[index] = 10
        sequence_end = next(
            end for start, end in zip(cumulative, cumulative[1:]) if start <= token_start < end
        )
        assert max(markers) < sequence_end, (name, markers, sequence_end)
        asset_id = f"grid_{grid[0]}x{grid[1]}x{grid[2]}_d{patch_dim}"
        if asset_id not in assets:
            values = pixel_values(grid, patch_dim)
            assets[asset_id] = {
                "shape": [patch_rows(grid), patch_dim],
                "generator": "((row*1009+col*1013)%997)/997.0",
                "row": "patch row in SpatialRowMajor order",
                "col": "patch element index 0..patch_dim-1",
                "synthetic_geometry_only": True,
                "sha256_f32le": pixel_sha256(values),
                "values_flat": values,
            }
        image = {
            "image_id": descriptor["image_id"],
            **geometry(grid, merge, patch_dim),
            "token_start": token_start,
            "token_span": [token_start, token_start + count],
            "marker_indices": markers,
            "marker_ids": [10] * count,
            "merge_source_rows": merge_source_rows(grid, merge),
            "pixel_asset": asset_id,
            "pixel_values_sha256": assets[asset_id]["sha256_f32le"],
        }
        if "duplicate_of" in descriptor:
            image["duplicate_of"] = descriptor["duplicate_of"]
        images.append(image)
    sequence_spans = [
        {"index": i, "start": start, "length": end - start}
        for i, (start, end) in enumerate(zip(cumulative, cumulative[1:]))
    ]
    return {
        "name": name,
        "synthetic_geometry_fixture": True,
        "cumulative_token_lengths": cumulative,
        "sequence_spans": sequence_spans,
        "ids": ids,
        "images": images,
        "expected": {
            "patch_dim": patch_dim,
            "patch_rows": [image["patch_rows"] for image in images],
            "token_rows": [image["token_rows"] for image in images],
            "marker_indices": [image["marker_indices"] for image in images],
        },
    }


def text_case(
    name: str,
    sequences: list[list[int]],
    vocab_size: int,
    duplicate_of: list[int | None] | None = None,
) -> dict[str, Any]:
    for sequence in sequences:
        assert all(0 <= token < vocab_size for token in sequence)
        assert 10 in sequence, "every text case carries image_token_index 10"
    cumulative = [0]
    for sequence in sequences:
        cumulative.append(cumulative[-1] + len(sequence))
    hashes = [sha256_bytes(json.dumps(seq, separators=(",", ":")).encode()) for seq in sequences]
    return {
        "name": name,
        "cumulative_token_lengths": cumulative,
        "sequences": sequences,
        "sequence_sha256": hashes,
        "duplicate_of": duplicate_of or [None] * len(sequences),
        "ids": [token for sequence in sequences for token in sequence],
        "vocab_size": vocab_size,
        "image_token_index_present": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=ROOT / "processor-cases.json")
    args = parser.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", None):
        raise SystemExit("run this fixture generator with CUDA_VISIBLE_DEVICES=''")

    payloads = {}
    provenance = {}
    for name in READ_FILES:
        data, meta = read_json(args.model / name)
        payloads[name] = data
        provenance[name] = meta

    config = payloads["config.json"]
    processor = payloads["processor_config.json"]
    tokenizer = payloads["tokenizer_config.json"]
    image_processor = processor["image_processor"]
    vision = config["vision_config"]
    text = config["text_config"]

    merge = int(processor["spatial_merge_size"])
    patch_size = int(processor["patch_size"])
    patch_dim = 3 * patch_size * patch_size
    assert patch_size == vision["patch_size"] == 14
    assert merge == 2
    assert patch_dim == 588

    text_special = {
        "bos_token_id": text.get("bos_token_id", config.get("bos_token_id")),
        "eos_token_id": text.get("eos_token_id", config.get("eos_token_id")),
        "pad_token_id": text.get("pad_token_id", config.get("pad_token_id")),
    }
    processor_view = {
        "processor_class": processor.get("processor_class"),
        "image_processor_type": image_processor.get("image_processor_type"),
        "patch_size": patch_size,
        "spatial_merge_size": merge,
        "image_mean": image_processor["image_mean"],
        "image_std": image_processor["image_std"],
        "longest_edge": image_processor["size"]["longest_edge"],
        "image_size": vision["image_size"],
        "image_token_index": config["image_token_index"],
        "vision_feature_layer": config["vision_feature_layer"],
        "projector_hidden_act": config["projector_hidden_act"],
        "multimodal_projector_bias": config["multimodal_projector_bias"],
        "image_token": processor.get("image_token"),
        "image_break_token": processor.get("image_break_token"),
        "image_end_token": processor.get("image_end_token"),
        "data_format": image_processor.get("data_format"),
        "do_convert_rgb": image_processor.get("do_convert_rgb"),
        "do_normalize": image_processor.get("do_normalize"),
        "do_rescale": image_processor.get("do_rescale"),
        "do_resize": image_processor.get("do_resize"),
        "resample": image_processor.get("resample"),
        "rescale_factor": image_processor.get("rescale_factor"),
        "num_channels": vision.get("num_channels"),
        "text_config": {
            "hidden_size": text["hidden_size"],
            "intermediate_size": text["intermediate_size"],
            "num_hidden_layers": text["num_hidden_layers"],
            "num_attention_heads": text["num_attention_heads"],
            "num_key_value_heads": text["num_key_value_heads"],
            "head_dim": text["head_dim"],
            "vocab_size": text["vocab_size"],
            "max_position_embeddings": text["max_position_embeddings"],
            "rms_norm_eps": text["rms_norm_eps"],
            "hidden_act": text["hidden_act"],
            "is_causal": text["is_causal"],
            "use_bidirectional_attention": text.get("use_bidirectional_attention"),
            "sliding_window": text["sliding_window"],
            "rope_theta": text.get("rope_theta", text["rope_parameters"].get("rope_theta")),
            "rope_parameters": text["rope_parameters"],
            "llama_4_scaling": text.get("llama_4_scaling"),
        },
        "tokenizer": {
            "tokenizer_class": tokenizer.get("tokenizer_class"),
            "backend": tokenizer.get("backend"),
            "model_max_length": tokenizer.get("model_max_length"),
            "special_token_strings": {
                "bos_token": tokenizer.get("bos_token"),
                "eos_token": tokenizer.get("eos_token"),
                "unk_token": tokenizer.get("unk_token"),
                "pad_token": tokenizer.get("pad_token"),
            },
            "special_token_ids_from_tokenizer_config": None,
            "special_token_ids_note": (
                "tokenizer_config.json stores token strings only; ids are not present "
                "there. tokenizer.json was deliberately not read."
            ),
            "text_config_special_token_ids": text_special,
            "extra_special_token_count": len(tokenizer.get("extra_special_tokens", [])),
            "extra_special_tokens": tokenizer.get("extra_special_tokens", []),
        },
    }

    assets: dict[str, Any] = {}
    image_cases = [
        image_case(
            "dynamic_grid_1x4x6",
            [0, 7],
            [{"image_id": "img_small", "grid_thw": [1, 4, 6], "token_start": 0}],
            merge,
            patch_dim,
            assets,
        ),
        image_case(
            "dynamic_grid_1x14x10",
            [0, 43],
            [{"image_id": "img_large", "grid_thw": [1, 14, 10], "token_start": 0}],
            merge,
            patch_dim,
            assets,
        ),
        image_case(
            "duplicate_same_image_twice",
            [0, 7, 14],
            [
                {"image_id": "img_duplicate_first", "grid_thw": [1, 4, 6], "token_start": 0},
                {
                    "image_id": "img_duplicate_second",
                    "grid_thw": [1, 4, 6],
                    "token_start": 7,
                    "duplicate_of": "img_duplicate_first",
                },
            ],
            merge,
            patch_dim,
            assets,
        ),
        image_case(
            "mixed_sizes_ragged_batch",
            [0, 7, 50],
            [
                {"image_id": "img_mixed_small", "grid_thw": [1, 4, 6], "token_start": 0},
                {"image_id": "img_mixed_large", "grid_thw": [1, 14, 10], "token_start": 7},
            ],
            merge,
            patch_dim,
            assets,
        ),
    ]

    vocab_size = int(text["vocab_size"])
    ragged_sequences = [
        [1, 10, 21, 22, 23, 24, 2],
        [10, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 2],
        [1, 200, 201, 10, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 2],
    ]
    duplicate_sequence = [1, 10, 21, 22, 23, 24, 2]
    text_cases = [
        text_case("ragged_cumulative_0_7_30_60", ragged_sequences, vocab_size),
        text_case(
            "duplicate_sequences",
            [duplicate_sequence, duplicate_sequence],
            vocab_size,
            duplicate_of=[None, 0],
        ),
    ]
    cumulative_lengths = [0]
    for sequence in ragged_sequences:
        cumulative_lengths.append(cumulative_lengths[-1] + len(sequence))
    assert cumulative_lengths == [0, 7, 30, 60]

    readout_cases = [
        {
            "name": "pooled_mean_indices",
            "source_text_case": "ragged_cumulative_0_7_30_60",
            "pooled_indices": [6, 29, 59],
            "pooling": "mean over each sequence span",
            "expected_rows": 3,
        },
        {
            "name": "raw_indices",
            "source_text_case": "ragged_cumulative_0_7_30_60",
            "raw_indices": [3, 15, 44],
            "expected_rows": 3,
        },
        {
            "name": "mixed_pooled_and_raw",
            "source_text_case": "ragged_cumulative_0_7_30_60",
            "pooled_indices": [6, 29, 59],
            "raw_indices": [3, 15, 44],
            "expected_rows": 6,
        },
        {
            "name": "classifier_label_case",
            "id2label": config.get("id2label", {}),
            "label2id": config.get("label2id", {}),
            "score_logits": [-0.75],
            "expected_label_id": 0,
            "expected_label": "LABEL_0",
        },
    ]

    error_cases = [
        {
            "name": "wrong_patch_dim",
            "expected_error_kind": "PatchDimension",
            "expected_error": "expected patch dimension 588, got 587",
            "grid_thw": [1, 4, 6],
            "merge_size": merge,
            "patch_dim": 587,
        },
        {
            "name": "grid_height_not_divisible_by_merge",
            "expected_error_kind": "MergeMismatch",
            "expected_error": "image grid [1, 5, 6] is not divisible by merge size 2",
            "grid_thw": [1, 5, 6],
            "merge_size": merge,
        },
        {
            "name": "token_span_crossing_sequence",
            "expected_error_kind": "SpanCrossesSequence",
            "expected_error": (
                "token span start=5 count=6 crosses sequence start=0 length=7"
            ),
            "grid_thw": [1, 4, 6],
            "merge_size": merge,
            "token_start": 5,
            "cumulative_token_lengths": [0, 7, 30],
        },
        {
            "name": "marker_id_mismatch",
            "expected_error_kind": "Mistral3MarkerMismatch",
            "expected_error": "Mistral3 image marker at 2 is 11, expected 10",
            "grid_thw": [1, 4, 6],
            "merge_size": merge,
            "token_start": 0,
            "bad_marker_index": 2,
            "bad_marker_value": 11,
        },
        {
            "name": "overlapping_image_spans",
            "expected_error_kind": "OverlappingSpans",
            "expected_error": "media spans overlap at 4 after ending at 6",
            "grid_thw": [1, 4, 6],
            "merge_size": merge,
            "image_token_starts": [0, 4],
        },
        {
            "name": "zero_grid",
            "expected_error_kind": "ZeroDimension",
            "expected_error": "image grid has a zero dimension",
            "grid_thw": [1, 0, 6],
            "merge_size": merge,
        },
    ]

    document = {
        "schema": "mistral3-processor-cases/v1",
        "generated_by": "fixtures/make_fixtures.py",
        "provenance": {
            "checkpoint_path": str(args.model),
            "files": provenance,
            "read_files": list(READ_FILES),
            "tensor_payload_read": False,
            "tokenizer_json_read": False,
            "network_used": False,
            "cuda_used": False,
            "torch_used": False,
        },
        "processor": processor_view,
        "pixel_assets": assets,
        "image_cases": image_cases,
        "text_cases": text_cases,
        "readout_cases": readout_cases,
        "error_cases": error_cases,
        "counts": {
            "image_cases": len(image_cases),
            "text_cases": len(text_cases),
            "readout_cases": len(readout_cases),
            "error_cases": len(error_cases),
            "pixel_assets": len(assets),
        },
        "labels": {
            "pixels": "deterministic synthetic geometry, not a real image decode",
            "image_patch_dim": patch_dim,
            "merge_source_order": "[merge_row * merge_col] row-major",
        },
    }
    args.output.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n")
    size = args.output.stat().st_size
    print(
        json.dumps(
            {
                "output": str(args.output),
                "bytes": size,
                **document["counts"],
                "image_case_names": [case["name"] for case in image_cases],
                "text_case_names": [case["name"] for case in text_cases],
                "readout_case_names": [case["name"] for case in readout_cases],
                "error_case_names": [case["name"] for case in error_cases],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import json, pathlib, hashlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)

def ids(n, offset=0):
    return [20 + ((i + offset) * 13) % 20000 for i in range(n)]

def batch(lengths, pooled, raw, *, offset=0, id_override=None, pos_override=None,
          cumulative_override=None, max_override=None):
    total = sum(lengths)
    input_ids = id_override if id_override is not None else ids(total, offset)
    positions = []
    for length in lengths:
        positions.extend(range(length))
    if pos_override is not None:
        positions = pos_override
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    if cumulative_override is not None:
        cumulative = cumulative_override
    return {
        "input_ids": input_ids,
        "position_ids": positions,
        "token_type_ids": [0] * len(input_ids),
        "cumulative_seq_lengths": cumulative,
        "pooled_indices": pooled,
        "raw_indices": raw,
        "max_length": max(lengths) if max_override is None else max_override,
    }

cases = []
def add(name, value, valid, expected_variant=None, special=None):
    path = OUT / f"{name}.json"
    data = json.dumps(value, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    path.write_bytes(data)
    cases.append({
        "name": name,
        "file": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "valid": valid,
        "expected_error_variant": expected_variant,
        "special": special,
        "token_count": len(value.get("input_ids", [])),
        "sequence_count": max(0, len(value.get("cumulative_seq_lengths", [])) - 1),
        "pooled_count": len(value.get("pooled_indices", [])),
        "raw_count": len(value.get("raw_indices", [])),
    })

add("t7_a", batch([7], [0], []), True)
add("t1", batch([1], [0], []), True)
add("t7_b", batch([7], [0], [], offset=101), True)
add("t8", batch([8], [0], []), True)
add("t9", batch([9], [0], []), True)
add("t7_no_selection", batch([7], [], []), True)
add("ragged_3_4", batch([3, 4], [0, 1], []), True)
add("raw_t7", batch([7], [], [0]), True)
add("mixed_ragged", batch([3, 4], [0], [1, 0]), True)
add("subset_ragged", batch([3, 4], [1], []), True)
add("duplicate_t7", batch([7], [0, 0], [0, 0]), True)
add("duplicate_pooled_t7", batch([7], [0, 0], []), True)
add("bounded_duplicate_ragged", batch([1, 6], [], [0, 0]), True)
add("rawoverride_t7", batch([7], [0], [0]), True)
add("duplicate_raw_t7", batch([7], [0], [0, 0]), True)
add("empty", batch([0], [0], []), False, "Inference", "empty")
add("invalid_token_id", batch([7], [0], [], id_override=[20, 21, 22, 300000, 24, 25, 26]), False, "Inference")
add("invalid_position", batch([7], [0], [], pos_override=[0, 1, 2, 3, 4, 5, 999999]), False, "Inference")
add("invalid_cumulative", batch([7], [0], [], cumulative_override=[0, 3, 6]), False, "Inference")
add("invalid_selection", batch([7], [0], [], ) | {"pooled_indices": [5]}, False, "Inference")
add("invalid_token_nosel", batch([7], [], [], id_override=[20, 21, 22, 300000, 24, 25, 26]), False, "Inference")
add("invalid_media", batch([7], [0], []), False, "Inference", "media")
add("invalid_media_nosel", batch([7], [], []), False, "Inference", "media")
add("invalid_compact_nosel", batch([7], [], []), False, "Inference", "compact")
add("invalid_compact", batch([7], [0], []), False, "Inference", "compact")

manifest = {
    "schema": "oxide-q35moe-unified-fixtures/v1",
    "batch_fields": ["input_ids", "position_ids", "token_type_ids", "cumulative_seq_lengths", "pooled_indices", "raw_indices", "max_length"],
    "ids_rule": "20 + (i*13) % 20000; t9 uses i=0..8 (20..124); t7_b offset +101 token indices",
    "cases": cases,
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps({"cases": len(cases), "out": str(OUT)}))

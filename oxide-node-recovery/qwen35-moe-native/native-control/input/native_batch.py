"""Pure-CPU contract for exact physical batches and native output layouts.

Nothing in this module imports Torch.  It owns the strict batch JSON contract,
physical-pooled-then-raw layout, canonical embedding-map semantics, guard
checks and the 19-argument ``tei_forward`` ABI description.
"""
import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path

MAX_BATCH_JSON = 32 << 20
BATCH_FIELDS = ("input_ids", "position_ids", "token_type_ids",
                "cumulative_seq_lengths", "pooled_indices", "raw_indices",
                "max_length")
U32_MAX = (1 << 32) - 1
GUARD_FLOATS = 16
GUARD_SENTINEL = struct.pack("<I", 0x7F800001) * GUARD_FLOATS
FORWARD_ABI = (
    ("void*", 8),
    ("constint64_t*", 8),
    ("constint64_t*", 8),
    ("constint64_t*", 8),
    ("constint32_t*", 8),
    ("int64_t", 8),
    ("int64_t", 8),
    ("int32_t", 4),
    ("constint64_t*", 8),
    ("size_t", 8),
    ("constint64_t*", 8),
    ("size_t", 8),
    ("float*", 8),
    ("size_t", 8),
    ("constint64_t*", 8),
    ("constTeiImage*", 8),
    ("size_t", 8),
    ("constTeiAudio*", 8),
    ("size_t", 8),
)


class BatchError(ValueError):
    """Raised for any malformed batch, layout or native-output contract."""


def unique_object(items):
    result = {}
    for name, value in items:
        if name in result:
            raise BatchError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def _u32(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise BatchError(f"{name} must be a JSON integer, got {value!r}")
    if value < 0 or value > U32_MAX:
        raise BatchError(f"{name} is outside u32: {value!r}")
    return value


def _u32_list(value, name):
    if not isinstance(value, list):
        raise BatchError(f"{name} must be a JSON array")
    return [_u32(item, f"{name}[{index}]") for index, item in enumerate(value)]


def validate_batch(raw):
    """Return an exact deep copy of the seven-field physical batch."""
    if not isinstance(raw, dict):
        raise BatchError("batch JSON must be an object")
    missing = set(BATCH_FIELDS) - set(raw)
    unknown = set(raw) - set(BATCH_FIELDS)
    if missing:
        raise BatchError(f"missing batch fields: {sorted(missing)}")
    if unknown:
        raise BatchError(f"unknown batch fields: {sorted(unknown)}")
    batch = {name: _u32_list(raw[name], name) for name in BATCH_FIELDS
             if name != "max_length"}
    batch["max_length"] = _u32(raw["max_length"], "max_length")
    ids = batch["input_ids"]
    if not (len(ids) == len(batch["position_ids"]) == len(batch["token_type_ids"])):
        raise BatchError("input_ids, position_ids and token_type_ids lengths differ")
    cumulative = batch["cumulative_seq_lengths"]
    if len(cumulative) < 2:
        raise BatchError("cumulative_seq_lengths must describe at least one sequence")
    if cumulative[0] != 0:
        raise BatchError("cumulative_seq_lengths must start at 0")
    if cumulative[-1] != len(ids):
        raise BatchError("cumulative_seq_lengths must end at the token count")
    if any(left >= right for left, right in zip(cumulative, cumulative[1:])):
        raise BatchError("cumulative_seq_lengths must be strictly increasing")
    lengths = [right - left for left, right in zip(cumulative, cumulative[1:])]
    if batch["max_length"] != max(lengths):
        raise BatchError("max_length must equal the actual maximum sequence length")
    count = len(lengths)
    for name in ("pooled_indices", "raw_indices"):
        for index in batch[name]:
            if index >= count:
                raise BatchError(f"{name} selection {index} is outside batch of {count}")
    return batch


def load_batch(path):
    """Read at most 32 MiB and return (exact-file sha256, validated batch)."""
    path = Path(path)
    if not path.is_file():
        raise BatchError(f"missing batch JSON: {path}")
    with path.open("rb") as handle:
        data = handle.read(MAX_BATCH_JSON + 1)
    if len(data) > MAX_BATCH_JSON:
        raise BatchError(f"batch JSON exceeds 32 MiB: {path}")
    try:
        raw = json.loads(data, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BatchError(f"invalid batch JSON: {exc}") from exc
    return hashlib.sha256(data).hexdigest(), validate_batch(raw)


def batch_facts(batch):
    cumulative = batch["cumulative_seq_lengths"]
    lengths = [right - left for left, right in zip(cumulative, cumulative[1:])]
    return {"sequence_count": len(lengths), "token_count": len(batch["input_ids"]),
            "sequence_lengths": lengths, "max_length": batch["max_length"],
            "pooled_count": len(batch["pooled_indices"]),
            "raw_count": len(batch["raw_indices"])}


def physical_layout(batch, width):
    """Pooled selections first, then raw token spans in request order."""
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise BatchError(f"invalid hidden width: {width!r}")
    cumulative = batch["cumulative_seq_lengths"]
    segments = []
    offset = 0
    for selection in batch["pooled_indices"]:
        segments.append({"kind": "pooled", "index": selection, "shape": [width],
                         "offset_floats": offset, "length_floats": width})
        offset += width
    raw_rows = 0
    for selection in batch["raw_indices"]:
        rows = cumulative[selection + 1] - cumulative[selection]
        length = rows * width
        segments.append({"kind": "raw", "index": selection,
                         "shape": [rows, width], "offset_floats": offset,
                         "length_floats": length})
        offset += length
        raw_rows += rows
    return {"pooled_count": len(batch["pooled_indices"]),
            "raw_count": len(batch["raw_indices"]),
            "raw_token_rows": raw_rows,
            "physical_float_count": offset,
            "physical_bytes": offset * 4,
            "segments": segments}


def _segment_values(raw, segment):
    return list(struct.unpack_from("<%df" % segment["length_floats"], raw,
                                   segment["offset_floats"] * 4))


def _segment_bits(raw, segment):
    start = segment["offset_floats"] * 4
    return raw[start:start + segment["length_floats"] * 4]


def canonical_outputs(batch, layout, flat_bytes):
    """Apply Rust Embeddings map semantics and verify duplicate bit identity."""
    raw = bytes(flat_bytes)
    if len(raw) != layout["physical_bytes"]:
        raise BatchError("native output bytes do not match the physical layout")
    seen = {}
    duplicate_indices = []
    for segment in layout["segments"]:
        key = (segment["kind"], segment["index"])
        bits = _segment_bits(raw, segment)
        if key in seen:
            if seen[key] != bits:
                raise BatchError(
                    f"duplicate {segment['kind']} selection {segment['index']} "
                    "is not bit-identical")
            duplicate_indices.append(segment["index"])
        else:
            seen[key] = bits
    canonical = {}
    overridden = []
    for segment in layout["segments"]:
        values = _segment_values(raw, segment)
        if not all(math.isfinite(value) for value in values):
            raise BatchError(f"non-finite {segment['kind']} output at index {segment['index']}")
        if (segment["kind"] == "raw" and segment["index"] in canonical
                and canonical[segment["index"]]["kind"] == "pooled"):
            overridden.append(segment["index"])
        canonical[segment["index"]] = {"index": segment["index"],
                                       "kind": segment["kind"],
                                       "shape": list(segment["shape"]),
                                       "values": values}
    outputs = [canonical[index] for index in sorted(canonical)]
    facts = {"entry_count": len(outputs),
             "float_count": sum(len(item["values"]) for item in outputs),
             "raw_overrides_pooled": sorted(set(overridden)),
             "duplicate_selection_indices": sorted(set(duplicate_indices))}
    return outputs, facts


def require_little_endian(byteorder=None):
    order = sys.byteorder if byteorder is None else byteorder
    if order != "little":
        raise BatchError("native F32 transport requires a little-endian host")
    return True


def guard_bytes():
    return GUARD_SENTINEL


def make_guarded_buffer(capacity_floats):
    if capacity_floats < 0:
        raise BatchError("negative output capacity")
    storage = bytearray(capacity_floats * 4 + len(GUARD_SENTINEL))
    storage[capacity_floats * 4:] = GUARD_SENTINEL
    return storage


def check_guards(storage, capacity_floats):
    expected = capacity_floats * 4 + len(GUARD_SENTINEL)
    if len(storage) != expected:
        raise BatchError("guarded buffer length is malformed")
    if bytes(storage[capacity_floats * 4:]) != GUARD_SENTINEL:
        raise BatchError("native guard floats changed after tei_forward")
    return True


def _normal_type(text):
    return re.sub(r"\s+", "", text)


def parse_forward_params(header_text):
    marker = "int32_t tei_forward("
    try:
        start = header_text.index(marker) + len(marker)
        end = header_text.index(");", start)
    except ValueError as exc:
        raise BatchError("bridge.h lacks a complete tei_forward declaration") from exc
    body = header_text[start:end]
    params = []
    for part in body.split(","):
        compact = part.strip()
        if not compact:
            continue
        if " " in compact:
            kind = compact.rsplit(None, 1)[0]
        else:
            kind = compact
        params.append(_normal_type(kind))
    return params


def check_forward_abi(header_text):
    params = parse_forward_params(header_text)
    expected = [kind for kind, _ in FORWARD_ABI]
    if len(params) != 19:
        raise BatchError(f"tei_forward must take 19 ABI arguments, got {len(params)}")
    if params != expected:
        raise BatchError(f"tei_forward ABI type mismatch: {params} != {expected}")
    widths = []
    for kind in params:
        if kind.endswith("*"):
            widths.append(8)
        elif "int64_t" in kind or "size_t" in kind:
            widths.append(8)
        elif "int32_t" in kind:
            widths.append(4)
        else:
            raise BatchError(f"unknown ABI type width: {kind}")
    expected_widths = [width for _, width in FORWARD_ABI]
    if widths != expected_widths:
        raise BatchError(f"tei_forward ABI widths mismatch: {widths} != {expected_widths}")
    return {"arg_count": len(params), "types": params, "widths": widths, "ok": True}

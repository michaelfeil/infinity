"""Metadata-only admission and bounded safetensors upload for native controls.

Five cached encoder/Qwen2 controls, two dense BF16 Qwen3 embedding controls,
the MiniLM-L6 BERT fixture and the TinyLlama BF16 fixture. No Torch import or
CUDA operation occurs in this module. NativeConfig follows libtorch/src/lib.rs;
the original JSON remains the source of C++ model options, and every recipe
field is checked against that JSON.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import mmap
from pathlib import Path
import re
import struct

MAX_JSON = 16 << 20
FLOAT_BYTES = {"F32": 4, "F16": 2, "BF16": 2}
DTYPE_BYTES = dict(FLOAT_BYTES, I64=8, I32=4, U8=1, BOOL=1)
CASES = {
    "distilbert": ("distilbert", 768, 12, 6, 3072, 30522, 512, "mean", "", 100, 1e-12),
    "roberta": ("roberta", 768, 12, 6, 3072, 50265, 514, "mean", "", 103, 1e-5),
    "camembert": ("camembert", 768, 12, 12, 3072, 32005, 514, "mean", "roberta.", 204, 1e-5),
    "xlmr": ("xlm-roberta", 1024, 16, 24, 4096, 250002, 8194, "cls", "", 391, 1e-5),
    "gte": ("qwen2", 1536, 12, 28, 8960, 151646, 131072, "last-token", "model.", 339, 1e-6),
    "qwen3-0.6b": ("qwen3", 1024, 16, 28, 3072, 151669, 32768, "last-token", "", 310, 1e-6),
    # Published 8B config.json uses vocab_size 151665 (the prompt's 151669 is
    # not the staged actual value); max_position_embeddings is 40960.
    "qwen3-8b": ("qwen3", 4096, 32, 36, 12288, 151665, 40960, "last-token", "", 398, 1e-6),
    "minilm": ("bert", 384, 12, 6, 1536, 30522, 512, "mean", "", 103, 1e-12),
    "tiny": ("llama", 2048, 32, 22, 5632, 32000, 2048, "last-token", "model.", 201, 1e-5),
}
MINILM = {"type_vocab_size": 2, "layer_norm_eps": 1e-12, "hidden_act": "gelu",
          "position_embedding_type": "absolute", "architectures": ["BertModel"]}
TINY = {"num_key_value_heads": 4, "head_dim": 64, "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5, "attention_bias": False,
        "tie_word_embeddings": False, "torch_dtype": "bfloat16",
        "rope_scaling": None, "architectures": ["LlamaForCausalLM"]}
# Exact dense Qwen3 facts that must agree with the model's own config.json.
QWEN3 = {
    "qwen3-0.6b": {"qwidth": 2048, "kv_width": 1024, "head_dim": 128,
                   "rope_theta": 1_000_000.0, "tie_word_embeddings": True},
    "qwen3-8b": {"qwidth": 4096, "kv_width": 1024, "head_dim": 128,
                 "rope_theta": 1_000_000.0, "tie_word_embeddings": False},
}


def unique_object(items):
    result = {}
    for name, value in items:
        if name in result:
            raise ValueError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def read_json(path):
    with Path(path).open("rb") as handle:
        data = handle.read(MAX_JSON + 1)
    if len(data) > MAX_JSON:
        raise ValueError(f"JSON exceeds 16 MiB: {path}")
    return json.loads(data, object_pairs_hook=unique_object)


def integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > (1 << 63) - 1:
        raise ValueError(f"invalid nonnegative int64 {name}: {value!r}")
    return value


def _geometry(raw):
    aliases = {"hidden_size": "dim", "num_attention_heads": "n_heads",
               "num_hidden_layers": "n_layers", "intermediate_size": "hidden_dim"}
    native = {key: integer(raw.get(key, raw.get(alias)), key) for key, alias in aliases.items()}
    for key in ("vocab_size", "max_position_embeddings"):
        native[key] = integer(raw.get(key), key)
    family = raw.get("model_type")
    signature = (family, *(native[key] for key in ["hidden_size", "num_attention_heads",
                 "num_hidden_layers", "intermediate_size", "vocab_size", "max_position_embeddings"]))
    matches = [name for name, spec in CASES.items() if signature == spec[:7]]
    if len(matches) != 1:
        raise ValueError(f"configuration is outside the cached control geometries: {signature}")
    return native, family, matches[0]


def case_for_config(raw):
    """Return the admitted control case without choosing an output pool."""
    return _geometry(raw)[2]


def normalize_config(raw, pool):
    native, family, case = _geometry(raw)
    spec = CASES[case]
    if pool != spec[7]:
        raise ValueError(f"{case} control requires {spec[7]} pooling, got {pool}")
    if raw.get("is_decoder", False) or raw.get("add_cross_attention", False):
        raise ValueError("encoder decoder/cross-attention modifiers are outside these controls")
    if raw.get("position_embedding_type", "absolute") != "absolute":
        raise ValueError("relative encoder positions are outside these controls")
    if raw.get("quantization_config") is not None:
        raise ValueError("quantized weights are outside the BF16 controls")
    activation = raw.get("hidden_act", "gelu")
    if activation != ("silu" if case in ("gte", "qwen3-0.6b", "qwen3-8b", "tiny")
                      else "gelu"):
        raise ValueError(f"unsupported native activation for {case}: {activation}")
    if case == "distilbert" and raw.get("activation", "gelu") != "gelu":
        raise ValueError("DistilBERT must retain its checkpoint GELU activation")
    if case == "minilm":
        actual = {"type_vocab_size": raw.get("type_vocab_size"),
                  "layer_norm_eps": raw.get("layer_norm_eps"),
                  "position_embedding_type": raw.get("position_embedding_type", "absolute"),
                  "architectures": raw.get("architectures")}
        expected = {"type_vocab_size": MINILM["type_vocab_size"],
                    "layer_norm_eps": MINILM["layer_norm_eps"],
                    "position_embedding_type": MINILM["position_embedding_type"],
                    "architectures": MINILM["architectures"]}
        if actual != expected:
            raise ValueError(f"minilm must match its public BERT fixture: {actual!r} != {expected!r}")
    if case == "gte":
        if (raw.get("is_causal") is not False or raw.get("num_key_value_heads") != 2
                or raw.get("head_dim", 128) != 128 or raw.get("rope_theta") != 1_000_000.0
                or raw.get("rms_norm_eps") != 1e-6 or raw.get("use_sliding_window") is not False
                or raw.get("rope_scaling") is not None or raw.get("rope_parameters") is not None):
            raise ValueError("GTE requires its noncausal Q12/KV2 d128 unscaled full-NeoX recipe")
    recipe = None
    if case in QWEN3:
        actual = {
            "num_key_value_heads": raw.get("num_key_value_heads"),
            "head_dim": raw.get("head_dim"),
            "rope_theta": raw.get("rope_theta"),
            "rms_norm_eps": raw.get("rms_norm_eps"),
            "attention_bias": raw.get("attention_bias"),
            "torch_dtype": raw.get("torch_dtype", raw.get("dtype")),
            "tie_word_embeddings": raw.get("tie_word_embeddings"),
            "use_sliding_window": raw.get("use_sliding_window", False),
            "sliding_window": raw.get("sliding_window"),
            "rope_scaling": raw.get("rope_scaling"),
            "rope_parameters": raw.get("rope_parameters"),
            "max_window_layers": raw.get("max_window_layers"),
            "attention_dropout": raw.get("attention_dropout", 0.0),
            "architectures": raw.get("architectures"),
        }
        expected = {"num_key_value_heads": 8, "head_dim": 128,
                    "rope_theta": QWEN3[case]["rope_theta"], "rms_norm_eps": 1e-6,
                    "attention_bias": False, "torch_dtype": "bfloat16",
                    "tie_word_embeddings": QWEN3[case]["tie_word_embeddings"],
                    "use_sliding_window": False, "sliding_window": None,
                    "rope_scaling": None, "rope_parameters": None,
                    "max_window_layers": spec[3], "attention_dropout": 0.0,
                    "architectures": ["Qwen3ForCausalLM"]}
        if actual != expected or raw.get("layer_norm_eps") is not None:
            raise ValueError(f"{case} must match its actual dense BF16 config JSON: "
                             f"{actual!r} != {expected!r}")
        recipe = dict(QWEN3[case], head_dim=actual["head_dim"],
                      qwidth=QWEN3[case]["qwidth"],
                      norm_epsilon=actual["rms_norm_eps"],
                      source_dtype="BF16")
    if case == "tiny":
        head_dim = raw.get("head_dim", native["hidden_size"] // native["num_attention_heads"])
        actual = {"num_key_value_heads": raw.get("num_key_value_heads"),
                  "head_dim": head_dim, "rope_theta": raw.get("rope_theta"),
                  "rms_norm_eps": raw.get("rms_norm_eps"),
                  "attention_bias": raw.get("attention_bias"),
                  "torch_dtype": raw.get("torch_dtype", raw.get("dtype")),
                  "tie_word_embeddings": raw.get("tie_word_embeddings"),
                  "rope_scaling": raw.get("rope_scaling"),
                  "architectures": raw.get("architectures")}
        expected = dict(TINY, head_dim=TINY["head_dim"])
        if actual != expected or raw.get("layer_norm_eps") is not None:
            raise ValueError(f"tiny must match its public Llama BF16 fixture: "
                             f"{actual!r} != {expected!r}")
        recipe = {"head_dim": head_dim,
                  "qwidth": native["num_attention_heads"] * head_dim,
                  "kv_width": TINY["num_key_value_heads"] * head_dim,
                  "rope_theta": TINY["rope_theta"],
                  "rms_norm_eps": TINY["rms_norm_eps"],
                  "tie_word_embeddings": TINY["tie_word_embeddings"],
                  "norm_epsilon": TINY["rms_norm_eps"], "source_dtype": "BF16"}
    if case == "minilm":
        recipe = {"norm_epsilon": MINILM["layer_norm_eps"], "source_dtype": "F32",
                  "type_vocab_size": MINILM["type_vocab_size"]}
    native.update(model_type=family, type_vocab_size=integer(raw.get("type_vocab_size", 1), "type_vocab_size"),
                  layer_norm_eps=raw.get("layer_norm_eps", 1e-5), hidden_act=activation)
    if case == "minilm":
        if native["type_vocab_size"] != 2 or native["layer_norm_eps"] != 1e-12:
            raise ValueError("minilm requires type_vocab_size 2 and epsilon 1e-12")
    elif native["type_vocab_size"] != 1 or native["layer_norm_eps"] != 1e-5:
        raise ValueError("unexpected NativeConfig type vocabulary or epsilon for this cached case")
    profile = {"case": case, "pool": pool, "backbone_root": spec[8],
               "floating_tensors": spec[9], "effective_norm_epsilon": spec[10],
               "position_start": 0,
               "position_policy": "explicit zero-based positions, identical to frozen oxide probe; not router HTTP offset",
               "http_position_start": 2 if case in ("roberta", "camembert", "xlmr") else 0}
    if recipe is not None:
        profile.update(recipe)
    return native, profile


def required_shapes(config, profile):
    h, i, v, p, layers = (config[key] for key in ["hidden_size", "intermediate_size", "vocab_size",
                                                "max_position_embeddings", "num_hidden_layers"])
    root = profile["backbone_root"]
    result = {}
    def tensor(name, shape):
        result[root + name] = tuple(shape)
    def linear(name, output, input_, bias=True):
        tensor(name + ".weight", (output, input_))
        if bias:
            tensor(name + ".bias", (output,))
    def norm(name, bias=True):
        tensor(name + ".weight", (h,))
        if bias:
            tensor(name + ".bias", (h,))
    if profile["case"] in QWEN3:
        qwidth, kv_width = profile["qwidth"], profile["kv_width"]
        head_dim = profile["head_dim"]
        tensor("embed_tokens.weight", (v, h))
        norm("norm", False)
        for layer in range(layers):
            base = f"layers.{layer}."
            norm(base + "input_layernorm", False)
            norm(base + "post_attention_layernorm", False)
            linear(base + "self_attn.q_proj", qwidth, h, False)
            linear(base + "self_attn.k_proj", kv_width, h, False)
            linear(base + "self_attn.v_proj", kv_width, h, False)
            linear(base + "self_attn.o_proj", h, qwidth, False)
            tensor(base + "self_attn.q_norm.weight", (head_dim,))
            tensor(base + "self_attn.k_norm.weight", (head_dim,))
            linear(base + "mlp.gate_proj", i, h, False)
            linear(base + "mlp.up_proj", i, h, False)
            linear(base + "mlp.down_proj", h, i, False)
    elif profile["case"] == "tiny":
        qwidth, kv_width = profile["qwidth"], profile["kv_width"]
        tensor("embed_tokens.weight", (v, h))
        norm("norm", False)
        for layer in range(layers):
            base = f"layers.{layer}."
            norm(base + "input_layernorm", False)
            norm(base + "post_attention_layernorm", False)
            linear(base + "self_attn.q_proj", qwidth, h, False)
            linear(base + "self_attn.k_proj", kv_width, h, False)
            linear(base + "self_attn.v_proj", kv_width, h, False)
            linear(base + "self_attn.o_proj", h, qwidth, False)
            linear(base + "mlp.gate_proj", i, h, False)
            linear(base + "mlp.up_proj", i, h, False)
            linear(base + "mlp.down_proj", h, i, False)
    elif profile["case"] == "gte":
        tensor("embed_tokens.weight", (v, h))
        norm("norm", False)
        for layer in range(layers):
            base = f"layers.{layer}."
            norm(base + "input_layernorm", False)
            norm(base + "post_attention_layernorm", False)
            linear(base + "self_attn.q_proj", h, h)
            linear(base + "self_attn.k_proj", 256, h)
            linear(base + "self_attn.v_proj", 256, h)
            linear(base + "self_attn.o_proj", h, h, False)
            linear(base + "mlp.gate_proj", i, h, False)
            linear(base + "mlp.up_proj", i, h, False)
            linear(base + "mlp.down_proj", h, i, False)
    else:
        tensor("embeddings.word_embeddings.weight", (v, h))
        tensor("embeddings.position_embeddings.weight", (p, h))
        norm("embeddings.LayerNorm")
        if profile["case"] != "distilbert":
            tensor("embeddings.token_type_embeddings.weight",
                   (config.get("type_vocab_size", 1), h))
        if profile["case"] == "minilm":
            linear("pooler.dense", h, h)
        for layer in range(layers):
            if profile["case"] == "distilbert":
                base = f"transformer.layer.{layer}."
                for name in ("q_lin", "k_lin", "v_lin", "out_lin"):
                    linear(base + "attention." + name, h, h)
                norm(base + "sa_layer_norm")
                linear(base + "ffn.lin1", i, h)
                linear(base + "ffn.lin2", h, i)
                norm(base + "output_layer_norm")
            else:
                base = f"encoder.layer.{layer}."
                for name in ("query", "key", "value"):
                    linear(base + "attention.self." + name, h, h)
                linear(base + "attention.output.dense", h, h)
                norm(base + "attention.output.LayerNorm")
                linear(base + "intermediate.dense", i, h)
                linear(base + "output.dense", h, i)
                norm(base + "output.LayerNorm")
    return result


def check_required_tensors(tensors, required, allowed=FLOAT_BYTES):
    """Pure metadata check used by validate_shapes and CPU unit tests."""
    for name, shape in required.items():
        info = tensors.get(name)
        if info is None or info.shape != shape or info.dtype not in allowed:
            raise ValueError(f"missing/wrong native backbone tensor {name}: expected {shape}, got {info}")


def safe_checkpoint_file(root, filename):
    if not isinstance(filename, str) or not filename or filename in (".", "..") or "/" in filename or "\\" in filename:
        raise ValueError(f"checkpoint filenames must be flat: {filename!r}")
    path = root / filename
    resolved = path.resolve(strict=True)
    inside = resolved.parent == root
    # Standard HF snapshots may point to this same repository's flat blob dir.
    snapshot = root.parent.name == "snapshots"
    blob = (snapshot and resolved.parent == root.parent.parent / "blobs"
            and re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", resolved.name))
    if not (inside or blob) or not resolved.is_file():
        raise ValueError(f"checkpoint path escapes model repository: {filename}")
    return path


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple
    start: int
    end: int


class Checkpoint:
    def __init__(self, model):
        self.root = Path(model).resolve(strict=True)
        self.index_path = self.root / "model.safetensors.index.json"
        if (self.root / "model.safetensors").exists():
            self.index = None
            names = ["model.safetensors"]
        elif self.index_path.is_file():
            self.index = read_json(self.index_path).get("weight_map")
            if (not isinstance(self.index, dict) or not self.index
                    or not all(isinstance(k, str) and k for k in self.index)
                    or not all(isinstance(v, str) and v for v in self.index.values())):
                raise ValueError("safetensors index must have a nonempty weight_map")
            names = sorted(set(self.index.values()))
        else:
            raise ValueError("expected single-file or indexed safetensors checkpoint")
        self.files = []
        self.tensors = {}
        for name in names:
            path = safe_checkpoint_file(self.root, name)
            with path.open("rb") as handle:
                stat = handle.fileno()
                import os
                stat = os.fstat(stat)
                prefix = handle.read(8)
                if len(prefix) != 8:
                    raise ValueError("truncated safetensors length")
                length = struct.unpack("<Q", prefix)[0]
                if length > MAX_JSON or length > stat.st_size - 8:
                    raise ValueError("invalid safetensors header length")
                data = handle.read(length)
                header = json.loads(data, object_pairs_hook=unique_object)
            if not isinstance(header, dict):
                raise ValueError("safetensors header must be an object")
            file_tensors = []
            for key, value in header.items():
                if key == "__metadata__":
                    continue
                if key in self.tensors or not isinstance(value, dict):
                    raise ValueError(f"duplicate or malformed tensor: {key}")
                dtype = value.get("dtype")
                shape = value.get("shape")
                offsets = value.get("data_offsets")
                if dtype not in DTYPE_BYTES or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
                    raise ValueError(f"invalid safetensors metadata: {key}")
                shape = tuple(integer(dimension, key) for dimension in shape)
                start, end = (integer(offset, key) for offset in offsets)
                if start > end or end > stat.st_size - 8 - length or end - start != math.prod(shape) * DTYPE_BYTES[dtype]:
                    raise ValueError(f"invalid tensor byte range: {key}")
                if self.index is not None and self.index.get(key) != name:
                    raise ValueError(f"tensor {key} is not owned by indexed shard {name}")
                info = TensorInfo(key, dtype, shape, start, end)
                self.tensors[key] = info
                file_tensors.append(info)
            cursor = 0
            for info in sorted(file_tensors, key=lambda item: (item.start, item.end)):
                if info.start != cursor:
                    raise ValueError(f"overlapping or missing safetensors payload at {info.name}")
                cursor = info.end
            if cursor != stat.st_size - 8 - length:
                raise ValueError(f"unused or truncated safetensors payload in {name}")
            self.files.append({"path": path, "offset": 8 + length, "tensors": file_tensors,
                               "size": stat.st_size, "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns,
                               "header_sha256": hashlib.sha256(data).hexdigest()})
        if self.index is not None and set(self.index) != set(self.tensors):
            raise ValueError("index references missing tensors")

    def validate_shapes(self, config, profile):
        required = required_shapes(config, profile)
        source_dtype = profile.get("source_dtype", "F32")
        allowed = {"BF16"} if source_dtype == "BF16" else FLOAT_BYTES
        check_required_tensors(self.tensors, required, allowed)
        optional = {}
        if profile["case"] == "tiny":
            if any(name.endswith(("self_attn.q_proj.bias", "self_attn.k_proj.bias",
                                  "self_attn.v_proj.bias")) for name in self.tensors):
                raise ValueError("tiny llama fixture must not carry QKV bias tensors")
            lm = self.tensors.get("lm_head.weight") or self.tensors.get("model.lm_head.weight")
            if lm is not None:
                if lm.shape != (config["vocab_size"], config["hidden_size"]) or lm.dtype != "BF16":
                    raise ValueError(f"unexpected tiny lm_head shape/dtype: {lm}")
                optional["lm_head"] = lm.name
        if profile["case"] == "minilm":
            position_ids = self.tensors.get("position_ids")
            if position_ids is not None and position_ids.dtype != "I64":
                raise ValueError("minilm position_ids fixture must remain skipped I64")
        floating = [info for info in self.tensors.values() if not skipped(info.name)]
        if len(floating) != profile["floating_tensors"] or any(info.dtype != source_dtype for info in floating):
            raise ValueError(f"{profile['case']} requires its verified count of {source_dtype} source tensors")
        return {"required_backbone_tensors": len(required), "floating_tensors": len(floating),
                "optional_tensors": optional,
                "source_dtypes": {source_dtype: len(floating)},
                "source_dtype": source_dtype,
                "skipped": sorted(name for name in self.tensors if skipped(name)),
                "files": [{"name": f["path"].name, "bytes": f["size"], "header_sha256": f["header_sha256"]} for f in self.files],
                "streaming_policy": "one shard mmap and one typed source tensor at a time; synchronous tei_weight copies then converts to engine BF16",
                "payload_hashes": "not computed; header hashes and source-file metadata recorded",
                "largest_source_tensor_bytes": max(info.end-info.start for info in floating)}

    def upload(self, torch, callback):
        import os
        types = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
        for file in self.files:
            with file["path"].open("rb") as handle:
                stat = os.fstat(handle.fileno())
                if (stat.st_size, stat.st_ino, stat.st_mtime_ns) != (file["size"], file["inode"], file["mtime_ns"]):
                    raise ValueError("checkpoint changed after metadata validation")
                with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_COPY) as storage:
                    for info in file["tensors"]:
                        if skipped(info.name):
                            continue
                        tensor = torch.frombuffer(storage, dtype=types[info.dtype], count=math.prod(info.shape),
                                                  offset=file["offset"] + info.start).reshape(info.shape)
                        try:
                            callback(info.name, tensor)
                        finally:
                            del tensor


def skipped(name):
    return name.endswith("position_ids") or name.endswith("token_type_ids")

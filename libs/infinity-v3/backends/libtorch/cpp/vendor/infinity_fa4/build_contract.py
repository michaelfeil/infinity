# SPDX-License-Identifier: MIT
"""Pure stdlib contract shared by the FA4 source build and CPU guards.

This module never imports torch, CuTe, flash_attn, or tvm_ffi.
"""
import hashlib
import json
import pathlib
import re

PROFILE_RE = r"FA4_PROFILE\((\w+),\s*(\d+),\s*(\d+),\s*(\d+),\s*([01]),\s*([01])\)"

CANONICAL26_PROFILES = (
    ("bert", 16, 16, 64, False, False),
    ("modern_local", 12, 12, 64, False, True),
    ("qwen", 32, 8, 128, True, False),
    ("voyage", 16, 8, 128, False, False),
    ("gqa6", 12, 2, 128, False, False),
    ("gqa8", 32, 4, 128, False, False),
    ("qwen35", 16, 4, 256, False, False),
    ("mha32", 12, 12, 32, False, False),
    ("media72", 16, 16, 72, False, False),
    ("mha128", 6, 6, 128, False, False),
    ("local128", 6, 6, 128, False, True),
    ("causal_gqa2", 16, 8, 128, True, False),
    ("gqa3", 24, 8, 128, False, False),
    ("gqa4", 32, 8, 128, False, False),
    ("gqa7", 28, 4, 128, False, False),
    ("causal_local128", 32, 8, 128, True, True),
    ("wide_gqa6", 24, 4, 256, False, False),
    ("causal_wide6", 24, 4, 256, True, False),
    ("causal_wide4", 16, 4, 256, True, False),
    ("causal_gqa8_d64", 32, 4, 64, True, False),
    ("mistral_local4096", 32, 8, 128, True, True),
    ("gemma3_local512", 4, 1, 256, True, True),
    ("causal_gqa8", 32, 4, 128, True, False),
    ("embedding_gemma2_local256", 4, 2, 256, False, True),
    ("gemma4_local256_causal", 8, 1, 256, True, True),
    ("gemma4_local256_noncausal", 8, 1, 256, False, True),
)
# Append-only minimal tail after the frozen canonical26 prefix. Only new
# ratios/mask families are appended: mode26 Qwen3.5-MoE group8/d256 causal,
# mode27/28 Qwen3 dense H5120 Q40/KV8 group5 d128 causal/noncausal, and mode29
# Mistral noncausal W4096 asymmetric local [4096,-1]. Absolute Q/KV heads are
# runtime tensor dimensions; Q64/Q8/Q12 duplicates are intentionally absent.
APPENDED_PROFILES = (
    ("qwen35_moe_gqa8", 16, 2, 256, True, False),
    ("qwen3_q40kv8_d128_causal", 40, 8, 128, True, False),
    ("qwen3_q40kv8_d128_noncausal", 40, 8, 128, False, False),
    ("mistral_local4096_noncausal", 32, 8, 128, False, True),
    ("qwen35_moe_gqa8_noncausal", 16, 2, 256, False, False),
)
# Canonical38 append: dense-gap decoders (31..33), unique Gemma3 global/local1024
# shapes (34..36), and Gemma4 E4B local1024 Q8/KV2 (37). Rune 26B Q16/KV8 d256
# local1024 deduplicates onto mode 35 because FA4 matches ratio + width + causal
# + window, not absolute head counts.
EXTENDED_PROFILES = (
    ("qwen2_q14kv2_d64_causal", 14, 2, 64, True, False),
    ("qwen2_q12kv2_d128_causal", 12, 2, 128, True, False),
    ("llama_q32kv8_d64_noncausal", 32, 8, 64, False, False),
    ("gemma3_global_d256_gqa2", 16, 8, 256, True, False),
    ("gemma3_local1024_d256_gqa2", 16, 8, 256, True, True),
    ("gemma3_local1024_d128_gqa2", 32, 16, 128, True, True),
    ("gemma4_e4b_local1024_d256_gqa4", 8, 2, 256, True, True),
)
QWEN35_MOE_PROFILE = APPENDED_PROFILES[0]
CANONICAL27_PROFILES = CANONICAL26_PROFILES + APPENDED_PROFILES[:1]
CANONICAL28_PROFILES = CANONICAL26_PROFILES + APPENDED_PROFILES[:2]
CANONICAL29_PROFILES = CANONICAL26_PROFILES + APPENDED_PROFILES[:3]
CANONICAL30_PROFILES = CANONICAL26_PROFILES + APPENDED_PROFILES[:4]
CANONICAL31_PROFILES = CANONICAL26_PROFILES + APPENDED_PROFILES
CANONICAL38_PROFILES = CANONICAL31_PROFILES + EXTENDED_PROFILES
CANONICAL19_PROFILES = CANONICAL26_PROFILES[:19]
CANONICAL20_PROFILES = CANONICAL26_PROFILES[:20]
CANONICAL21_PROFILES = CANONICAL26_PROFILES[:21]
CANONICAL22_PROFILES = CANONICAL26_PROFILES[:22]
CANONICAL23_PROFILES = CANONICAL26_PROFILES[:23]
CANONICAL24_PROFILES = CANONICAL26_PROFILES[:24]
CANONICAL25_PROFILES = CANONICAL26_PROFILES[:25]
CANONICAL_PROFILE_SETS = {
    19: CANONICAL19_PROFILES,
    20: CANONICAL20_PROFILES,
    21: CANONICAL21_PROFILES,
    22: CANONICAL22_PROFILES,
    23: CANONICAL23_PROFILES,
    24: CANONICAL24_PROFILES,
    25: CANONICAL25_PROFILES,
    26: CANONICAL26_PROFILES,
    27: CANONICAL27_PROFILES,
    28: CANONICAL28_PROFILES,
    29: CANONICAL29_PROFILES,
    30: CANONICAL30_PROFILES,
    31: CANONICAL31_PROFILES,
    38: CANONICAL38_PROFILES,
}
CANONICAL38_PROFILE_SEQUENCE_SHA256 = (
    "5308427d3c2f149dba100ef7235843df3693b11dd125ccb78288616bd7dcd99d"
)
CANONICAL26_PROFILE_SEQUENCE_SHA256 = (
    "3b214841c169c3bcaad0574ee9df917c9fc8a05be1e64602cf5b91d99a67ff1e"
)
CANONICAL_PREFIX_SEQUENCE_SHA256 = {
    19: "064e20f61405380889e1d88511d3f52d01caf654a2df983aa407f4bc5e737ea3",
    20: "827958cc0e82a0bccf0cf2ac96d25492a9fb40e7e36c54843618eaaf26807ebd",
    21: "b7d9e06b0fa290812b4e0018d8ff919af29762a47dd68ea525799a86b81ec0fa",
    22: "7ca9a15773f4aae33b55bb17619afa65edb9bf2c14b25d7e89338c7d8907e490",
    23: "268db2182d75fcda41f7d6e52b35dc967c1e9652ac5ad86e543ceed922fe7f80",
    24: "99c39dc70c2cd22532df08d4edbb6a3f62cfd10a910c1da66321cbb434a233b6",
    25: "8443cd2fefe5c627da27c305442f5aff1a1d7cdfb1ffc1cbbd19149f6cde4245",
    26: CANONICAL26_PROFILE_SEQUENCE_SHA256,
    27: "7cb60381f0383fdcdd8fd4da8b0dbe0f31252e2236de40aa78ebd54987376114",
    28: "e8cf5c714796adedf2562b4d0cd2070af237a4d5b2df39a869b2ca9607bc5e7e",
    29: "8d08ad78acc3a31e3b8714503c0c399fa2934f2245534ef2cc59765a017d6a79",
    30: "4647be9d34a82927cdaf37387f5314372c7cac9213f2d52a317a9f2ed16a2d26",
    31: "187defaacc9f9b9b326b392fbeda675545a73660f44f69e21743743f782a87f1",
    38: CANONICAL38_PROFILE_SEQUENCE_SHA256,
}
PROFILE_SEQUENCE_SHA256 = CANONICAL_PREFIX_SEQUENCE_SHA256[38]
CANONICAL26_REGISTRY = {
    "namespace": "canonical26",
    "mode_count": 26,
    "mode_mask": "0x3ffffff",
    "profile_sequence_sha256": CANONICAL26_PROFILE_SEQUENCE_SHA256,
}
CANONICAL27_REGISTRY = {
    "namespace": "canonical27",
    "mode_count": 27,
    "mode_mask": "0x7ffffff",
    "profile_sequence_sha256": CANONICAL_PREFIX_SEQUENCE_SHA256[27],
}
CANONICAL30_REGISTRY = {
    "namespace": "canonical30",
    "mode_count": 30,
    "mode_mask": "0x3fffffff",
    "profile_sequence_sha256": CANONICAL_PREFIX_SEQUENCE_SHA256[30],
}
CANONICAL31_REGISTRY = {
    "namespace": "canonical31",
    "mode_count": 31,
    "mode_mask": "0x7fffffff",
    "profile_sequence_sha256": CANONICAL_PREFIX_SEQUENCE_SHA256[31],
}
CANONICAL38_REGISTRY = {
    "namespace": "canonical38",
    "mode_count": 38,
    "mode_mask": f"{(1 << 38) - 1:#x}",
    "profile_sequence_sha256": PROFILE_SEQUENCE_SHA256,
}
CANONICAL_PREFIX_NAMESPACES = {
    19: "legacy19", 20: "published20", 21: "private23", 22: "private23",
    23: "private23", 24: "registry24", 25: "canonical26", 26: "canonical26",
    27: "canonical27", 28: "canonical28", 29: "canonical29", 30: "canonical30",
    31: "canonical31", 38: "canonical38",
}
CANONICAL_PREFIX_REGISTRIES = {
    count: {
        "namespace": CANONICAL_PREFIX_NAMESPACES[count],
        "mode_count": count,
        "mode_mask": f"{(1 << count) - 1:#x}",
        "profile_sequence_sha256": fingerprint,
    }
    for count, fingerprint in CANONICAL_PREFIX_SEQUENCE_SHA256.items()
}
CANONICAL_PROFILES = CANONICAL38_PROFILES
CANONICAL_REGISTRY = CANONICAL38_REGISTRY
HELPER_FAMILIES = (
    ("deberta", ("fa4_deberta_fp16", "fa4_deberta_bf16")),
    ("alibi", ("fa4_alibi_model_bias_fp16", "fa4_alibi_model_bias_bf16")),
    ("jinacode", (
        "fa4_jinacode_flash_alibi_fp16",
        "fa4_jinacode_flash_alibi_bf16",
    )),
    ("deberta_exact_mul", (
        "fa4_deberta_exact_mul_both_bf16",
        "fa4_deberta_exact_mul_single_bf16",
    )),
)
HELPER_SYMBOLS = dict(HELPER_FAMILIES)
LINK_HELPER_ORDER = ("deberta", "deberta_exact_mul", "alibi", "jinacode")
DTYPES = ("fp16", "bf16")
WINDOW_CONTRACT = "compiled_equals_canonical_runtime"
WINDOW_CONTRACT_MODES = (23, 24, 25)
CANONICAL26_APPENDED = {
    20: ("mistral_local4096", (4095, 0)),
    21: ("gemma3_local512", (511, 0)),
    22: ("causal_gqa8", (None, None)),
    23: ("embedding_gemma2_local256", (512, 512)),
    24: ("gemma4_local256_causal", (511, 0)),
    25: ("gemma4_local256_noncausal", (256, 256)),
}
CANONICAL30_APPENDED = {
    **CANONICAL26_APPENDED,
    26: ("qwen35_moe_gqa8", (None, None)),
    27: ("qwen3_q40kv8_d128_causal", (None, None)),
    28: ("qwen3_q40kv8_d128_noncausal", (None, None)),
    29: ("mistral_local4096_noncausal", (4096, -1)),
    30: ("qwen35_moe_gqa8_noncausal", (None, None)),
}
CANONICAL38_APPENDED = {
    **CANONICAL30_APPENDED,
    31: ("qwen2_q14kv2_d64_causal", (None, None)),
    32: ("qwen2_q12kv2_d128_causal", (None, None)),
    33: ("llama_q32kv8_d64_noncausal", (None, None)),
    34: ("gemma3_global_d256_gqa2", (None, None)),
    35: ("gemma3_local1024_d256_gqa2", (1023, 0)),
    36: ("gemma3_local1024_d128_gqa2", (1023, 0)),
    37: ("gemma4_e4b_local1024_d256_gqa4", (1023, 0)),
}
CANONICAL27_APPENDED = CANONICAL30_APPENDED
SHA256_RE = re.compile(r"[0-9a-f]{64}")
LEGACY_METADATA_INSTRUCTION = (
    "legacy V2 manifest rejected: derive canonical metadata first "
    "(canonical helper order, explicit link_object_order, and the six "
    "runtime_window/window_contract annotations); do not relink missing metadata")


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def parse_profile_rows(profiles_inc):
    rows = []
    for line in pathlib.Path(profiles_inc).read_text().splitlines():
        line = line.strip()
        if not line.startswith("FA4_PROFILE("):
            continue
        match = re.fullmatch(PROFILE_RE, line)
        if match is None:
            raise RuntimeError(f"invalid profile row: {line}")
        name, heads, kv_heads, head_dim, causal, local = match.groups()
        rows.append((name, int(heads), int(kv_heads), int(head_dim),
                     bool(int(causal)), bool(int(local))))
    return rows


def parse_profile_names(profiles_inc):
    return [row[0] for row in parse_profile_rows(profiles_inc)]


def profile_sequence_sha(rows):
    sequence = "|".join(
        f"{index}:{name}:{heads}:{kv_heads}:{head_dim}:{int(causal)}:{int(local)}"
        for index, (name, heads, kv_heads, head_dim, causal, local) in enumerate(rows))
    return hashlib.sha256(sequence.encode()).hexdigest()


def validate_profile_rows(rows):
    """Validate a complete known canonical prefix.

    Active builds write canonical38. Complete canonical26/canonical31 (and the
    intermediate canonical27..30) prefixes remain accepted for legacy bundles.
    """
    expected = CANONICAL_PROFILE_SETS.get(len(rows))
    if expected is None:
        raise RuntimeError(
            f"profiles.inc row count {len(rows)} is not a known canonical prefix")
    for mode, actual in enumerate(rows):
        if tuple(actual) != expected[mode]:
            raise RuntimeError(
                f"profiles.inc mode{mode} geometry/identity {tuple(actual)!r} "
                f"!= {expected[mode]!r}")
    sequence_sha = profile_sequence_sha(rows)
    expected_sha = CANONICAL_PREFIX_SEQUENCE_SHA256[len(rows)]
    if sequence_sha != expected_sha:
        raise RuntimeError(
            f"profiles.inc fingerprint {sequence_sha} != {expected_sha}")
    return {"profiles": len(rows), "profile_sequence_sha256": sequence_sha}


def validate_profiles_inc(profiles_inc):
    rows = parse_profile_rows(profiles_inc)
    validate_profile_rows(rows)
    return rows


def canonical_rows_for(count):
    rows = CANONICAL_PROFILE_SETS.get(count)
    if rows is None:
        raise RuntimeError(f"unsupported canonical profile count {count}")
    return rows


def canonical_registry_for(count):
    if count == 38:
        return dict(CANONICAL38_REGISTRY)
    if count in CANONICAL_PREFIX_SEQUENCE_SHA256:
        return dict(CANONICAL_PREFIX_REGISTRIES[count])
    raise RuntimeError(f"unsupported canonical registry count {count}")


def validate_registry_metadata(registry, declared=None):
    if not isinstance(registry, dict):
        raise RuntimeError("canonical manifest lacks registry metadata")
    count = registry.get("mode_count", declared)
    if type(count) is not int:
        raise RuntimeError(f"registry mode_count is not an int: {count!r}")
    expected = canonical_registry_for(count)
    extra = sorted(set(registry) - set(expected))
    if extra:
        raise RuntimeError(f"canonical{count} registry has unexpected fields: {extra}")
    for key, expected_value in expected.items():
        if key not in registry:
            raise RuntimeError(f"canonical{count} registry lacks {key}")
        actual = registry[key]
        if key == "mode_count":
            if type(actual) is not int:
                raise RuntimeError(f"registry mode_count is not an int: {actual!r}")
        elif not isinstance(actual, str):
            raise RuntimeError(f"registry {key} is not a string: {actual!r}")
        if actual != expected_value:
            raise RuntimeError(
                f"canonical{count} registry {key}={actual!r} != {expected_value!r}")
    return dict(expected)


def canonical_window(mode, name, causal, local):
    appended = CANONICAL38_APPENDED.get(mode)
    if appended is not None:
        expected_name, expected_window = appended
        if name != expected_name:
            raise RuntimeError(
                f"mode{mode} profile identity {name!r} != {expected_name!r}")
        derived = expected_window
    else:
        derived = ((64, 0) if causal and local
                   else (64, 64) if local else (None, None))
    if bool(local) != (derived[0] is not None):
        raise RuntimeError(
            f"mode{mode} {name}: local={bool(local)} disagrees with canonical window {derived}")
    if bool(causal) != (derived[1] == 0 if local else bool(causal)):
        raise RuntimeError(
            f"mode{mode} {name}: causal={bool(causal)} disagrees with canonical window {derived}")
    return derived


def profile_kernel(mode, name, heads, kv_heads, d, causal, window, dtype):
    derived = canonical_window(mode, name, bool(causal), window[0] is not None)
    if tuple(window) != derived:
        raise RuntimeError(
            f"mode{mode} {name}: compiler window {tuple(window)} != canonical {derived}")
    kernel = {
        "symbol": f"fa4_{name}_{dtype}",
        "heads": heads,
        "kv_heads": kv_heads,
        "d": d,
        "causal": bool(causal),
        "window": list(derived),
        "dtype": dtype,
    }
    if mode in WINDOW_CONTRACT_MODES:
        kernel["runtime_window"] = list(derived)
        kernel["window_contract"] = WINDOW_CONTRACT
    return kernel


def _helper_contracts():
    contracts = {}
    for dtype in DTYPES:
        contracts[f"fa4_deberta_{dtype}"] = {
            "symbol": f"fa4_deberta_{dtype}", "heads": 12, "kv_heads": 12,
            "d": 64, "causal": False, "window": [None, None], "dtype": dtype,
        }
        contracts[f"fa4_alibi_model_bias_{dtype}"] = {
            "symbol": f"fa4_alibi_model_bias_{dtype}", "heads": 8,
            "kv_heads": 8, "d": 64, "causal": False, "window": [None, None],
            "dtype": dtype, "kind": "alibi_model_bias",
            "score_source": "alibi_score.py", "score_abi": 1,
            "qk_scale": 0.125, "external_softmax_scale": 1.0,
            "bias_rounding": "model_dtype",
        }
        contracts[f"fa4_jinacode_flash_alibi_{dtype}"] = {
            "symbol": f"fa4_jinacode_flash_alibi_{dtype}", "heads": 12,
            "kv_heads": 12, "d": 64, "causal": False, "window": [None, None],
            "dtype": dtype, "kind": "alibi12_native_flash",
            "score_source": "alibi12_native_score.py", "score_abi": 1,
            "qk_scale": 0.125, "external_softmax_scale": 1.0,
            "bias_rounding": "float32_fma_unscaled_then_pow2_scale",
            "alibi_heads": 12, "alibi_slope_count": 12,
        }
    for name, divisor, reciprocal in (
            ("both", 13.875, 0x3d939a86),
            ("single", 11.3125, 0x3db509e7)):
        symbol = f"fa4_deberta_exact_mul_{name}_bf16"
        contracts[symbol] = {
            "symbol": symbol, "heads": 12, "kv_heads": 12, "d": 64,
            "causal": False, "window": [None, None], "dtype": "bf16",
            "relative_score_input": "raw_bf16_matmul",
            "relative_divisor_f32": divisor,
            "relative_reciprocal_f32_bits": reciprocal,
            "relative_score_transform":
                "mul.rn.f32+cvt.rn.bf16x2.f32+bit_widen_f32",
        }
    return contracts


HELPER_CONTRACTS = _helper_contracts()


def _same_contract_value(expected, actual):
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual == expected
    if isinstance(expected, int):
        return type(actual) is int and actual == expected
    if isinstance(expected, float):
        return (type(actual) in (int, float) and not isinstance(actual, bool)
                and actual == expected)
    if isinstance(expected, str):
        return isinstance(actual, str) and actual == expected
    if expected is None:
        return actual is None
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(_same_contract_value(left, right)
                        for left, right in zip(expected, actual)))
    raise RuntimeError(f"unsupported contract value {expected!r}")


def _validate_exact_kernel(actual, expected, label):
    if not isinstance(actual, dict):
        raise RuntimeError(f"{label}: kernel entry is not an object")
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise RuntimeError(
            f"{label}: kernel fields differ missing={missing} extra={extra}")
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if not _same_contract_value(expected_value, actual_value):
            raise RuntimeError(
                f"{label}: field {key}={actual_value!r} != {expected_value!r}")


def _helper_flags(symbols):
    symbol_set = set(symbols)
    flags = {}
    for family, family_symbols in HELPER_FAMILIES:
        present = [symbol in symbol_set for symbol in family_symbols]
        if any(present) and not all(present):
            raise RuntimeError(
                f"incomplete {family} helper set: "
                f"missing={[s for s, ok in zip(family_symbols, present) if not ok]}")
        flags[f"include_{family}"] = all(present)
    return flags


def expected_kernel_symbols(
    profile_names,
    include_deberta=True,
    include_deberta_exact_mul=True,
    include_alibi=True,
    include_jinacode=True,
):
    symbols = [f"fa4_{name}_{dtype}" for dtype in DTYPES for name in profile_names]
    flags = {
        "deberta": include_deberta,
        "deberta_exact_mul": include_deberta_exact_mul,
        "alibi": include_alibi,
        "jinacode": include_jinacode,
    }
    for family, family_symbols in HELPER_FAMILIES:
        if flags[family]:
            symbols.extend(family_symbols)
    return symbols


def expected_link_symbols(profile_names, **flags):
    symbols = [f"fa4_{name}_{dtype}" for dtype in DTYPES for name in profile_names]
    for family in LINK_HELPER_ORDER:
        if flags.get(f"include_{family}", True):
            symbols.extend(HELPER_SYMBOLS[family])
    return symbols


def canonical_metadata_order(kernels, profile_names, **flags):
    expected = expected_kernel_symbols(profile_names, **flags)
    by_symbol = {}
    for kernel in kernels:
        symbol = kernel.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            raise RuntimeError(f"invalid kernel symbol: {symbol!r}")
        if symbol in by_symbol:
            raise RuntimeError(f"duplicate canonical31 kernel symbol: {symbol}")
        by_symbol[symbol] = kernel
    missing = [symbol for symbol in expected if symbol not in by_symbol]
    extra = [symbol for symbol in by_symbol if symbol not in expected]
    if missing or extra or len(kernels) != len(expected):
        raise RuntimeError(
            "canonical31 kernel inventory differs: "
            f"missing={missing} extra={extra} "
            f"expected_count={len(expected)} actual_count={len(kernels)}")
    return [by_symbol[symbol] for symbol in expected]


def validate_kernel_order(kernels, profile_names, **flags):
    expected = expected_kernel_symbols(profile_names, **flags)
    actual = [str(kernel.get("symbol", "")) for kernel in kernels]
    if actual != expected:
        missing = [symbol for symbol in expected if symbol not in actual]
        extra = [symbol for symbol in actual if symbol not in expected]
        raise RuntimeError(
            "canonical31 kernel inventory differs: "
            f"missing={missing} extra={extra} "
            f"expected_count={len(expected)} actual_count={len(actual)}")
    return expected


def validate_kernel_contracts(kernels, profile_rows):
    names = [row[0] for row in profile_rows]
    by_symbol = {}
    for kernel in kernels:
        if not isinstance(kernel, dict):
            raise RuntimeError("kernel entry is not an object")
        symbol = kernel.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            raise RuntimeError(f"invalid kernel symbol: {symbol!r}")
        if symbol in by_symbol:
            raise RuntimeError(f"duplicate canonical31 kernel symbol: {symbol}")
        by_symbol[symbol] = kernel
    flags = _helper_flags(by_symbol)
    expected = expected_kernel_symbols(names, **flags)
    if list(by_symbol) != expected:
        raise RuntimeError(
            "canonical31 kernel metadata order differs from qualified source order")
    ordinary = {}
    for mode, row in enumerate(profile_rows):
        name, heads, kv_heads, d, causal, local = row
        window = canonical_window(mode, name, causal, local)
        for dtype in DTYPES:
            symbol = f"fa4_{name}_{dtype}"
            expected_kernel = {
                "symbol": symbol, "heads": heads, "kv_heads": kv_heads,
                "d": d, "causal": causal, "window": list(window),
                "dtype": dtype,
            }
            if mode in WINDOW_CONTRACT_MODES:
                expected_kernel["runtime_window"] = list(window)
                expected_kernel["window_contract"] = WINDOW_CONTRACT
            ordinary[symbol] = expected_kernel
    for symbol in expected:
        expected_kernel = ordinary.get(symbol) or HELPER_CONTRACTS.get(symbol)
        if expected_kernel is None:
            raise RuntimeError(f"unknown canonical31 kernel symbol {symbol}")
        _validate_exact_kernel(by_symbol[symbol], expected_kernel, symbol)
    return {"kernels": len(expected), "helpers": sum(flags.values())}


def validate_window_contract(kernels, profile_names):
    by_symbol = {}
    for kernel in kernels:
        symbol = kernel.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            raise RuntimeError(f"invalid kernel symbol: {symbol!r}")
        if symbol in by_symbol:
            raise RuntimeError(f"duplicate canonical31 kernel symbol: {symbol}")
        by_symbol[symbol] = kernel
    annotated = set()
    for mode in WINDOW_CONTRACT_MODES:
        name, window = CANONICAL30_APPENDED[mode]
        if profile_names[mode] != name:
            raise RuntimeError(
                f"mode{mode} profile identity {profile_names[mode]!r} != {name!r}")
        for dtype in DTYPES:
            symbol = f"fa4_{name}_{dtype}"
            kernel = by_symbol.get(symbol)
            if kernel is None:
                raise RuntimeError(f"missing canonical30 window kernel {symbol}")
            annotated.add(symbol)
            if kernel.get("window") != list(window):
                raise RuntimeError(
                    f"{symbol}: window={kernel.get('window')!r} != {list(window)!r}")
            if kernel.get("runtime_window") != list(window):
                raise RuntimeError(
                    f"{symbol}: runtime_window={kernel.get('runtime_window')!r} "
                    f"!= {list(window)!r}")
            if kernel.get("window_contract") != WINDOW_CONTRACT:
                raise RuntimeError(
                    f"{symbol}: window_contract={kernel.get('window_contract')!r} "
                    f"!= {WINDOW_CONTRACT!r}")
    for symbol, kernel in by_symbol.items():
        if symbol in annotated:
            continue
        if "runtime_window" in kernel or "window_contract" in kernel:
            raise RuntimeError(
                f"{symbol}: pre-canonical kernel metadata was extended")
    return sorted(annotated)


def _declared_profile_count(manifest, observed):
    registry = manifest.get("registry")
    if isinstance(registry, dict) and "mode_count" in registry:
        count = registry["mode_count"]
        if type(count) is not int:
            raise RuntimeError(f"registry mode_count is not an int: {count!r}")
        return count
    # Legacy manifests may omit registry metadata. Infer the longest known
    # ordinary prefix matching the observed export-name prefix.
    prefix = []
    for symbol in observed:
        if not symbol.startswith("fa4_") or symbol.startswith(
                ("fa4_deberta_", "fa4_alibi_", "fa4_jinacode_")):
            break
        prefix.append(symbol)
    best = None
    for count in sorted(CANONICAL_PROFILE_SETS):
        names = [row[0] for row in CANONICAL_PROFILE_SETS[count]]
        expected = [f"fa4_{name}_{dtype}" for dtype in DTYPES for name in names]
        if prefix[:len(expected)] == expected:
            best = count
    if best is None:
        raise RuntimeError("manifest lacks a known canonical profile prefix")
    return best


def inspect_manifest_inventory(manifest, profile_rows=None):
    kernels = manifest.get("kernels")
    if not isinstance(kernels, list) or not kernels:
        raise RuntimeError("manifest lacks kernel inventory")
    observed = []
    for kernel in kernels:
        if not isinstance(kernel, dict):
            raise RuntimeError("manifest kernel entry is not an object")
        symbol = kernel.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            raise RuntimeError(f"invalid kernel symbol: {symbol!r}")
        observed.append(symbol)
    if len(set(observed)) != len(observed):
        raise RuntimeError("manifest kernel inventory contains duplicate symbols")
    count = _declared_profile_count(manifest, observed)
    canonical_rows = canonical_rows_for(count)
    if profile_rows is not None and list(profile_rows) != list(canonical_rows):
        # The caller's profiles.inc must still describe the manifest prefix.
        raise RuntimeError(
            "manifest canonical prefix differs from the linked profiles.inc")
    names = [row[0] for row in canonical_rows]
    flags = _helper_flags(observed)
    expected_metadata = expected_kernel_symbols(names, **flags)
    expected_link = [f"{symbol}.o" for symbol in expected_link_symbols(names, **flags)]
    expected_legacy = [name.removesuffix(".o") for name in expected_link]
    if observed == expected_metadata:
        order_kind = "canonical"
    elif observed == expected_legacy:
        order_kind = "legacy"
    else:
        order_kind = "unknown"
        raise RuntimeError(
            "manifest kernel order is neither canonical source order nor V2 link order")
    inventory_header = manifest.get("canonical31_inventory")
    if inventory_header is not None and not isinstance(inventory_header, dict):
        raise RuntimeError("canonical31_inventory is not an object")
    declared = (inventory_header or {}).get("symbols")
    if order_kind == "canonical":
        if inventory_header.get("exact_order") is not True:
            raise RuntimeError(
                "canonical31_inventory.exact_order must be boolean true")
        if declared != expected_metadata:
            raise RuntimeError(
                "canonical kernel order requires canonical31_inventory.symbols "
                "in the same canonical order")
        if "link_object_order" not in manifest:
            raise RuntimeError("canonical metadata requires explicit link_object_order")
        link_order = manifest["link_object_order"]
        if link_order != expected_link:
            raise RuntimeError(
                "manifest link_object_order differs from the preserved V2 link order")
    else:
        if declared is not None and declared != expected_legacy:
            raise RuntimeError(
                "legacy kernel order requires legacy canonical31_inventory.symbols")
        link_order = manifest.get("link_object_order", expected_link)
        if link_order != expected_link:
            raise RuntimeError(
                "manifest link_object_order differs from the preserved V2 link order")
    objects = manifest.get("objects")
    if not isinstance(objects, dict):
        raise RuntimeError("manifest lacks object hash mapping")
    missing = [name for name in expected_link if name not in objects]
    extra = [name for name in objects if name not in expected_link]
    if missing or extra:
        raise RuntimeError(
            f"manifest object inventory differs: missing={missing} extra={extra}")
    for symbol in observed:
        object_name = f"{symbol}.o"
        if object_name not in objects:
            raise RuntimeError(f"{symbol}: object entry missing")
        value = objects[object_name]
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise RuntimeError(f"{object_name}: object hash is not a sha256 digest")
    return {
        "observed_symbols": observed,
        "order_kind": order_kind,
        "declared_symbols": declared,
        "helper_flags": flags,
        "expected_metadata": expected_metadata,
        "expected_link": expected_link,
        "link_object_order": link_order,
    }


def validate_manifest_inventory(manifest, profile_rows=None, require_all_helpers=True):
    count = _declared_profile_count(manifest, manifest.get("kernels", []) and [
        kernel.get("symbol") for kernel in manifest["kernels"]
        if isinstance(kernel, dict)
    ])
    canonical_rows = canonical_rows_for(count)
    inventory = inspect_manifest_inventory(manifest, canonical_rows)
    if inventory["order_kind"] != "canonical":
        raise RuntimeError(LEGACY_METADATA_INSTRUCTION)
    if require_all_helpers and not all(inventory["helper_flags"].values()):
        raise RuntimeError(
            "canonical38 requires every source-known helper family: "
            f"{inventory['helper_flags']}")
    validate_registry_metadata(manifest.get("registry"), count)
    validate_kernel_contracts(manifest["kernels"], canonical_rows)
    return inventory


def validate_compile_env(environ, compile_only):
    if not compile_only:
        raise RuntimeError("canonical30 source export requires --compile-only")
    if environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("canonical30 source export requires CUDA_VISIBLE_DEVICES=''")
    if environ.get("CUTE_DSL_PTXAS_PATH"):
        raise RuntimeError("refusing external CUTE_DSL_PTXAS_PATH for offline export")
    enabled = environ.get("FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED", "0")
    if enabled not in ("", "0", "false", "False", "no", "off"):
        raise RuntimeError("refusing persistent FA4 JIT cache during canonical30 export")
    environ["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] = "0"
    environ.pop("FLASH_ATTENTION_CUTE_DSL_CACHE_DIR", None)
    environ.pop("CUTE_DSL_PTXAS_PATH", None)
    return {
        "cuda_visible_devices": "",
        "persistent_jit_cache_enabled": False,
        "external_ptxas": False,
        "compile_only": True,
    }


def source_hashes(root):
    root = pathlib.Path(root)
    return {str(path.relative_to(root)): digest(path) for path in sorted(root.rglob("*.py"))}


def load_source_lock(path):
    return json.loads(pathlib.Path(path).read_text())


def expected_source_hashes(lock, capability):
    expected = dict(lock["base_sources"])
    if capability == 100:
        expected["interface.py"] = lock["sm100_gemma_patch"]["after_sha256"]
    return expected


def verify_source_identity(staged_cute, capability, lock):
    expected = expected_source_hashes(lock, capability)
    actual = source_hashes(staged_cute)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(path for path in expected if path in actual and expected[path] != actual[path])
        raise RuntimeError(
            "staged FA4 source identity differs: "
            f"missing={missing} extra={extra} changed={changed}")
    return {"files": len(expected), "capability": capability}


def runtime_pins(ffi_root, runtime_dir):
    return {
        "libtvm_ffi.so": digest(pathlib.Path(ffi_root) / "lib/libtvm_ffi.so"),
        "libcute_dsl_runtime.so": digest(pathlib.Path(runtime_dir) / "libcute_dsl_runtime.so"),
    }


def verify_runtime_pins(manifest, ffi_root, runtime_dir):
    pins = manifest.get("export_runtime_pins")
    if not isinstance(pins, dict) or not pins:
        raise RuntimeError("export manifest lacks resolved runtime pins")
    actual = runtime_pins(ffi_root, runtime_dir)
    if actual != pins:
        raise RuntimeError(
            "link runtime binaries differ from export pins: "
            f"expected={pins} actual={actual}")
    return actual

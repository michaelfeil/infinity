# SPDX-License-Identifier: MIT
# Adapted from candle-flash-attn-v4; see provenance.json and LICENSE-MIT.
"""Stage the pinned FA4 sources with FA2-compatible SM90 softmax arithmetic.

Only the private build copy is modified. The installed Python package remains
unchanged. Re-evaluate these compatibility choices when updating upstream.
"""
import hashlib
import importlib.util
import pathlib
import shutil
import sys
import tempfile

_ORIGINAL = "        row_sum.store(utils.warp_reduce(row_sum.load(), operator.add, width=4))"
_REPLACEMENT = """        # Match FA2's four-lane denominator reduction: offsets 2, then 1.
        # FP32 addition is not associative. Reversing the shuffles can change
        # FP16/BF16 rounding, amplified by subsequent encoder layers.
        for r in cutlass.range(cute.size(row_sum), unroll_full=True):
            total = row_sum[r]
            total = total + cute.arch.shuffle_sync_bfly(total, offset=2)
            total = total + cute.arch.shuffle_sync_bfly(total, offset=1)
            row_sum[r] = total"""

_TILE_ORIGINAL = """    elif head_dim <= 128:
        return FwdConfig(128, 128, True, True)"""
_TILE_REPLACEMENT = """    elif head_dim <= 128:
        # Match FA2's online-softmax block boundaries for causal d128.
        # A different key tile changes low-precision probability rounding.
        tile_n = 64 if head_dim == 128 and is_causal else 128
        return FwdConfig(128, tile_n, True, True)"""


_SM8X_TILE_ORIGINAL = """        elif arch // 10 == 8:
            cfg = FwdConfig(128, 64, True, True)  # SM80, should tune"""
_SM8X_TILE_REPLACEMENT = """        elif arch // 10 == 8:
            # Match FA2's d64 online-softmax block boundaries.
            cfg = FwdConfig(128, 128 if head_dim == 64 else 64, True, True)"""

_SM8X_SCORE_ORIGINAL = """        if arch // 10 == 8:
            raise NotImplementedError("Custom user-provided score_mod is not supported on SM8x architectures.")"""
_SM8X_SCORE_REPLACEMENT = """        if arch // 10 == 8 and score_mod.__name__ != "deberta_score":
            raise NotImplementedError("InfinityFA4 only exports the DeBERTa score hook on SM8x.")"""

_SCHEDULER_IMPORT = "from cutlass.utils.hardware_info import HardwareInfo"
_SCHEDULER_REPLACEMENT = """from cutlass.utils.hardware_info import HardwareInfo
import os


def _infinityfa4_sm_count():
    # Persistent grids need a target SM count, not a build-host CUDA device.
    # The exporter sets this alongside FLASH_ATTENTION_ARCH for offline builds.
    override = os.getenv("FLASH_ATTENTION_NUM_SMS")
    if override is not None:
        count = int(override)
        if count <= 0:
            raise ValueError("FLASH_ATTENTION_NUM_SMS must be positive")
        return count
    return HardwareInfo().get_device_multiprocessor_count()
"""


def stage_fa4_sources():
    if any(name == "flash_attn" or name.startswith("flash_attn.") for name in sys.modules):
        raise RuntimeError("Stage the FA4 backport before importing flash_attn")
    spec = importlib.util.find_spec("flash_attn")
    locations = [] if spec is None else list(spec.submodule_search_locations or [])
    candidates = [pathlib.Path(p) for p in locations if (pathlib.Path(p) / "cute/softmax.py").is_file()]
    if len(candidates) != 1:
        raise RuntimeError("Install one pinned FA4 build package before exporting")
    original = candidates[0]
    text = (original / "cute/softmax.py").read_text()
    if text.count(_ORIGINAL) != 1:
        raise RuntimeError("FA4 softmax changed: review the denominator backport before exporting")
    patched = text.replace(_ORIGINAL, _REPLACEMENT)
    interface = (original / "cute/interface.py").read_text()
    if interface.count(_TILE_ORIGINAL) != 1:
        raise RuntimeError("FA4 SM90 tile selection changed: review the causal d128 backport")
    if interface.count(_SM8X_TILE_ORIGINAL) != 1:
        raise RuntimeError("FA4 SM8x tile selection changed: review the d64 backport")
    if interface.count(_SM8X_SCORE_ORIGINAL) != 2:
        raise RuntimeError("FA4 SM8x score dispatch changed: review the DeBERTa export")
    patched_interface = interface.replace(_TILE_ORIGINAL, _TILE_REPLACEMENT).replace(
        _SM8X_TILE_ORIGINAL, _SM8X_TILE_REPLACEMENT
    ).replace(_SM8X_SCORE_ORIGINAL, _SM8X_SCORE_REPLACEMENT, 1)
    scheduler = (original / "cute/tile_scheduler.py").read_text()
    if (scheduler.count(_SCHEDULER_IMPORT) != 1
            or scheduler.count("hardware_info = cutlass.utils.HardwareInfo()") != 1
            or scheduler.count("hardware_info = HardwareInfo()") != 2
            or scheduler.count("hardware_info.get_device_multiprocessor_count()") != 3):
        raise RuntimeError("FA4 persistent scheduling changed: review the offline SM-count patch")
    patched_scheduler = "\n".join(
        line for line in scheduler.split("\n")
        if line.strip() not in {
            "hardware_info = cutlass.utils.HardwareInfo()",
            "hardware_info = HardwareInfo()",
        }
    ).replace(_SCHEDULER_IMPORT, _SCHEDULER_REPLACEMENT).replace(
        "hardware_info.get_device_multiprocessor_count()", "_infinityfa4_sm_count()"
    )
    stage = tempfile.TemporaryDirectory(prefix="fa4-softmax-")
    target = pathlib.Path(stage.name) / "flash_attn"
    shutil.copytree(original, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target / "cute/softmax.py").write_text(patched)
    (target / "cute/interface.py").write_text(patched_interface)
    (target / "cute/tile_scheduler.py").write_text(patched_scheduler)
    sys.path.insert(0, stage.name)
    return stage, {
        "name": "fa2-compatible-four-lane-softmax-reduction",
        "original_softmax_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "patched_softmax_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "sm90_causal_d128_tile_n": 64,
        "sm8x_d64_tile_n": 128,
        "sm8x_deberta_score": True,
        "original_interface_sha256": hashlib.sha256(interface.encode()).hexdigest(),
        "patched_interface_sha256": hashlib.sha256(patched_interface.encode()).hexdigest(),
        "original_scheduler_sha256": hashlib.sha256(scheduler.encode()).hexdigest(),
        "patched_scheduler_sha256": hashlib.sha256(patched_scheduler.encode()).hexdigest(),
    }

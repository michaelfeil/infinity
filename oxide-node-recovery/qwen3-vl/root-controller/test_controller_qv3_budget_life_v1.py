#!/usr/bin/env python3
"""Focused CPU contract tests for the bounded Rust-only controller.

No Docker, GPU, CUDA, Torch, network, or model execution. Only the controller's
explicit pins/plan/stage manifest and owning-life report checker are exercised.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import controller_qv3_budget_life_v1 as controller


def test_pins_and_stage_manifest():
    pins = controller.pins()
    assert pins["pins_sha256"] == controller.canonical_hash(
        {key: value for key, value in pins.items() if key != "pins_sha256"}
    )
    assert pins["unified_consumer"]["sha256"] == controller.UNIFIED_CONSUMER_SHA
    assert pins["native_v6_gold"]["observations"] == 8
    assert len(pins["native_v6_gold"]["f32_files"]) == 40
    assert len(pins["fixtures"]["assets"]) == 13
    assert len(pins["runtime_paths_24"]) == 24
    assert pins["no_native_job"] is True
    manifest = controller.stage_manifest()
    assert len(manifest) == 41
    assert sum(1 for entry in manifest if entry["kind"] == "runtime") == 24
    assert sum(1 for entry in manifest if entry["kind"] == "copy") == 16
    assert sum(1 for entry in manifest if entry["kind"] == "generate") == 1
    for entry in manifest:
        if entry["kind"] == "runtime":
            assert entry["copied"] is False
        if entry["source"] is not None:
            assert "*" not in entry["source"]
            assert "?" not in entry["source"]


def test_plan_is_rust_first_then_host_compare_and_life():
    plan = controller.plan("test-run", "/tmp/qwen3vl-budget-life-test-out")
    assert plan["no_native_job"] is True
    assert plan["one_cold_backend_load"] is True
    assert [job["job"] for job in plan["jobs"]] == [
        "quality-life-vectors",
        "quality-compare-native-v6-gold",
        "quality-owning-life-contract",
    ]
    gpu, compare, life = plan["jobs"]
    assert gpu["kind"] == "container"
    assert "CUDA_VISIBLE_DEVICES=0" in gpu["argv"]
    assert "gpu-quality-life" in gpu["argv"]
    assert any("remote_guard.py" in arg for arg in gpu["argv"])
    assert "184-191" in gpu["argv"]
    assert any(arg.endswith("unified-consumer") for arg in gpu["argv"])
    assert compare["kind"] == "host"
    assert str(controller.NATIVE_GOLD) in compare["argv"]
    assert str(controller.COMPARE) in compare["argv"]
    assert "--allow-raw" in compare["argv"]
    assert life["kind"] == "host"
    assert "--check-life-report" in life["argv"]
    assert all("native-consumer" not in arg for job in plan["jobs"] for arg in job["argv"])


def _write_baselines(directory: Path, names: list[str]):
    baseline_dir = directory / "life-baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    baselines = []
    for name in names:
        path = baseline_dir / f"{name}.baseline.f32"
        payload = name.encode() * 4
        path.write_bytes(payload)
        baselines.append(
            {
                "fixture": name,
                "shape": f"shape-{name}",
                "sequences": 1,
                "width": 4,
                "values": 4,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "file": {
                    "fixture": name,
                    "kind": "baseline",
                    "width": 4,
                    "sequences": 1,
                    "values": 4,
                    "cumulative": [0, 4],
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "path": f"/remote/{name}.baseline.f32",
                    "finite": True,
                },
            }
        )
    return baselines


def test_life_check_accepts_good_and_rejects_mutations():
    manifest = json.loads(controller.FIXTURES_MANIFEST.read_text())
    names = [item["name"] for item in manifest["fixtures"]]
    with tempfile.TemporaryDirectory(prefix="qv3-life-check-") as tmp_name:
        tmp = Path(tmp_name)
        report = {
            "schema": "qwen3-vl-host-consumer-gpu-owning-life/v1",
            "route": "Qwen3VlBackend::submit_embed",
            "backend_constructions": 1,
            "fixture_count": 8,
            "width": 4,
            "multi_vector": False,
            "baselines": _write_baselines(tmp, names),
            "held": {"a": names[0], "a_shape": "a", "b": names[1], "b_shape": "b"},
            "third_submit_while_two_held": False,
            "drop_without_completion_fenced": True,
            "complete_b_exact": True,
            "changed_shape": {"fixture": names[2], "shape": "c", "exact": True},
            "invalid_media_typed": True,
            "invalid_media_zero_selectors": True,
            "recovered": {"fixture": names[3], "exact": True},
            "health_after_recovery": True,
        }
        (tmp / "gpu-owning-life.json").write_text(json.dumps(report))
        (tmp / "gpu-quality-life.json").write_text(
            json.dumps(
                {
                    "schema": "qwen3-vl-host-consumer-gpu-quality-life/v1",
                    "backend_constructions": 1,
                    "quality": {},
                    "owning_life": {},
                }
            )
        )
        good = controller.check_life(tmp / "gpu-owning-life.json", tmp / "check-good.json")
        assert good["passed"] is True, good["failures"]

        mutated = dict(report)
        mutated["complete_b_exact"] = False
        (tmp / "gpu-owning-life.json").write_text(json.dumps(mutated))
        bad = controller.check_life(tmp / "gpu-owning-life.json", tmp / "check-bad.json")
        assert bad["passed"] is False
        assert any("complete_b_exact" in failure for failure in bad["failures"])

        mutated = dict(report)
        mutated["fake"] = "Overloaded"
        (tmp / "gpu-owning-life.json").write_text(json.dumps(mutated))
        bad = controller.check_life(tmp / "gpu-owning-life.json", tmp / "check-overloaded.json")
        assert bad["passed"] is False
        assert any("Overloaded" in failure for failure in bad["failures"])

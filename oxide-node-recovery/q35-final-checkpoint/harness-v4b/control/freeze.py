#!/usr/bin/env python3
import hashlib, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
EXCLUDE = {"HARNESS-FREEZE-v2.json", "PINS.json"}
manifest = json.loads((ROOT / "fixtures/manifest.json").read_text())
pending_sha = [case["name"] for case in manifest["cases"]
               if case.get("sha256") == "GENERATE-WITH-control/generate_fixtures.py"]
if pending_sha:
    raise SystemExit(f"fixture sha256 not generated; run control/generate_fixtures.py first: {pending_sha}")
contract = json.loads((ROOT / "CONTRACT.json").read_text())
product_source = contract["product_source"]
product_freeze = product_source["freeze_sha256"]
product_freeze_policy = product_source.get(
    "freeze_sha256_policy",
    "explicit UNBOUND placeholder; Codex binds the final product sha256 before any build",
)
if product_freeze != "UNBOUND" and len(product_freeze) != 64:
    raise SystemExit("CONTRACT.json product_source.freeze_sha256 must be UNBOUND or a 64-char sha256")
paths = []
for path in ROOT.rglob("*"):
    if not path.is_file():
        continue
    rel = path.relative_to(ROOT)
    if rel.parts[0] in {"consumer-target", "control/__pycache__"}:
        continue
    if str(rel) in EXCLUDE or rel.name.endswith(".pyc"):
        continue
    paths.append(rel)
files = {str(rel): {"bytes": (ROOT / rel).stat().st_size,
                    "sha256": hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()}
         for rel in sorted(paths)}
freeze = {
    "schema": "oxide-q35moe-unified-harness-freeze/v2",
    "status": "v4b-source-updated-unbuilt",
    "preserves_v1_harness_freeze": "HARNESS-FREEZE-v1.json",
    "product_source": product_source["path"],
    "product_freeze_sha256": product_freeze,
    "product_freeze_sha256_policy": product_freeze_policy,
    "product_read_or_compiled": False,
    "cpu_build_run": False,
    "tests_run": False,
    "optimized_build_run": False,
    "gpu_run": False,
    "files": files,
}
(ROOT / "HARNESS-FREEZE-v2.json").write_text(json.dumps(freeze, indent=2) + "\n")
before = json.loads((ROOT / "control/product-pins-before.json").read_text())
after = json.loads((ROOT / "control/product-pins-after.json").read_text())
product_match = before == after and all(
    entry.get("match") is True for entry in before.values()
)
pins = {
    "schema": "oxide-q35moe-unified-harness-pins/v2",
    "status": "v4b-source-updated-unbuilt",
    "harness_freeze_sha256": hashlib.sha256((ROOT / "HARNESS-FREEZE-v2.json").read_bytes()).hexdigest(),
    "product_source": product_source["path"],
    "product_freeze_sha256": product_freeze,
    "product_pins_before_after_match": product_match,
    "product_pins_status": "historical v1 product evidence; v2 product is unsealed and was not read or compiled",
    "source_file_count": len(files),
    "dev_test": "not run for this source update",
    "dev_clippy": "not run for this source update",
    "dev_fmt": "not run for this source update",
    "optimized_build_run": False,
    "gpu_run": False,
    "cpu_build_run": False,
    "tests_run": False,
}
(ROOT / "PINS.json").write_text(json.dumps(pins, indent=2) + "\n")
print(json.dumps(pins, indent=2))

# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
"""Mock-boundary regression tests for device/build.sh.

The real compiler is replaced by a tiny PTX producer, and ptxas/cuobjdump are
replaced by deterministic fakes. These tests cover scheduling, locking,
fail-closed manifest handling, native-cubin checks, and config visibility
without requiring CUDA, a GPU, or a Cargo build.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import textwrap
import unittest

DEVICE = Path(__file__).resolve().parents[1]
BUILD = DEVICE / "build.sh"
ALL_ARCHITECTURES = ["sm_90", "sm_100", "sm_120"]

SYMBOLS = """
infinity_add_rms_norm infinity_add_rms_norm_1024_exact_vector infinity_add_rms_norm_1536_cached infinity_add_rms_norm_1536_vector
infinity_add_rms_norm_2048_exact infinity_add_rms_norm_2048_vector infinity_add_rms_norm_4096_vector infinity_add_rms_norm_rounded
infinity_attention infinity_base_rope_tables_64_bf16 infinity_bf16_to_f32 infinity_classifier_bias_bf16
infinity_classifier_embedding_norm_384_exact infinity_classifier_gather_first_bf16 infinity_classifier_layer_norm_384_exact infinity_deberta_divide
infinity_deberta_embedding_word_norm_768 infinity_deberta_layer_norm_768 infinity_deberta_relative_transpose infinity_deberta_signed_buckets
infinity_deberta_unpack_scaled infinity_dense_tanh_bf16_inplace infinity_embedding_gather infinity_embedding_gather_scaled
infinity_embedding_norm infinity_embedding_norm_1024_rounded_native infinity_embedding_norm_384 infinity_embedding_norm_384_f16
infinity_embedding_norm_768_exact infinity_embedding_norm_768_exact_vector infinity_embedding_norm_mpnet_768_aten infinity_embedding_norm_rounded
infinity_embedding_norm_word_768_aten infinity_embedding_word_norm_768_candle infinity_embedding_word_type_norm_512_exact infinity_embedding_word_type_norm_768_exact
infinity_encoder_activation_gelu_tanh_bf16 infinity_encoder_activation_relu_bf16 infinity_encoder_activation_silu_bf16 infinity_encoder_activation_tanh_bf16
infinity_f16_to_f32 infinity_f32_to_bf16 infinity_gated_gelu_up_first infinity_gelu_bias
infinity_gelu_exact infinity_gemma2_gated_gelu_bf16 infinity_gemma2_mean_pool_bf16 infinity_gemma2_norm_bf16_f32scale
infinity_gemma2_ple_gate_bf16 infinity_gemma2_residual_norm_bf16_f32scale infinity_gemma2_rope_tables_bf16 infinity_gemma2_scale_bf16
infinity_gemma2_unit_norm_bf16 infinity_jina_code_qk_norm_768 infinity_layer_norm infinity_layer_norm_1024_rounded_native
infinity_layer_norm_384 infinity_layer_norm_384_f16 infinity_layer_norm_512_rounded_native infinity_layer_norm_768_exact
infinity_layer_norm_768_exact_vector infinity_layer_norm_mpnet_768_aten infinity_layer_norm_rounded infinity_mean_pool
infinity_mean_pool_f16 infinity_plain_norm_384_f16 infinity_pool infinity_pool_f16
infinity_qk_norm_rope_128_bf16 infinity_qkv infinity_qkv_f16 infinity_rms_norm
infinity_rms_norm_1024_exact_vector infinity_rms_norm_1536_cached infinity_rms_norm_1536_vector infinity_rms_norm_2048_exact
infinity_rms_norm_2048_vector infinity_rms_norm_4096_vector infinity_rms_norm_rounded infinity_rope_interleaved
infinity_rope_interleaved_hfma infinity_rope_neox infinity_rope_neox_hfma infinity_rounded_residual_norm_768_candle
infinity_selected_rows infinity_silu_mul infinity_silu_mul_bf16_pairs infinity_silu_mul_bf16_pairs_native
infinity_silu_mul_bf16_vector8_native infinity_silu_mul_f32
""".split()
for leaf in (
    "encoder_activation_exports.txt",
    "qwen35_exports.txt",
    "qk256_exports.txt",
    "qwen3vl_exports.txt",
    "qwen35vl_exports.txt",
    "mistral3_exports.txt",
    "qwen35_moe_exports.txt",
    "rms5120_aten_exports.txt", "mistral3_hostfix_exports.txt",
):
    SYMBOLS += (DEVICE / leaf).read_text().split()
SYMBOLS = sorted(set(SYMBOLS))

FAKE_COMPILER = textwrap.dedent(r'''#!/usr/bin/env python3
import json, os, pathlib, sys, time
arch = sys.argv[sys.argv.index("--arch") + 1]
mode = os.environ.get("FAKE_MODE", "ok")
record = os.environ.get("FAKE_RECORD")
config = pathlib.Path(".cargo/cuda-oxide.toml")
config_text = config.read_text() if config.is_file() else None

def log(event, **extra):
    if record:
        line = json.dumps({"event": event, "arch": arch, "pid": os.getpid(),
                           "time": time.monotonic(),
                           "cargo_jobs": int(os.environ.get("CARGO_BUILD_JOBS", "0")),
                           **extra}, sort_keys=True) + "\n"
        fd = os.open(record, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode())
        finally:
            os.close(fd)

main = pathlib.Path("src/main.rs")
if main.is_file():
    main.write_bytes(main.read_bytes())
log("start", cwd=str(pathlib.Path.cwd()), cwd_config=config_text is not None,
    cwd_config_text=config_text)
if mode == "fail":
    time.sleep(0.05)
    log("end", status="failed")
    sys.exit(7)
entries = os.environ["FAKE_SYMBOLS"].split()
target = arch
if mode == "wrong_target":
    target = "sm_999"
elif mode == "wrong_exports":
    entries[0] = "fake_wrong_export"
ptx = ".version 8.0\n.target " + target + "\n.address_size 64\n"
ptx += "\n".join(f".visible .entry {name}() {{ ret; }}" for name in entries) + "\n"
out = pathlib.Path(os.environ["CUDA_OXIDE_PTX_DIR"])
out.mkdir(parents=True, exist_ok=True)
(out / "infinity_oxide_kernels.ptx").write_text(ptx)
time.sleep(float(os.environ.get("FAKE_SLEEP", "0.2")))
log("end", status="ok")
''')

FAKE_PTXAS = textwrap.dedent(r'''#!/usr/bin/env python3
import pathlib, sys
source = pathlib.Path(sys.argv[sys.argv.index("-v") + 1])
target = pathlib.Path(sys.argv[sys.argv.index("-o") + 1])
target.write_text(source.read_text())
''')

FAKE_CUOBJDUMP = textwrap.dedent(r'''#!/usr/bin/env python3
import os, pathlib, re, sys
text = pathlib.Path(sys.argv[-1]).read_text()
arch = re.search(r"^\.target\s+(\w+)", text, re.MULTILINE).group(1)
entries = re.findall(r"\.visible\s+\.entry\s+(\w+)\s*\(", text)
if "--dump-elf-symbols" in sys.argv:
    if os.environ.get("FAKE_DROP_NATIVE"):
        entries = entries[1:]
    print("\n".join("STO_ENTRY " + name for name in entries))
else:
    number = "999" if os.environ.get("FAKE_BAD_SM") else arch.split("_", 1)[1]
    print(f"64-bit ELF: type=ET_EXEC, ABI=8, sm={number}, toolkit=13.1")
''')


def write_executable(path, text):
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def source_digest(root):
    files = [root / name for name in (
        "build.sh", "Cargo.toml", "Cargo.lock", "rust-toolchain.toml",
        "encoder_activation_exports.txt", "qwen35_exports.txt", "qk256_exports.txt",
        "qwen3vl_exports.txt", "qwen35vl_exports.txt", "mistral3_exports.txt",
        "qwen35_moe_exports.txt", "rms5120_aten_exports.txt", "mistral3_hostfix_exports.txt",
        "LICENSE-Candle-BSD3", "LICENSE-PyTorch-BSD3",
        "LICENSE-Qwen-VL-MIT", "LICENSE-Qwen-VL-APACHE", "NOTICE.qwen-vl")]
    files += sorted((root / "src").rglob("*.rs"))
    files.append(root.parent / "NOTICE")
    files.append(root.parent / "NOTICE.qwen35")
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def tree_state(root):
    state = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        name = str(path.relative_to(root))
        if path.is_symlink():
            state[name] = ("symlink", os.readlink(path),
                           info.st_mtime_ns, info.st_ctime_ns)
        elif path.is_file():
            state[name] = ("file", hashlib.sha256(path.read_bytes()).hexdigest(),
                           info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        elif path.is_dir():
            state[name] = ("dir", info.st_mtime_ns, info.st_ctime_ns)
    return state


def read_intervals(path):
    events = [json.loads(line) for line in path.read_text().splitlines() if line]
    starts, intervals = {}, []
    for event in events:
        key = (event["pid"], event["arch"])
        if event["event"] == "start":
            starts[key] = event["time"]
        else:
            intervals.append((starts[key], event["time"]))
    return intervals


def read_starts(path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line and json.loads(line)["event"] == "start"]


def max_concurrency(intervals):
    points = [point for pair in intervals for point in pair]
    return max((sum(start <= point < end for start, end in intervals)
                for point in points), default=0)


def weighted_max_concurrency(path):
    events = [json.loads(line) for line in path.read_text().splitlines() if line]
    starts, intervals = {}, []
    for event in events:
        key = (event["pid"], event["arch"])
        if event["event"] == "start":
            starts[key] = (event["time"], event["cargo_jobs"])
        else:
            start, jobs = starts[key]
            intervals.append((start, event["time"], jobs))
    points = [point for start, end, _jobs in intervals for point in (start, end)]
    return max((sum(jobs for start, end, jobs in intervals if start <= point < end)
                for point in points), default=0)


def expected_job_counts(total, targets=3):
    workers = min(total, targets)
    base, extra = divmod(total, workers)
    return sorted(base + (index < extra) for index in range(targets))


class BuildScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="oxide-build-tests-"))
        cls.compiler = cls.tmp / "fake-cargo-oxide"
        cls.ptxas = cls.tmp / "fake-ptxas"
        cls.cuobjdump = cls.tmp / "fake-cuobjdump"
        write_executable(cls.compiler, FAKE_COMPILER)
        write_executable(cls.ptxas, FAKE_PTXAS)
        write_executable(cls.cuobjdump, FAKE_CUOBJDUMP)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def build_env(self, record, jobs, extra=None):
        env = os.environ.copy()
        env.update({
            "CARGO_OXIDE": str(self.compiler),
            "PTXAS": str(self.ptxas),
            "CUOBJDUMP": str(self.cuobjdump),
            "FAKE_SYMBOLS": " ".join(SYMBOLS),
            "FAKE_SLEEP": "0.25",
            "FAKE_RECORD": str(record),
            "CARGO_BUILD_JOBS": str(jobs),
            "CARGO_NET_OFFLINE": "true",
        })
        if extra:
            env.update(extra)
        return env

    def run_build(self, output, record, jobs=1, source=DEVICE, extra=None,
                  architectures=None):
        command = ["bash", str(source / "build.sh"), str(output)]
        if architectures is not None:
            command += ["--architectures", *architectures]
        return subprocess.run(
            command, cwd=str(source), env=self.build_env(record, jobs, extra),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=120)

    def assert_manifest(self, output, architectures=ALL_ARCHITECTURES):
        manifest_path = output / "manifest.json"
        self.assertEqual(stat.S_IMODE(manifest_path.stat().st_mode), 0o644)
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual([item["architecture"] for item in manifest["architectures"]],
                         architectures)
        self.assertEqual(manifest["requested_architectures"], architectures)
        self.assertEqual(manifest["complete_target_set"],
                         architectures == ALL_ARCHITECTURES)
        self.assertEqual((output / "NOTICE.qwen35").read_bytes(),
                         (DEVICE.parent / "NOTICE.qwen35").read_bytes())
        self.assertEqual(manifest["sources"]["NOTICE.qwen35"],
                         hashlib.sha256((output / "NOTICE.qwen35").read_bytes()).hexdigest())
        for item in manifest["architectures"]:
            self.assertEqual(len(item["symbols_verified"]), len(SYMBOLS))
            candidate = (output / item["architecture"] / ".kernels.cubin").resolve()
            self.assertEqual(item["assemble_command"][-1], str(candidate))
            self.assertFalse(candidate.exists())
        return manifest

    def test_bounded_jobs_1_through_5(self):
        for jobs in (1, 2, 3, 4, 5):
            with self.subTest(jobs=jobs):
                output = self.tmp / f"jobs-{jobs}-out"
                record = self.tmp / f"jobs-{jobs}.jsonl"
                result = self.run_build(output, record, jobs=jobs)
                self.assertEqual(result.returncode, 0, result.stdout)
                intervals = read_intervals(record)
                self.assertEqual(len(intervals), 3)
                self.assertEqual(max_concurrency(intervals), min(jobs, 3))
                starts = read_starts(record)
                self.assertEqual(sorted(event["cargo_jobs"] for event in starts),
                                 expected_job_counts(jobs))
                self.assertLessEqual(weighted_max_concurrency(record), jobs)
                self.assert_manifest(output)

    def test_default_selection_reports_complete_target_set(self):
        output = self.tmp / "default-selection-out"
        record = self.tmp / "default-selection.jsonl"
        result = self.run_build(output, record, jobs=1)
        self.assertEqual(result.returncode, 0, result.stdout)
        manifest = self.assert_manifest(output)
        self.assertIs(manifest["complete_target_set"], True)

    def test_architecture_subset_adapts_jobs_and_preserves_canonical_order(self):
        cases = [
            (["sm_120"], 5, ["sm_120"], [5]),
            (["sm_120", "sm_90", "sm_120"], 5, ["sm_90", "sm_120"], [2, 3]),
        ]
        for index, (requested, jobs, selected, job_counts) in enumerate(cases):
            with self.subTest(requested=requested):
                output = self.tmp / f"subset-{index}-out"
                record = self.tmp / f"subset-{index}.jsonl"
                result = self.run_build(
                    output, record, jobs=jobs, architectures=requested)
                self.assertEqual(result.returncode, 0, result.stdout)
                intervals = read_intervals(record)
                self.assertEqual(len(intervals), len(selected))
                self.assertEqual(max_concurrency(intervals), min(jobs, len(selected)))
                starts = read_starts(record)
                self.assertEqual(sorted(event["arch"] for event in starts),
                                 sorted(selected))
                self.assertEqual(sorted(event["cargo_jobs"] for event in starts),
                                 job_counts)
                self.assertLessEqual(weighted_max_concurrency(record), jobs)
                self.assert_manifest(output, selected)
                self.assertEqual(sorted(path.name for path in output.glob("sm_*")),
                                 sorted(selected))
                self.assertIn("complete_target_set=false", result.stdout)

    def test_invalid_architecture_selection_fails_before_compile(self):
        cases = [
            ("unsupported", ["sm_90", "sm_999"],
             "Unsupported architecture(s): sm_999"),
            ("empty", [], "--architectures requires at least one target"),
        ]
        for name, requested, expected in cases:
            with self.subTest(case=name):
                output = self.tmp / f"invalid-{name}-out"
                record = self.tmp / f"invalid-{name}.jsonl"
                result = self.run_build(output, record, architectures=requested)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stdout)
                self.assertFalse(output.exists())
                self.assertFalse(record.exists())

    def test_subset_rejects_reused_all_target_output_without_mutation(self):
        output = self.tmp / "reused-full-output"
        first = self.run_build(output, self.tmp / "reused-full.jsonl", jobs=2)
        self.assertEqual(first.returncode, 0, first.stdout)
        self.assert_manifest(output)
        before = tree_state(output)
        record = self.tmp / "reused-subset.jsonl"
        second = self.run_build(
            output, record, jobs=5, architectures=["sm_90"])
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("unselected architecture path(s): sm_100, sm_120",
                      second.stdout)
        self.assertIn("dedicated output directory", second.stdout)
        self.assertEqual(tree_state(output), before)
        self.assert_manifest(output)
        self.assertFalse(record.exists())

    def test_subset_rejects_symlinked_excluded_target_without_touching_it(self):
        output = self.tmp / "reused-symlink-output"
        output.mkdir()
        link = output / "sm_120"
        link.symlink_to(self.tmp / "missing-target")
        result = self.run_build(
            output, self.tmp / "reused-symlink.jsonl", architectures=["sm_90"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unselected architecture path(s): sm_120", result.stdout)
        self.assertTrue(link.is_symlink())
        self.assertEqual(os.readlink(link), str(self.tmp / "missing-target"))
        self.assertFalse((output / "manifest.json").exists())

    def test_selected_arch_symlink_is_rejected_and_external_bundle_preserved(self):
        external = self.tmp / "external-bundle/sm_90"
        external.mkdir(parents=True)
        external_cubin = external / "kernels.cubin"
        external_cubin.write_bytes(b"EXTERNAL_KERNEL\n")
        output = self.tmp / "selected-symlink-output"
        output.mkdir()
        link = output / "sm_90"
        link.symlink_to(external, target_is_directory=True)
        record = self.tmp / "selected-symlink.jsonl"
        result = self.run_build(output, record, architectures=["sm_90"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Selected architecture path(s) are symlinks: sm_90",
                      result.stdout)
        self.assertTrue(link.is_symlink())
        self.assertEqual(os.readlink(link), str(external))
        self.assertEqual(external_cubin.read_bytes(), b"EXTERNAL_KERNEL\n")
        self.assertEqual(sorted(path.name for path in external.iterdir()),
                         ["kernels.cubin"])
        self.assertFalse((output / "manifest.json").exists())
        self.assertFalse(record.exists())

    def test_same_subset_output_repeats_and_recovers_after_failure(self):
        output = self.tmp / "same-subset-output"
        first = self.run_build(
            output, self.tmp / "same-subset-first.jsonl", architectures=["sm_90"])
        self.assertEqual(first.returncode, 0, first.stdout)
        self.assert_manifest(output, ["sm_90"])
        second = self.run_build(
            output, self.tmp / "same-subset-second.jsonl",
            architectures=["sm_90"])
        self.assertEqual(second.returncode, 0, second.stdout)
        self.assert_manifest(output, ["sm_90"])
        failed = self.run_build(
            output, self.tmp / "same-subset-failed.jsonl",
            architectures=["sm_90"], extra={"FAKE_MODE": "fail"})
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse((output / "manifest.json").exists())
        self.assertFalse((output / "sm_90/kernels.ptx").exists())
        self.assertFalse((output / "sm_90/kernels.cubin").exists())
        retry = self.run_build(
            output, self.tmp / "same-subset-retry.jsonl",
            architectures=["sm_90"])
        self.assertEqual(retry.returncode, 0, retry.stdout)
        manifest = self.assert_manifest(output, ["sm_90"])
        self.assertFalse(manifest["complete_target_set"])
        for arch in ("sm_100", "sm_120"):
            self.assertFalse((output / arch).exists())

    def stale_fixture(self, output):
        output.mkdir()
        (output / "manifest.json").write_text('{"stale": true}\n')
        stale = output.parent / (output.name + ".work") / "sm_90/ptx"
        stale.mkdir(parents=True)
        marker = "// STALE_PTX_MUST_NOT_BE_ACCEPTED\n"
        (stale / "infinity_oxide_kernels.ptx").write_text(marker)
        return stale / "infinity_oxide_kernels.ptx", marker

    def test_compiler_failure_clears_manifest_and_rejects_stale_ptx(self):
        output = self.tmp / "compiler-failure-out"
        record = self.tmp / "compiler-failure.jsonl"
        stale, _marker = self.stale_fixture(output)
        result = self.run_build(output, record, jobs=1, extra={"FAKE_MODE": "fail"})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("returned non-zero exit status 7", result.stdout)
        self.assertFalse((output / "manifest.json").exists())
        self.assertFalse((output / "sm_90/kernels.ptx").exists())
        self.assertFalse(stale.exists())
        self.assertEqual(list(output.glob(".manifest.*")), [])

    def test_wrong_outputs_and_native_checks_fail_closed(self):
        cases = [
            ("wrong_target", "Incorrect compiler target or kernel exports: sm_90", {}),
            ("wrong_exports", "Incorrect compiler target or kernel exports: sm_90", {}),
            ("ok", "Incorrect native cubin architecture for sm_90: 999",
             {"FAKE_BAD_SM": "1"}),
            ("ok", "Incorrect native cubin exports for sm_90",
             {"FAKE_DROP_NATIVE": "1"}),
        ]
        for index, (mode, expected, extra) in enumerate(cases):
            with self.subTest(case=mode, extra=extra):
                output = self.tmp / f"failure-{index}-out"
                record = self.tmp / f"failure-{index}.jsonl"
                stale, marker = self.stale_fixture(output)
                result = self.run_build(
                    output, record, jobs=1, extra={"FAKE_MODE": mode, **extra})
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stdout)
                self.assertFalse((output / "manifest.json").exists())
                if stale.exists():
                    self.assertNotIn(marker, stale.read_text())
                published = output / "sm_90/kernels.ptx"
                if published.exists():
                    self.assertNotIn(marker, published.read_text())
                self.assertFalse((output / "sm_90/kernels.cubin").exists())
                self.assertEqual(list(output.glob(".manifest.*")), [])

    def test_concurrent_outputs_serialize_and_source_stays_untouched(self):
        before = source_digest(DEVICE)
        main_before = (DEVICE / "src/main.rs").stat()
        first_out, second_out = self.tmp / "concurrent-a", self.tmp / "concurrent-b"
        first_rec, second_rec = self.tmp / "concurrent-a.jsonl", self.tmp / "concurrent-b.jsonl"
        processes = [
            subprocess.Popen(
                ["bash", str(BUILD), str(output)], cwd=str(DEVICE),
                env=self.build_env(record, 2),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for output, record in ((first_out, first_rec), (second_out, second_rec))
        ]
        outputs = [process.communicate(timeout=120)[0] for process in processes]
        for process, text in zip(processes, outputs):
            self.assertEqual(process.returncode, 0, text)
        combined = read_intervals(first_rec) + read_intervals(second_rec)
        self.assertLessEqual(max_concurrency(combined), 2)
        self.assert_manifest(first_out)
        self.assert_manifest(second_out)
        self.assertEqual(source_digest(DEVICE), before)
        main_after = (DEVICE / "src/main.rs").stat()
        self.assertEqual((main_after.st_mtime_ns, main_after.st_size),
                         (main_before.st_mtime_ns, main_before.st_size))

    def test_local_config_keeps_original_cwd_and_forces_sequential(self):
        tree = self.tmp / "config-tree/libs/infinity-v3/backends/oxide/device"
        shutil.copytree(
            DEVICE, tree,
            ignore=shutil.ignore_patterns(".oxide-artifacts", "target",
                                          "*.ptx", "*.ll", "*.cubin", "__pycache__"))
        shutil.copy2(DEVICE.parent / "NOTICE", tree.parent / "NOTICE")
        shutil.copy2(DEVICE.parent / "NOTICE.qwen35", tree.parent / "NOTICE.qwen35")
        (tree / ".cargo").mkdir()
        (tree / ".cargo/cuda-oxide.toml").write_text(
            '[env]\nOXIDE_LOCAL_CONFIG_MARKER = "visible"\n')
        output = self.tmp / "config-out"
        record = self.tmp / "config.jsonl"
        result = self.run_build(output, record, jobs=3, source=tree)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("Detected device/.cargo/cuda-oxide.toml", result.stdout)
        intervals = read_intervals(record)
        self.assertEqual(max_concurrency(intervals), 1)
        starts = read_starts(record)
        self.assertEqual(len(starts), 3)
        for event in starts:
            self.assertTrue(event["cwd_config"])
            self.assertEqual(event["cwd"], str(tree))
            self.assertIn("OXIDE_LOCAL_CONFIG_MARKER", event["cwd_config_text"])
        self.assert_manifest(output)

    def test_missing_qwen_notice_fails_before_compiler(self):
        tree = self.tmp / "missing-notice/libs/infinity-v3/backends/oxide/device"
        shutil.copytree(DEVICE, tree, ignore=shutil.ignore_patterns(
            ".oxide-artifacts", "target", "*.ptx", "*.ll", "*.cubin", "__pycache__"))
        shutil.copy2(DEVICE.parent / "NOTICE", tree.parent / "NOTICE")
        output = self.tmp / "missing-notice-output"
        record = self.tmp / "missing-notice-compiler.jsonl"
        result = self.run_build(output, record, source=tree, architectures=["sm_90"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("NOTICE.qwen35", result.stdout)
        self.assertFalse(record.exists())
        self.assertFalse((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()

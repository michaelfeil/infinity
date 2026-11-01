import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import build

PACKAGE = Path(__file__).resolve().parent
GOOD_URL = "https://raw.githubusercontent.com/pytorch/pytorch/{}/torch/headeronly/util/Half.h"


class FakeResponse:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def read(self, size=-1):
        if self.pos >= len(self.data):
            return b""
        end = len(self.data) if size < 0 else self.pos + size
        chunk = self.data[self.pos:end]
        self.pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class Opener:
    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def __call__(self, url, timeout=60):
        self.calls.append(url)
        return FakeResponse(self.mapping[url])


def entry(data, url="https://raw.githubusercontent.com/pytorch/pytorch/x/file.h"):
    return {"sha256": hashlib.sha256(data).hexdigest(), "url": url}


def write_manifest(path, files, pytorch=build.PYTORCH_COMMIT, cutlass=build.CUTLASS_COMMIT):
    path.write_text(json.dumps({
        "pytorch_commit": pytorch,
        "cutlass_commit": cutlass,
        "files": files,
    }))


def manifest_file(tmp, files):
    path = tmp / "manifest.json"
    write_manifest(path, files)
    return path


def make_package(tmp, header="bias_table_ptr relative_bucket"):
    package = tmp / "package"
    package.mkdir()
    (package / build.ADAPTER).write_text("adapter")
    (package / "kernel_forward.h").write_text(header)
    return package


def source_file(rel, data):
    return build.SourceFile(rel, hashlib.sha256(data).hexdigest(), GOOD_URL.format("x"))


class BuildUtilityTests(unittest.TestCase):
    def test_staged_manifest_is_exact(self):
        manifest = build.load_manifest(PACKAGE / "public-source-manifest.json")
        self.assertEqual(len(manifest.files), build.EXPECTED_FILE_COUNT)
        self.assertEqual(manifest.pytorch_commit, build.PYTORCH_COMMIT)
        self.assertEqual(manifest.cutlass_commit, build.CUTLASS_COMMIT)

    def test_valid_subset_manifest(self):
        data = b"half"
        rel = "pytorch/torch/headeronly/util/Half.h"
        with tempfile.TemporaryDirectory() as tmp:
            path = manifest_file(Path(tmp), {rel: entry(data)})
            manifest = build.load_manifest(path, allow_subset=True)
            self.assertEqual(set(manifest.files), {rel})
            self.assertEqual(manifest.files[rel].sha256, hashlib.sha256(data).hexdigest())

    def test_missing_and_corrupt_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(build.BuildError):
                build.load_manifest(root / "missing.json", allow_subset=True)
            bad = root / "bad.json"
            bad.write_text("{")
            with self.assertRaises(build.BuildError):
                build.load_manifest(bad, allow_subset=True)

    def test_manifest_format_rejections(self):
        sha = "a" * 64
        rel = "pytorch/x.h"
        documents = [
            {},
            {"pytorch_commit": build.PYTORCH_COMMIT, "cutlass_commit": build.CUTLASS_COMMIT, "files": {}},
            {"pytorch_commit": "bad", "cutlass_commit": build.CUTLASS_COMMIT, "files": {rel: {"sha256": sha}}},
            {"pytorch_commit": build.PYTORCH_COMMIT, "cutlass_commit": build.CUTLASS_COMMIT,
             "files": {rel: {"sha256": "bad"}}},
            {"pytorch_commit": build.PYTORCH_COMMIT, "cutlass_commit": build.CUTLASS_COMMIT,
             "files": {rel: {"sha256": sha, "url": 7}}},
            {"pytorch_commit": build.PYTORCH_COMMIT, "cutlass_commit": build.CUTLASS_COMMIT,
             "files": {"../x": {"sha256": sha}}},
        ]
        for document in documents:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "manifest.json"
                path.write_text(json.dumps(document))
                with self.subTest(document=document), self.assertRaises(build.BuildError):
                    build.load_manifest(path, allow_subset=True)

    def test_relative_path_validation(self):
        for rel in ("/abs", "../x", "a/../b", "a//b", "a/./b", "a\\b", "C:/x", "", "."):
            with self.subTest(rel=rel), self.assertRaises(build.BuildError):
                build.validate_relative_path(rel)
        self.assertEqual(build.validate_relative_path("pytorch/a.h"), "pytorch/a.h")

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            outside = Path(tmp) / "outside.h"
            outside.write_text("x")
            (root / "link.h").symlink_to(outside)
            with self.assertRaisesRegex(build.BuildError, "symlink"):
                build.resolve_source_file(root, "link.h")
            (root / "dir").symlink_to(Path(tmp))
            with self.assertRaisesRegex(build.BuildError, "symlink"):
                build.resolve_source_file(root, "dir/outside.h")

    def test_verify_source_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            rel = "pytorch/a.h"
            data = b"data"
            manifest = build.Manifest(build.PYTORCH_COMMIT, build.CUTLASS_COMMIT,
                                      {rel: source_file(rel, data)})
            with self.assertRaisesRegex(build.BuildError, "missing"):
                build.verify_source_tree(root, manifest)
            path = root / rel
            path.parent.mkdir(parents=True)
            path.write_bytes(b"bad")
            with self.assertRaisesRegex(build.BuildError, "hash mismatch"):
                build.verify_source_tree(root, manifest)
            path.write_bytes(data)
            self.assertEqual(build.verify_source_tree(root, manifest)[rel], manifest.files[rel].sha256)

    def test_fetch_only_missing_and_verifies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            rel1, rel2 = "pytorch/a.h", "pytorch/b.h"
            data1, data2 = b"one", b"two"
            path1 = root / rel1
            path1.parent.mkdir(parents=True)
            path1.write_bytes(data1)
            url2 = "https://raw.githubusercontent.com/pytorch/pytorch/x/" + rel2
            manifest = build.Manifest(build.PYTORCH_COMMIT, build.CUTLASS_COMMIT, {
                rel1: source_file(rel1, data1),
                rel2: build.SourceFile(rel2, hashlib.sha256(data2).hexdigest(), url2),
            })
            opener = Opener({url2: data2})
            self.assertEqual(build.fetch_missing(root, manifest, opener=opener), 1)
            self.assertEqual((root / rel2).read_bytes(), data2)
            self.assertEqual(opener.calls, [url2])

    def test_fetch_hash_failure_is_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            rel = "pytorch/a.h"
            url = "https://raw.githubusercontent.com/pytorch/pytorch/x/" + rel
            manifest = build.Manifest(build.PYTORCH_COMMIT, build.CUTLASS_COMMIT, {
                rel: build.SourceFile(rel, hashlib.sha256(b"good").hexdigest(), url),
            })
            opener = Opener({url: b"bad"})
            with self.assertRaisesRegex(build.BuildError, "hash mismatch"):
                build.fetch_missing(root, manifest, opener=opener)
            self.assertFalse((root / rel).exists())
            self.assertEqual(list(root.rglob(".mpnet-fetch-*")), [])

    def test_fetch_refuses_corrupt_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            rel = "pytorch/a.h"
            path = root / rel
            path.parent.mkdir(parents=True)
            path.write_bytes(b"bad")
            url = "https://raw.githubusercontent.com/pytorch/pytorch/x/" + rel
            manifest = build.Manifest(build.PYTORCH_COMMIT, build.CUTLASS_COMMIT, {
                rel: build.SourceFile(rel, hashlib.sha256(b"good").hexdigest(), url),
            })
            opener = Opener({url: b"good"})
            with self.assertRaisesRegex(build.BuildError, "corrupt"):
                build.fetch_missing(root, manifest, opener=opener)
            self.assertEqual(opener.calls, [])
            self.assertEqual(path.read_bytes(), b"bad")

    def test_fetch_rejects_unpinned_url(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            rel = "pytorch/a.h"
            manifest = build.Manifest(build.PYTORCH_COMMIT, build.CUTLASS_COMMIT, {
                rel: build.SourceFile(rel, hashlib.sha256(b"x").hexdigest(), "https://example.com/x"),
            })
            opener = Opener({})
            with self.assertRaisesRegex(build.BuildError, "raw.githubusercontent.com"):
                build.fetch_missing(root, manifest, opener=opener)
            self.assertEqual(opener.calls, [])

    def test_cutlass_url_derivation_and_url_validation(self):
        rel = "cutlass/include/cutlass/bfloat16.h"
        manifest = build.Manifest(build.PYTORCH_COMMIT, build.CUTLASS_COMMIT,
                                  {rel: build.SourceFile(rel, "a" * 64)})
        self.assertEqual(
            build.source_url(manifest.files[rel], manifest),
            "https://raw.githubusercontent.com/NVIDIA/cutlass/"
            + build.CUTLASS_COMMIT + "/include/cutlass/bfloat16.h",
        )
        with self.assertRaises(build.BuildError):
            build.source_url(build.SourceFile("pytorch/a.h", "a" * 64), manifest)
        with self.assertRaises(build.BuildError):
            build.validate_download_url("https://codeload.github.com/x")
        with self.assertRaises(build.BuildError):
            build.validate_download_url("http://raw.githubusercontent.com/x")

    def test_fetch_byte_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            rel = "pytorch/a.h"
            payload = b"x" * 32
            url = "https://raw.githubusercontent.com/pytorch/pytorch/x/" + rel
            manifest = build.Manifest(build.PYTORCH_COMMIT, build.CUTLASS_COMMIT, {
                rel: build.SourceFile(rel, hashlib.sha256(payload).hexdigest(), url),
            })
            opener = Opener({url: payload})
            with self.assertRaisesRegex(build.BuildError, "byte limit"):
                build.fetch_missing(root, manifest, opener=opener, max_bytes=8)
            self.assertFalse((root / rel).exists())
            self.assertEqual(list(root.rglob(".mpnet-fetch-*")), [])

    def test_fetch_rejects_symlink_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            outside = Path(tmp) / "outside"
            outside.mkdir()
            (root / "pytorch").symlink_to(outside)
            rel = "pytorch/a.h"
            manifest = build.Manifest(build.PYTORCH_COMMIT, build.CUTLASS_COMMIT, {
                rel: build.SourceFile(rel, hashlib.sha256(b"x").hexdigest(),
                                      "https://raw.githubusercontent.com/pytorch/pytorch/x/" + rel),
            })
            with self.assertRaisesRegex(build.BuildError, "symlink"):
                build.fetch_missing(root, manifest, opener=Opener({}))

    def test_default_check_only_uses_no_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "root"
            root.mkdir()
            package = make_package(Path(tmp))
            rel = "pytorch/a.h"
            data = b"data"
            path = root / rel
            path.parent.mkdir(parents=True)
            path.write_bytes(data)
            manifest_path = manifest_file(Path(tmp), {rel: entry(data)})
            with mock.patch.object(build.urllib.request, "urlopen",
                                   side_effect=AssertionError("network")):
                result = build.check_only(package, manifest_path=manifest_path,
                                          source_root=root, allow_subset=True)
            self.assertFalse(result["network_used"])
            self.assertFalse(result["cuda_used"])
            self.assertEqual(result["verified_files"], 1)

    def test_cli_check_only_defaults(self):
        with mock.patch.object(build.urllib.request, "urlopen",
                               side_effect=AssertionError("network")):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(build.main(["--check-only"]), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["mode"], "check-only")
        self.assertFalse(result["network_used"])
        self.assertFalse(result["cuda_used"])

    def test_check_only_requires_patched_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            package = make_package(Path(tmp), header="stock header")
            manifest_path = manifest_file(Path(tmp), {
                "pytorch/a.h": entry(b"data"),
            })
            with self.assertRaisesRegex(build.BuildError, "bucket-bias"):
                build.check_only(package, manifest_path=manifest_path, allow_subset=True)

    def test_architectures(self):
        self.assertEqual(build.parse_architectures(["90,100", "120"]), [90, 100, 120])
        for value in ([], ["80"], ["90,80"], ["abc"]):
            with self.subTest(value=value), self.assertRaises(build.BuildError):
                build.parse_architectures(value)

    def test_fresh_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "out"
            existing.mkdir()
            (existing / "x").write_text("x")
            with self.assertRaisesRegex(build.BuildError, "fresh"):
                build.require_fresh_output(existing)
            empty = Path(tmp) / "empty"
            empty.mkdir()
            self.assertEqual(build.require_fresh_output(empty), empty)
            fresh = Path(tmp) / "fresh"
            self.assertEqual(build.require_fresh_output(fresh), fresh)
            self.assertTrue(fresh.is_dir())

    def test_elf_audit_rejects_forbidden_dependencies_and_symbols(self):
        dynamic = " 0x1 (NEEDED) Shared library: [libcuda.so.1]\n"
        undefined = "                 U pthread_create\n"
        defined = "\n".join(f"0000000000000000 T {name}" for name in build.EXPORTS)
        result = build.audit_text(dynamic, undefined, defined)
        self.assertIn("libcuda.so.1", result["needed"])
        with self.assertRaisesRegex(build.BuildError, "ELF dependencies"):
            build.audit_text(dynamic.replace("libcuda.so.1", "libc10.so"), undefined, defined)
        with self.assertRaisesRegex(build.BuildError, "undefined symbols"):
            build.audit_text(dynamic, "                 U at::Tensor", defined)
        with self.assertRaisesRegex(build.BuildError, "exported symbols"):
            build.audit_text(dynamic, undefined, defined.replace(build.EXPORTS[0], "other"))

    def test_manifest_layout(self):
        rel = "pytorch/a.h"
        manifest = build.Manifest(build.PYTORCH_COMMIT, build.CUTLASS_COMMIT,
                                  {rel: build.SourceFile(rel, "a" * 64)})
        document = build.make_arch_manifest(
            arch=90, architectures=build.ARCHITECTURES, manifest=manifest,
            library_sha="b" * 64, command=["nvcc", "x"], tool_version="v",
            provenance={"manifest_sha256": "c" * 64}, cuda_toolkit="/cuda",
            licenses=["licenses/LICENSE-PyTorch-BSD3", "licenses/LICENSE-CUTLASS-BSD3"],
        )
        self.assertEqual(document["abi_version"], 1)
        self.assertEqual(document["dtype"], "bf16")
        self.assertEqual(document["num_heads"], 12)
        self.assertEqual(document["head_dim"], 64)
        self.assertEqual(document["qkv_row_stride"], 2304)
        self.assertEqual(document["bias_num_buckets"], 32)
        self.assertFalse(document["causal"])
        self.assertEqual(document["architecture"], 90)
        self.assertEqual(document["architectures"], [90, 100, 120])
        self.assertEqual(document["symbols"], dict(zip(("prepare","validate","launch","smem_bytes"),build.EXPORTS)))
        self.assertEqual(document["library_sha256"], "b" * 64)
        self.assertEqual(document["source"]["pytorch_commit"], build.PYTORCH_COMMIT)
        self.assertFalse(document["qualification"]["numerical"])
        self.assertFalse(document["qualification"]["performance"])

    def test_install_patch_tree_uses_public_relative_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = make_package(root)
            output = root / "out"
            output.mkdir()
            dest = build.install_patch_tree(package, output)
            self.assertEqual(dest, output / build.PATCH_HEADER)
            self.assertEqual(dest.read_text(), "bias_table_ptr relative_bucket")

    def test_install_licenses_copies_expected_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = make_package(root)
            public = root / "public"
            (public / "pytorch").mkdir(parents=True)
            (public / "cutlass").mkdir()
            (public / "pytorch" / "LICENSE").write_text("pytorch license")
            (public / "cutlass" / "LICENSE.txt").write_text("cutlass license")
            output = root / "out"
            output.mkdir()
            names = build.install_licenses(package, public, output)
            self.assertEqual(names, ["licenses/LICENSE-PyTorch-BSD3",
                                     "licenses/LICENSE-CUTLASS-BSD3"])
            self.assertEqual((output / "licenses" / "LICENSE-PyTorch-BSD3").read_text(),
                             "pytorch license")
            self.assertEqual((output / "licenses" / "LICENSE-CUTLASS-BSD3").read_text(),
                             "cutlass license")
            self.assertTrue((output / "NOTICE").is_file())
            self.assertTrue((output / "licenses" / "NOTICE").is_file())

    def test_build_command_flags(self):
        command = build.build_command(
            "/cuda/bin/nvcc", 100, "/out", "/public",
            "/out/src/standalone_mpnet_attention.cu", "/out/sm_100/libmpnet_attention.so",
        )
        for flag in ("-std=c++17", "-O3", "-shared", "--cudart=shared",
                     "-Xcompiler", "-fPIC", "-Xlinker", "-z", "defs",
                     "-DMPNET_ATTENTION_HAVE_BUCKET_BIAS=1",
                     "-DC10_USING_CUSTOM_GENERATED_MACROS", "-arch=sm_100", "-lcuda"):
            self.assertIn(flag, command)
        for include in ("/out/pytorch/aten/src", "/public/pytorch/aten/src",
                        "/public/pytorch", "/public/cutlass/include"):
            self.assertIn(include, command)

    def test_build_requires_source_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = make_package(root)
            manifest_path = manifest_file(root, {"pytorch/a.h": entry(b"data")})
            with self.assertRaisesRegex(build.BuildError, "source-root"):
                build.run_build(
                    package_dir=package, manifest_path=manifest_path,
                    source_root=None, output=root / "out",
                    cuda_toolkit=build.DEFAULT_CUDA, arches=[90],
                    allow_subset=True,
                )

    def test_exact_host_provider_abi_manifest_contract(self):
        manifest = build.load_manifest(PACKAGE / "public-source-manifest.json")
        d = build.make_arch_manifest(arch=90,architectures=(90,100,120),manifest=manifest,library_sha="a"*64,command=[],tool_version="test",provenance={},cuda_toolkit="/cuda",licenses=[])
        self.assertEqual(d["dtype"],"bf16")
        self.assertEqual(set(d["source"]),{"pytorch_commit","cutlass_commit"})
        self.assertEqual(d["symbols"],dict(zip(("prepare","validate","launch","smem_bytes"),build.EXPORTS)))
        self.assertEqual(d["library_sha256"],"a"*64)


if __name__ == "__main__":
    unittest.main()

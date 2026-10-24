# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
"""Exercise the host ISA audit without importing CUDA, PyTorch or CuTe."""
import importlib.util
import platform
import subprocess
import tempfile
import unittest
from pathlib import Path

vendor = Path(__file__).resolve().parents[1] / "backends/libtorch/cpp/vendor/infinity_fa4"
spec = importlib.util.spec_from_file_location("portable_host", vendor / "portable_host.py")
portable_host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(portable_host)


@unittest.skipUnless(platform.machine() == "x86_64", "x86 instruction encodings")
class HostAuditTests(unittest.TestCase):
    def test_instruction_encodings(self):
        # Assemble, but never execute: CI need not support AVX or AVX-512.
        cases = {
            "baseline": ("pxor %xmm0, %xmm0", False),
            "vex": ("vxorps %xmm0, %xmm0, %xmm0", True),
            "evex": ("vmovaps %zmm0, %zmm1", True),
            "masked_xmm": ("vmovaps %xmm0, %xmm1{%k1}", True),
            # A prefix-looking immediate byte is not an instruction prefix.
            "immediate": ("movabs $0x6262626262626262, %rax", False),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, (instruction, rejected) in cases.items():
                with self.subTest(name=name):
                    obj = Path(directory) / (name + ".o")
                    subprocess.run(
                        ["cc", "-x", "assembler", "-c", "-o", str(obj), "-"],
                        input=f".text\n{instruction}\nret\n", text=True, check=True,
                        capture_output=True,
                    )
                    if rejected:
                        with self.assertRaisesRegex(RuntimeError, "Nonportable AVX"):
                            portable_host.validate_host_object(obj)
                    else:
                        portable_host.validate_host_object(obj)


if __name__ == "__main__":
    unittest.main()

# Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
"""Export baseline host code without changing the embedded GPU binary."""
import platform
import re
import subprocess


def host_target():
    targets = {
        "x86_64": ("x86-64", "+sse2,-avx,-avx2,-avx512f"),
        "aarch64": ("generic", "+neon"),
    }
    machine = platform.machine()
    if machine not in targets:
        raise RuntimeError(f"No portable FA4 host target for {machine}")
    cpu, features = targets[machine]
    return dict(architecture=machine, cpu=cpu, features=features)


def export_portable(compiled, path, symbol):
    # CuTe 4.7.1 rejects --host-target on its TVM-FFI export path. Set the
    # LLVM function attributes on its exported host module instead; otherwise
    # the exporter silently specializes device-load code to the build CPU.
    from cutlass._mlir import ir
    from cutlass._mlir._mlir_libs._cutlass_ir import _aot_support
    from cutlass.base_dsl.export import get_export_module
    from cutlass.base_dsl.tvm_ffi_builder import rename_tvm_ffi_function

    module = get_export_module(
        compiled.ir_module, "__cute_internal_" + symbol,
        preserve_symbols={f"__tvm_ffi_{compiled.function_name}"},
    )
    rename_tvm_ffi_function(module, compiled.function_name, symbol)
    target = host_target()
    with module.context:
        features = ", ".join(f'"{f}"' for f in target["features"].split(","))
        for op in module.body.operations:
            if op.name == "llvm.func":
                op.attributes["target_cpu"] = ir.StringAttr.get(target["cpu"])
                op.attributes["target_features"] = ir.Attribute.parse(
                    f"#llvm.target_features<[{features}]>"
                )
    path.write_bytes(_aot_support.export_module_to_bytes(
        module, format="o", opt_level=3, enable_pic=True,
    ))


def validate_host_object(path):
    if host_target()["architecture"] != "x86_64":
        return
    # VEX/EVEX prefixes identify AVX-family instructions, including masked
    # XMM operations that a search for YMM/ZMM registers alone would miss.
    disassembly = subprocess.check_output(["objdump", "-d", "--insn-width=16", str(path)], text=True)
    # Address-size and segment overrides can precede VEX/EVEX. Match only
    # instruction prefixes, not prefix-looking immediate or displacement bytes.
    legacy = r"(?:(?:26|2e|36|3e|64|65|66|67|f0|f2|f3)\s+)*"
    if re.search(r"^\s*[0-9a-f]+:\s+" + legacy + r"(?:c4|c5|62)\s", disassembly, re.MULTILINE):
        raise RuntimeError(f"Nonportable AVX host instructions in {path}")

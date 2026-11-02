"""Execute the exact pinned pure option closures without importing Torch."""
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "native-control-v2" / "input" / "native_base.py"
GOLD = ROOT / "native-control-v2" / "native_gold.py"
MODEL = Path("/workspace/model-performance/michaelfeil/onejev-perf/models/qwen35-35b-a3b")


@contextmanager
def option_environment(values, names):
    saved = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ.pop(name, None)
        os.environ.update(values)
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main():
    assert "torch" not in sys.modules
    parsed = ast.parse(SOURCE.read_bytes(), filename=str(SOURCE))
    wanted = {"env_flag", "env_int", "env_byte_size", "build_options"}
    functions = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    pool = [node for node in parsed.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "POOL_OPTIONS" for target in node.targets)]
    assert {node.name for node in functions} == wanted and len(pool) == 1
    # Preserve the actual source AST bodies and locations; no parser/test double.
    module = types.ModuleType("native_base")
    module.__dict__.update(os=os, json=json)
    exec(compile(ast.Module(body=pool + functions, type_ignores=[]), str(SOURCE), "exec"), module.__dict__)
    old = sys.modules.get("native_base")
    sys.modules["native_base"] = module
    sys.path.insert(0, str(GOLD.parent))
    try:
        spec = importlib.util.spec_from_file_location("q35_gold_parser_source", GOLD)
        gold = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gold)
        config = json.loads((MODEL / "config.json").read_bytes())
        names = {node.value for node in ast.walk(parsed) if isinstance(node, ast.Constant)
                 and isinstance(node.value, str) and node.value.startswith("INFINITY_TORCH_")}
        common = {"INFINITY_TORCH_CUDA_GRAPHS": "1", "INFINITY_TORCH_FP8_SCOPE": "mlp"}
        with option_environment({**common, "INFINITY_TORCH_CUDA_GRAPH_MAX_BYTES": "512MiB"}, names):
            try:
                gold.build_pool_options(config, "mean")
            except ValueError as error:
                assert "INFINITY_TORCH_CUDA_GRAPH_MAX_BYTES" in str(error)
            else:
                raise AssertionError("actual integer parser wrongly accepted 512MiB")
        with option_environment({**common, "INFINITY_TORCH_CUDA_GRAPH_MAX_BYTES": str(512 << 20)}, names):
            options = gold.build_pool_options(config, "mean")
        assert options["_cuda_graphs"] is True and options["_fp8_dynamic"] is False
        for key, value in {"_cuda_graph_max_bytes": 512 << 20, "_cuda_graph_max_tokens": 4096,
                           "_cuda_graph_entries": 4, "_cuda_graph_reuse": 3,
                           "_cuda_graph_max_captures": 8}.items():
            assert options[key] == value, key
        assert options["text_config"] == config["text_config"]
        assert "torch" not in sys.modules
        report = {"schema": "q35moe-native-options-source-regression/v1", "status": "passed",
                  "execution": "verbatim AST pure closures from pinned native_base plus actual native_gold.build_pool_options",
                  "parser_mocked": False, "native_base_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  "native_gold_sha256": hashlib.sha256(GOLD.read_bytes()).hexdigest(),
                  "numeric_max_bytes": options["_cuda_graph_max_bytes"], "old_unit_string_rejected": True,
                  "default_tokens_entries_reuse_captures": [4096, 4, 3, 8],
                  "torch_imported": False, "engine_created": False, "gpu_executed": False}
        (ROOT / "native-options-parser-receipt-v1.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        if old is None:
            sys.modules.pop("native_base", None)
        else:
            sys.modules["native_base"] = old


if __name__ == "__main__":
    main()

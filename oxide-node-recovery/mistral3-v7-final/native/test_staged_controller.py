#!/usr/bin/env python3
"""CPU-only source tests for the staged controller command construction."""
import os, sys, unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import root_staged_controller as controller
import container_sdk_got as got

class StagedControllerSourceTests(unittest.TestCase):
    def test_allowlist_stages_exact_bridge_and_runner_header(self):
        self.assertIn("include/bridge.h", controller.ALLOWLIST)
        self.assertIn("src/mistral3_runner.cpp", controller.ALLOWLIST)
        runner = (HERE / "src/mistral3_runner.cpp").read_text()
        self.assertIn('#include "bridge.h"', runner)
        self.assertNotIn('../../libs/infinity-v3/backends/libtorch/cpp/bridge.h', runner)

    def test_docker_uses_explicit_socket_and_preserves_proxy_ca_auth(self):
        command = controller.docker_argv("exec", "container", "true")
        self.assertEqual(command[:3], ["docker", "--host=unix:///var/run/docker.sock", "exec"])
        base = {
            "DOCKER_HOST": "tcp://wrong:2376",
            "DOCKER_CONTEXT": "wrong",
            "DOCKER_TLS": "1",
            "DOCKER_TLS_VERIFY": "1",
            "DOCKER_CERT_PATH": "/wrong",
            "DOCKER_CONFIG": "/auth/config",
            "HTTP_PROXY": "http://proxy",
            "HTTPS_PROXY": "http://proxy",
            "SSL_CERT_FILE": "/ca.pem",
        }
        env = controller.docker_env(base)
        for key in controller.DOCKER_CLEAR:
            self.assertNotIn(key, env)
        self.assertEqual(env["DOCKER_CONFIG"], "/auth/config")
        self.assertEqual(env["HTTP_PROXY"], "http://proxy")
        self.assertEqual(env["HTTPS_PROXY"], "http://proxy")
        self.assertEqual(env["SSL_CERT_FILE"], "/ca.pem")

    def test_container_python_and_runtime_libraries_are_actual(self):
        self.assertEqual(controller.CONTAINER_PYTHON, "/opt/torch-sdk/bin/python3.12")
        env = controller.runtime_env("/sdk/libtei_torch.so", "/vendor", "0")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(env["LD_BIND_NOW"], "1")
        self.assertIn("/vendor", env["LD_LIBRARY_PATH"])
        self.assertIn("/sdk", env["LD_LIBRARY_PATH"])
        self.assertIn("/opt/torch-sdk/lib", env["LD_LIBRARY_PATH"])
        self.assertIn("/vendor/libcublasLt.so.13", env["LD_PRELOAD"])
        self.assertIn("/vendor/libcublas.so.13", env["LD_PRELOAD"])

    def test_build_and_got_commands_use_staged_include_and_actual_python(self):
        build = controller.build_runner_argv("/work", "/sdk/libtei_torch.so")
        self.assertIn("-I/work/include", build)
        self.assertIn("/work/src/mistral3_runner.cpp", build)
        self.assertIn("-ltei_torch", build)
        got = controller.got_argv("/work", "/sdk/libtei_torch.so", "abc", "/vendor")
        self.assertEqual(got[0], controller.CONTAINER_PYTHON)
        self.assertIn("--bridge-header", got)
        self.assertIn("/work/include/bridge.h", got)

    def test_bridge_header_reports_the_actual_19_arg_tei_forward(self):
        self.assertEqual(
            got.bridge_forward_args(HERE / "include" / "bridge.h"),
            19,
        )

    def test_host_never_uses_bench_and_gpu_stage_is_explicit(self):
        self.assertFalse(controller.HOST_MODEL_DEFAULT.startswith("/bench"))
        self.assertTrue(controller.CONTAINER_MODEL_DEFAULT.startswith("/bench"))
        self.assertFalse(controller.stage_requires_gpu("cpu"))
        self.assertTrue(controller.stage_requires_gpu("quality"))
        self.assertTrue(controller.stage_requires_gpu("perf"))

if __name__ == "__main__":
    unittest.main()

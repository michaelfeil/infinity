"""Check a built server image before publishing it; inference runs on CPU."""

import argparse
import json
import math
from pathlib import Path
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
import uuid


def docker(*args):
    return subprocess.check_output(["docker", *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--model-dir", type=Path, help="Use a local MiniLM checkpoint offline")
    args = parser.parse_args()
    name = f"infinity-release-smoke-{uuid.uuid4().hex}"
    command = ["run", "--detach", "--name", name, "-p", "127.0.0.1::8000"]
    if args.model_dir:
        command += ["--mount", f"type=bind,src={args.model_dir.resolve()},dst=/model,readonly"]
        model = "/model"
        revision = []
    else:
        model = "sentence-transformers/all-MiniLM-L6-v2"
        revision = ["--revision", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"]
    command += [
        args.image, "--model-id", model, *revision, "--served-model-name", "smoke",
        "--torch-device", "cpu", "--dtype", "float32", "--max-batch-tokens", "512",
        "--hostname", "0.0.0.0", "--port", "8000",
    ]
    try:
        docker(*command)
        address = docker("port", name, "8000/tcp")
        base = f"http://{address}"
        deadline = time.monotonic() + 300
        while True:
            try:
                with urlopen(f"{base}/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError, ConnectionError):
                pass
            if docker("inspect", "--format", "{{.State.Running}}", name) != "true":
                raise RuntimeError("Server exited before readiness")
            if time.monotonic() >= deadline:
                raise TimeoutError("Server did not become ready within five minutes")
            time.sleep(1)
        with urlopen(f"{base}/info", timeout=10) as response:
            info = json.load(response)
        if info["version"] != args.expected_version:
            raise ValueError(f"Unexpected server version: {info['version']}")
        payload = {"model": "smoke", "input": ["A small embedding test.", "A different sentence."]}
        request = Request(
            f"{base}/v1/embeddings", json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=30) as response:
            result = json.load(response)
        vectors = [entry["embedding"] for entry in result["data"]]
        if len(vectors) != 2 or any(
            len(vector) != 384 or not all(math.isfinite(x) for x in vector)
            or sum(x * x for x in vector) == 0 for vector in vectors
        ) or vectors[0] == vectors[1]:
            raise ValueError("Invalid MiniLM embeddings")
        print(f"{args.image}: version {args.expected_version}, readiness and embeddings passed")
    finally:
        subprocess.run(["docker", "logs", name], check=False)
        subprocess.run(["docker", "rm", "--force", name], check=False)


if __name__ == "__main__":
    main()

# Qwen3-30B-A3B paired LibTorch vs public Oxide performance harness V2

Source-only harness against public commit `252789b05ea2e09c14971c04dfcce458f863fead`.
Author ran no GPU/Torch/Docker/device/AOT/optimized build.

## Environment correction

- Native SDK, Torch Python and vendor libs exist ONLY in Docker container
  `infinity-torch214-compare-sdk`:
  - SDK `/tmp/native214-fa4-mode20/install/lib/libtei_torch.so` SHA `b66d70b6…`
  - Python `/opt/torch-sdk/bin/python3.12`
  - vendor `/tmp/oxide-host-cublas-130200` (`libcublas.so.13`, `libcublasLt.so.13`)
- Host model `/tmp/infinity-oxide-bench/models/Qwen--Qwen3-30B-A3B-ad44e777`
  is the qualified Oxide bundle.
- Native uses container model `/bench/michaelfeil/oxide-staged-qwen3-moe-ad44e777`
  from the V8 Native report. Weights are not copied or flattened.
- Native support scripts/helper/cases are staged into the container with
  `docker cp`; cleanup runs in every `finally` path.
- Docker CLI uses explicit `--host=unix:///var/run/docker.sock`. Exactly five
  selectors are cleared (`DOCKER_HOST`, `DOCKER_CONTEXT`, `DOCKER_TLS_VERIFY`,
  `DOCKER_CERT_PATH`, `DOCKER_API_VERSION`); `DOCKER_CONFIG`, proxy variables
  and SSL/CA bundle names are preserved.
- The remote guard records host docker-exec pid/start time/argv/deadline and
  cleans the unique remote directory in `finally`.

## Oxide host runtime

- Host vendor `/usr/local/cuda-13.1/lib64` is checked as cublasLt 130200.
- `--kernel-root /tmp/oxide-moe-root629-v8-public126-bundle-v1` already contains
  the qualified `moe/` sidecar; the controller verifies its manifest and DSOs.
  No `INFINITY_OXIDE_MOE_KERNELS` override is used. Oxide env only sets
  `INFINITY_OXIDE_KERNEL_MODULE` and `INFINITY_FA4_ROOT`.
- Oxide graphs are off; Native graphs/defaults are preserved.

## Workloads and timing

T7 then ragged129, then stable2k if fit. Balanced `N,O,O,N`, one engine at a
time, GPU0. First call separate; ABA variable-shape warm separate; 20 warmups;
200 timed. Both Native and Oxide timed sections include canonical result pack
and hash; quality validation is outside the hot loop. Larger timing requires
`--justify-large`.

`--skip-stable2k` now filters the case before quality-reference construction, so
it no longer requires a stable2k reference when stable2k is skipped.

Native graph eligibility/counters are recorded. Replay hits are not forced to be
positive if the SDK/model is legitimately ineligible; counters are never
suppressed.

## Quality

T7/ragged129 exact against existing Native mean gold; stable2k uses the first
Native window as reference. Every window/backend must match exact bits.

## Files

- `oxide-perf/` Rust normal-backend timing consumer
- `native-perf/native_perf_process.py` container native timing process
- `controller/run_paired_v2.py` balanced controller
- `PERF-MANIFEST.json`, `SOURCE-FREEZE.json`, `CASES-SOURCE.json`
- `evidence/` CPU preflight, Rust tests, controller tests, clippy/fmt logs

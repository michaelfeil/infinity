# Qwen3.5-MoE normal factory successor

Isolated source successor on the current composed Qwen35+Dense+Qwen3Moe worktree.
The frozen baseline is untouched.

- `init_backend` detects `qwen3_5_moe`/`qwen3_5_moe_text` before the dense
  Qwen35 route and returns an owning `Qwen35MoeBackend` to `BackendThread`.
- Owner wraps the tested `ExperimentalMoEEngine`; constructor runs at cold
  startup, not per request.
- `submit_embed` returns `Inference::Deferred`; one request-specific host lease
  is acquired per submit and one geometry-keyed Work buffer is reused.
- Native BF16 embedding supports raw and CLS/mean/last-token pools.
- Classifier/Decision/SPLADE return explicit typed unsupported.
- CPU tests: 738 passed, 0 failed, 133 ignored; Clippy and fmt pass.
- No optimized host build, GPU, Torch, Docker, network or model admission.

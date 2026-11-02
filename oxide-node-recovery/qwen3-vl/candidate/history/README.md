# Frozen pre-P1 history

- `qwen3_vl_engine_v1.rs`: the reviewed pre-P1 engine where `DeferredVl` was
  constructed inside the deferred closure.
- Frozen host consumer binary: `/tmp/opencode/qwen-vl-host-consumer-target/release/examples/qwen3_vl_host_consumer`
  SHA-256 `6e4f7f53afcb9aaac0ac855c65a1c6a18aa1a6084da5abb6eb97b92d0c0702a1`
  (opt3, LTO=false, CPU 72-79, jobs 4, offline, CVD empty).
- Root has not executed this binary on a GPU.
This history is preserved; the successor source and binary are separate.

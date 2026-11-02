# Head512 oracle and sanitizer sources

- `head512_provider_test.cu`: Torch-free cudart test using ABI v2 and the
  mandatory cold `infinity_head512_prepare_v2` call on the bound context. Covers both dtypes,
  causal/noncausal global, causal/noncausal local, independent ragged Q/KV,
  runtime GQA, padded row/head strides, tile-boundary sequence lengths
  (1,31,32,33,63,64,65,127,128,129), invalid geometry/capacity rejection, FP64
  output reference and LSE layout `[heads, total_q]`. This is the
  compute-sanitizer target.
- `head512_native_compare.cpp`: actual Native LibTorch `torch_flash_fma` /
  `torch_flash_fma_varlen` comparison for global d512 masks, both dtypes,
  GQA and padded strides, including ragged separate Q/KV boundaries. Native
  intentionally rejects d512 local masks, so local is not compared here.
- `../root/run_root.sh` in the handoff output boundary performs device
  compilation, DT_NEEDED gating, both sanitizer runs and the Native oracle.

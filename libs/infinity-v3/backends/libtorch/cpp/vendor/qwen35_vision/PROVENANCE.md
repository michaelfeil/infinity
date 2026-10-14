# Qwen3.5 vision kernels

Adapted from [huggingface/candle](https://github.com/huggingface/candle), revision `036f39e`, `candle-kernels/src/reduce.cu`.

Original source SHA256: `7154bdc5e962f16c99f046ca9aa86262e46f5e49ef8aafb817d8ef513ff1e0e5`.

License: Apache-2.0 OR MIT. Both original license texts and the original ggml attribution are retained.

The CUDA FP16 Qwen3.5 vision path uses the original core LayerNorm and F32 warp softmax arithmetic. Tensor dimensions stay unchanged; no token, frame, or head padding is introduced. Compile with default FMA and precise division/square root, without fast math.

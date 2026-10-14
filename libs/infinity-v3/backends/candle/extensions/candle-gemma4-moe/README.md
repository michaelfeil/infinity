# Gemma4 routed experts

Experimental BF16 execution for the text decoder of `google/gemma-4-26B-A4B-it`:
128 experts, top eight routing, hidden size 2816, expert intermediate size 704.
Requires a CUDA build targeting SM80 or later. Only H100 has been exercised.
Other Gemma4 expert shapes and quantized weights are rejected.

SM90 builds additionally compile a Hopper SM90a TMA/WGMMA implementation.
It is selected for at least 2048 compact tokens on runtime SM90 devices;
smaller batches and other builds use the portable grouped GEMM. Both paths
share routing, packing, GELU and combination kernels, including scalar fallbacks
for unaligned input/output views. Hopper reserves 2 MiB of additional GEMM
workspace and checks the CUTLASS requirement before launch. The Hopper tile is
128 x 256 x 64; this improved the tested long-request medians by another
2–6% over the initial 128 x 128 x 64 tile in the former decision-scoring prototype.
Those full-model timings are historical and are not measurements of this
embedding-only integration.

Routing, token grouping, both expert GEMMs, GELU gating and weighted reduction
run on the caller's CUDA stream. Routing counts stay on the device. The Rust
binding checks tensor shapes, dtypes, contiguous storage and device identity,
and owns the scratch/output allocations. Shared dense MLP and expert branch
normalization are composed in `models/gemma4.rs`.

CUTLASS v4.5.0 is fetched at pinned commit
`e406c186f510a15091cce01f782020ceb7ba8eb5` by cudaforge. Its license is included
in `LICENSE.cutlass`.

GELU rounds to BF16 before multiplying the up branch. The down projection keeps
FP32 output until routing weights are applied, then rounds each expert's weighted
output to BF16 before summing in FP32. This follows vLLM's intermediate rounding.
Different GEMM/attention accumulation orders can still produce different model
scores; routing agreement alone does not qualify full-model accuracy.

## Expert-block diagnostic

From this directory, with CUDA, the pinned CUTLASS checkout, PyTorch, safetensors
and vLLM 0.30 installed:

```sh
nvcc -std=c++17 -O3 --expt-relaxed-constexpr -arch=sm_90 \
  -shared -Xcompiler -fPIC -I/path/to/cutlass/include \
  tests/probe.cu -o /tmp/gemma4_moe.so
CUDA_VISIBLE_DEVICES=0 python tests/compare_vllm.py \
  --model /path/to/gemma-4-26B-A4B-it \
  --library /tmp/gemma4_moe.so --output /tmp/gemma4-moe-results.json
```

This compares actual layer-zero weights at 1, 17, 257 and 1024 tokens, with
relative RMS error below 0.1% and cosine above 0.999999 as primitive-test gates.
It also records CUDA-event timings. These tolerances are not model-quality
acceptance criteria. Validate embedding behavior separately from these primitive checks.

The harness also checks exact GELU-plus-multiply agreement with vLLM across six
input scales, to catch changes to intermediate BF16 rounding. It additionally
checks every BF16 gate bit pattern against the original scalar arithmetic,
with unit/random up values, widths 13/704 and aligned/offset buffers. NaNs
are treated as equivalent; finite outputs must match bit for bit.

For the Hopper implementation, build both translation units and select its
entry points in the harness:

```sh
nvcc -std=c++17 -O3 -DNDEBUG --expt-relaxed-constexpr -arch=sm_90a \
  -shared -Xcompiler -fPIC -I/path/to/cutlass/include \
  -I/path/to/cutlass/tools/util/include \
  tests/probe.cu kernels/grouped_gemm_hopper.cu -o /tmp/gemma4_moe_hopper.so
CUDA_VISIBLE_DEVICES=0 python tests/compare_vllm.py \
  --model /path/to/gemma-4-26B-A4B-it --hopper --tokens 2048 4096 \
  --library /tmp/gemma4_moe_hopper.so --output /tmp/gemma4-moe-hopper.json
```

Use `--unaligned-io` and `--concentrated` separately to exercise offset views
and routing concentrated on eight experts (including 120 empty experts).
`-DNDEBUG` avoids CUTLASS device assertions that serialize WGMMA instructions;
host shape, workspace and launch-status checks remain enabled.

## Qwen3-MoE

The same grouped-GEMM library also implements Qwen3's 128-expert/top-8 MLP,
with configurable hidden/intermediate widths, SiLU gating and optional routing
probability renormalization. It does not apply Gemma's learned expert scales.
The `qwen3_moe` model type uses the Qwen3 attention and pooling paths.
Both per-expert `gate_proj/up_proj/down_proj.weight` and fused expert tensors
are accepted, including configurations with dense MLP layers between MoE layers.

CUDA BF16 with 128 experts/top-8 uses device-only routing and grouped GEMMs.
Other dtypes, expert counts and CPU/Metal use the tensor reference path;
that path copies routing probabilities to the host and is slower. Scaled RoPE
and quantized checkpoints are rejected. Qwen3-Next and Qwen3.5 are distinct
architectures and are not selected by this model-type alias.

Example for the unquantized checkpoint (one full model per visible GPU):

```sh
infinity_v3 --model-id Qwen/Qwen3-30B-A3B \
  --dtype bfloat16 --pooling last-token --max-batch-tokens 8192
```

For the real-weight Qwen3 diagnostic, add `tests/qwen3_probe.cu` to the Hopper
probe build above, then run `tests/compare_qwen3_vllm.py` with `--model`,
`--library` and `--output`. The test checks routing with/without renormalization,
SiLU rounding, and portable/Hopper expert outputs against vLLM. Full-model
embedding comparisons are still required; primitive agreement alone
is not a model-quality result.

## Qwen3.5-MoE text inference

The `qwen3_5_moe` wrapper and `qwen3_5_moe_text` model types use a separate hybrid
model implementation: Gated DeltaNet layers, periodic gated full attention with
partial RoPE, 256 routed experts/top-8, and a sigmoid-gated shared expert. The
initial target is the BF16 text decoder of `Qwen/Qwen3.5-35B-A3B` on CUDA with
FlashAttention. Vision, MTP, quantized weights and scaled RoPE are not implemented.
The embedding path is causal and has no learned output projection; checkpoints
requesting bidirectional attention or an output projection are rejected, including
flags in wrapped text configurations. Linear key/value head dimensions must be 128; unsupported configurations fail
at loading. Each visible GPU holds an independent full model.

Linear attention uses a stateless variable-length prefill kernel with FP32
recurrent state. Each sequence starts from zero. Sequences are unfolded
for convolution/recurrent attention and folded back for projections and experts;
state never crosses sequence or request boundaries. This first kernel
walks tokens recurrently, so it does not yet provide a chunk-parallel prefill
implementation. Full attention uses the configured FlashAttention backend.

```sh
infinity_v3 --model-id Qwen/Qwen3.5-35B-A3B \
  --dtype bfloat16 --pooling last-token --max-batch-tokens 4096
```

Native diagnostics live under `tests/qwen35/`: compile `gdn_probe.cu` as a CUDA
shared library and run `compare_gdn.py --library ... --output ...` to compare
convolution, gated recurrence and output normalization with Transformers.
Compile `moe_probe.cu` together with `kernels/grouped_gemm_hopper.cu`, using the
CUTLASS include paths and flags above, then run `compare_moe.py --model ...
--library ... --output ...` for actual checkpoint expert comparisons with vLLM.
These primitive checks supplement full-model validation.

// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#include "env_config.h"
#include "helpers/dense_mlp.h"
#include "helpers/nvfp4_linear.h"
#include "kernels/fast_kernels.h"
#include "kernels/qwen35_kernels.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <cmath>
#include <cstdlib>
#include <vector>

namespace {
void set_nvfp4_flag(const char* value) {
  if (value)
    setenv(INFINITY_ENV("TORCH_NVFP4_MLP"), value, 1);
  else
    unsetenv(INFINITY_ENV("TORCH_NVFP4_MLP"));
}

uint8_t e4m3_byte(float value) {
  return at::tensor({value}, at::TensorOptions().dtype(at::kFloat))
      .to(at::kFloat8_e4m3fn)
      .view(at::kByte)
      .item<uint8_t>();
}

float e4m3_value(uint8_t byte) {
  return at::tensor({int64_t(byte)}, at::TensorOptions().dtype(at::kByte))
      .view(at::kFloat8_e4m3fn)
      .to(at::kFloat)
      .item<float>();
}

// M=129 K=32: row 128 is the E2M1 codebook + 2688 outlier (S=1); tail row tests swizzle.
void native_quant_golden() {
  const auto opts = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  std::vector<float> values = {
      0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
  values.resize(32, 2688.f);
  auto x = at::zeros({129, 32}, opts);
  x.narrow(0, 128, 1).copy_(
      at::tensor(values, at::TensorOptions().dtype(at::kFloat)).to(opts).reshape({1, 32}));
  const uint8_t expected[16] = {0x10,
                                0x32,
                                0x54,
                                0x76,
                                0x98,
                                0xBA,
                                0xDC,
                                0xFE,
                                0x77,
                                0x77,
                                0x77,
                                0x77,
                                0x77,
                                0x77,
                                0x77,
                                0x77};
  for (bool rowwise : {false, true}) {
    auto q = infinity::nvfp4_quantize_rows(x, rowwise);
    TORCH_CHECK(q.rows == 129 && q.cols == 32 &&
                    std::abs(q.global_scale.item<float>() - 1.f) < 1e-9f,
                "NVFP4 golden global scale/layout mismatch");
    TORCH_CHECK(q.row_scales.defined() == rowwise,
                "NVFP4 golden row-scale presence does not match the scale mode");
    if (rowwise)
      TORCH_CHECK(q.row_scales.numel() == 129 && (q.row_scales == 1.f).all().item<bool>(),
                  "NVFP4 golden per-row scales are not all one");
    auto packed = q.values.view(at::kByte).to(at::kCPU).contiguous().narrow(0, 128, 1);
    auto scales = q.block_scales.view(at::kByte).to(at::kCPU).contiguous();
    TORCH_CHECK(scales.numel() == infinity::nvfp4_scale_elements(129, 32),
                "NVFP4 golden block-scale metadata size mismatch");
    const auto* bytes = packed.const_data_ptr<uint8_t>();
    for (int i = 0; i < 16; ++i)
      TORCH_CHECK(bytes[i] == expected[i], "NVFP4 golden packed nibble mismatch at ", i);
    TORCH_CHECK(scales[512].item<uint8_t>() == e4m3_byte(1.f) &&
                    scales[513].item<uint8_t>() == e4m3_byte(448.f),
                "NVFP4 golden tail-row E4M3 block scale mismatch");
  }
}

void batch_invariance() {
  const auto opts = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  auto row = at::randn({1, 32}, opts);
  auto batch = at::cat({row, at::full({3, 32}, 2688.f, opts)}, 0);
  auto alone = infinity::nvfp4_quantize_rows(row, true);
  auto cobatch = infinity::nvfp4_quantize_rows(batch, true);
  TORCH_CHECK(alone.global_scale.item<float>() == 1.f && cobatch.global_scale.item<float>() == 1.f,
              "Row-wise activation tensor scale must stay one");
  auto a_codes = alone.values.view(at::kByte).to(at::kCPU).contiguous();
  auto b_codes = cobatch.values.view(at::kByte).to(at::kCPU).contiguous().narrow(0, 0, 1);
  TORCH_CHECK(at::equal(a_codes, b_codes), "Per-row S changed first-row FP4 codes");
  TORCH_CHECK(alone.row_scales.defined() && cobatch.row_scales.defined(),
              "Per-row activation scales were not retained");
  const float a_row = alone.row_scales[0].item<float>();
  const float b_row = cobatch.row_scales[0].item<float>();
  TORCH_CHECK(a_row == b_row, "Per-row S depends on the co-batched rows");
  auto a_scales = alone.block_scales.view(at::kByte).to(at::kCPU).contiguous();
  auto b_scales = cobatch.block_scales.view(at::kByte).to(at::kCPU).contiguous();
  for (int block = 0; block < 2; ++block) {
    const float a_effective = e4m3_value(a_scales[block].item<uint8_t>()) * a_row;
    const float b_effective = e4m3_value(b_scales[block].item<uint8_t>()) * b_row;
    TORCH_CHECK(std::abs(a_effective - b_effective) <= 1e-6f * std::max(a_effective, b_effective),
                "Per-row S changed decoded first-row block scale at ",
                block);
  }
}

// Exact E4M3-subnormal co-batch counterexample: the target row [2^-7, 0, ...]
// alone gets S=2^-18, block scale 123 (352) and E2M1 value 6 (nibble 7). Under
// the old batch-global power-of-two S a co-row of 2048 chose value 4 while 4096
// chose 0. Per-row S must make the target byte-for-byte independent of either
// co-row.
void subnormal_cobatch_invariance() {
  const auto opts = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  auto target = at::zeros({1, 16}, opts);
  target[0][0].fill_(std::ldexp(1.f, -7));
  auto make = [&](float high) {
    auto co_row = at::zeros({1, 16}, opts);
    co_row[0][0].fill_(high);
    return infinity::nvfp4_quantize_rows(at::cat({target, co_row}, 0), true);
  };
  auto low = make(2048.f);
  auto high = make(4096.f);
  for (const auto* q : {&low, &high}) {
    TORCH_CHECK(q->global_scale.item<float>() == 1.f &&
                    q->row_scales[0].item<float>() == std::ldexp(1.f, -18),
                "Subnormal row scale is not the row-local power of two");
    TORCH_CHECK(q->block_scales.view(at::kByte)[0].item<uint8_t>() == 123,
                "Subnormal row E4M3 block scale changed");
    TORCH_CHECK(q->values.view(at::kByte)[0][0].item<uint8_t>() == 0x07,
                "Subnormal row E2M1 code changed");
  }
  TORCH_CHECK(low.row_scales[0].item<float>() == high.row_scales[0].item<float>() &&
                  low.block_scales.view(at::kByte)[0].item<uint8_t>() ==
                      high.block_scales.view(at::kByte)[0].item<uint8_t>() &&
                  at::equal(low.values.view(at::kByte).narrow(0, 0, 1),
                            high.values.view(at::kByte).narrow(0, 0, 1)),
              "Subnormal target row depends on its co-batched row");
}

// One row with four blocks keeps the E4M3 subnormal scale and the value just
// below the minimum-normal scale inside a single row. With the row amax 2688
// the row scale is one, so the block bytes match the previous per-row golden.
void subnormal_scale_golden() {
  const auto opts = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  auto x = at::zeros({1, 64}, opts);
  x[0].narrow(0, 0, 16).fill_(.03125f);
  x[0].narrow(0, 16, 16).fill_(.09326171875f);
  x[0].narrow(0, 48, 16).fill_(2688.f);
  auto q = infinity::nvfp4_quantize_rows(x, true);
  TORCH_CHECK(q.global_scale.item<float>() == 1.f && q.row_scales[0].item<float>() == 1.f,
              "Unexpected edge-case row scale");
  auto scales = q.block_scales.view(at::kByte).cpu();
  auto values = q.values.view(at::kByte).cpu();
  const int expected[] = {3, 8, 0, 126};
  for (int block = 0; block < 4; ++block) {
    TORCH_CHECK(scales[block].item<int>() == expected[block], "Edge-case scale mismatch");
    TORCH_CHECK(at::all(values[0].narrow(0, block * 8, 8) == (block == 2 ? 0 : 0x77)).item<bool>(),
                "Edge-case FP4 code mismatch");
  }
}

void check_same_quantized(const infinity::Nvfp4Quantized& actual,
                          const infinity::Nvfp4Quantized& expected,
                          const char* label) {
  TORCH_CHECK(actual.rows == expected.rows && actual.cols == expected.cols &&
                  actual.values.sizes() == expected.values.sizes() &&
                  actual.block_scales.sizes() == expected.block_scales.sizes(),
              label,
              ": NVFP4 geometry mismatch");
  TORCH_CHECK(at::equal(actual.values.view(at::kByte), expected.values.view(at::kByte)),
              label,
              ": NVFP4 packed values differ");
  TORCH_CHECK(at::equal(actual.block_scales.view(at::kByte), expected.block_scales.view(at::kByte)),
              label,
              ": NVFP4 block scales differ");
  TORCH_CHECK(
      at::equal(actual.global_scale, expected.global_scale), label, ": NVFP4 global scale differs");
  TORCH_CHECK(actual.row_scales.defined() && expected.row_scales.defined() &&
                  at::equal(actual.row_scales, expected.row_scales),
              label,
              ": NVFP4 row scales differ");
}

infinity::Nvfp4Quantized swiglu_rowwise_reference(const at::Tensor& gate_up,
                                                  const at::Tensor& row_scales) {
  const int64_t width = gate_up.size(1) / 2;
  auto gate = gate_up.narrow(1, 0, width).contiguous();
  auto up = gate_up.narrow(1, width, width).contiguous();
  if (row_scales.defined()) {
    infinity::nvfp4_apply_row_scales_(gate, row_scales);
    infinity::nvfp4_apply_row_scales_(up, row_scales);
  }
  return infinity::nvfp4_quantize_rows(infinity::fused_gated_activation(at::cat({gate, up}, 1), 5),
                                       true);
}

// Five rows exercise the 128-row padded scale tile, a zero row, non-finite
// gate/up lanes, and very different restore scales. The fused SwiGLU staging
// must be byte-identical to explicit restore -> rounded activation -> rowwise
// quantization at the representative intermediate widths.
void swiglu_fused_equivalence() {
  const auto opts = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  for (int64_t width : {9728, 12288, 17408}) {
    auto gate_up = at::randn({5, 2 * width}, opts) * .03f;
    gate_up[1].zero_();
    gate_up[2][0].fill_(INFINITY);
    gate_up[2][width + 1].fill_(-INFINITY);
    gate_up[2][3].fill_(NAN);
    auto scales = at::tensor(
        {std::ldexp(1.f, -20), 1.f, std::ldexp(1.f, 7), std::ldexp(1.f, -3), std::ldexp(1.f, 12)},
        opts.dtype(at::kFloat));
    check_same_quantized(infinity::nvfp4_swiglu_quantize(gate_up, scales),
                         swiglu_rowwise_reference(gate_up, scales),
                         "Fused NVFP4 SwiGLU with mixed restore scales");
    check_same_quantized(infinity::nvfp4_swiglu_quantize(gate_up),
                         swiglu_rowwise_reference(gate_up, {}),
                         "Fused NVFP4 SwiGLU without restore scales");
  }
  auto empty = at::empty({0, 32}, opts);
  check_same_quantized(infinity::nvfp4_swiglu_quantize(empty),
                       swiglu_rowwise_reference(empty, {}),
                       "Empty NVFP4 SwiGLU");
}

// Exercise both ends of the reciprocal range and the reachable fallback for
// subnormal denominators. Build tiny BF16 values on CPU before copying them.
void row_scale_range_golden() {
  const auto opts = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  auto x = at::zeros({3, 64}, opts.device(at::kCPU));
  x[0].narrow(0, 0, 16).fill_(std::ldexp(255.f, 120));
  x[0].narrow(0, 16, 16).fill_(std::ldexp(.09326171875f, 117));
  x[0].narrow(0, 32, 16).fill_(std::ldexp(.03125f, 117));
  x[1].narrow(0, 0, 16).fill_(std::ldexp(2688.f, -128));
  x[2].narrow(0, 0, 16).fill_(std::ldexp(1.f, -133));
  auto q = infinity::nvfp4_quantize_rows(x.to(opts), true);
  auto rows = q.row_scales.cpu();
  auto scales = q.block_scales.view(at::kByte).cpu();
  TORCH_CHECK(rows[0].item<float>() == std::ldexp(1.f, 117) &&
                  rows[1].item<float>() == std::ldexp(1.f, -128) &&
                  rows[2].item<float>() == std::ldexp(1.f, -144),
              "NVFP4 extreme row scales changed");
  TORCH_CHECK(scales[0].item<int>() == 123 && scales[1].item<int>() == 8 &&
                  scales[2].item<int>() == 3 && scales[16].item<int>() == 126 &&
                  scales[32].item<int>() == 123,
              "NVFP4 extreme or subnormal block scales changed");
}

// The last fused width and the first fallback width. The fallback keeps the
// original two-pass activation/amax/quantize path byte-identical.
void swiglu_shared_memory_boundary() {
  const auto opts = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  constexpr int64_t fused_width = (48 * 1024 - 256) / 2;
  for (int64_t width : {fused_width, fused_width + 16}) {
    auto gate_up = at::randn({3, 2 * width}, opts) * .02f;
    gate_up[2][0].fill_(NAN);
    auto scales =
        at::tensor({std::ldexp(1.f, -9), 1.f, std::ldexp(1.f, 5)}, opts.dtype(at::kFloat));
    check_same_quantized(infinity::nvfp4_swiglu_quantize(gate_up, scales),
                         swiglu_rowwise_reference(gate_up, scales),
                         width == fused_width ? "Boundary NVFP4 SwiGLU fused"
                                              : "Boundary NVFP4 SwiGLU fallback");
  }
}

void dense_mlp_forward_and_guards() {
  const auto opts = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  auto gate_up = at::randn({64, 32}, opts) * .03f;
  auto down = at::randn({16, 32}, opts) * .03f;
  auto recipe = infinity::MlpActivationRecipe::numerical(
      [](const at::Tensor& rows) { return infinity::fused_gated_activation(rows, 5); }, true);
  recipe.fusion = infinity::MlpFusion::RoundedSwiGlu;
  auto make = [&] { return infinity::DenseMlp({gate_up}, {down}, recipe); };
  auto x = at::randn({4, 32}, opts) * .5f;
  set_nvfp4_flag("0");
  auto off_mlp = make();
  off_mlp.prepare(infinity::MlpPrecision::Fp8);
  TORCH_CHECK(!off_mlp.input().method().accepts_native_quantized_rows() &&
                  at::isfinite(off_mlp.forward(x)).all().item<bool>(),
              "Default MXFP8 path changed after NVFP4 opt-in was disabled");
  set_nvfp4_flag("gate");
  bool rejected = false;
  try {
    auto bad_mlp = make();
    bad_mlp.prepare(infinity::MlpPrecision::Fp8);
  } catch (const c10::Error&) {
    rejected = true;
  }
  TORCH_CHECK(rejected, "Unknown NVFP4 flag value was silently accepted");
  set_nvfp4_flag("1");
  auto nv_mlp = make();
  nv_mlp.prepare(infinity::MlpPrecision::Fp8);
  nv_mlp.prepare(infinity::MlpPrecision::Fp8); // Released native weights remain prepared.
  TORCH_CHECK(nv_mlp.input().method().accepts_native_quantized_rows() &&
                  nv_mlp.output().method().accepts_native_quantized_rows() &&
                  nv_mlp.input().method().supports_native_swiglu_quantized(),
              "NVFP4 did not select the fused both-projections method");
  infinity::Nvfp4Linear in(gate_up), out(down);
  auto gu = in.forward(x);
  auto fused_stage = infinity::nvfp4_swiglu_quantize(gu);
  auto code5_stage = infinity::nvfp4_quantize_rows(infinity::fused_gated_activation(gu, 5), true);
  auto gemma_stage = infinity::nvfp4_quantize_rows(infinity::gemma_gated_cuda(gu, false), true);
  auto same = [](const auto& a, const auto& b) {
    return at::equal(a.values.view(at::kByte), b.values.view(at::kByte)) &&
           at::equal(a.block_scales.view(at::kByte), b.block_scales.view(at::kByte)) &&
           at::equal(a.global_scale, b.global_scale) && a.row_scales.defined() &&
           b.row_scales.defined() && at::equal(a.row_scales, b.row_scales);
  };
  TORCH_CHECK(same(fused_stage, code5_stage) && same(fused_stage, gemma_stage),
              "Fused NVFP4 SwiGLU changed exact quantized staging or row scales");
  auto fused = nv_mlp.forward(x);
  auto code5 = out.forward(infinity::fused_gated_activation(gu, 5));
  auto gemma = out.forward(infinity::gemma_gated_cuda(gu, false));
  TORCH_CHECK(at::equal(fused, code5) && at::equal(fused, gemma),
              "Fused NVFP4 MLP changed exact code-5/Gemma quantized output");

  auto bias = at::randn({32}, opts).slice(0, 0, 32, 2);
  infinity::DenseMlp biased({gate_up}, {down, bias}, recipe);
  biased.prepare(infinity::MlpPrecision::Fp8);
  TORCH_CHECK(at::equal(biased.forward(x), code5 + bias),
              "NVFP4 MLP did not preserve a strided down-projection bias");

  // NVFP4 tiles wide activations; the rowwise FP8 fusion's width limit does not apply.
  constexpr int64_t intermediate = 32896;
  auto wide_gate = at::randn({2 * intermediate, 128}, opts) * .02f;
  auto wide_down = at::randn({128, intermediate}, opts) * .02f;
  auto wide_x = at::randn({2, 128}, opts) * .5f;
  infinity::DenseMlp wide({wide_gate}, {wide_down}, recipe);
  TORCH_CHECK(wide.supports_nvfp4(), "Wide tiled NVFP4 MLP was rejected");
  wide.prepare(infinity::MlpPrecision::Fp8);
  infinity::Nvfp4Linear wide_in(wide_gate), wide_out(wide_down);
  auto reference = wide_out.forward(infinity::fused_gated_activation(wide_in.forward(wide_x), 5));
  TORCH_CHECK(at::equal(wide.forward(wide_x), reference),
              "Wide NVFP4 MLP changed exact composed output");

  // cuBLASLt cannot execute a packed K that is only 8-byte aligned. Reject it at preparation.
  infinity::DenseMlp unaligned({at::randn({64, 48}, opts)}, {at::randn({48, 32}, opts)}, recipe);
  rejected = false;
  try {
    unaligned.prepare(infinity::MlpPrecision::Fp8);
  } catch (const c10::Error&) {
    rejected = true;
  }
  TORCH_CHECK(rejected && !unaligned.fp8(), "Unsupported packed K was accepted at preparation");
  set_nvfp4_flag(nullptr);
}

// A 16x32 all-ones weight against one- and two-valued activation rows. Row 0
// decodes with S=2^-11 and row 1 with S=2^-10, both at block scale 352/code 6;
// the weight uses its exact 1/2688 global scale at 448/code 6. The raw GEMM is
// 67584 for both rows, so the per-row post-scale must produce 33 and 66. The
// original K=32 ones/ones geometry probe measured 33, not an idealized 32.
void row_scale_gemm_golden() {
  const auto opts = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  infinity::Nvfp4Linear linear(at::ones({16, 32}, opts));
  auto x = at::cat({at::ones({1, 32}, opts), at::full({1, 32}, 2.f, opts)}, 0);
  auto expected = at::cat({at::full({1, 16}, 33.f, opts), at::full({1, 16}, 66.f, opts)}, 0);
  TORCH_CHECK(at::equal(linear.forward(x), expected),
              "NVFP4 row-scale GEMM golden changed decoded output");
  auto restore = at::full({2, 16}, std::ldexp(1.f, 120), opts);
  auto factors = at::tensor({std::ldexp(1.f, -140), std::ldexp(1.f, -139)}, opts.dtype(at::kFloat));
  infinity::nvfp4_apply_row_scales_(restore, factors);
  auto expected_restore = at::cat({at::full({1, 16}, std::ldexp(1.f, -20), opts),
                                   at::full({1, 16}, std::ldexp(1.f, -19), opts)},
                                  0);
  TORCH_CHECK(at::equal(restore, expected_restore),
              "NVFP4 row-scale restore must keep F32 subnormal factors");
}
} // namespace

int main() {
  if (at::cuda::device_count() == 0)
    return 77;
  if (!infinity::nvfp4_provider_supported(at::Device(at::kCUDA, 0))) {
    return 77;
  }
  row_scale_gemm_golden();
  native_quant_golden();
  batch_invariance();
  subnormal_cobatch_invariance();
  subnormal_scale_golden();
  swiglu_fused_equivalence();
  row_scale_range_golden();
  swiglu_shared_memory_boundary();
  dense_mlp_forward_and_guards();
  return 0;
}

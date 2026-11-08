// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
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
    setenv("INFINITY_TORCH_NVFP4_MLP", value, 1);
  else
    unsetenv("INFINITY_TORCH_NVFP4_MLP");
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
  for (bool power2 : {false, true}) {
    auto q = infinity::nvfp4_quantize_rows(x, power2);
    TORCH_CHECK(q.rows == 129 && q.cols == 32 &&
                    std::abs(q.global_scale.item<float>() - 1.f) < 1e-9f,
                "NVFP4 golden global scale/layout mismatch");
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
  auto a_codes = alone.values.view(at::kByte).to(at::kCPU).contiguous();
  auto b_codes = cobatch.values.view(at::kByte).to(at::kCPU).contiguous().narrow(0, 0, 1);
  TORCH_CHECK(at::equal(a_codes, b_codes), "Power-of-two S changed first-row FP4 codes");
  auto a_scales = alone.block_scales.view(at::kByte).to(at::kCPU).contiguous();
  auto b_scales = cobatch.block_scales.view(at::kByte).to(at::kCPU).contiguous();
  const float a_global = alone.global_scale.item<float>();
  const float b_global = cobatch.global_scale.item<float>();
  for (int block = 0; block < 2; ++block) {
    const float a_effective = e4m3_value(a_scales[block].item<uint8_t>()) * a_global;
    const float b_effective = e4m3_value(b_scales[block].item<uint8_t>()) * b_global;
    TORCH_CHECK(std::abs(a_effective - b_effective) <= 1e-6f * std::max(a_effective, b_effective),
                "Power-of-two S changed decoded first-row block scale at ",
                block);
  }
}

// Include a subnormal scale and a value just below the minimum-normal scale:
// the latter rounds up to a normal scale but still needs exact re-encoding.
void subnormal_scale_golden() {
  auto x = at::tensor({.03125f, .09326171875f, 0.f, 2688.f}, at::TensorOptions().dtype(at::kFloat))
               .to(at::Device(at::kCUDA), at::kBFloat16)
               .unsqueeze(1)
               .expand({4, 16})
               .contiguous();
  auto q = infinity::nvfp4_quantize_rows(x, true);
  TORCH_CHECK(q.global_scale.item<float>() == 1.f, "Unexpected edge-case global scale");
  auto scales = q.block_scales.view(at::kByte).cpu();
  auto values = q.values.view(at::kByte).cpu();
  const int expected[] = {3, 8, 0, 126};
  for (int row = 0; row < 4; ++row) {
    TORCH_CHECK(scales[row * 16].item<int>() == expected[row], "Edge-case scale mismatch");
    TORCH_CHECK(at::all(values[row] == (row == 2 ? 0 : 0x77)).item<bool>(),
                "Edge-case FP4 code mismatch");
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
           at::equal(a.global_scale, b.global_scale);
  };
  TORCH_CHECK(same(fused_stage, code5_stage) && same(fused_stage, gemma_stage),
              "Fused NVFP4 SwiGLU changed exact quantized staging");
  auto fused = nv_mlp.forward(x);
  auto code5 = out.forward(infinity::fused_gated_activation(gu, 5));
  auto gemma = out.forward(infinity::gemma_gated_cuda(gu, false));
  TORCH_CHECK(at::equal(fused, code5) && at::equal(fused, gemma),
              "Fused NVFP4 MLP changed exact code-5/Gemma quantized output");
  set_nvfp4_flag(nullptr);
}
} // namespace

int main() {
  if (at::cuda::device_count() == 0)
    return 77;
  if (!infinity::nvfp4_provider_supported(at::Device(at::kCUDA, 0))) {
    return 77;
  }
  native_quant_golden();
  batch_invariance();
  subnormal_scale_golden();
  dense_mlp_forward_and_guards();
  return 0;
}

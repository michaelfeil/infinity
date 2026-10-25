#include "kernels/qwen35_chunk.h"
#include "kernels/flashinfer_gdn.h"
#include "kernels/qwen35_kernels.h"
#include "kernels/qwen35_reference.h"
#include "batching/prefix_cache.h"
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <iostream>

void check(const at::Tensor& actual, const at::Tensor& expected, double tolerance) {
  auto x = actual.cpu().to(at::kDouble), reference = expected.cpu().to(at::kDouble);
  TORCH_CHECK(at::isfinite(x).all().item<bool>(), "Prefill produced nonfinite values");
  TORCH_CHECK((x - reference).norm().item<double>() /
                      reference.norm().clamp_min(1e-12).item<double>() <
                  tolerance,
              "Prefill disagrees with recurrent reference");
}

// Exercise fresh packed dispatch with ragged tails.
// The separate-rounding recurrence supplies independent outputs and state.
void production_prefill(int64_t heads) {
  constexpr int64_t tokens = 519, kh = 16;
  auto opts = at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  auto qkv = at::randn({tokens, (2 * kh + heads) * 128}, opts) * .2;
  auto z = at::randn({tokens, heads * 128}, opts) * .2,
       ab = at::randn({tokens, 2 * heads}, opts) * .2;
  auto conv = at::randn({(2 * kh + heads) * 128, 4}, opts) * .2;
  auto alog = at::zeros({heads}, opts.dtype(at::kFloat)), dt = at::zeros_like(alog),
       norm = at::ones({128}, opts);
  std::vector<int32_t> offsets{0, 1, 14, tokens};
  infinity::PackedInput input;
  input.batch = 3;
  input.offsets = offsets.data();
  input.max_sequence = tokens - 14;
  input.cumulative = at::tensor(offsets, opts.dtype(at::kInt));
  auto budget = std::make_shared<infinity::PrefixBudget>();
  budget->reserved_bytes = 64 << 20;
  std::vector<std::shared_ptr<infinity::PrefixEntry>> saved;
  for (int row = 0; row < 3; ++row) {
    auto entry = std::make_shared<infinity::PrefixEntry>();
    entry->budget = budget;
    entry->reserved_bytes = (64 << 20) / 3;
    entry->identity.tokens.resize(offsets[row + 1] - offsets[row]);
    saved.push_back(entry);
  }
  infinity::PrefixContext capture{
      input, offsets, {0, 0, 0}, {nullptr, nullptr, nullptr}, saved, 64 << 20, budget};
  auto reference = input;
  reference.prefix = &capture;
  auto expected = infinity::qwen35_reference_delta_cuda(
      qkv, z, ab, conv, alog, dt, norm, reference, kh, heads, 1e-6, 0);
  at::Tensor state;
  auto actual = infinity::qwen35_delta_cuda(
      qkv, z, ab, conv, alog, dt, norm, input, kh, heads, 1e-6, {}, &state);
  std::vector<at::Tensor> states;
  for (const auto& entry : saved)
    states.push_back(entry->layers.at(0).recurrent);
  check(actual, expected, .01);
  check(state, at::stack(states), .006);
  auto forward = [&] {
    return infinity::qwen35_delta_cuda(qkv,
                                       z,
                                       ab,
                                       conv,
                                       alog,
                                       dt,
                                       norm,
                                       input,
                                       kh,
                                       heads,
                                       1e-6,
                                       {},
                                       nullptr,
                                       {},
                                       {},
                                       false,
                                       -1,
                                       true);
  };
  // Ordinary decision requests use the same provider without retaining state.
  check(forward(), expected, .01);
  auto stream = c10::cuda::getStreamFromPool();
  c10::cuda::CUDAStreamGuard guard(stream);
  forward();
  stream.synchronize();
  at::cuda::CUDAGraph graph;
  graph.capture_begin();
  auto replayed = forward();
  graph.capture_end();
  qkv.mul_(.8);
  graph.replay();
  TORCH_CHECK(at::equal(replayed, forward()), "Transient GDN graph replay lost changed inputs");
}

int main() {
  int devices = 0, tested = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess)
    return 77;
  at::manual_seed(42);
  at::set_num_threads(8);
  // Reuse the same process across devices to catch process-wide CUDA initialization caches.
  for (int index = 0; index < devices; ++index) {
    cudaDeviceProp device{};
    if (cudaGetDeviceProperties(&device, index) != cudaSuccess)
      continue;
    c10::cuda::CUDAGuard guard(index);
    bool supported = infinity::flashinfer_gdn_available(c10::Device(at::kCUDA, index));
#ifdef TEI_TORCH_GDN_HOPPER
    supported = supported || device.major == 9;
#endif
    if (!supported)
      continue;
    ++tested;
    for (int64_t heads : {32, 48})
      for (int mode : {0, 1, 2}) {
        constexpr int64_t tokens = 143, kh = 16;
        std::vector<int32_t> offsets{0, 1, 14, tokens};
        auto options = at::TensorOptions().dtype(at::kFloat);
        auto q = at::randn({tokens, kh, 128}, options), k = at::randn_like(q);
        q = q / q.square().sum(-1, true).sqrt() / std::sqrt(128.);
        k = k / k.square().sum(-1, true).sqrt();
        auto v = at::randn({tokens, heads, 128}, options).to(at::kBFloat16) * .2;
        auto decay = at::rand({tokens, heads}, options), beta = at::rand_like(decay);
        if (mode < 2)
          decay.fill_(mode);
        else
          for (int64_t t = 0; t < tokens; t += 3)
            decay[t].zero_();
        std::vector<at::Tensor> outputs, states;
        for (int row = 0; row < 3; ++row) {
          auto state = at::zeros({heads, 128, 128}, options.dtype(at::kDouble));
          for (int t = offsets[row]; t < offsets[row + 1]; ++t) {
            auto qt = at::repeat_interleave(q[t], heads / kh, 0).to(at::kDouble);
            auto kt = at::repeat_interleave(k[t], heads / kh, 0).to(at::kDouble);
            state = state * decay[t].to(at::kDouble).view({heads, 1, 1});
            auto delta = (v[t].to(at::kDouble) - (state * kt.unsqueeze(-1)).sum(1)) *
                         beta[t].to(at::kDouble).unsqueeze(-1);
            state = state + kt.unsqueeze(-1) * delta.unsqueeze(1);
            outputs.push_back((state * qt.unsqueeze(-1)).sum(1));
          }
          states.push_back(state);
        }
        infinity::PackedInput input;
        input.batch = 3;
        input.cumulative = at::tensor(offsets, options.dtype(at::kInt)).to(at::kCUDA);
        auto values =
            at::cat({at::zeros({tokens, kh * 256}, v.options()), v.flatten(1)}, 1).to(at::kCUDA);
        auto qk = at::cat({q, k}, -1).reshape({tokens * kh, 256}).to(at::kCUDA);
        auto gates = at::cat({decay, beta}, -1).to(at::kCUDA);
        auto output = at::empty({tokens, heads, 128}, values.options());
        at::Tensor state;
        if (!infinity::flashinfer_gdn_prefill(
                values, qk, gates, input, output, kh, heads, &state)) {
#ifdef TEI_TORCH_GDN_HOPPER
          infinity::qwen35_prefill_cuda(values, qk, gates, input, output, kh, heads, &state);
#else
          TORCH_CHECK(false, "Registered FlashInfer GDN bundle was unavailable");
#endif
        }
        check(output, at::stack(outputs), .007);
        check(state, at::stack(states), .006);
      }
    for (int64_t heads : {32, 48})
      production_prefill(heads);
  }
  if (!tested)
    return 77;
  std::cout << "Ragged prefill outputs and final states match FP64 recurrence tolerance\n";
}

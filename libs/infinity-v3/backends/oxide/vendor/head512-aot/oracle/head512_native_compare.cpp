// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// Test-only comparison of the Torch-free provider against the actual Native
// LibTorch head512 adapter (same kernel family and launch contract).
//
// Native intentionally gates d512 local windows out, so this file compares
// global causal/noncausal only. Local d512 is covered by the Torch-free
// provider test against an FP64 reference.
#include "head512_provider.h"

#include "kernels/torch_flash_fma.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
constexpr int64_t kD = 512;

[[noreturn]] void fail(const std::string& message) {
  std::fprintf(stderr, "head512 native compare FAIL: %s\n", message.c_str());
  std::exit(1);
}

struct Padded {
  at::Tensor backing;
  at::Tensor view;
};

Padded padded(at::IntArrayRef sizes, at::IntArrayRef strides, at::TensorOptions options,
              uint32_t& state) {
  int64_t last = 0;
  for (size_t i = 0; i < sizes.size(); ++i)
    last += (sizes[i] - 1) * strides[i];
  Padded p;
  p.backing = at::empty({last + 1}, options);
  p.view = p.backing.as_strided(sizes, strides);
  p.view.copy_(at::randn(sizes, options));
  // Touch the padding too, so stale storage cannot hide an indexing defect.
  p.backing.add_(0.0);
  state = state * 1664525u + 1013904223u;
  return p;
}

struct Outputs {
  at::Tensor native;
  at::Tensor provider_view;
  at::Tensor provider_backing;
  at::Tensor lse;
};

Outputs run(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
            const std::vector<int32_t>& q_offsets, const std::vector<int32_t>& k_offsets,
            int64_t max_q, int64_t max_k, bool causal, bool separate, int64_t device) {
  const auto options = q.options();
  const auto cuq = at::tensor(q_offsets, at::TensorOptions().dtype(at::kInt)).to(q.device());
  const auto cuk = at::tensor(k_offsets, at::TensorOptions().dtype(at::kInt)).to(q.device());
  at::Tensor native;
  if (separate)
    native = infinity::torch_flash_fma_varlen(
        q, k, v, cuq, cuk, max_q, max_k, 1.0, causal, -1, -1);
  else
    native = infinity::torch_flash_fma(q, k, v, cuq, max_q, 1.0, causal, -1, -1);
  if (!at::isfinite(native).all().item<bool>())
    fail("Native output is non-finite");

  const int64_t total_q = q.size(0);
  const int64_t heads = q.size(1);
  const int64_t o_head = kD + 16;
  const int64_t o_row = heads * o_head + 8;
  const int64_t out_last = (total_q - 1) * o_row + (heads - 1) * o_head + kD;
  auto provider_backing = at::empty({out_last + 1}, options);
  auto provider_view =
      provider_backing.as_strided({total_q, heads, kD}, {o_row, o_head, 1});
  auto lse = at::empty({heads, total_q}, options.dtype(at::kFloat));

  InfinityHead512Batch batch{};
  batch.q = reinterpret_cast<uint64_t>(q.data_ptr());
  batch.k = reinterpret_cast<uint64_t>(k.data_ptr());
  batch.v = reinterpret_cast<uint64_t>(v.data_ptr());
  batch.q_offsets = reinterpret_cast<uint64_t>(cuq.data_ptr());
  batch.k_offsets = reinterpret_cast<uint64_t>(cuk.data_ptr());
  batch.total_q = total_q;
  batch.total_k = k.size(0);
  batch.batch = static_cast<int64_t>(q_offsets.size()) - 1;
  batch.max_q = max_q;
  batch.max_k = max_k;
  batch.heads = heads;
  batch.kv_heads = k.size(1);
  batch.scale = 1.0;
  batch.strides[0] = q.stride(0); batch.strides[1] = q.stride(1); batch.strides[2] = q.stride(2);
  batch.strides[3] = k.stride(0); batch.strides[4] = k.stride(1); batch.strides[5] = k.stride(2);
  batch.strides[6] = v.stride(0); batch.strides[7] = v.stride(1); batch.strides[8] = v.stride(2);
  batch.strides[9] = o_row; batch.strides[10] = o_head; batch.strides[11] = 1;
  batch.dtype = q.scalar_type() == at::kHalf ? INFINITY_HEAD512_DTYPE_FP16
                                            : INFINITY_HEAD512_DTYPE_BF16;
  batch.causal = causal ? 1 : 0;
  batch.window_left = -1;
  batch.window_right = -1;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  if (infinity_head512_prepare_v2() != 0)
    fail(std::string("provider cold prepare failed: ") + infinity_head512_error_v2());
  if (infinity_head512_enqueue_v2(&batch,
                                  provider_view.data_ptr(),
                                  provider_backing.nbytes(),
                                  lse.data_ptr(),
                                  lse.nbytes(),
                                  stream) != 0)
    fail(std::string("provider enqueue failed: ") + infinity_head512_error_v2());
  if (cudaDeviceSynchronize() != cudaSuccess)
    fail("provider synchronization failed");
  if (!at::isfinite(provider_view).all().item<bool>())
    fail("provider output is non-finite");
  if (!at::isfinite(lse).all().item<bool>())
    fail("provider lse is non-finite");
  (void)device;
  return {native.contiguous(), provider_view.contiguous(), provider_backing, lse};
}

void compare_case(bool causal, bool grouped, const std::vector<int32_t>& q_lengths,
                  const std::vector<int32_t>& k_lengths, bool separate, int64_t device) {
  if (q_lengths.size() != k_lengths.size())
    fail("batch mismatch");
  for (auto dtype : {at::kHalf, at::kBFloat16}) {
    auto options = at::TensorOptions().device(at::kCUDA, device).dtype(dtype);
    std::vector<int32_t> q_offsets{0}, k_offsets{0};
    for (int32_t n : q_lengths) q_offsets.push_back(q_offsets.back() + n);
    for (int32_t n : k_lengths) k_offsets.push_back(k_offsets.back() + n);
    const int64_t total_q = q_offsets.back(), total_k = k_offsets.back();
    const int64_t heads = grouped ? 4 : 8;
    const int64_t kv_heads = grouped ? 1 : 2;
    const int64_t q_head = kD + 32, q_row = heads * q_head + 16;
    const int64_t kv_head = kD + 24, kv_row = kv_heads * kv_head + 8;
    uint32_t state = 0xabc00000u ^ static_cast<uint32_t>(causal) ^
                     (static_cast<uint32_t>(grouped) << 1);
    auto q = padded({total_q, heads, kD}, {q_row, q_head, 1}, options, state);
    auto k = padded({total_k, kv_heads, kD}, {kv_row, kv_head, 1}, options, state);
    auto v = padded({total_k, kv_heads, kD}, {kv_row, kv_head, 1}, options, state);
    const int64_t max_q = *std::max_element(q_lengths.begin(), q_lengths.end());
    const int64_t max_k = *std::max_element(k_lengths.begin(), k_lengths.end());
    auto outputs = run(q.view, k.view, v.view, q_offsets, k_offsets, max_q, max_k,
                       causal, separate, device);
    if (!at::equal(outputs.native, outputs.provider_view)) {
      auto diff = (outputs.native.to(at::kFloat) - outputs.provider_view.to(at::kFloat)).abs();
      fail("provider differs from Native global head512; max=" +
           std::to_string(diff.max().item<double>()));
    }
    std::printf("PASS native d512 dtype=%s causal=%d grouped=%d separate=%d "
                "total_q=%lld total_k=%lld padded=1\n",
                dtype == at::kHalf ? "fp16" : "bf16", int(causal), int(grouped),
                int(separate), static_cast<long long>(total_q),
                static_cast<long long>(total_k));
  }
}
}  // namespace

int main() {
  if (at::cuda::device_count() <= 0)
    fail("no CUDA device");
  const int64_t device = 0;
  c10::cuda::CUDAGuard guard(device);
  auto probe = at::zeros({3, 1, kD}, at::TensorOptions().device(at::kCUDA, device).dtype(at::kHalf));
  if (!infinity::torch_flash_fma_supported(probe, probe, probe, false, -1, -1))
    fail("Native head512 global support is not available on this device");
  compare_case(false, false, {3, 17, 9}, {3, 17, 9}, false, device);
  compare_case(true, false, {3, 17, 9}, {3, 17, 9}, false, device);
  compare_case(false, true, {32, 129, 17}, {32, 129, 17}, false, device);
  compare_case(true, true, {32, 129, 17}, {32, 129, 17}, false, device);
  compare_case(true, true, {3, 17, 9}, {5, 2, 13}, true, device);
  compare_case(false, true, {3, 17, 9}, {5, 2, 13}, true, device);
  compare_case(true, false, {1, 31, 32, 33, 63, 64, 65, 127, 128, 129},
               {1, 31, 32, 33, 63, 64, 65, 127, 128, 129}, false, device);
  std::printf("head512 native compare PASS\n");
  return 0;
}

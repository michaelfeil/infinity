// SPDX-License-Identifier: MIT
/// \file
/// Native routed-MoE dispatch across compiled expert implementations.
/// Validates layouts and supplies on-device routing/workspace buffers.

#include "kernels/moe_kernels.h"
#ifdef TEI_TORCH_MOE_KERNELS
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include "kernels/moe_symbols.h"
extern "C" size_t qwen3_moe_workspace_bytes(int, int, int);
extern "C" size_t qwen35_moe_workspace_bytes(int, int, int);
extern "C" size_t gemma4_moe_workspace_bytes(int, int, int);
extern "C" int qwen3_moe_forward_bf16(const float*,
                                      const void*,
                                      void*,
                                      void*,
                                      void*,
                                      int,
                                      int,
                                      int,
                                      int,
                                      void*,
                                      size_t,
                                      cudaStream_t);
extern "C" int qwen35_moe_forward_bf16(const float*,
                                       const void*,
                                       void*,
                                       void*,
                                       void*,
                                       int,
                                       int,
                                       int,
                                       int,
                                       void*,
                                       size_t,
                                       cudaStream_t);
extern "C" int gemma4_moe_forward_bf16(const float*,
                                       const float*,
                                       const void*,
                                       void*,
                                       void*,
                                       void*,
                                       int,
                                       int,
                                       int,
                                       void*,
                                       size_t,
                                       cudaStream_t);
#ifdef TEI_TORCH_MOE_HOPPER
extern "C" size_t hopper_qwen3_moe_workspace_bytes(int, int, int);
extern "C" size_t hopper_qwen35_moe_workspace_bytes(int, int, int);
extern "C" size_t hopper_gemma4_moe_workspace_bytes(int, int, int);
extern "C" int hopper_qwen3_moe_forward_bf16(const float*,
                                             const void*,
                                             void*,
                                             void*,
                                             void*,
                                             int,
                                             int,
                                             int,
                                             int,
                                             void*,
                                             size_t,
                                             cudaStream_t);
extern "C" int hopper_qwen35_moe_forward_bf16(const float*,
                                              const void*,
                                              void*,
                                              void*,
                                              void*,
                                              int,
                                              int,
                                              int,
                                              int,
                                              void*,
                                              size_t,
                                              cudaStream_t);
extern "C" int hopper_gemma4_moe_forward_bf16(const float*,
                                              const float*,
                                              const void*,
                                              void*,
                                              void*,
                                              void*,
                                              int,
                                              int,
                                              int,
                                              void*,
                                              size_t,
                                              cudaStream_t);
#endif
#endif
namespace tei {
bool native_hopper_moe_available(const c10::Device& device) {
#if defined(TEI_TORCH_MOE_KERNELS) && defined(TEI_TORCH_MOE_HOPPER)
  if (!device.is_cuda())
    return false;
  auto props = device.has_index() ? at::cuda::getDeviceProperties(device.index())
                                  : at::cuda::getCurrentDeviceProperties();
  return props->major == 9;
#else
  return false;
#endif
}

at::Tensor routed_moe_cuda(const at::Tensor& x,
                           const at::Tensor& logits,
                           const at::Tensor& gu,
                           const at::Tensor& down,
                           bool renormalize,
                           const at::Tensor& scales) {
#ifdef TEI_TORCH_MOE_KERNELS
  if (!x.is_cuda() || x.scalar_type() != at::kBFloat16 || x.dim() != 2 || x.size(0) == 0 ||
      gu.dim() != 3 || down.dim() != 3 || logits.dim() != 2)
    return {};
  auto e = gu.size(0), h = x.size(1), i = down.size(2), t = x.size(0);
  if ((e != 128 && e != 256) || h % 8 || i % 8 || h > INT_MAX || i > INT_MAX / 2 || t > INT_MAX / 8)
    return {};
  if (scales.defined() && (e != 128 || h != 2816 || i != 704))
    return {};
  const c10::cuda::CUDAGuard guard(x.device());
  auto props = at::cuda::getCurrentDeviceProperties();
  if (props->major < 8)
    return {};
  TORCH_CHECK(gu.sizes() == at::IntArrayRef({e, 2 * i, h}) &&
                  down.sizes() == at::IntArrayRef({e, h, i}) &&
                  logits.sizes() == at::IntArrayRef({t, e}),
              "Invalid routed MoE dimensions");
  for (const auto& w : {gu, down})
    TORCH_CHECK(w.device() == x.device() && w.scalar_type() == at::kBFloat16 && w.is_contiguous(),
                "MoE weights must be contiguous BF16 on input device");
  TORCH_CHECK(x.is_contiguous() && logits.is_contiguous() && logits.device() == x.device() &&
                  logits.scalar_type() == at::kFloat,
              "MoE input/logits must be contiguous BF16/FP32 on same device");
  if (scales.defined())
    TORCH_CHECK(scales.device() == x.device() && scales.scalar_type() == at::kFloat &&
                    scales.is_contiguous() && scales.numel() == e,
                "Invalid MoE expert scales");
  auto bytes = scales.defined() ? gemma4_moe_workspace_bytes(t, h, i)
                                : (e == 128 ? qwen3_moe_workspace_bytes(t, h, i)
                                            : qwen35_moe_workspace_bytes(t, h, i));
  bool hopper = false;
#ifdef TEI_TORCH_MOE_HOPPER
  hopper = props->major == 9;
  if (hopper)
    bytes = scales.defined() ? hopper_gemma4_moe_workspace_bytes(t, h, i)
                             : (e == 128 ? hopper_qwen3_moe_workspace_bytes(t, h, i)
                                         : hopper_qwen35_moe_workspace_bytes(t, h, i));
#endif
  TORCH_CHECK(bytes, "Unsupported routed MoE workspace dimensions");
  auto scratch = at::empty({static_cast<int64_t>(bytes)}, x.options().dtype(at::kByte));
  auto out = at::empty_like(x);
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  int status;
#ifdef TEI_TORCH_MOE_HOPPER
  if (hopper) {
    if (scales.defined())
      status = hopper_gemma4_moe_forward_bf16(logits.data_ptr<float>(),
                                              scales.data_ptr<float>(),
                                              x.data_ptr(),
                                              gu.data_ptr(),
                                              down.data_ptr(),
                                              out.data_ptr(),
                                              t,
                                              h,
                                              i,
                                              scratch.data_ptr(),
                                              bytes,
                                              stream);
    else {
      auto fn = e == 128 ? hopper_qwen3_moe_forward_bf16 : hopper_qwen35_moe_forward_bf16;
      status = fn(logits.data_ptr<float>(),
                  x.data_ptr(),
                  gu.data_ptr(),
                  down.data_ptr(),
                  out.data_ptr(),
                  t,
                  h,
                  i,
                  renormalize,
                  scratch.data_ptr(),
                  bytes,
                  stream);
    }
  } else
#endif
  {
    if (scales.defined())
      status = gemma4_moe_forward_bf16(logits.data_ptr<float>(),
                                       scales.data_ptr<float>(),
                                       x.data_ptr(),
                                       gu.data_ptr(),
                                       down.data_ptr(),
                                       out.data_ptr(),
                                       t,
                                       h,
                                       i,
                                       scratch.data_ptr(),
                                       bytes,
                                       stream);
    else {
      auto fn = e == 128 ? qwen3_moe_forward_bf16 : qwen35_moe_forward_bf16;
      status = fn(logits.data_ptr<float>(),
                  x.data_ptr(),
                  gu.data_ptr(),
                  down.data_ptr(),
                  out.data_ptr(),
                  t,
                  h,
                  i,
                  renormalize,
                  scratch.data_ptr(),
                  bytes,
                  stream);
    }
  }
  TORCH_CHECK(status == 0, "Native routed MoE kernel failed: ", status);
  return out;
#else
  return {};
#endif
}
}

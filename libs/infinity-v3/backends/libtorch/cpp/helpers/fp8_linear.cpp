/// \file
/// Native FP8 linear execution and qualified ordinary cuBLASLt plan reuse.
/// Keeps plan, stream, workspace, and model-dtype rounding policies explicit.

#include "helpers/fp8_linear.h"
#include "helpers/workspace_admission.h"
#include "helpers/vendor_error.h"
#include <ATen/ops/_scaled_mm.h>
#include <string>
#include <mutex>
#include <thread>
#include <vector>
#include <cstdint>
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/tunable/Tunable.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/Context.h>
#include <ATen/cuda/Exceptions.h>
#include <cublasLt.h>
#include <cuda.h>
#include <c10/cuda/driver_api.h>
#include <c10/cuda/CUDAGuard.h>
#endif

namespace infinity {
/*
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

*/
// Ordinary Modern projections: no quantization or scale descriptors.
struct OrdinaryLinearExecutor::Impl {
#if defined(TEI_TORCH_CUDA_KERNELS) && CUDA_VERSION >= 12090
  /// Owns cuBLASLt descriptors, a selected algorithm, and optional retained workspace.
  struct Plan {
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t a{}, b{}, c{}, d{};
    cublasLtMatmulAlgo_t algorithm{};
    std::shared_ptr<WorkspaceMemoryPermit> memory_permit;
    at::Tensor workspace;
    bool rejected = false;

    ~Plan() {
      if (d)
        cublasLtMatrixLayoutDestroy(d);
      if (c)
        cublasLtMatrixLayoutDestroy(c);
      if (b)
        cublasLtMatrixLayoutDestroy(b);
      if (a)
        cublasLtMatrixLayoutDestroy(a);
      if (operation)
        cublasLtMatmulDescDestroy(operation);
    }
  };

  /// Execution-lane identity, cuBLASLt handle, and retained workspace for ordinary GEMMs.
  struct Lane {
    int device;
    CUcontext context;
    std::thread::id thread;
    cudaStream_t stream;
    cublasLtHandle_t handle;
    std::shared_ptr<WorkspaceMemoryPermit> memory_permit;
    at::Tensor workspace;
  };

  /// Cached GEMM geometry and algorithm plan associated with an execution lane.
  struct Entry {
    size_t lane, workspace_bytes;
    int64_t m, n, k;
    at::ScalarType output_dtype;
    std::unique_ptr<Plan> plan;
  };

  std::mutex mutex;
  std::vector<Lane> lanes;
  std::vector<Entry> entries;
  // Model-wide bounds: four execution lanes, 64 shapes/policies in total.
  // Capacity overflow uses ATen; existing plans/workspaces are never evicted.
  // Every lane allocates at most32MiB, shared by all its plans. Each plan
  // retains the allocation; changing workspace policy cannot free old storage.
  static constexpr size_t max_lanes = 4, max_plans = 64, max_workspace = 32 * 1024 * 1024;
#endif
};

OrdinaryLinearExecutor::OrdinaryLinearExecutor() : impl_(std::make_shared<Impl>()) {}

/// Execution identity and shared executor state retained by an ordinary-linear session.
struct OrdinaryLinearExecutor::Session::State {
#if defined(TEI_TORCH_CUDA_KERNELS) && CUDA_VERSION >= 12090
  std::shared_ptr<Impl> owner;
  int device;
  at::ScalarType dtype;
  c10::cuda::CUDAStream stream = c10::cuda::getDefaultCUDAStream();
  CUcontext context{};
  decltype(c10::cuda::DriverAPI::get()->cuCtxGetCurrent_) get_context{};
  cublasLtHandle_t handle{};
  size_t workspace_bytes{};
  std::thread::id thread;

  explicit State(std::shared_ptr<Impl> value) : owner(std::move(value)) {}
#endif
};

/// Snapshot an eligible device/context/stream/policy once for a synchronous forward. Unsupported or
/// capturing contexts produce an empty session and retain ordinary ATen fallback.
OrdinaryLinearExecutor::Session OrdinaryLinearExecutor::prepare(const at::Tensor& x) const {
  Session session;
#if defined(TEI_TORCH_CUDA_KERNELS) && CUDA_VERSION >= 12090
  if (!x.is_cuda() || (x.scalar_type() != at::kHalf && x.scalar_type() != at::kBFloat16))
    return session;
  c10::cuda::CUDAGuard guard(x.device());
  auto state = std::make_shared<Session::State>(impl_);
  const auto output_dtype = x.scalar_type();
  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  cudaStreamCaptureStatus capture;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(), &capture));
  if (capture != cudaStreamCaptureStatusNone)
    return session;
  if (at::cuda::tunable::getTuningContext()->IsTunableOpEnabled() ||
      at::globalContext()._SMCarveout_EXPERIMENTAL().has_value() ||
      (output_dtype == at::kHalf && at::globalContext().allowFP16AccumulationCuBLAS()) ||
      (output_dtype == at::kHalf ? at::globalContext().allowFP16ReductionCuBLAS()
                                 : at::globalContext().allowBF16ReductionCuBLAS()) !=
          at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK ||
      at::globalContext().deterministicAlgorithms() ||
      at::globalContext().blasPreferredBackend() != at::BlasBackend::Cublaslt)
    return session;
  // Reuse Torch's dynamically loaded driver API; no direct libcuda dependency.
  const auto get_context = c10::cuda::DriverAPI::get()->cuCtxGetCurrent_;
  CUcontext context{};
  if (!get_context || get_context(&context) != CUDA_SUCCESS || !context)
    return session;
  const auto workspace_bytes = at::cuda::getCUDABlasLtWorkspaceSize();
  if (workspace_bytes > Impl::max_workspace)
    return session;
  state->device = x.get_device();
  state->dtype = output_dtype;
  state->stream = stream;
  state->context = context;
  state->get_context = get_context;
  state->handle = at::cuda::getCurrentCUDABlasLtHandle();
  state->workspace_bytes = workspace_bytes;
  state->thread = std::this_thread::get_id();
  session.state_ = std::move(state);
#endif
  return session;
}

at::Tensor OrdinaryLinearExecutor::forward(const at::Tensor& x, const at::Tensor& weight) const {
#if defined(TEI_TORCH_CUDA_KERNELS) && CUDA_VERSION >= 12090
  c10::cuda::OptionalCUDAGuard guard;
  if (x.is_cuda())
    guard.set_device(x.device());
#endif
  auto session = prepare(x);
  return forward(x, weight, session);
}

at::Tensor OrdinaryLinearExecutor::forward(const at::Tensor& x,
                                           const at::Tensor& weight,
                                           const Session& session) const {
#if defined(TEI_TORCH_CUDA_KERNELS) && CUDA_VERSION >= 12090
  // Independent saved Half/BF16 ordinary packets define these layouts.
  // Unknown dtype/strides/policies retain original ATen.
  if (!x.is_cuda() || x.dim() != 2 || !weight.is_cuda() || weight.dim() != 2 ||
      (x.scalar_type() != at::kHalf && x.scalar_type() != at::kBFloat16) ||
      weight.scalar_type() != x.scalar_type() || x.device() != weight.device() || x.size(0) <= 0 ||
      x.size(1) != weight.size(1) || !x.is_contiguous() || !weight.is_contiguous())
    return {};
  const auto m = x.size(0), n = weight.size(0), k = x.size(1);
  const auto output_dtype = x.scalar_type();
  const auto* state = session.state_.get();
  if (!state || state->owner.get() != impl_.get() || state->device != x.get_device() ||
      state->dtype != output_dtype || state->thread != std::this_thread::get_id() ||
      state->stream != c10::cuda::getCurrentCUDAStream(x.get_device()))
    return {};
  // A stream guard or explicit context/device switch invalidates this request
  // snapshot. Decline before allocating or submitting to the captured handle.
  CUcontext active_context{};
  if (state->get_context(&active_context) != CUDA_SUCCESS || active_context != state->context)
    return {};
  const auto stream = state->stream;
  const auto context = state->context;
  const auto handle = state->handle;
  const auto workspace_bytes = state->workspace_bytes;
  const auto thread = state->thread;
  const auto aligned = [](const at::Tensor& tensor) {
    return reinterpret_cast<uintptr_t>(tensor.const_data_ptr()) % 256 == 0;
  };
  // Cached preferences below promise 256-byte operand alignment to cuBLASLt. Keep this pointer
  // check consistent with those descriptors; shape divisibility alone does
  // not prove address alignment.
  if (!aligned(x) || !aligned(weight))
    return {};
  auto& cache = *impl_;
  std::lock_guard<std::mutex> lock(cache.mutex);
  size_t lane = 0;
  for (; lane < cache.lanes.size(); ++lane) {
    const auto& v = cache.lanes[lane];
    if (v.device == x.get_device() && v.context == context && v.thread == thread &&
        v.stream == stream.stream() && v.handle == handle)
      break;
  }
  if (lane == cache.lanes.size()) {
    // Lane/plan caps bound retained descriptors and workspace. Reaching a cap declines caching, not
    // inference; the caller retains the ordinary projection path.
    if (lane == Impl::max_lanes)
      return {};
    cache.lanes.push_back({x.get_device(), context, thread, stream.stream(), handle, {}, {}});
  }
  Impl::Plan* plan = nullptr;
  for (auto& entry : cache.entries)
    if (entry.lane == lane && entry.workspace_bytes == workspace_bytes && entry.m == m &&
        entry.n == n && entry.k == k && entry.output_dtype == output_dtype) {
      plan = entry.plan.get();
      break;
    }
  if (!plan) {
    if (cache.entries.size() == Impl::max_plans)
      return {};
    auto owned = std::make_unique<Impl::Plan>();
    plan = owned.get();
    // Cache rejected geometry/policy too; no unbounded repeated host setup.
    cache.entries.push_back({lane, workspace_bytes, m, n, k, output_dtype, std::move(owned)});
    plan->rejected = true;
    const auto ok = [](cublasStatus_t status) { return status == CUBLAS_STATUS_SUCCESS; };
    if (!ok(cublasLtMatmulDescCreate(&plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F)))
      return {};
    const auto set = [&](cublasLtMatmulDescAttributes_t attr, const auto& value) {
      return ok(cublasLtMatmulDescSetAttribute(plan->operation, attr, &value, sizeof(value)));
    };
    const cublasOperation_t transpose = CUBLAS_OP_T, normal = CUBLAS_OP_N;
    const int8_t fast = 0;
    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    if (!set(CUBLASLT_MATMUL_DESC_TRANSA, transpose) || !set(CUBLASLT_MATMUL_DESC_TRANSB, normal) ||
        !set(CUBLASLT_MATMUL_DESC_FAST_ACCUM, fast) ||
        !set(CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue))
      return {};
    const auto type = output_dtype == at::kHalf ? CUDA_R_16F : CUDA_R_16BF;
    if (!ok(cublasLtMatrixLayoutCreate(&plan->a, type, k, n, k)) ||
        !ok(cublasLtMatrixLayoutCreate(&plan->b, type, k, m, k)) ||
        !ok(cublasLtMatrixLayoutCreate(&plan->c, type, n, m, n)) ||
        !ok(cublasLtMatrixLayoutCreate(&plan->d, type, n, m, n)))
      return {};

    /// RAII owner of a cuBLASLt heuristic preference descriptor.
    struct Preference {
      cublasLtMatmulPreference_t value{};

      ~Preference() {
        if (value)
          cublasLtMatmulPreferenceDestroy(value);
      }
    } preference;

    if (!ok(cublasLtMatmulPreferenceCreate(&preference.value)) ||
        !ok(cublasLtMatmulPreferenceSetAttribute(preference.value,
                                                 CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                 &workspace_bytes,
                                                 sizeof(workspace_bytes))))
      return {};
    // Remaining preference defaults match Torch: alignment256, reductionmask7.
    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    if (!ok(cublasLtMatmulAlgoGetHeuristic(handle,
                                           plan->operation,
                                           plan->a,
                                           plan->b,
                                           plan->c,
                                           plan->d,
                                           preference.value,
                                           1,
                                           &heuristic,
                                           &returned)) ||
        returned != 1 || heuristic.state != CUBLAS_STATUS_SUCCESS ||
        heuristic.workspaceSize > workspace_bytes)
      return {};
    plan->algorithm = heuristic.algo; // Original first tactic, not a custom search.
    plan->rejected = false;
  }
  if (plan->rejected)
    return {};
  // A supported plan remains cached when another executor temporarily owns
  // the shared workspace allowance. Retry admission without repeating setup.
  auto& workspace = cache.lanes[lane].workspace;
  if (workspace_bytes && !workspace.defined()) {
    auto permit = reserve_workspace_memory(x.get_device(), Impl::max_workspace);
    if (!permit)
      return {}; // Preserve the existing ATen fallback.
    workspace =
        at::empty({static_cast<int64_t>(Impl::max_workspace)}, x.options().dtype(at::kByte));
    cache.lanes[lane].memory_permit = std::move(permit);
  }
  plan->workspace = workspace; // Keep this exact allocation alive independently.
  plan->memory_permit = cache.lanes[lane].memory_permit;
  auto output = at::empty({m, n}, x.options().dtype(output_dtype));
  if (!aligned(output))
    return {};
  if (plan->workspace.defined())
    c10::cuda::CUDACachingAllocator::recordStream(plan->workspace.storage().data_ptr(), stream);
  const float alpha = 1.f, beta = 0.f;
  const auto status = cublasLtMatmul(handle,
                                     plan->operation,
                                     &alpha,
                                     weight.const_data_ptr(),
                                     plan->a,
                                     x.const_data_ptr(),
                                     plan->b,
                                     &beta,
                                     output.const_data_ptr(),
                                     plan->c,
                                     output.mutable_data_ptr(),
                                     plan->d,
                                     &plan->algorithm,
                                     workspace_bytes ? plan->workspace.mutable_data_ptr() : nullptr,
                                     workspace_bytes,
                                     stream.stream());
  if (status == CUBLAS_STATUS_ALLOC_FAILED)
    throw VendorAllocationError{"Ordinary packed projection cublasLtMatmul",
                                static_cast<int32_t>(status)};
  // Other failed tactics retain ATen's fallback through cuBLAS.
  if (status != CUBLAS_STATUS_SUCCESS) {
    plan->rejected = true;
    return {};
  }
  return output;
#else
  return {};
#endif
}

#ifndef TEI_TORCH_CUDA_KERNELS
std::pair<at::Tensor, at::Tensor> fp8_quantize_rows(const at::Tensor&) {
  TORCH_CHECK(false, "Dynamic FP8 requires a CUDA LibTorch build");
}
#endif
#if defined(TEI_TORCH_CUDA_KERNELS) && CUDA_VERSION >= 12090
namespace {
// Reuse the vendor's first heuristic for repeated projection geometries. Plans
// are thread-local; scale pointers are rebound for each invocation. Workspace
// remains owned by ATen, and capture retains ATen's graph-aware path.
struct ScaledPlan {
  int device;
  CUcontext context;
  cudaStream_t stream;
  cublasLtHandle_t handle;
  int64_t m, n, k;
  at::ScalarType dtype;
  size_t workspace_bytes;
  cublasLtMatmulDesc_t operation{};
  cublasLtMatrixLayout_t a{}, b{}, c{}, d{};
  cublasLtMatmulAlgo_t algorithm{};

  ~ScaledPlan() {
    if (a)
      cublasLtMatrixLayoutDestroy(a);
    if (b)
      cublasLtMatrixLayoutDestroy(b);
    if (c)
      cublasLtMatrixLayoutDestroy(c);
    if (d)
      cublasLtMatrixLayoutDestroy(d);
    if (operation)
      cublasLtMatmulDescDestroy(operation);
  }
};

at::Tensor scaled_mm_cached(const at::Tensor& input,
                            const at::Tensor& weight,
                            const at::Tensor& input_scale,
                            const at::Tensor& weight_scale,
                            at::ScalarType dtype) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  cudaStreamCaptureStatus capture;
  C10_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
  if (capture != cudaStreamCaptureStatusNone ||
      at::cuda::tunable::getTuningContext()->IsTunableOpEnabled() ||
      at::globalContext()._SMCarveout_EXPERIMENTAL().has_value() ||
      at::globalContext().deterministicAlgorithms())
    return {};
  // Match the default 256-byte alignment used by ATen's cuBLASLt heuristic.
  if ((reinterpret_cast<uintptr_t>(input.const_data_ptr()) % 256) ||
      (reinterpret_cast<uintptr_t>(weight.const_data_ptr()) % 256))
    return {};
  CUcontext context{};
  auto get_context = c10::cuda::DriverAPI::get()->cuCtxGetCurrent_;
  if (!get_context || get_context(&context) != CUDA_SUCCESS || !context)
    return {};
  auto handle = at::cuda::getCurrentCUDABlasLtHandle();
  auto workspace_bytes = at::cuda::getCUDABlasLtWorkspaceSize();
  const auto m = input.size(0), n = weight.size(0), k = input.size(1);
  static thread_local std::vector<std::unique_ptr<ScaledPlan>> plans;
  ScaledPlan* plan = nullptr;
  for (auto& entry : plans)
    if (entry->device == input.get_device() && entry->context == context &&
        entry->stream == stream.stream() && entry->handle == handle &&
        entry->workspace_bytes == workspace_bytes && entry->dtype == dtype && entry->m == m &&
        entry->n == n && entry->k == k) {
      plan = entry.get();
      break;
    }
  if (!plan) {
    auto entry = std::make_unique<ScaledPlan>();
    entry->device = input.get_device();
    entry->context = context;
    entry->stream = stream.stream();
    entry->handle = handle;
    entry->m = m;
    entry->n = n;
    entry->k = k;
    entry->dtype = dtype;
    entry->workspace_bytes = workspace_bytes;
    TORCH_CUDABLAS_CHECK(
        cublasLtMatmulDescCreate(&entry->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    auto set = [&](auto attribute, const auto& value) {
      TORCH_CUDABLAS_CHECK(
          cublasLtMatmulDescSetAttribute(entry->operation, attribute, &value, sizeof(value)));
    };
    const auto transpose = CUBLAS_OP_T, normal = CUBLAS_OP_N;
    const int8_t fast = 1;
    const int mode = CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    set(CUBLASLT_MATMUL_DESC_TRANSA, transpose);
    set(CUBLASLT_MATMUL_DESC_TRANSB, normal);
    set(CUBLASLT_MATMUL_DESC_FAST_ACCUM, fast);
    set(CUBLASLT_MATMUL_DESC_A_SCALE_MODE, mode);
    set(CUBLASLT_MATMUL_DESC_B_SCALE_MODE, mode);
    auto ws = weight_scale.const_data_ptr(), xs = input_scale.const_data_ptr();
    set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, ws);
    set(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, xs);
    auto output_type = dtype == at::kBFloat16 ? CUDA_R_16BF : CUDA_R_16F;
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&entry->a, CUDA_R_8F_E4M3, k, n, k));
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&entry->b, CUDA_R_8F_E4M3, k, m, k));
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&entry->c, output_type, n, m, n));
    TORCH_CUDABLAS_CHECK(cublasLtMatrixLayoutCreate(&entry->d, output_type, n, m, n));
    cublasLtMatmulPreference_t preference{};
    TORCH_CUDABLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    auto status = cublasLtMatmulPreferenceSetAttribute(preference,
                                                       CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                       &workspace_bytes,
                                                       sizeof(workspace_bytes));
    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    if (status == CUBLAS_STATUS_SUCCESS)
      status = cublasLtMatmulAlgoGetHeuristic(handle,
                                              entry->operation,
                                              entry->a,
                                              entry->b,
                                              entry->c,
                                              entry->d,
                                              preference,
                                              1,
                                              &heuristic,
                                              &returned);
    cublasLtMatmulPreferenceDestroy(preference);
    TORCH_CUDABLAS_CHECK(status);
    if (!returned)
      return {}; // Preserve ATen's supported full-accumulation handling.
    entry->algorithm = heuristic.algo;
    if (plans.size() == 128)
      plans.erase(plans.begin());
    plan = entry.get();
    plans.push_back(std::move(entry));
  }
  auto ws = weight_scale.const_data_ptr(), xs = input_scale.const_data_ptr();
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &ws, sizeof(ws)));
  TORCH_CUDABLAS_CHECK(cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &xs, sizeof(xs)));
  auto output = at::empty({m, n}, input.options().dtype(dtype));
  const float alpha = 1.f, beta = 0.f;
  auto status = cublasLtMatmul(handle,
                               plan->operation,
                               &alpha,
                               weight.const_data_ptr(),
                               plan->a,
                               input.const_data_ptr(),
                               plan->b,
                               &beta,
                               weight.const_data_ptr(),
                               plan->c,
                               output.mutable_data_ptr(),
                               plan->d,
                               &plan->algorithm,
                               at::cuda::getCUDABlasLtWorkspace(),
                               workspace_bytes,
                               stream);
  if (status == CUBLAS_STATUS_NOT_SUPPORTED)
    return {};
  TORCH_CUDABLAS_CHECK(status);
  return output;
}
}
#endif

Fp8Linear::Fp8Linear(const at::Tensor& weight) : dtype_(weight.scalar_type()), rowwise_(false) {
  TORCH_CHECK(weight.is_cuda() && weight.dim() == 2 && weight.size(0) > 0 && weight.size(1) > 0,
              "Dynamic FP8 requires nonempty CUDA matrix weights");
  TORCH_CHECK(dtype_ == at::kHalf || dtype_ == at::kBFloat16,
              "Dynamic FP8 requires an unquantized float16/bfloat16 checkpoint");
  // This FP8 projection interface requires aligned input/output feature widths. Its token-row count
  // is independent of that constraint and must not be padded to satisfy it.
  TORCH_CHECK(weight.size(0) % 16 == 0 && weight.size(1) % 16 == 0,
              "Dynamic FP8 projection widths must be divisible by16; tokens are never padded");
#ifdef TEI_TORCH_CUDA_KERNELS
  c10::cuda::CUDAGuard guard(weight.device());
  const auto* props = at::cuda::getDeviceProperties(weight.device().index());
  const int capability = props->major * 10 + props->minor;
  TORCH_CHECK(capability == 89 || capability == 90 || capability == 100 || capability == 120,
              "Dynamic FP8 requires SM89/90/100/120");
  rowwise_ = capability == 90;
#else
  TORCH_CHECK(false, "Dynamic FP8 requires a CUDA LibTorch build");
#endif
  TORCH_CHECK(at::isfinite(weight).all().item<bool>(),
              "Dynamic FP8 checkpoint weights must be finite");
  auto quantized = fp8_quantize_rows(weight.contiguous());
  matrix_ = std::move(quantized.first);
  scales_ = std::move(quantized.second);
  unit_ = at::ones({}, weight.options().dtype(at::kFloat));
}

at::Tensor Fp8Linear::forward(const at::Tensor& input, const at::Tensor& bias) const {
  TORCH_CHECK(input.dim() == 2 && input.device() == matrix_.device() &&
                  input.scalar_type() == dtype_ && input.size(1) == input_width(),
              "Invalid dynamic FP8 packed activations");
  TORCH_CHECK(!bias.defined() || (bias.dim() == 1 && bias.size(0) == output_width() &&
                                  bias.device() == input.device() && bias.scalar_type() == dtype_),
              "Invalid dynamic FP8 bias");
  if (input.size(0) == 0)
    return at::empty({0, output_width()}, input.options());
#ifdef TEI_TORCH_CUDA_KERNELS
  c10::cuda::CUDAGuard guard(input.device());
#endif
  auto quantized = fp8_quantize_rows(input.contiguous());
  return forward_quantized(quantized.first, quantized.second, bias);
}

/// Consume E4M3 [rows, input_width] activations and F32 [rows, 1] scales. Hopper uses rowwise
/// scaled GEMM; other qualified devices scale FP32 output before model-dtype rounding. Bias is
/// added after GEMM rounding, matching the unfused checkpoint recipe.
at::Tensor Fp8Linear::forward_quantized(const at::Tensor& input,
                                        const at::Tensor& row_scales,
                                        const at::Tensor& bias) const {
  TORCH_CHECK(input.dim() == 2 && input.device() == matrix_.device() &&
                  input.scalar_type() == at::kFloat8_e4m3fn && input.is_contiguous() &&
                  input.size(1) == input_width(),
              "Invalid prequantized FP8 activations");
  TORCH_CHECK(row_scales.dim() == 2 && row_scales.size(0) == input.size(0) &&
                  row_scales.size(1) == 1 && row_scales.device() == input.device() &&
                  row_scales.scalar_type() == at::kFloat && row_scales.is_contiguous(),
              "Invalid prequantized FP8 row scales");
  TORCH_CHECK(!bias.defined() || (bias.dim() == 1 && bias.size(0) == output_width() &&
                                  bias.device() == input.device() && bias.scalar_type() == dtype_),
              "Invalid dynamic FP8 bias");
  if (input.size(0) == 0)
    return at::empty({0, output_width()}, input.options().dtype(dtype_));
#ifdef TEI_TORCH_CUDA_KERNELS
  c10::cuda::CUDAGuard guard(input.device());
#endif
  at::Tensor output;
  if (rowwise_) {
#if defined(TEI_TORCH_CUDA_KERNELS) && CUDA_VERSION >= 12090
    output = scaled_mm_cached(input, matrix_, row_scales, scales_, dtype_);
#endif
    if (!output.defined()) {
      try {
        // Candle's Hopper recipe prefers fast accumulation. Some CUDA versions
        // lack fast tactics for short, wide-K shapes; preserve its full fallback.
        output = at::_scaled_mm(input, matrix_.t(), row_scales, scales_.t(), {}, {}, dtype_, true);
      } catch (const c10::Error& error) {
        if (std::string(error.what_without_backtrace()).find("CUBLAS_STATUS_NOT_SUPPORTED") ==
            std::string::npos)
          throw;
        output = at::_scaled_mm(input, matrix_.t(), row_scales, scales_.t(), {}, {}, dtype_, false);
      }
    }
  } else {
    // Match the portable recipe: scale full FP32 accumulation before the
    // single model-dtype rounding. No tensorwide activation scale is inferred.
    output = (at::_scaled_mm(input, matrix_.t(), unit_, unit_, {}, {}, at::kFloat, false) *
              row_scales * scales_.t())
                 .to(dtype_);
  }
  // Candle's quantized MLP applies bias after GEMM model-dtype rounding.
  return bias.defined() ? output + bias : output;
}
}

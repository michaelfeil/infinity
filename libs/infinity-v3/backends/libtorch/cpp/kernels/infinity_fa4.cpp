// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
/// \file
/// Loading and dispatch for optional architecture-specific InfinityFA4 bundles.
/// Validates supported layouts and keeps loaded bundle code resident for later calls.

#include "kernels/infinity_fa4.h"

#if defined(TEI_TORCH_CUDA_KERNELS) && defined(INFINITY_FA4_ROOT)
#include "env_config.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>

namespace tei {
namespace {
using Forward = int (*)(int,
                        int,
                        int,
                        void*,
                        void*,
                        void*,
                        void*,
                        void*,
                        void*,
                        void*,
                        int64_t,
                        int64_t,
                        int64_t,
                        int64_t,
                        int64_t,
                        double,
                        int,
                        int,
                        const int64_t*,
                        int);
using Deberta = int (*)(int,
                        int,
                        void*,
                        void*,
                        void*,
                        void*,
                        void*,
                        void*,
                        void*,
                        void*,
                        void*,
                        int64_t,
                        int64_t,
                        int64_t,
                        int64_t,
                        int64_t,
                        int);

/// Loaded bundle handle and exported attention/error entry points kept resident.
struct Bundle {
  void* handle;
  Forward forward;
  Deberta deberta;
  const char* (*error)();
  // Keep loaded code alive for CUDA graphs and concurrent in-flight requests.
};

bool enabled() {
  static const bool value = [] {
    const auto* setting = std::getenv(INFINITY_ENV("TORCH_FA4"));
    if (!setting)
      return true;
    const std::string s(setting);
    TORCH_CHECK(s == "0" || s == "1" || s == "false" || s == "true",
                "TORCH_FA4 must be 0, 1, false or true");
    return s == "1" || s == "true";
  }();
  return value;
}

void library_anchor() {}

Bundle* bundle(const at::Tensor& q) {
  if (!enabled() || !q.is_cuda() ||
      (q.scalar_type() != at::kHalf && q.scalar_type() != at::kBFloat16))
    return nullptr;
  auto p = at::cuda::getDeviceProperties(q.get_device());
  int sm = p->major * 10 + p->minor;
  // Other CUDA kernels may run through PTX on a GPU omitted from this build.
  constexpr int architectures[] = {INFINITY_FA4_ARCHITECTURES};
  if (std::find(std::begin(architectures), std::end(architectures), sm) == std::end(architectures))
    return nullptr;
  static std::mutex mutex;
  static std::map<int, std::unique_ptr<Bundle>> bundles;
  std::lock_guard lock(mutex);
  if (auto found = bundles.find(sm); found != bundles.end())
    return found->second.get();
  Dl_info info{};
  TORCH_CHECK(dladdr(reinterpret_cast<void*>(&library_anchor), &info) && info.dli_fname,
              "Cannot locate the InfinityFA4 runtime");
  auto relative =
      std::filesystem::path("infinityfa4") / ("sm_" + std::to_string(sm)) / "libinfinityfa4.so";
  auto path = std::filesystem::path(info.dli_fname).parent_path() / relative;
  if (!std::filesystem::exists(path))
    path = std::filesystem::path(INFINITY_FA4_ROOT) / ("sm_" + std::to_string(sm)) /
           "libinfinityfa4.so";
  TORCH_CHECK(
      std::filesystem::exists(path), "Missing InfinityFA4 bundle for SM", sm, ": ", path.string());
  auto handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  TORCH_CHECK(handle, "Cannot load InfinityFA4: ", dlerror());
  auto portable = reinterpret_cast<int (*)()>(dlsym(handle, "infinity_fa4_portable_host_v1"));
  if (!portable || portable() != 1) {
    TORCH_WARN("Ignoring FA4 bundle without portable host code; using attention fallback: ",
               path.string());
    dlclose(handle);
    bundles.emplace(sm, nullptr);
    return nullptr;
  }
  auto capability =
      reinterpret_cast<int (*)()>(dlsym(handle, "infinity_fa4_compute_capability_v1"));
  auto forward = reinterpret_cast<Forward>(dlsym(handle, "infinity_fa4_forward_v1"));
  auto deberta = reinterpret_cast<Deberta>(dlsym(handle, "infinity_fa4_deberta_v1"));
  auto error = reinterpret_cast<const char* (*)()>(dlsym(handle, "infinity_fa4_error_v1"));
  TORCH_CHECK(capability && forward && error && capability() == sm,
              "Incompatible InfinityFA4 bundle: ",
              path.string());
  auto result = std::make_unique<Bundle>(Bundle{handle, forward, deberta, error});
  auto* ptr = result.get();
  bundles.emplace(sm, std::move(result));
  return ptr;
}

bool layout(const at::Tensor& q) {
  return q.dim() == 3 && q.size(0) > 0 && q.size(1) > 0 && q.stride(2) == 1 &&
         q.stride(1) == q.size(2) && q.stride(0) >= q.size(1) * q.size(2) && q.stride(0) % 8 == 0 &&
         reinterpret_cast<uintptr_t>(q.const_data_ptr()) % 16 == 0;
}

void check_offsets(const at::Tensor& q, const at::Tensor& cu) {
  TORCH_CHECK(cu.device() == q.device() && cu.scalar_type() == at::kInt && cu.dim() == 1 &&
                  cu.numel() > 1 && cu.is_contiguous(),
              "InfinityFA4 requires CUDA int32 cumulative lengths");
}
}

/// Map eligible geometry/masks to exported kernel modes; undefined means no compatible export.
/// After an export is chosen, validate metadata and surface its launch
/// errors without retrying another provider.
at::Tensor infinity_fa4(const at::Tensor& q,
                        const at::Tensor& k,
                        const at::Tensor& v,
                        const at::Tensor& cu_q,
                        const at::Tensor& cu_k,
                        int64_t max_q,
                        double scale,
                        bool causal,
                        int64_t left,
                        int64_t right) {
  if (!layout(q) || !layout(k) || !layout(v) || k.sizes() != v.sizes() || q.size(2) != k.size(2) ||
      q.device() != k.device() || q.device() != v.device() || q.scalar_type() != k.scalar_type() ||
      q.scalar_type() != v.scalar_type() || !std::isfinite(scale) || scale <= 0)
    return {};
  int mode = -1;
  auto d = q.size(2), h = q.size(1), hk = k.size(1);
  if (d == 64 && h == hk && !causal) {
    if (left < 0 && right < 0)
      mode = 0;
    else if (left >= 0 && right >= 0)
      mode = 1;
  }
  if (d == 128 && h == 4 * hk && causal && left < 0 && (right < 0 || right == 0))
    mode = 2;
  if (d == 128 && h == 2 * hk && !causal && left < 0 && right < 0)
    mode = 3;
  if (mode < 0)
    return {};
  auto* native = bundle(q);
  if (!native)
    return {};
  c10::cuda::CUDAGuard guard(q.device());
  check_offsets(q, cu_q);
  check_offsets(q, cu_k);
  TORCH_CHECK(cu_q.numel() == cu_k.numel() && max_q > 0 && max_q <= INT32_MAX &&
                  q.size(0) <= INT32_MAX && k.size(0) <= INT32_MAX,
              "Invalid InfinityFA4 packed geometry");
  auto output = at::empty(q.sizes(), q.options());
  std::array<int64_t, 12> strides{};
  int at = 0;
  for (const auto& t : {q, k, v, output})
    for (auto stride : t.strides())
      strides[at++] = stride;
  auto ptr = [](const at::Tensor& t) { return const_cast<void*>(t.const_data_ptr()); };
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  int status = native->forward(mode,
                               q.scalar_type() == at::kHalf ? 0 : 1,
                               q.get_device(),
                               stream,
                               ptr(q),
                               ptr(k),
                               ptr(v),
                               ptr(output),
                               ptr(cu_q),
                               ptr(cu_k),
                               q.size(0),
                               k.size(0),
                               cu_q.numel() - 1,
                               h,
                               hk,
                               scale,
                               static_cast<int>(std::clamp<int64_t>(left, -1, INT32_MAX)),
                               static_cast<int>(std::clamp<int64_t>(right, -1, INT32_MAX)),
                               strides.data(),
                               static_cast<int>(max_q));
  TORCH_CHECK(status == 0, "InfinityFA4 attention failed: ", native->error());
  return output;
}

bool infinity_fa4_deberta_supported(const at::Tensor& q, int64_t head_dim) {
  if (head_dim != 64 || q.numel() == 0)
    return false;
  auto* native = bundle(q);
  return native && native->deberta;
}

at::Tensor infinity_fa4_deberta(const at::Tensor& q,
                                const at::Tensor& k,
                                const at::Tensor& v,
                                const at::Tensor& cumulative,
                                const at::Tensor& c2p,
                                const at::Tensor& p2c,
                                const at::Tensor& buckets,
                                int64_t max_sequence) {
  auto* native = bundle(q);
  TORCH_CHECK(native && native->deberta && q.dim() == 3 && q.size(2) == 64 &&
                  q.sizes() == k.sizes() && q.sizes() == v.sizes(),
              "Unsupported InfinityFA4 DeBERTa geometry");
  c10::cuda::CUDAGuard guard(q.device());
  check_offsets(q, cumulative);
  TORCH_CHECK(c2p.dim() == 3 && c2p.sizes() == p2c.sizes() && c2p.size(0) == q.size(1) &&
                  c2p.size(1) == q.size(0) && c2p.size(2) % 2 == 0 && buckets.dim() == 1 &&
                  buckets.scalar_type() == at::kInt && buckets.numel() == 2 * max_sequence - 1 &&
                  max_sequence > 0 && max_sequence <= INT32_MAX,
              "Invalid InfinityFA4 DeBERTa relative tables");
  for (const auto& t : {k, v, c2p, p2c})
    TORCH_CHECK(t.device() == q.device() && t.scalar_type() == q.scalar_type(),
                "InfinityFA4 DeBERTa tensor mismatch");
  TORCH_CHECK(buckets.device() == q.device() && buckets.is_contiguous(),
              "Invalid InfinityFA4 DeBERTa buckets");
  auto qc = q.contiguous(), kc = k.contiguous(), vc = v.contiguous();
  auto a = c2p.contiguous(), b = p2c.contiguous(), out = at::empty(q.sizes(), q.options());
  auto ptr = [](const at::Tensor& t) { return const_cast<void*>(t.const_data_ptr()); };
  auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
  int status = native->deberta(q.scalar_type() == at::kHalf ? 0 : 1,
                               q.get_device(),
                               stream,
                               ptr(qc),
                               ptr(kc),
                               ptr(vc),
                               ptr(out),
                               ptr(cumulative),
                               ptr(a),
                               ptr(b),
                               ptr(buckets),
                               q.size(0),
                               cumulative.numel() - 1,
                               q.size(1),
                               c2p.size(2) / 2,
                               buckets.numel(),
                               static_cast<int>(max_sequence));
  TORCH_CHECK(status == 0, "InfinityFA4 DeBERTa failed: ", native->error());
  return out;
}
}
#else
namespace tei {
at::Tensor infinity_fa4(const at::Tensor&,
                        const at::Tensor&,
                        const at::Tensor&,
                        const at::Tensor&,
                        const at::Tensor&,
                        int64_t,
                        double,
                        bool,
                        int64_t,
                        int64_t) {
  return {};
}

bool infinity_fa4_deberta_supported(const at::Tensor&, int64_t) {
  return false;
}

at::Tensor infinity_fa4_deberta(const at::Tensor&,
                                const at::Tensor&,
                                const at::Tensor&,
                                const at::Tensor&,
                                const at::Tensor&,
                                const at::Tensor&,
                                const at::Tensor&,
                                int64_t) {
  TORCH_CHECK(false, "InfinityFA4 was not included in this build");
}
}
#endif

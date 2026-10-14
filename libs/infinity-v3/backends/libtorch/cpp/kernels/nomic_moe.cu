// SPDX-License-Identifier: MIT
// BF16 routed expert GEMM. All problem sizes, matrix pointers and strides live
// on the GPU; the grouped scheduler does not read routing counts on the host.
#include <cutlass/cutlass.h>
#include <ATen/ATen.h>
#include "kernels/nomic_moe.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/epilogue/thread/linear_combination.h>

using Element = cutlass::bfloat16_t;
template <typename Output>
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    Output, 128 / cutlass::sizeof_bits<Output>::value, float, float>;
template <typename Output>
using Kernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
    Element, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 8,
    Output, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>, Epilogue<Output>,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle, 3,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;
template <typename Output>
using Gemm = cutlass::gemm::device::GemmGrouped<Kernel<Output>>;

template <typename Output>
int nomic_split_grouped_gemm(
    void *problem_sizes, int groups, void *a_ptrs, void *b_ptrs, void *c_ptrs,
    void *lda, void *ldb, void *ldc, cudaStream_t stream) {
    if (groups <= 0) return int(cutlass::Status::kErrorInvalidProblem);
    using Operation = Gemm<Output>;
    const int blocks = Operation::sufficient();
    if (blocks <= 0) return int(cutlass::Status::kErrorInternal);
    typename Operation::Arguments args(
        static_cast<cutlass::gemm::GemmCoord *>(problem_sizes), groups, blocks,
        {1.f, 0.f}, static_cast<Element **>(a_ptrs), static_cast<Element **>(b_ptrs),
        static_cast<Output **>(c_ptrs), static_cast<Output **>(c_ptrs),
        static_cast<int64_t *>(lda), static_cast<int64_t *>(ldb),
        static_cast<int64_t *>(ldc), static_cast<int64_t *>(ldc));
    Operation op;
    auto status = op.initialize(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) return int(status);
    return int(op.run(stream));
}

int nomic_grouped_f32(void* sizes, int groups, void* a, void* b, void* c,
    void* lda, void* ldb, void* ldc, cudaStream_t stream) {
  return nomic_split_grouped_gemm<float>(sizes, groups, a, b, c, lda, ldb, ldc, stream);
}

namespace tei {
__global__ void setup_split(const char* input, const char* weights, float* output,
    const int32_t* ends, const uint8_t* policy, int32_t* coords, int64_t* ptrs,
    int64_t* strides, int E, int rows, int K, int N, int64_t weight_stride) {
  int group = threadIdx.x;
  if (group >= 2 * E) return;
  int e = group % E, part = group / E;
  int start = e ? ends[e - 1] : 0, count = ends[e] - start;
  int factor = policy[count], span = K / factor;
  int kstart = factor == 2 ? part * span : 0;
  // A second group is empty when the reference GEMM does not split K.
  coords[3 * group] = part < factor ? count : 0;
  coords[3 * group + 1] = N;
  coords[3 * group + 2] = span;
  ptrs[group] = reinterpret_cast<int64_t>(input + 2 * (int64_t(start) * K + kstart));
  ptrs[2 * E + group] = reinterpret_cast<int64_t>(
      weights + 2 * (e * weight_stride + int64_t(kstart) * N));
  ptrs[4 * E + group] = reinterpret_cast<int64_t>(
      output + int64_t(part) * rows * N + int64_t(start) * N);
  strides[group] = K;
  strides[2 * E + group] = N;
  strides[4 * E + group] = N;
}

__global__ void reduce_split(const float* partials, __nv_bfloat16* output,
    const int32_t* ends, const uint8_t* policy, int E, int rows, int N) {
  int e = blockIdx.y, start = e ? ends[e - 1] : 0;
  int count = ends[e] - start, factor = policy[count];
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < int64_t(count) * N; index += int64_t(gridDim.x) * blockDim.x) {
    int64_t offset = int64_t(start) * N + index;
    float value = partials[offset];
    if (factor == 2) value = __fadd_rn(value, partials[int64_t(rows) * N + offset]);
    output[offset] = __float2bfloat16_rn(value);
  }
}

at::Tensor nomic_moe_down(const at::Tensor& a, const at::Tensor& b,
    const at::Tensor& ends, const at::Tensor& policy) {
  auto E = b.size(0), rows = a.size(0), K = a.size(1), N = b.size(2);
  // Scratch contains the two possible K partials for each actual route, not
  // padded expert rows. The final BF16 conversion follows the reference split.
  auto partials = at::empty({2, rows, N}, a.options().dtype(at::kFloat));
  auto result = at::empty({rows, N}, a.options());
  auto coords = at::empty({2 * E, 3}, a.options().dtype(at::kInt));
  auto ptrs = at::empty({3, 2 * E}, a.options().dtype(at::kLong));
  auto strides = at::empty_like(ptrs);
  auto stream = at::cuda::getCurrentCUDAStream();
  setup_split<<<1, 32, 0, stream>>>(static_cast<const char*>(a.data_ptr()),
      static_cast<const char*>(b.data_ptr()), partials.mutable_data_ptr<float>(),
      ends.const_data_ptr<int32_t>(), policy.const_data_ptr<uint8_t>(),
      coords.mutable_data_ptr<int32_t>(), ptrs.mutable_data_ptr<int64_t>(),
      strides.mutable_data_ptr<int64_t>(), E, rows, K, N, b.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto* p = ptrs.mutable_data_ptr<int64_t>();
  auto* l = strides.mutable_data_ptr<int64_t>();
  int status = nomic_grouped_f32(coords.data_ptr(), 2 * E, p, p + 2 * E,
      p + 4 * E, l, l + 2 * E, l + 4 * E, stream);
  TORCH_CHECK(status == 0, "Split group failed:", status);
  reduce_split<<<dim3(64, E), 256, 0, stream>>>(partials.const_data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(result.mutable_data_ptr()),
      ends.const_data_ptr<int32_t>(), policy.const_data_ptr<uint8_t>(), E, rows, N);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return result;
}
} // namespace tei

extern "C" int tei_gemma4_grouped_gemm_bf16(
    void*, int, void*, void*, void*, void*, void*, void*, cudaStream_t);

namespace tei {
namespace {
__global__ void setup_up(const char* input, const char* weights, char* output,
    const int32_t* ends, int32_t* coords, int64_t* pointers, int64_t* strides,
    int E, int K, int N, int64_t weight_stride) {
  int e = threadIdx.x;
  if (e >= E) return;
  int start = e ? ends[e - 1] : 0;
  coords[3 * e] = ends[e] - start;
  coords[3 * e + 1] = N;
  coords[3 * e + 2] = K;
  pointers[e] = reinterpret_cast<int64_t>(input + 2 * int64_t(start) * K);
  pointers[E + e] = reinterpret_cast<int64_t>(weights + 2 * e * weight_stride);
  pointers[2 * E + e] = reinterpret_cast<int64_t>(output + 2 * int64_t(start) * N);
  strides[e] = K;
  strides[E + e] = K;
  strides[2 * E + e] = N;
}
}

at::Tensor nomic_moe_up(const at::Tensor& a, const at::Tensor& first,
    const at::Tensor& ends) {
  const auto E = first.size(0), rows = a.size(0), K = a.size(1), N = first.size(1);
  auto result = at::empty({rows, N}, a.options());
  auto coords = at::empty({E, 3}, a.options().dtype(at::kInt));
  auto pointers = at::empty({3, E}, a.options().dtype(at::kLong));
  auto strides = at::empty_like(pointers);
  auto stream = at::cuda::getCurrentCUDAStream();
  setup_up<<<1, 32, 0, stream>>>(static_cast<const char*>(a.data_ptr()),
      static_cast<const char*>(first.data_ptr()), static_cast<char*>(result.data_ptr()),
      ends.const_data_ptr<int32_t>(), coords.mutable_data_ptr<int32_t>(),
      pointers.mutable_data_ptr<int64_t>(), strides.mutable_data_ptr<int64_t>(),
      E, K, N, first.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto* p = pointers.mutable_data_ptr<int64_t>();
  auto* l = strides.mutable_data_ptr<int64_t>();
  int status = tei_gemma4_grouped_gemm_bf16(coords.data_ptr(), E, p, p + E,
      p + 2 * E, l, l + E, l + 2 * E, stream);
  TORCH_CHECK(status == 0, "Grouped Nomic up failed:", status);
  return result;
}
} // namespace tei

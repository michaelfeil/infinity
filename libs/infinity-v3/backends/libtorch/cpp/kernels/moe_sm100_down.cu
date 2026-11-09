// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
/// \file
/// CUTLASS SM100 grouped down-projection with FP32 output.
/// The FP32 accumulator reaches the weighted readout, so no route is rounded
/// to BF16 before its routing weight is applied.

#include "kernels/moe_symbols.h"

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>
#include <cuda_runtime.h>

namespace infinity_moe_sm100 {
constexpr size_t kSm100WorkspaceBytes = 4 * 1024 * 1024;
using Element = cutlass::bfloat16_t;
using Problem = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
using Tile = cute::Shape<cute::_128, cute::_256, cute::_64>;
using Cluster = cute::Shape<cute::_2, cute::_1, cute::_1>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassTensorOp,
    Tile,
    Cluster,
    cutlass::epilogue::collective::EpilogueTileAuto,
    float,
    float,
    float,
    cutlass::layout::RowMajor*,
    4,
    float,
    cutlass::layout::RowMajor*,
    4,
    cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm,
    cutlass::epilogue::fusion::LinearCombination<float, float>>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassTensorOp,
    Element,
    cutlass::layout::RowMajor*,
    8,
    Element,
    cutlass::layout::ColumnMajor*,
    8,
    float,
    Tile,
    Cluster,
    cutlass::gemm::collective::StageCountAutoCarveout< static_cast<int>(
        sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Problem, Mainloop, Epilogue>;
using Op = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
using StrideA = typename Kernel::InternalStrideA;
using StrideB = typename Kernel::InternalStrideB;
using StrideC = typename Kernel::InternalStrideC;
using StrideD = typename Kernel::InternalStrideD;

struct Workspace {
  char* base;
  size_t offset = 0;

  template <typename T> T* take(size_t count) {
    offset = (offset + 255) & ~size_t(255);
    auto result = base ? reinterpret_cast<T*>(base + offset) : nullptr;
    offset += sizeof(T) * count;
    return result;
  }

  char* tail() {
    offset = (offset + 255) & ~size_t(255);
    return base ? base + offset : nullptr;
  }

  size_t remaining() const {
    return kSm100WorkspaceBytes > offset ? kSm100WorkspaceBytes - offset : 0;
  }
};

__global__ void setup_problems(const int* counts,
                               const int* offsets,
                               Element* activated,
                               Element* down,
                               float* output,
                               int hidden,
                               int intermediate,
                               Problem::UnderlyingProblemShape* problems,
                               const Element** a,
                               const Element** b,
                               const float** c,
                               float** d,
                               StrideA* lda,
                               StrideB* ldb,
                               StrideC* ldc,
                               StrideD* ldd) {
  const int e = threadIdx.x, rows = counts[e];
  problems[e] = Problem::UnderlyingProblemShape(rows, hidden, intermediate);
  a[e] = activated + int64_t(offsets[e]) * intermediate;
  b[e] = down + int64_t(e) * hidden * intermediate;
  c[e] = output + int64_t(offsets[e]) * hidden;
  d[e] = output + int64_t(offsets[e]) * hidden;
  lda[e] = cutlass::make_cute_packed_stride(StrideA{}, {rows, intermediate, 1});
  ldb[e] = cutlass::make_cute_packed_stride(StrideB{}, {hidden, intermediate, 1});
  ldc[e] = cutlass::make_cute_packed_stride(StrideC{}, {rows, hidden, 1});
  ldd[e] = cutlass::make_cute_packed_stride(StrideD{}, {rows, hidden, 1});
}
} // namespace infinity_moe_sm100

namespace infinity_moe {

bool sm100_down_available(int major, int minor) {
#if defined(INFINITY_TORCH_MOE_SM100_FAMILY)
  return major == 10 && (minor == 0 || minor == 3);
#else
  return major == 10 && minor == 0;
#endif
}

size_t sm100_down_workspace_bytes() {
  return infinity_moe_sm100::kSm100WorkspaceBytes;
}

int gemma4_moe_down_f32_sm100(const void* activated,
                              const void* down,
                              const int* counts,
                              const int* offsets,
                              float* output,
                              int experts,
                              int hidden,
                              int intermediate,
                              void* workspace,
                              size_t workspace_bytes,
                              cudaStream_t stream) {
  if (!workspace || workspace_bytes < infinity_moe_sm100::kSm100WorkspaceBytes)
    return -1;
  infinity_moe_sm100::Workspace w{static_cast<char*>(workspace)};
  auto* problems = w.take<infinity_moe_sm100::Problem::UnderlyingProblemShape>(experts);
  auto** a = w.take<const infinity_moe_sm100::Element*>(experts);
  auto** b = w.take<const infinity_moe_sm100::Element*>(experts);
  auto** c = w.take<const float*>(experts);
  auto** d = w.take<float*>(experts);
  auto* lda = w.take<infinity_moe_sm100::StrideA>(experts);
  auto* ldb = w.take<infinity_moe_sm100::StrideB>(experts);
  auto* ldc = w.take<infinity_moe_sm100::StrideC>(experts);
  auto* ldd = w.take<infinity_moe_sm100::StrideD>(experts);
  infinity_moe_sm100::setup_problems<<<1, experts, 0, stream>>>(
      counts,
      offsets,
      static_cast<infinity_moe_sm100::Element*>(const_cast<void*>(activated)),
      static_cast<infinity_moe_sm100::Element*>(const_cast<void*>(down)),
      output,
      hidden,
      intermediate,
      problems,
      a,
      b,
      c,
      d,
      lda,
      ldb,
      ldc,
      ldd);
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess)
    return -9;
  auto hardware =
      cutlass::KernelHardwareInfo::make_kernel_hardware_info< infinity_moe_sm100::Kernel>(device);
  typename infinity_moe_sm100::Op::Arguments args;
  typename infinity_moe_sm100::Kernel::TileSchedulerArguments scheduler{};
  decltype(args.epilogue.thread) fusion{};
  fusion.alpha = 1.f;
  fusion.beta = 0.f;
  args = {cutlass::gemm::GemmUniversalMode::kGrouped,
          {experts, problems, nullptr},
          {a, lda, b, ldb},
          {fusion, c, ldc, d, ldd},
          hardware,
          scheduler};
  char* gemm_workspace = w.tail();
  if (infinity_moe_sm100::Op::get_workspace_size(args) > w.remaining())
    return -8;
  infinity_moe_sm100::Op op;
  auto status = op.can_implement(args);
  if (status != cutlass::Status::kSuccess)
    return int(status);
  status = op.initialize(args, gemm_workspace, stream);
  if (status != cutlass::Status::kSuccess)
    return int(status);
  return int(op.run(stream));
}
} // namespace infinity_moe

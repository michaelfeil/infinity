// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
/// \file
/// CUTLASS 4.5 SM100 grouped FP8 expert GEMMs with rowwise FP32 scales.
#include "gemma_moe_fp8_grouped_sm100.h"

#include <cstdint>

#include <cutlass/cutlass.h>
#include <cutlass/functional.h>
#include <cutlass/numeric_types.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/dispatch_policy.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <cute/tensor.hpp>

namespace infinity_moe_fp8 {
namespace {

using ElementA = cutlass::float_e4m3_t;
using ElementB = cutlass::float_e4m3_t;
using DtypeScale = float;
using DtypeAccum = float;
using DtypeEpilogue = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;
constexpr int AlignmentB = 16;

using Problem = cutlass::gemm::GroupProblemShape<cute::Shape<int32_t, int32_t, int32_t>>;
using Tile = cute::Shape<cute::_128, cute::_128, cute::_128>;
using Cluster = cute::Shape<cute::_2, cute::_1, cute::_1>;

using Multiply =
    cutlass::epilogue::fusion::Sm90Compute< cutlass::multiplies,
                                            DtypeEpilogue,
                                            DtypeEpilogue,
                                            cutlass::FloatRoundStyle::round_to_nearest>;

using ScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<
    0,
    Tile,
    DtypeScale*,
    DtypeScale,
    cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>>;

using ScaleB = cutlass::epilogue::fusion::Sm90RowBroadcast<
    0,
    Tile,
    DtypeScale*,
    DtypeScale,
    cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>>;

using Accum = cutlass::epilogue::fusion::Sm90AccFetch;

template <typename ElementD> struct GroupedKernel {
  static constexpr int AlignmentD = 16 / sizeof(ElementD);

  using Cast = cutlass::epilogue::fusion::Sm90Compute< cutlass::epilogue::thread::Identity,
                                                       ElementD,
                                                       DtypeEpilogue,
                                                       cutlass::FloatRoundStyle::round_to_nearest>;

  using AccumScale = cutlass::epilogue::fusion::
      Sm90EVT< Multiply, ScaleB, cutlass::epilogue::fusion::Sm90EVT<Multiply, ScaleA, Accum>>;
  using EpilogueEVT = cutlass::epilogue::fusion::Sm90EVT<Cast, AccumScale>;

  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100,
      cutlass::arch::OpClassTensorOp,
      Tile,
      Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      DtypeAccum,
      DtypeEpilogue,
      void,
      LayoutD*,
      AlignmentD,
      ElementD,
      LayoutD*,
      AlignmentD,
      cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm,
      EpilogueEVT>::CollectiveOp;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100,
      cutlass::arch::OpClassTensorOp,
      ElementA,
      LayoutA*,
      AlignmentA,
      ElementB,
      LayoutB*,
      AlignmentB,
      DtypeAccum,
      Tile,
      Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout< static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmSm100>::CollectiveOp;

  using Kernel = cutlass::gemm::kernel::GemmUniversal<Problem, Mainloop, Epilogue>;
  using Op = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using StrideA = typename Kernel::InternalStrideA;
  using StrideB = typename Kernel::InternalStrideB;
  using StrideD = typename Kernel::InternalStrideD;
};

constexpr size_t kWorkspaceBytes = 32 * 1024 * 1024;

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
    return kWorkspaceBytes > offset ? kWorkspaceBytes - offset : 0;
  }
};

template <typename ElementD, typename StrideD>
__global__ void setup_problems(const int* counts,
                               const int* offsets,
                               const ElementA* a,
                               const ElementB* b,
                               ElementD* d,
                               const DtypeScale* scale_a,
                               const DtypeScale* scale_b,
                               int experts,
                               int hidden,
                               int width,
                               Problem::UnderlyingProblemShape* problems,
                               const ElementA** a_ptrs,
                               const ElementB** b_ptrs,
                               ElementD** d_ptrs,
                               DtypeScale** scale_a_ptrs,
                               DtypeScale** scale_b_ptrs,
                               typename GroupedKernel<ElementD>::StrideA* lda,
                               typename GroupedKernel<ElementD>::StrideB* ldb,
                               StrideD* ldd) {
  using Traits = GroupedKernel<ElementD>;
  using StrideA = typename Traits::StrideA;
  using StrideB = typename Traits::StrideB;
  const int e = threadIdx.x;
  if (e >= experts)
    return;
  const int rows = counts[e];
  const int start = offsets[e];
  problems[e] = Problem::UnderlyingProblemShape(rows, width, hidden);
  a_ptrs[e] = a + int64_t(start) * hidden;
  b_ptrs[e] = b + int64_t(e) * width * hidden;
  d_ptrs[e] = d + int64_t(start) * width;
  scale_a_ptrs[e] = const_cast<DtypeScale*>(scale_a) + start;
  scale_b_ptrs[e] = const_cast<DtypeScale*>(scale_b) + int64_t(e) * width;
  lda[e] = cutlass::make_cute_packed_stride(StrideA{}, {rows, hidden, 1});
  ldb[e] = cutlass::make_cute_packed_stride(StrideB{}, {width, hidden, 1});
  ldd[e] = cutlass::make_cute_packed_stride(StrideD{}, {rows, width, 1});
}

template <typename ElementD>
int grouped_gemm_impl(const void* packed_a,
                      const void* expert_b,
                      const float* scale_a,
                      const float* scale_b,
                      const int* counts,
                      const int* offsets,
                      void* output,
                      int experts,
                      int hidden,
                      int width,
                      void* workspace,
                      size_t workspace_bytes,
                      cudaStream_t stream) {
  if (!packed_a || !expert_b || !scale_a || !scale_b || !counts || !offsets || !output)
    return -2;
  if (!workspace || workspace_bytes < kWorkspaceBytes)
    return -1;
  if (experts <= 0 || experts > 1024 || hidden <= 0 || width <= 0 || (hidden % 64) != 0 ||
      (width % 8) != 0)
    return -2;

  using Selected = GroupedKernel<ElementD>;
  using SelectedKernel = typename Selected::Kernel;
  using SelectedOp = typename Selected::Op;
  Workspace w{static_cast<char*>(workspace)};
  auto* problems = w.take<Problem::UnderlyingProblemShape>(experts);
  auto** a_ptrs = w.take<const ElementA*>(experts);
  auto** b_ptrs = w.take<const ElementB*>(experts);
  auto** d_ptrs = w.take<ElementD*>(experts);
  auto** scale_a_ptrs = w.take<DtypeScale*>(experts);
  auto** scale_b_ptrs = w.take<DtypeScale*>(experts);
  auto* lda = w.take<typename Selected::StrideA>(experts);
  auto* ldb = w.take<typename Selected::StrideB>(experts);
  auto* ldd = w.take<typename Selected::StrideD>(experts);

  setup_problems<ElementD, typename Selected::StrideD>
      <<<1, experts, 0, stream>>>(counts,
                                  offsets,
                                  static_cast<const ElementA*>(packed_a),
                                  static_cast<const ElementB*>(expert_b),
                                  static_cast<ElementD*>(output),
                                  scale_a,
                                  scale_b,
                                  experts,
                                  hidden,
                                  width,
                                  problems,
                                  a_ptrs,
                                  b_ptrs,
                                  d_ptrs,
                                  scale_a_ptrs,
                                  scale_b_ptrs,
                                  lda,
                                  ldb,
                                  ldd);
  if (cudaGetLastError() != cudaSuccess)
    return -3;

  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess)
    return -9;

  auto hardware = cutlass::KernelHardwareInfo::make_kernel_hardware_info<SelectedKernel>(device);
  typename SelectedOp::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {experts, problems, nullptr},
      {a_ptrs, lda, b_ptrs, ldb},
      {{{{scale_b_ptrs}, {{scale_a_ptrs}, {}, {}}, {}}, {}}, nullptr, ldd, d_ptrs, ldd},
      hardware,
      typename SelectedKernel::TileSchedulerArguments{}};

  char* gemm_workspace = w.tail();
  if (SelectedOp::get_workspace_size(arguments) > w.remaining())
    return -8;

  SelectedOp op;
  auto status = op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess)
    return int(status);
  status = op.initialize(arguments, gemm_workspace, stream);
  if (status != cutlass::Status::kSuccess)
    return int(status);
  return int(op.run(stream));
}

} // namespace

size_t grouped_workspace_bytes() {
  return kWorkspaceBytes;
}

bool grouped_available(int major, int minor) {
#if defined(INFINITY_MOE_FP8_GROUPED_SM100_FAMILY)
  return major == 10 && (minor == 0 || minor == 3);
#else
  return major == 10 && minor == 0;
#endif
}

int grouped_gemm_bf16(const void* packed_a,
                      const void* expert_b,
                      const float* scale_a,
                      const float* scale_b,
                      const int* counts,
                      const int* offsets,
                      void* output,
                      int experts,
                      int hidden,
                      int width,
                      void* workspace,
                      size_t workspace_bytes,
                      cudaStream_t stream) {
  return grouped_gemm_impl<cutlass::bfloat16_t>(packed_a,
                                                expert_b,
                                                scale_a,
                                                scale_b,
                                                counts,
                                                offsets,
                                                output,
                                                experts,
                                                hidden,
                                                width,
                                                workspace,
                                                workspace_bytes,
                                                stream);
}

int grouped_gemm_f32(const void* packed_a,
                     const void* expert_b,
                     const float* scale_a,
                     const float* scale_b,
                     const int* counts,
                     const int* offsets,
                     void* output,
                     int experts,
                     int hidden,
                     int width,
                     void* workspace,
                     size_t workspace_bytes,
                     cudaStream_t stream) {
  return grouped_gemm_impl<float>(packed_a,
                                  expert_b,
                                  scale_a,
                                  scale_b,
                                  counts,
                                  offsets,
                                  output,
                                  experts,
                                  hidden,
                                  width,
                                  workspace,
                                  workspace_bytes,
                                  stream);
}

} // namespace infinity_moe_fp8

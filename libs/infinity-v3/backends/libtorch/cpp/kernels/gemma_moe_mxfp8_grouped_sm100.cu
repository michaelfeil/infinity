// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
/// \file
/// Standalone CUTLASS 4.5 SM100 MXFP8 grouped-expert provider.

#include "gemma_moe_mxfp8_grouped_sm100.h"

#include <cutlass/cutlass.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/dispatch_policy.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>
#include <cute/tensor.hpp>

#include <cstdint>

#include <cuda_runtime.h>

namespace infinity_moe_mxfp8 {
namespace {

constexpr size_t kWorkspaceBytes = 64 * 1024 * 1024;
constexpr int64_t kSFVecSize = 32;
constexpr int64_t kSFBlkMN = 128;
constexpr int64_t kSFBlk = 4;

__host__ __device__ inline int64_t round_up(int64_t value, int64_t multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

__host__ __device__ inline int64_t inner_blocks(int64_t cols) {
  return round_up(cols / kSFVecSize, kSFBlk);
}

// CUTLASS 4.5 SM100 block-scaled grouped configuration. The element pair and
// schedules follow example 92's MXF8F6F4 path, with GroupProblemShape and both
// A/B pointer arrays per expert.
using ElementInput = cutlass::float_e4m3_t;
using ElementA = cutlass::mx_float8_t<ElementInput>;
using ElementB = cutlass::mx_float8_t<ElementInput>;
using ElementSF = cutlass::float_ue8m0_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;
using Problem = cutlass::gemm::GroupProblemShape<cute::Shape<int32_t, int32_t, int32_t>>;
using Tile = cute::Shape<cute::_128, cute::_128, cute::_128>;
using Cluster = cute::Shape<cute::_1, cute::_1, cute::_1>;
constexpr int AlignmentAB = 16;
constexpr int AlignmentD = 16;

template <typename ElementD> struct GroupedKernel {
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100,
      cutlass::arch::OpClassTensorOp,
      Tile,
      Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto,
      float,
      float,
      void,
      LayoutD*,
      AlignmentD,
      ElementD,
      LayoutD*,
      AlignmentD,
      cutlass::epilogue::PtrArrayTmaWarpSpecialized1Sm>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100,
      cutlass::arch::OpClassBlockScaledTensorOp,
      ElementA,
      LayoutA*,
      AlignmentAB,
      ElementB,
      LayoutB*,
      AlignmentAB,
      float,
      Tile,
      Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout< static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Problem, Mainloop, Epilogue>;
  using Op = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using DataA = typename Kernel::ElementA;
  using DataB = typename Kernel::ElementB;
  using DataSF = typename Kernel::CollectiveMainloop::ElementSF;
  using StrideA = typename Kernel::InternalStrideA;
  using StrideB = typename Kernel::InternalStrideB;
  using StrideC = typename Kernel::InternalStrideC;
  using StrideD = typename Kernel::InternalStrideD;
  using LayoutSFA = typename Kernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB = typename Kernel::CollectiveMainloop::InternalLayoutSFB;
  using Sm1xx = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
};

struct Workspace {
  char* base;
  size_t offset = 0;

  template <typename T> T* take(size_t count) {
    offset = (offset + 255) & ~size_t(255);
    auto* result = base ? reinterpret_cast<T*>(base + offset) : nullptr;
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

template <typename ElementD>
__global__ void setup_problems(const int* counts,
                               const int* offsets,
                               const typename GroupedKernel<ElementD>::DataA* a,
                               const typename GroupedKernel<ElementD>::DataB* b,
                               const typename GroupedKernel<ElementD>::DataSF* sfa,
                               const typename GroupedKernel<ElementD>::DataSF* sfb,
                               ElementD* d,
                               int experts,
                               int hidden,
                               int width,
                               Problem::UnderlyingProblemShape* problems,
                               const typename GroupedKernel<ElementD>::DataA** a_ptrs,
                               const typename GroupedKernel<ElementD>::DataB** b_ptrs,
                               ElementD** d_ptrs,
                               const typename GroupedKernel<ElementD>::DataSF** sfa_ptrs,
                               const typename GroupedKernel<ElementD>::DataSF** sfb_ptrs,
                               typename GroupedKernel<ElementD>::StrideA* lda,
                               typename GroupedKernel<ElementD>::StrideB* ldb,
                               typename GroupedKernel<ElementD>::StrideD* ldd,
                               typename GroupedKernel<ElementD>::LayoutSFA* layouts_sfa,
                               typename GroupedKernel<ElementD>::LayoutSFB* layouts_sfb) {
  using Traits = GroupedKernel<ElementD>;
  using Sm1xx = typename Traits::Sm1xx;
  const int e = threadIdx.x;
  if (e >= experts)
    return;
  const int rows = counts[e];
  const int start = offsets[e];
  const int64_t inner = inner_blocks(hidden);
  const int64_t sfa_tile_bytes = inner * kSFBlkMN;
  // Zero-row groups never execute. Give their local SF layout one valid tile
  // so descriptor construction stays well-formed.
  const int32_t layout_rows = rows > 0 ? rows : static_cast<int32_t>(kSFBlkMN);

  problems[e] = Problem::UnderlyingProblemShape(rows, width, hidden);
  a_ptrs[e] = a + int64_t(start) * hidden;
  b_ptrs[e] = b + int64_t(e) * width * hidden;
  d_ptrs[e] = d + int64_t(start) * width;
  sfa_ptrs[e] = sfa + (int64_t(start) / kSFBlkMN) * sfa_tile_bytes;
  const auto shape = cute::make_shape(layout_rows, width, hidden, 1);
  const auto layout_sfa = Sm1xx::tile_atom_to_shape_SFA(shape);
  const auto layout_sfb = Sm1xx::tile_atom_to_shape_SFB(shape);
  layouts_sfa[e] = layout_sfa;
  layouts_sfb[e] = layout_sfb;
  const int64_t sfb_expert_bytes = static_cast<int64_t>(cute::size(cute::filter_zeros(layout_sfb)));
  sfb_ptrs[e] = sfb + int64_t(e) * sfb_expert_bytes;
  lda[e] = cutlass::make_cute_packed_stride(typename Traits::StrideA{}, {rows, hidden, 1});
  ldb[e] = cutlass::make_cute_packed_stride(typename Traits::StrideB{}, {width, hidden, 1});
  ldd[e] = cutlass::make_cute_packed_stride(typename Traits::StrideD{}, {rows, width, 1});
}

template <typename ElementD>
int grouped_gemm_impl(const void* packed_a,
                      const void* expert_b,
                      const void* packed_sfa,
                      const void* expert_sfb,
                      const int* counts,
                      const int* offsets,
                      void* output,
                      int experts,
                      int hidden,
                      int width,
                      void* workspace,
                      size_t workspace_bytes,
                      cudaStream_t stream) {
  if (!packed_a || !expert_b || !packed_sfa || !expert_sfb || !counts || !offsets || !output)
    return -2;
  if (!workspace || workspace_bytes < kWorkspaceBytes)
    return -1;
  if (experts <= 0 || experts > 1024 || hidden <= 0 || hidden % kSFVecSize != 0 || width <= 0 ||
      width % kSFBlkMN != 0)
    return -2;

  using Selected = GroupedKernel<ElementD>;
  using SelectedKernel = typename Selected::Kernel;
  using SelectedOp = typename Selected::Op;
  Workspace w{static_cast<char*>(workspace)};
  auto* problems = w.take<Problem::UnderlyingProblemShape>(experts);
  auto** a_ptrs = w.take<const typename Selected::DataA*>(experts);
  auto** b_ptrs = w.take<const typename Selected::DataB*>(experts);
  auto** d_ptrs = w.take<ElementD*>(experts);
  auto** sfa_ptrs = w.take<const typename Selected::DataSF*>(experts);
  auto** sfb_ptrs = w.take<const typename Selected::DataSF*>(experts);
  auto* lda = w.take<typename Selected::StrideA>(experts);
  auto* ldb = w.take<typename Selected::StrideB>(experts);
  auto* ldd = w.take<typename Selected::StrideD>(experts);
  auto* layouts_sfa = w.take<typename Selected::LayoutSFA>(experts);
  auto* layouts_sfb = w.take<typename Selected::LayoutSFB>(experts);

  setup_problems<ElementD>
      <<<1, experts, 0, stream>>>(counts,
                                  offsets,
                                  static_cast<const typename Selected::DataA*>(packed_a),
                                  static_cast<const typename Selected::DataB*>(expert_b),
                                  static_cast<const typename Selected::DataSF*>(packed_sfa),
                                  static_cast<const typename Selected::DataSF*>(expert_sfb),
                                  static_cast<ElementD*>(output),
                                  experts,
                                  hidden,
                                  width,
                                  problems,
                                  a_ptrs,
                                  b_ptrs,
                                  d_ptrs,
                                  sfa_ptrs,
                                  sfb_ptrs,
                                  lda,
                                  ldb,
                                  ldd,
                                  layouts_sfa,
                                  layouts_sfb);
  if (cudaGetLastError() != cudaSuccess)
    return -3;

  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess)
    return -9;
  auto hardware = cutlass::KernelHardwareInfo::make_kernel_hardware_info<SelectedKernel>(device);
  typename SelectedOp::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {experts, problems, nullptr},
      {a_ptrs, lda, b_ptrs, ldb, sfa_ptrs, layouts_sfa, sfb_ptrs, layouts_sfb},
      {{}, nullptr, nullptr, d_ptrs, ldd},
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
#if defined(INFINITY_MOE_MXFP8_GROUPED_SM100_FAMILY)
  return major == 10 && (minor == 0 || minor == 3);
#else
  return major == 10 && minor == 0;
#endif
}

size_t row_scale_inner_blocks(int64_t cols) {
  return cols > 0 && cols % kSFVecSize == 0 ? static_cast<size_t>(inner_blocks(cols)) : 0;
}

size_t row_scale_tile_bytes(int64_t cols) {
  return row_scale_inner_blocks(cols) * kSFBlkMN;
}

size_t row_scale_elements(int64_t rows, int64_t cols) {
  const auto inner = row_scale_inner_blocks(cols);
  return rows > 0 && inner ? static_cast<size_t>(round_up(rows, kSFBlkMN)) * inner : 0;
}

size_t row_scale_expert_offset(int64_t physical_row_start, int64_t cols) {
  if (physical_row_start < 0 || (physical_row_start % kSFBlkMN) != 0)
    return 0;
  return static_cast<size_t>(physical_row_start / kSFBlkMN) * row_scale_tile_bytes(cols);
}

size_t weight_scale_expert_bytes(int64_t width, int64_t hidden) {
  if (width <= 0 || width % kSFBlkMN != 0)
    return 0;
  return static_cast<size_t>(width / kSFBlkMN) * row_scale_tile_bytes(hidden);
}

int grouped_gemm_bf16(const void* packed_a,
                      const void* expert_b,
                      const void* packed_sfa,
                      const void* expert_sfb,
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
                                                packed_sfa,
                                                expert_sfb,
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
                     const void* packed_sfa,
                     const void* expert_sfb,
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
                                  packed_sfa,
                                  expert_sfb,
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

} // namespace infinity_moe_mxfp8

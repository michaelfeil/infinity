// SPDX-License-Identifier: MIT
// Hopper TMA/WGMMA expert GEMMs. Routing and pack/combine kernels are shared
// with the portable path; no routing metadata is transferred to the host.
#include "../../../src/kernels/gemma4_moe_kernels.cuh"

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_types.h>
#include <limits.h>
namespace gemma4_hopper {
// Fixed upper bound checked against CUTLASS before either expert GEMM launches.
constexpr size_t kGemmWorkspaceBytes = 2 * 1024 * 1024;
using namespace cute;
using Element = cutlass::bfloat16_t;
using Problem = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
template <typename Output, bool Weighted = false> struct Hopper {
    // Wider N tiles reuse input tiles across more output columns.
    using Tile = Shape<_128, _256, _64>;
    using Cluster = Shape<_1, _1, _1>;
    // Multiply FP32 accumulators by routing weights before the BF16 store.
    // This preserves the original combine kernel's per-expert rounding.
    using WeightedFusion = cutlass::epilogue::fusion::Sm90EVT<
        cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, Output, float,
                                               cutlass::FloatRoundStyle::round_to_nearest>,
        cutlass::epilogue::fusion::Sm90AccFetch,
        cutlass::epilogue::fusion::Sm90ColBroadcast<0, Tile, float *, float, Stride<_1, _0, _0>,
                                                    1>>;
    using Fusion = cute::conditional_t<Weighted, WeightedFusion,
                                       cutlass::epilogue::fusion::LinearCombination<Output, float>>;
    using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster,
        cutlass::epilogue::collective::EpilogueTileAuto, float, float, Output,
        cutlass::layout::RowMajor *, 128 / cutlass::sizeof_bits<Output>::value, Output,
        cutlass::layout::RowMajor *, 128 / cutlass::sizeof_bits<Output>::value,
        cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative, Fusion>::CollectiveOp;
    using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Element, cutlass::layout::RowMajor *,
        8, Element, cutlass::layout::ColumnMajor *, 8, float, Tile, Cluster,
        cutlass::gemm::collective::StageCountAutoCarveout<sizeof(typename Epilogue::SharedStorage)>,
        cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;
    using Kernel = cutlass::gemm::kernel::GemmUniversal<Problem, Mainloop, Epilogue>;
    using Op = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};
using StrideA = typename Hopper<Element>::Kernel::InternalStrideA;
using StrideB = typename Hopper<Element>::Kernel::InternalStrideB;
using StrideC = typename Hopper<Element>::Kernel::InternalStrideC;
template <typename Output, bool Weighted = false>
int grouped_gemm(void *shapes, int groups, void *a, void *b, void *c, void *lda, void *ldb,
                 void *ldc, void *workspace, cudaStream_t stream, float **row_weights = nullptr) {
    using G = typename Hopper<Output, Weighted>::Op;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess)
        return -9;
    auto hw = cutlass::KernelHardwareInfo::make_kernel_hardware_info<
        typename Hopper<Output, Weighted>::Kernel>(dev);
    typename G::Arguments args;
    decltype(args.epilogue.thread) fusion{};
    if constexpr (Weighted) {
        fusion = {{}, {row_weights, 1.f, {}}, {}};
    } else {
        fusion.alpha = 1.f;
        fusion.beta = 0.f;
    }
    args = {cutlass::gemm::GemmUniversalMode::kGrouped,
            {groups, static_cast<Problem::UnderlyingProblemShape *>(shapes), nullptr},
            {static_cast<const Element **>(a), static_cast<StrideA *>(lda),
             static_cast<const Element **>(b), static_cast<StrideB *>(ldb)},
            {fusion, static_cast<const Output **>(c), static_cast<StrideC *>(ldc),
             static_cast<Output **>(c), static_cast<StrideC *>(ldc)},
            hw};
    if (G::get_workspace_size(args) > kGemmWorkspaceBytes)
        return -8;
    G op;
    auto status = op.can_implement(args);
    if (status != cutlass::Status::kSuccess)
        return int(status);
    status = op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess)
        return int(status);
    return int(op.run(stream));
}

namespace hopper_detail {
struct Workspace {
    char *base;
    size_t offset = 0;
    template <typename T> T *take(size_t count) {
        offset = (offset + 255) & ~size_t(255);
        auto result = base ? reinterpret_cast<T *>(base + offset) : nullptr;
        offset += sizeof(T) * count;
        return result;
    }
};
struct Buffers {
    int *counts, *offsets, *cursors;
    uint32_t *ids, *mapping;
    float *weights;
    Element *packed, *gate_up, *activated;
    void *expert_output;
    float *sorted_weights;
    float **row_weights;
    Problem::UnderlyingProblemShape *problems;
    Element **a, **b, **c;
    float **c_f32;
    StrideA *lda;
    StrideB *ldb;
    StrideC *ldc;
    void *workspace;
    Buffers(Workspace &w, size_t tokens, size_t hidden, size_t intermediate,
            size_t experts = 128, bool weighted = false) {
        const size_t slots = tokens * 8;
        counts = w.take<int>(experts);
        offsets = w.take<int>(experts + 1);
        cursors = w.take<int>(experts);
        ids = w.take<uint32_t>(slots);
        mapping = w.take<uint32_t>(slots);
        weights = w.take<float>(slots);
        packed = w.take<Element>(slots * hidden);
        gate_up = w.take<Element>(slots * 2 * intermediate);
        activated = w.take<Element>(slots * intermediate);
        expert_output = weighted ? static_cast<void *>(w.take<Element>(slots * hidden))
                                 : static_cast<void *>(w.take<float>(slots * hidden));
        sorted_weights = weighted ? w.take<float>(slots) : nullptr;
        row_weights = weighted ? w.take<float *>(experts) : nullptr;
        problems = w.take<Problem::UnderlyingProblemShape>(experts);
        a = w.take<Element *>(experts);
        b = w.take<Element *>(experts);
        c = w.take<Element *>(experts);
        c_f32 = w.take<float *>(experts);
        lda = w.take<StrideA>(experts);
        ldb = w.take<StrideB>(experts);
        ldc = w.take<StrideC>(experts);
        workspace = w.take<char>(kGemmWorkspaceBytes);
    }
};
template <typename Output>
__global__ void setup_problems(const int *counts, const int *offsets, Element *input,
                               Element *weight, Output *output, int input_width, int output_width,
                               Problem::UnderlyingProblemShape *problems, Element **a, Element **b,
                               Output **c, StrideA *lda, StrideB *ldb, StrideC *ldc) {
    int e = threadIdx.x;
    problems[e] = Problem::UnderlyingProblemShape(counts[e], output_width, input_width);
    a[e] = input + int64_t(offsets[e]) * input_width;
    b[e] = weight + int64_t(e) * input_width * output_width;
    c[e] = output + int64_t(offsets[e]) * output_width;
    lda[e] = StrideA{int64_t(input_width), _1{}, _0{}};
    ldb[e] = StrideB{int64_t(input_width), _1{}, _0{}};
    ldc[e] = StrideC{int64_t(output_width), _1{}, _0{}};
}
__global__ void sort_weights(const uint32_t *mapping, const float *weights, float *sorted,
                             int slots) {
    const uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < uint64_t(slots))
        sorted[mapping[i]] = weights[i];
}
__global__ void weight_pointers(const int *offsets, float *sorted, float **rows) {
    int e = threadIdx.x;
    rows[e] = sorted + offsets[e];
}
__global__ void combine_weighted(const __nv_bfloat16 *input, const uint32_t *mapping,
                                 __nv_bfloat16 *output, int tokens, int hidden) {
    uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < uint64_t(tokens) * hidden) {
        uint64_t t = i / hidden, c = i % hidden;
        float sum = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k)
            sum += __bfloat162float(input[uint64_t(mapping[t * 8 + k]) * hidden + c]);
        output[i] = __float2bfloat16_rn(sum);
    }
}
} // namespace hopper_detail

using namespace hopper_detail;
static size_t routed_workspace_bytes(int tokens, int hidden, int intermediate, int experts = 128,
                                     bool weighted = false) {
    if (tokens <= 0 || tokens > INT_MAX / 8 || hidden <= 0 || hidden % 8 || intermediate <= 0 ||
        intermediate % 8 || intermediate > INT_MAX / 2)
        return 0;
    if (uint64_t(tokens) * 8 >
        (SIZE_MAX - 4 * 1024 * 1024) / (uint64_t(hidden) * 6 + uint64_t(intermediate) * 6 + 64))
        return 0;
    Workspace w{nullptr};
    Buffers b(w, tokens, hidden, intermediate, experts, weighted);
    return (w.offset + 255) & ~size_t(255);
}

extern "C" size_t hopper_gemma4_moe_workspace_bytes(int tokens, int hidden, int intermediate) {
    return hidden == 2816 && intermediate == 704
               ? routed_workspace_bytes(tokens, hidden, intermediate, 128, true)
               : 0;
}
extern "C" size_t hopper_qwen3_moe_workspace_bytes(int tokens, int hidden, int intermediate) {
    return routed_workspace_bytes(tokens, hidden, intermediate);
}

// The caller owns scratch storage and its stream lifetime. No allocation,
// host transfer of routing metadata, or device synchronization occurs here.
static int routed_forward_bf16(const float *logits, const float *scales, const void *input,
                               void *gate_up_weight, void *down_weight, void *output, int tokens,
                               int hidden, int intermediate, void *scratch, size_t scratch_bytes,
                               cudaStream_t stream, bool qwen, bool renormalize,
                               int experts = 128) {
    const size_t required = qwen ? routed_workspace_bytes(tokens, hidden, intermediate, experts)
                                 : hopper_gemma4_moe_workspace_bytes(tokens, hidden, intermediate);
    if (!required || scratch_bytes < required || !scratch)
        return -1;
    Workspace w{static_cast<char *>(scratch)};
    Buffers b(w, tokens, hidden, intermediate, experts, !qwen);
    int slots = tokens * 8;
    if (cudaMemsetAsync(b.counts, 0, experts * sizeof(int), stream) != cudaSuccess ||
        cudaMemsetAsync(b.cursors, 0, experts * sizeof(int), stream) != cudaSuccess)
        return -2;
    if (experts == 256)
        qwen35_moe_route_256_8_f32<<<tokens, 256, 0, stream>>>(logits, b.ids, b.weights,
                                                               renormalize);
    else if (qwen)
        qwen3_moe_route_128_8_f32<<<tokens, 128, 0, stream>>>(logits, b.ids, b.weights,
                                                              renormalize);
    else
        gemma4_moe_route_128_8_f32<<<tokens, 128, 0, stream>>>(logits, scales, b.ids, b.weights);
    gemma4_moe_count<<<(slots + 255) / 256, 256, 0, stream>>>(b.ids, b.counts, slots);
    gemma4_moe_offsets<<<1, experts, 0, stream>>>(b.counts, b.offsets);
    gemma4_moe_assign<<<(slots + 255) / 256, 256, 0, stream>>>(b.ids, b.offsets, b.cursors,
                                                               b.mapping, slots);
    if ((reinterpret_cast<uintptr_t>(input) & 15) == 0) {
        gemma4_moe_pack_vec8<<<(uint64_t(slots) * (hidden / 8) + 255) / 256, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(input), b.mapping,
            reinterpret_cast<__nv_bfloat16 *>(b.packed), slots, hidden);
    } else {
        gemma4_moe_pack<<<(uint64_t(slots) * hidden + 255) / 256, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(input), b.mapping,
            reinterpret_cast<__nv_bfloat16 *>(b.packed), slots, hidden);
    }
    setup_problems<<<1, experts, 0, stream>>>(
        b.counts, b.offsets, b.packed, static_cast<Element *>(gate_up_weight), b.gate_up, hidden,
        2 * intermediate, b.problems, b.a, b.b, b.c, b.lda, b.ldb, b.ldc);
    int status = grouped_gemm<Element>(b.problems, experts, b.a, b.b, b.c, b.lda, b.ldb, b.ldc,
                                       b.workspace, stream);
    if (status)
        return status;
    if (qwen) {
        qwen3_moe_silu_mul<<<slots, 128, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(b.gate_up),
            reinterpret_cast<__nv_bfloat16 *>(b.activated), slots, intermediate);
    } else {
        gemma4_moe_gelu_mul<<<slots, 128, 0, stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(b.gate_up),
            reinterpret_cast<__nv_bfloat16 *>(b.activated), slots, intermediate);
    }
    if (qwen) {
        setup_problems<<<1, experts, 0, stream>>>(
            b.counts, b.offsets, b.activated, static_cast<Element *>(down_weight),
            static_cast<float *>(b.expert_output), intermediate, hidden, b.problems, b.a, b.b, b.c_f32, b.lda, b.ldb, b.ldc);
        status = grouped_gemm<float>(b.problems, experts, b.a, b.b, b.c_f32, b.lda, b.ldb, b.ldc,
                                     b.workspace, stream);
        if (status)
            return status;
        if ((reinterpret_cast<uintptr_t>(output) & 15) == 0) {
            gemma4_moe_combine_vec8<<<(uint64_t(tokens) * (hidden / 8) + 255) / 256, 256, 0, stream>>>(
                static_cast<float *>(b.expert_output), b.mapping, b.weights,
                static_cast<__nv_bfloat16 *>(output), tokens, hidden);
        } else {
            gemma4_moe_unpermute_combine<<<(uint64_t(tokens) * hidden + 255) / 256, 256, 0, stream>>>(
                static_cast<float *>(b.expert_output), b.mapping, b.weights,
                static_cast<__nv_bfloat16 *>(output), tokens, hidden);
        }
    } else {
        setup_problems<<<1, experts, 0, stream>>>(
            b.counts, b.offsets, b.activated, static_cast<Element *>(down_weight),
            static_cast<Element *>(b.expert_output), intermediate, hidden, b.problems, b.a, b.b,
            b.c, b.lda, b.ldb, b.ldc);
        sort_weights<<<(uint64_t(slots) + 255) / 256, 256, 0, stream>>>(
            b.mapping, b.weights, b.sorted_weights, slots);
        weight_pointers<<<1, experts, 0, stream>>>(b.offsets, b.sorted_weights, b.row_weights);
        status = grouped_gemm<Element, true>(b.problems, experts, b.a, b.b, b.c, b.lda, b.ldb,
                                           b.ldc, b.workspace, stream, b.row_weights);
        if (status)
            return status;
        combine_weighted<<<(uint64_t(tokens) * hidden + 255) / 256, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16 *>(b.expert_output), b.mapping,
            static_cast<__nv_bfloat16 *>(output), tokens, hidden);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

extern "C" int hopper_gemma4_moe_forward_bf16(const float *logits, const float *scales,
                                              const void *input, void *gate_up_weight,
                                              void *down_weight, void *output, int tokens,
                                              int hidden, int intermediate, void *scratch,
                                              size_t scratch_bytes, cudaStream_t stream) {
    return routed_forward_bf16(logits, scales, input, gate_up_weight, down_weight, output, tokens,
                               hidden, intermediate, scratch, scratch_bytes, stream, false, true);
}
extern "C" int hopper_qwen3_moe_forward_bf16(const float *logits, const void *input,
                                             void *gate_up_weight, void *down_weight, void *output,
                                             int tokens, int hidden, int intermediate,
                                             int renormalize, void *scratch, size_t scratch_bytes,
                                             cudaStream_t stream) {
    return routed_forward_bf16(logits, nullptr, input, gate_up_weight, down_weight, output, tokens,
                               hidden, intermediate, scratch, scratch_bytes, stream, true,
                               renormalize != 0);
}

extern "C" size_t hopper_qwen35_moe_workspace_bytes(int tokens, int hidden, int intermediate) {
    return routed_workspace_bytes(tokens, hidden, intermediate, 256);
}
extern "C" int hopper_qwen35_moe_forward_bf16(const float *logits, const void *input,
                                              void *gate_up_weight, void *down_weight, void *output,
                                              int tokens, int hidden, int intermediate,
                                              int renormalize, void *scratch, size_t scratch_bytes,
                                              cudaStream_t stream) {
    return routed_forward_bf16(logits, nullptr, input, gate_up_weight, down_weight, output, tokens,
                               hidden, intermediate, scratch, scratch_bytes, stream, true,
                               renormalize != 0, 256);
}

} // namespace gemma4_hopper

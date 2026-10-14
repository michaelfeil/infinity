// SPDX-License-Identifier: MIT
// BF16 routed expert GEMM. All problem sizes, matrix pointers and strides live
// on the GPU; the grouped scheduler does not read routing counts on the host.
#include <cutlass/cutlass.h>
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
    Element, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 8,
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
int grouped_gemm(
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

// Standalone primitive entry point used by the grouped-GEMM diagnostic.
extern "C" int gemma4_grouped_gemm_bf16(
    void *sizes, int groups, void *a, void *b, void *c,
    void *lda, void *ldb, void *ldc, cudaStream_t stream) {
    return grouped_gemm<Element>(sizes, groups, a, b, c, lda, ldb, ldc, stream);
}

#include "../../../src/kernels/gemma4_moe.cu"
#include <limits.h>

namespace {
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
    float *expert_output;
    cutlass::gemm::GemmCoord *problems;
    Element **a, **b, **c;
    float **c_f32;
    int64_t *lda, *ldb, *ldc;
    Buffers(Workspace &w, size_t tokens, size_t hidden, size_t intermediate, size_t experts = 128) {
        const size_t slots = tokens * 8;
        counts = w.take<int>(experts); offsets = w.take<int>(experts + 1); cursors = w.take<int>(experts);
        ids = w.take<uint32_t>(slots); mapping = w.take<uint32_t>(slots); weights = w.take<float>(slots);
        packed = w.take<Element>(slots * hidden);
        gate_up = w.take<Element>(slots * 2 * intermediate);
        activated = w.take<Element>(slots * intermediate);
        expert_output = w.take<float>(slots * hidden);
        problems = w.take<cutlass::gemm::GemmCoord>(experts);
        a = w.take<Element *>(experts); b = w.take<Element *>(experts); c = w.take<Element *>(experts);
        c_f32 = w.take<float *>(experts);
        lda = w.take<int64_t>(experts); ldb = w.take<int64_t>(experts); ldc = w.take<int64_t>(experts);
    }
};
template <typename Output>
__global__ void setup_problems(const int *counts, const int *offsets,
    Element *input, Element *weight, Output *output, int input_width, int output_width,
    cutlass::gemm::GemmCoord *problems, Element **a, Element **b, Output **c,
    int64_t *lda, int64_t *ldb, int64_t *ldc) {
    int e = threadIdx.x;
    problems[e] = cutlass::gemm::GemmCoord(counts[e], output_width, input_width);
    a[e] = input + int64_t(offsets[e]) * input_width;
    b[e] = weight + int64_t(e) * input_width * output_width;
    c[e] = output + int64_t(offsets[e]) * output_width;
    lda[e] = input_width; ldb[e] = input_width; ldc[e] = output_width;
}
}

static size_t routed_workspace_bytes(int tokens, int hidden, int intermediate, int experts = 128) {
    if (tokens <= 0 || tokens > INT_MAX / 8 || hidden <= 0 || hidden % 8 || intermediate <= 0 || intermediate % 8 || intermediate > INT_MAX / 2) return 0;
    if (uint64_t(tokens) * 8 > (SIZE_MAX - 4 * 1024 * 1024) /
        (uint64_t(hidden) * 6 + uint64_t(intermediate) * 6 + 64)) return 0;
    Workspace w{nullptr}; Buffers b(w, tokens, hidden, intermediate, experts);
    return (w.offset + 255) & ~size_t(255);
}

extern "C" size_t gemma4_moe_workspace_bytes(int tokens, int hidden, int intermediate) {
    return hidden == 2816 && intermediate == 704 ? routed_workspace_bytes(tokens, hidden, intermediate) : 0;
}
extern "C" size_t qwen3_moe_workspace_bytes(int tokens, int hidden, int intermediate) {
    return routed_workspace_bytes(tokens, hidden, intermediate);
}

// The caller owns scratch storage and its stream lifetime. No allocation,
// host transfer of routing metadata, or device synchronization occurs here.
static int routed_forward_bf16(
    const float *logits, const float *scales, const void *input,
    void *gate_up_weight, void *down_weight, void *output,
    int tokens, int hidden, int intermediate, void *scratch, size_t scratch_bytes,
    cudaStream_t stream, bool qwen, bool renormalize, int experts = 128) {
    const size_t required = qwen ? routed_workspace_bytes(tokens, hidden, intermediate, experts) : gemma4_moe_workspace_bytes(tokens, hidden, intermediate);
    if (!required || scratch_bytes < required || !scratch) return -1;
    Workspace w{static_cast<char *>(scratch)}; Buffers b(w, tokens, hidden, intermediate, experts);
    int slots = tokens * 8;
    if (cudaMemsetAsync(b.counts, 0, experts * sizeof(int), stream) != cudaSuccess ||
        cudaMemsetAsync(b.cursors, 0, experts * sizeof(int), stream) != cudaSuccess) return -2;
    if (experts == 256) qwen35_moe_route_256_8_f32<<<tokens,256,0,stream>>>(logits,b.ids,b.weights,renormalize);
    else if (qwen) qwen3_moe_route_128_8_f32<<<tokens,128,0,stream>>>(logits,b.ids,b.weights,renormalize);
    else gemma4_moe_route_128_8_f32<<<tokens,128,0,stream>>>(logits,scales,b.ids,b.weights);
    gemma4_moe_count<<<(slots+255)/256,256,0,stream>>>(b.ids,b.counts,slots);
    gemma4_moe_offsets<<<1,experts,0,stream>>>(b.counts,b.offsets);
    gemma4_moe_assign<<<(slots+255)/256,256,0,stream>>>(b.ids,b.offsets,b.cursors,b.mapping,slots);
    // Contiguous tensor views may have an unaligned starting offset.
    if ((reinterpret_cast<uintptr_t>(input) & 15) == 0) {
        gemma4_moe_pack_vec8<<<(uint64_t(slots)*(hidden/8)+255)/256,256,0,stream>>>(
            static_cast<const __nv_bfloat16 *>(input),b.mapping,
            reinterpret_cast<__nv_bfloat16 *>(b.packed),slots,hidden);
    } else {
        gemma4_moe_pack<<<(uint64_t(slots)*hidden+255)/256,256,0,stream>>>(
            static_cast<const __nv_bfloat16 *>(input),b.mapping,
            reinterpret_cast<__nv_bfloat16 *>(b.packed),slots,hidden);
    }
    setup_problems<<<1,experts,0,stream>>>(b.counts,b.offsets,b.packed,static_cast<Element *>(gate_up_weight),b.gate_up,
        hidden,2*intermediate,b.problems,b.a,b.b,b.c,b.lda,b.ldb,b.ldc);
    int status = grouped_gemm<Element>(b.problems,experts,b.a,b.b,b.c,b.lda,b.ldb,b.ldc,stream);
    if (status) return status;
    if (qwen) {
    qwen3_moe_silu_mul<<<slots,128,0,stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(b.gate_up),reinterpret_cast<__nv_bfloat16 *>(b.activated),slots,intermediate);
    } else {
    gemma4_moe_gelu_mul<<<slots,128,0,stream>>>(
        reinterpret_cast<__nv_bfloat16 *>(b.gate_up),reinterpret_cast<__nv_bfloat16 *>(b.activated),slots,intermediate);
    }
    setup_problems<<<1,experts,0,stream>>>(b.counts,b.offsets,b.activated,static_cast<Element *>(down_weight),b.expert_output,
        intermediate,hidden,b.problems,b.a,b.b,b.c_f32,b.lda,b.ldb,b.ldc);
    status = grouped_gemm<float>(b.problems,experts,b.a,b.b,b.c_f32,b.lda,b.ldb,b.ldc,stream);
    if (status) return status;
    if ((reinterpret_cast<uintptr_t>(output) & 15) == 0) {
        gemma4_moe_combine_vec8<<<(uint64_t(tokens)*(hidden/8)+255)/256,256,0,stream>>>(
            b.expert_output,b.mapping,b.weights,
            static_cast<__nv_bfloat16 *>(output),tokens,hidden);
    } else {
        gemma4_moe_unpermute_combine<<<(uint64_t(tokens)*hidden+255)/256,256,0,stream>>>(
            b.expert_output,b.mapping,b.weights,
            static_cast<__nv_bfloat16 *>(output),tokens,hidden);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

extern "C" int gemma4_moe_forward_bf16(
    const float *logits, const float *scales, const void *input,
    void *gate_up_weight, void *down_weight, void *output,
    int tokens, int hidden, int intermediate, void *scratch, size_t scratch_bytes,
    cudaStream_t stream) {
    return routed_forward_bf16(logits, scales, input, gate_up_weight, down_weight, output,
        tokens, hidden, intermediate, scratch, scratch_bytes, stream, false, true);
}
extern "C" int qwen3_moe_forward_bf16(
    const float *logits, const void *input, void *gate_up_weight, void *down_weight, void *output,
    int tokens, int hidden, int intermediate, int renormalize, void *scratch, size_t scratch_bytes,
    cudaStream_t stream) {
    return routed_forward_bf16(logits, nullptr, input, gate_up_weight, down_weight, output,
        tokens, hidden, intermediate, scratch, scratch_bytes, stream, true, renormalize != 0);
}

extern "C" size_t qwen35_moe_workspace_bytes(int tokens, int hidden, int intermediate) {
    return routed_workspace_bytes(tokens, hidden, intermediate, 256);
}
extern "C" int qwen35_moe_forward_bf16(
    const float *logits, const void *input, void *gate_up_weight, void *down_weight, void *output,
    int tokens, int hidden, int intermediate, int renormalize, void *scratch, size_t scratch_bytes,
    cudaStream_t stream) {
    return routed_forward_bf16(logits, nullptr, input, gate_up_weight, down_weight, output,
        tokens, hidden, intermediate, scratch, scratch_bytes, stream, true, renormalize != 0, 256);
}

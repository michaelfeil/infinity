// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
/// \file
/// Standalone CUTLASS 4.5 SM100 MXFP8 grouped-expert provider for Gemma MoE.
///
/// Operands are E4M3 values with UE8M0 block scales for every 32 adjacent K
/// elements. Scale bytes use the CUTLASS 4.5 32x4_4 SF layout, which is the
/// same layout produced by the repository's mxfp8_fusions row quantizer.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace infinity_moe_mxfp8 {

/// Fixed caller-provided workspace. One grouped launch per GEMM; routing
/// counts, offsets, problem shapes, pointers, strides and SF layouts are all
/// prepared on the device.
size_t grouped_workspace_bytes();

bool grouped_available(int major, int minor);

/// Number of inner SF blocks: round_up(K / 32, 4). K=704 gives 22 blocks,
/// padded to 24 entries per 128-row/column SF tile.
size_t row_scale_inner_blocks(int64_t cols);
/// Bytes per 128-row/column SF tile: row_scale_inner_blocks(cols) * 128.
size_t row_scale_tile_bytes(int64_t cols);
/// Scale bytes required for a [rows, cols] row-wise quantized matrix.
size_t row_scale_elements(int64_t rows, int64_t cols);
/// Scale bytes for one [width, hidden] expert weight matrix. Requires
/// width % 128 == 0 so a global [experts*width, hidden] quantizer produces
/// contiguous per-expert scale slices.
size_t weight_scale_expert_bytes(int64_t width, int64_t hidden);

/// Grouped MXFP8 GEMM with BF16 output (gate/up) or FP32 output (down).
///
/// packed_a:       E4M3 [padded_rows, hidden], expert-sorted physical rows
/// expert_b:       E4M3 [experts, width, hidden], contiguous row-major
/// packed_sfa:     UE8M0 scales for [padded_rows, hidden] in this SF layout
///                 (produced by the generic mxfp8_fusions row quantizer)
/// expert_sfb:     UE8M0 scales for [experts*width, hidden] in this SF layout
///                 (produced by the generic mxfp8_fusions row quantizer)
/// counts:         int32 [experts], logical rows per expert (device)
/// offsets:        int32 [experts], physical padded row starts (device)
/// output:         BF16 or FP32 [padded_rows, width]
///
/// Requirement: offsets[e] % 128 == 0 and
/// offsets[e+1] == offsets[e] + round_up(counts[e], 128).
/// The GEMM problem M is the logical counts[e]; pointers use the physical
/// padded offsets. SFA slices therefore begin on 128-row CUTLASS SF tiles.
/// No host route-count read or per-expert launch loop occurs in the provider.
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
                      cudaStream_t stream);

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
                     cudaStream_t stream);

} // namespace infinity_moe_mxfp8

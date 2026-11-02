// SPDX-License-Identifier: MIT
// Standalone C ABI for the supplied routed-MoE CUDA operators.
//
// This header is intentionally free of CUDA, Torch, ATen, c10 and Python
// dependencies. Streams are passed as an opaque void* whose value is the
// caller's externally-owned cudaStream_t.
#ifndef OXIDE_MOE_H
#define OXIDE_MOE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OXIDE_MOE_ABI_VERSION 1u

#if defined(_WIN32)
#define OXIDE_MOE_API
#else
#define OXIDE_MOE_API __attribute__((visibility("default")))
#endif

typedef int32_t oxide_moe_status_t;
enum oxide_moe_status {
  OXIDE_MOE_STATUS_OK = 0,
  OXIDE_MOE_STATUS_INVALID_ARGUMENT = 1,
  OXIDE_MOE_STATUS_UNSUPPORTED_ABI = 2,
  OXIDE_MOE_STATUS_UNSUPPORTED_MODEL = 3,
  OXIDE_MOE_STATUS_UNSUPPORTED_ARCH = 4,
  OXIDE_MOE_STATUS_UNSUPPORTED_KERNEL = 5,
  OXIDE_MOE_STATUS_SHAPE_MISMATCH = 6,
  OXIDE_MOE_STATUS_OVERFLOW = 7,
  OXIDE_MOE_STATUS_WORKSPACE_TOO_SMALL = 8,
  OXIDE_MOE_STATUS_KERNEL_ERROR = 9,
  OXIDE_MOE_STATUS_INTERNAL_ERROR = 10
};

typedef int32_t oxide_moe_model_kind;
enum oxide_moe_model {
  OXIDE_MOE_MODEL_QWEN3_E128_TOP8 = 1,
  OXIDE_MOE_MODEL_QWEN35_E256_TOP8 = 2,
  OXIDE_MOE_MODEL_GEMMA4_E128_TOP8 = 3
};

typedef int32_t oxide_moe_kernel_kind;
enum oxide_moe_kernel {
  OXIDE_MOE_KERNEL_AUTO = 0,
  OXIDE_MOE_KERNEL_PORTABLE = 1,
  OXIDE_MOE_KERNEL_HOPPER = 2
};

typedef int32_t oxide_moe_arch;
enum oxide_moe_architecture {
  OXIDE_MOE_ARCH_SM90 = 90,
  OXIDE_MOE_ARCH_SM100 = 100,
  OXIDE_MOE_ARCH_SM120 = 120
};

#define OXIDE_MOE_KERNEL_MASK_PORTABLE (1u << OXIDE_MOE_KERNEL_PORTABLE)
#define OXIDE_MOE_KERNEL_MASK_HOPPER (1u << OXIDE_MOE_KERNEL_HOPPER)
#define OXIDE_MOE_MODEL_MASK_QWEN3 (1u << OXIDE_MOE_MODEL_QWEN3_E128_TOP8)
#define OXIDE_MOE_MODEL_MASK_QWEN35 (1u << OXIDE_MOE_MODEL_QWEN35_E256_TOP8)
#define OXIDE_MOE_MODEL_MASK_GEMMA4 (1u << OXIDE_MOE_MODEL_GEMMA4_E128_TOP8)

enum oxide_moe_build_flags {
  OXIDE_MOE_BUILD_FLAG_PORTABLE = 1u << 0,
  OXIDE_MOE_BUILD_FLAG_HOPPER = 1u << 1,
  OXIDE_MOE_BUILD_FLAG_COMPILE_ONLY = 1u << 2
};

enum oxide_moe_plan_flags {
  OXIDE_MOE_PLAN_FLAG_RENORMALIZE = 1u << 0,
  OXIDE_MOE_PLAN_FLAG_EXPERT_SCALES = 1u << 1
};

typedef struct oxide_moe_build_info {
  uint32_t struct_size;
  uint32_t abi_version;
  oxide_moe_arch arch;
  oxide_moe_kernel_kind default_kernel;
  uint32_t kernel_mask;
  uint32_t model_mask;
  uint32_t flags;
  uint32_t reserved[3];
} oxide_moe_build_info;

typedef struct oxide_moe_model_info {
  uint32_t struct_size;
  uint32_t abi_version;
  oxide_moe_model_kind model;
  int32_t experts;
  int32_t top_k;
  int32_t hidden;
  int32_t intermediate;
  int32_t requires_expert_scales;
  int32_t supports_renormalize;
  uint32_t kernel_mask;
  uint32_t reserved[3];
} oxide_moe_model_info;

typedef struct oxide_moe_shape {
  uint32_t struct_size;
  uint32_t abi_version;
  oxide_moe_model_kind model;
  oxide_moe_kernel_kind kernel;
  int32_t tokens;
  int32_t hidden;
  int32_t intermediate;
  int32_t experts;
  int32_t top_k;
  int32_t renormalize;
  uint32_t reserved[2];
} oxide_moe_shape;

typedef struct oxide_moe_workspace_info {
  uint32_t struct_size;
  uint32_t abi_version;
  oxide_moe_arch arch;
  oxide_moe_kernel_kind kernel;
  size_t workspace_bytes;
  uint32_t flags;
  uint32_t reserved[3];
} oxide_moe_workspace_info;

typedef struct oxide_moe_plan {
  uint32_t struct_size;
  uint32_t abi_version;
  oxide_moe_arch arch;
  oxide_moe_kernel_kind kernel;
  oxide_moe_model_kind model;
  int32_t tokens;
  int32_t hidden;
  int32_t intermediate;
  int32_t experts;
  int32_t top_k;
  int32_t renormalize;
  size_t workspace_bytes;
  uint32_t flags;
  uint32_t reserved[3];
} oxide_moe_plan;

typedef struct oxide_moe_buffers {
  uint32_t struct_size;
  uint32_t abi_version;
  const float* logits;
  const void* input_bf16;
  const void* gate_up_weight_bf16;
  const void* down_weight_bf16;
  const float* expert_scales;
  void* output_bf16;
  void* workspace;
  size_t workspace_bytes;
  void* stream;
  uint32_t flags;
  uint32_t reserved[3];
} oxide_moe_buffers;

OXIDE_MOE_API uint32_t oxide_moe_abi_version(void);

OXIDE_MOE_API oxide_moe_status_t oxide_moe_query_build(oxide_moe_build_info* out);

OXIDE_MOE_API oxide_moe_status_t oxide_moe_query_model(oxide_moe_model_kind model,
                                                       oxide_moe_model_info* out);

OXIDE_MOE_API oxide_moe_status_t oxide_moe_query_workspace(const oxide_moe_shape* shape,
                                                           oxide_moe_workspace_info* out);

OXIDE_MOE_API oxide_moe_status_t oxide_moe_prepare(const oxide_moe_shape* shape,
                                                   oxide_moe_plan* out);

OXIDE_MOE_API oxide_moe_status_t oxide_moe_forward_bf16(const oxide_moe_plan* plan,
                                                        const oxide_moe_buffers* buffers,
                                                        int32_t* out_kernel_status);

#ifdef __cplusplus
}
#endif

#endif

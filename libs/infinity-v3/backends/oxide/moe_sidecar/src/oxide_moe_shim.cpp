// SPDX-License-Identifier: MIT
// Small ABI shim over the supplied, unmodified routed-MoE CUDA operators.
// The shim performs CPU-only validation and forwards to either the portable
// SM80-CUTLASS path or the SM90-CUTLASS path selected at build time.

#include "oxide_moe.h"
#include "moe_symbols.h"

#include <climits>
#include <cstddef>
#include <cstdint>

// Match the raw CUDA runtime stream type without making the public C header depend on CUDA.
struct CUstream_st;
using RawStream = CUstream_st*;

#ifndef OXIDE_MOE_BUILD_ARCH
#error "OXIDE_MOE_BUILD_ARCH must be defined as 90, 100, or 120"
#endif

#if OXIDE_MOE_BUILD_ARCH == 90
#define OXIDE_MOE_HAS_HOPPER 1
#elif OXIDE_MOE_BUILD_ARCH == 100 || OXIDE_MOE_BUILD_ARCH == 120
#define OXIDE_MOE_HAS_HOPPER 0
#else
#error "OXIDE_MOE_BUILD_ARCH must be 90, 100, or 120"
#endif

// The raw units are C-linkage and have been renamed by moe_symbols.h. Streams
// are ABI-compatible pointer values; the shim deliberately avoids CUDA headers.
extern "C" {
size_t qwen3_moe_workspace_bytes(int tokens, int hidden, int intermediate);
size_t qwen35_moe_workspace_bytes(int tokens, int hidden, int intermediate);
size_t gemma4_moe_workspace_bytes(int tokens, int hidden, int intermediate);

int qwen3_moe_forward_bf16(const float* logits,
                           const void* input,
                           void* gate_up_weight,
                           void* down_weight,
                           void* output,
                           int tokens,
                           int hidden,
                           int intermediate,
                           int renormalize,
                           void* scratch,
                           size_t scratch_bytes,
                           RawStream stream);
int qwen35_moe_forward_bf16(const float* logits,
                            const void* input,
                            void* gate_up_weight,
                            void* down_weight,
                            void* output,
                            int tokens,
                            int hidden,
                            int intermediate,
                            int renormalize,
                            void* scratch,
                            size_t scratch_bytes,
                            RawStream stream);
int gemma4_moe_forward_bf16(const float* logits,
                            const float* scales,
                            const void* input,
                            void* gate_up_weight,
                            void* down_weight,
                            void* output,
                            int tokens,
                            int hidden,
                            int intermediate,
                            void* scratch,
                            size_t scratch_bytes,
                            RawStream stream);

#if OXIDE_MOE_HAS_HOPPER
size_t hopper_qwen3_moe_workspace_bytes(int tokens, int hidden, int intermediate);
size_t hopper_qwen35_moe_workspace_bytes(int tokens, int hidden, int intermediate);
size_t hopper_gemma4_moe_workspace_bytes(int tokens, int hidden, int intermediate);

int hopper_qwen3_moe_forward_bf16(const float* logits,
                                  const void* input,
                                  void* gate_up_weight,
                                  void* down_weight,
                                  void* output,
                                  int tokens,
                                  int hidden,
                                  int intermediate,
                                  int renormalize,
                                  void* scratch,
                                  size_t scratch_bytes,
                                  RawStream stream);
int hopper_qwen35_moe_forward_bf16(const float* logits,
                                   const void* input,
                                   void* gate_up_weight,
                                   void* down_weight,
                                   void* output,
                                   int tokens,
                                   int hidden,
                                   int intermediate,
                                   int renormalize,
                                   void* scratch,
                                   size_t scratch_bytes,
                                   RawStream stream);
int hopper_gemma4_moe_forward_bf16(const float* logits,
                                   const float* scales,
                                   const void* input,
                                   void* gate_up_weight,
                                   void* down_weight,
                                   void* output,
                                   int tokens,
                                   int hidden,
                                   int intermediate,
                                   void* scratch,
                                   size_t scratch_bytes,
                                   RawStream stream);
#endif
}

namespace {

constexpr uint32_t kAbiVersion = OXIDE_MOE_ABI_VERSION;
constexpr int32_t kBuildArch = OXIDE_MOE_BUILD_ARCH;
constexpr bool kHasHopper = OXIDE_MOE_HAS_HOPPER != 0;

struct ModelDesc {
  int32_t experts;
  int32_t top_k;
  int32_t hidden;
  int32_t intermediate;
  bool requires_scales;
  bool supports_renormalize;
};

bool model_desc(int32_t model, ModelDesc* out) {
  if (!out)
    return false;
  switch (model) {
  case OXIDE_MOE_MODEL_QWEN3_E128_TOP8:
    *out = ModelDesc{128, 8, 0, 0, false, true};
    return true;
  case OXIDE_MOE_MODEL_QWEN35_E256_TOP8:
    *out = ModelDesc{256, 8, 0, 0, false, true};
    return true;
  case OXIDE_MOE_MODEL_GEMMA4_E128_TOP8:
    *out = ModelDesc{128, 8, 2816, 704, true, false};
    return true;
  default:
    return false;
  }
}

uint32_t kernel_mask() {
  return static_cast<uint32_t>(OXIDE_MOE_KERNEL_MASK_PORTABLE) |
         (kHasHopper ? static_cast<uint32_t>(OXIDE_MOE_KERNEL_MASK_HOPPER) : 0u);
}

uint32_t model_mask() {
  return OXIDE_MOE_MODEL_MASK_QWEN3 | OXIDE_MOE_MODEL_MASK_QWEN35 |
         OXIDE_MOE_MODEL_MASK_GEMMA4;
}

uint32_t build_flags() {
  return static_cast<uint32_t>(OXIDE_MOE_BUILD_FLAG_PORTABLE) |
         (kHasHopper ? static_cast<uint32_t>(OXIDE_MOE_BUILD_FLAG_HOPPER) : 0u) |
         (kBuildArch == OXIDE_MOE_ARCH_SM90
              ? 0u
              : static_cast<uint32_t>(OXIDE_MOE_BUILD_FLAG_COMPILE_ONLY));
}

bool checked_mul_size(size_t a, size_t b, size_t* out) {
  if (a != 0 && b > SIZE_MAX / a)
    return false;
  *out = a * b;
  return true;
}

bool valid_header(uint32_t struct_size, uint32_t abi_version, size_t expected) {
  return struct_size == expected && abi_version == kAbiVersion;
}

oxide_moe_status_t resolve_kernel(oxide_moe_kernel_kind requested,
                                  oxide_moe_kernel_kind* resolved) {
  if (!resolved)
    return OXIDE_MOE_STATUS_INVALID_ARGUMENT;
  switch (requested) {
  case OXIDE_MOE_KERNEL_AUTO:
    *resolved = kHasHopper ? OXIDE_MOE_KERNEL_HOPPER : OXIDE_MOE_KERNEL_PORTABLE;
    return OXIDE_MOE_STATUS_OK;
  case OXIDE_MOE_KERNEL_PORTABLE:
    *resolved = OXIDE_MOE_KERNEL_PORTABLE;
    return OXIDE_MOE_STATUS_OK;
  case OXIDE_MOE_KERNEL_HOPPER:
    if (!kHasHopper)
      return OXIDE_MOE_STATUS_UNSUPPORTED_KERNEL;
    *resolved = OXIDE_MOE_KERNEL_HOPPER;
    return OXIDE_MOE_STATUS_OK;
  default:
    return OXIDE_MOE_STATUS_INVALID_ARGUMENT;
  }
}

size_t raw_workspace_bytes(int32_t model,
                           int32_t kernel,
                           int32_t tokens,
                           int32_t hidden,
                           int32_t intermediate) {
  if (kernel == OXIDE_MOE_KERNEL_HOPPER) {
#if OXIDE_MOE_HAS_HOPPER
    switch (model) {
    case OXIDE_MOE_MODEL_QWEN3_E128_TOP8:
      return hopper_qwen3_moe_workspace_bytes(tokens, hidden, intermediate);
    case OXIDE_MOE_MODEL_QWEN35_E256_TOP8:
      return hopper_qwen35_moe_workspace_bytes(tokens, hidden, intermediate);
    case OXIDE_MOE_MODEL_GEMMA4_E128_TOP8:
      return hopper_gemma4_moe_workspace_bytes(tokens, hidden, intermediate);
    default:
      return 0;
    }
#else
    return 0;
#endif
  }

  switch (model) {
  case OXIDE_MOE_MODEL_QWEN3_E128_TOP8:
    return qwen3_moe_workspace_bytes(tokens, hidden, intermediate);
  case OXIDE_MOE_MODEL_QWEN35_E256_TOP8:
    return qwen35_moe_workspace_bytes(tokens, hidden, intermediate);
  case OXIDE_MOE_MODEL_GEMMA4_E128_TOP8:
    return gemma4_moe_workspace_bytes(tokens, hidden, intermediate);
  default:
    return 0;
  }
}

oxide_moe_status_t validate_shape(const oxide_moe_shape* shape,
                                  ModelDesc* desc,
                                  oxide_moe_kernel_kind* resolved_kernel) {
  if (!shape || !desc || !resolved_kernel)
    return OXIDE_MOE_STATUS_INVALID_ARGUMENT;
  if (!valid_header(shape->struct_size, shape->abi_version, sizeof(*shape)))
    return shape->abi_version == kAbiVersion ? OXIDE_MOE_STATUS_INVALID_ARGUMENT
                                             : OXIDE_MOE_STATUS_UNSUPPORTED_ABI;
  if (!model_desc(shape->model, desc))
    return OXIDE_MOE_STATUS_UNSUPPORTED_MODEL;
  if (shape->experts != desc->experts || shape->top_k != desc->top_k)
    return OXIDE_MOE_STATUS_SHAPE_MISMATCH;
  if (shape->tokens <= 0 || shape->hidden <= 0 || shape->intermediate <= 0)
    return OXIDE_MOE_STATUS_INVALID_ARGUMENT;
  if ((shape->hidden & 7) != 0 || (shape->intermediate & 7) != 0)
    return OXIDE_MOE_STATUS_SHAPE_MISMATCH;
  if ((desc->hidden != 0 && shape->hidden != desc->hidden) ||
      (desc->intermediate != 0 && shape->intermediate != desc->intermediate))
    return OXIDE_MOE_STATUS_SHAPE_MISMATCH;
  if (shape->tokens > INT_MAX / 8 || shape->intermediate > INT_MAX / 2)
    return OXIDE_MOE_STATUS_OVERFLOW;
  if (shape->renormalize != 0 && shape->renormalize != 1)
    return OXIDE_MOE_STATUS_INVALID_ARGUMENT;
  return resolve_kernel(shape->kernel, resolved_kernel);
}

oxide_moe_status_t checked_logical_bytes(const oxide_moe_shape* shape) {
  size_t bytes = 0;
  size_t elems = 0;

  if (!checked_mul_size(static_cast<size_t>(shape->tokens),
                        static_cast<size_t>(shape->hidden), &elems) ||
      !checked_mul_size(elems, sizeof(uint16_t), &bytes))
    return OXIDE_MOE_STATUS_OVERFLOW;
  if (bytes > static_cast<size_t>(INT64_MAX))
    return OXIDE_MOE_STATUS_OVERFLOW;

  if (!checked_mul_size(static_cast<size_t>(shape->tokens),
                        static_cast<size_t>(shape->experts), &elems) ||
      !checked_mul_size(elems, sizeof(float), &bytes))
    return OXIDE_MOE_STATUS_OVERFLOW;
  if (bytes > static_cast<size_t>(INT64_MAX))
    return OXIDE_MOE_STATUS_OVERFLOW;

  if (!checked_mul_size(static_cast<size_t>(shape->experts), 2u, &elems) ||
      !checked_mul_size(elems, static_cast<size_t>(shape->intermediate), &elems) ||
      !checked_mul_size(elems, static_cast<size_t>(shape->hidden), &elems) ||
      !checked_mul_size(elems, sizeof(uint16_t), &bytes))
    return OXIDE_MOE_STATUS_OVERFLOW;
  if (bytes > static_cast<size_t>(INT64_MAX))
    return OXIDE_MOE_STATUS_OVERFLOW;

  if (!checked_mul_size(static_cast<size_t>(shape->experts),
                        static_cast<size_t>(shape->hidden), &elems) ||
      !checked_mul_size(elems, static_cast<size_t>(shape->intermediate), &elems) ||
      !checked_mul_size(elems, sizeof(uint16_t), &bytes))
    return OXIDE_MOE_STATUS_OVERFLOW;
  if (bytes > static_cast<size_t>(INT64_MAX))
    return OXIDE_MOE_STATUS_OVERFLOW;

  if (!checked_mul_size(static_cast<size_t>(shape->experts), sizeof(float), &bytes))
    return OXIDE_MOE_STATUS_OVERFLOW;
  if (bytes > static_cast<size_t>(INT64_MAX))
    return OXIDE_MOE_STATUS_OVERFLOW;
  return OXIDE_MOE_STATUS_OK;
}

oxide_moe_status_t validate_plan(const oxide_moe_plan* plan, ModelDesc* desc) {
  if (!plan || !desc)
    return OXIDE_MOE_STATUS_INVALID_ARGUMENT;
  if (!valid_header(plan->struct_size, plan->abi_version, sizeof(*plan)))
    return plan->abi_version == kAbiVersion ? OXIDE_MOE_STATUS_INVALID_ARGUMENT
                                            : OXIDE_MOE_STATUS_UNSUPPORTED_ABI;
  if (plan->arch != kBuildArch)
    return OXIDE_MOE_STATUS_UNSUPPORTED_ARCH;
  if (plan->kernel != OXIDE_MOE_KERNEL_PORTABLE &&
      plan->kernel != OXIDE_MOE_KERNEL_HOPPER)
    return OXIDE_MOE_STATUS_INVALID_ARGUMENT;
  if (plan->kernel == OXIDE_MOE_KERNEL_HOPPER && !kHasHopper)
    return OXIDE_MOE_STATUS_UNSUPPORTED_KERNEL;

  oxide_moe_shape shape{};
  shape.struct_size = sizeof(shape);
  shape.abi_version = kAbiVersion;
  shape.model = plan->model;
  shape.kernel = plan->kernel;
  shape.tokens = plan->tokens;
  shape.hidden = plan->hidden;
  shape.intermediate = plan->intermediate;
  shape.experts = plan->experts;
  shape.top_k = plan->top_k;
  shape.renormalize = plan->renormalize;

  oxide_moe_kernel_kind resolved = OXIDE_MOE_KERNEL_AUTO;
  const oxide_moe_status_t shape_status = validate_shape(&shape, desc, &resolved);
  if (shape_status != OXIDE_MOE_STATUS_OK)
    return shape_status;
  if (resolved != plan->kernel)
    return OXIDE_MOE_STATUS_INVALID_ARGUMENT;

  const oxide_moe_status_t logical_status = checked_logical_bytes(&shape);
  if (logical_status != OXIDE_MOE_STATUS_OK)
    return logical_status;

  const size_t required = raw_workspace_bytes(plan->model,
                                              plan->kernel,
                                              plan->tokens,
                                              plan->hidden,
                                              plan->intermediate);
  if (required == 0)
    return OXIDE_MOE_STATUS_OVERFLOW;
  if (plan->workspace_bytes != required)
    return OXIDE_MOE_STATUS_INVALID_ARGUMENT;

  const uint32_t expected_flags =
      (plan->renormalize ? static_cast<uint32_t>(OXIDE_MOE_PLAN_FLAG_RENORMALIZE) : 0u) |
      (desc->requires_scales ? static_cast<uint32_t>(OXIDE_MOE_PLAN_FLAG_EXPERT_SCALES) : 0u);
  if (plan->flags != expected_flags)
    return OXIDE_MOE_STATUS_INVALID_ARGUMENT;
  return OXIDE_MOE_STATUS_OK;
}

oxide_moe_status_t validate_buffers(const oxide_moe_plan* plan,
                                    const ModelDesc& desc,
                                    const oxide_moe_buffers* buffers) {
  if (!buffers)
    return OXIDE_MOE_STATUS_INVALID_ARGUMENT;
  if (!valid_header(buffers->struct_size, buffers->abi_version, sizeof(*buffers)))
    return buffers->abi_version == kAbiVersion ? OXIDE_MOE_STATUS_INVALID_ARGUMENT
                                               : OXIDE_MOE_STATUS_UNSUPPORTED_ABI;
  if (buffers->flags != 0)
    return OXIDE_MOE_STATUS_INVALID_ARGUMENT;
  if (!buffers->logits || !buffers->input_bf16 || !buffers->gate_up_weight_bf16 ||
      !buffers->down_weight_bf16 || !buffers->output_bf16 || !buffers->workspace)
    return OXIDE_MOE_STATUS_INVALID_ARGUMENT;
  if (desc.requires_scales && !buffers->expert_scales)
    return OXIDE_MOE_STATUS_INVALID_ARGUMENT;
  if (!desc.requires_scales && buffers->expert_scales)
    return OXIDE_MOE_STATUS_INVALID_ARGUMENT;
  if (buffers->workspace_bytes < plan->workspace_bytes)
    return OXIDE_MOE_STATUS_WORKSPACE_TOO_SMALL;
  // Required alignments: workspace 256B, vector weights 16B, logits/scales 4B,
  // BF16 input/output 2B. The shim does not align, allocate or copy buffers.
  if (reinterpret_cast<uintptr_t>(buffers->workspace) % 256u != 0 ||
      reinterpret_cast<uintptr_t>(buffers->gate_up_weight_bf16) % 16u != 0 ||
      reinterpret_cast<uintptr_t>(buffers->down_weight_bf16) % 16u != 0 ||
      reinterpret_cast<uintptr_t>(buffers->logits) % 4u != 0 ||
      reinterpret_cast<uintptr_t>(buffers->expert_scales) % 4u != 0 ||
      reinterpret_cast<uintptr_t>(buffers->input_bf16) % 2u != 0 ||
      reinterpret_cast<uintptr_t>(buffers->output_bf16) % 2u != 0)
    return OXIDE_MOE_STATUS_INVALID_ARGUMENT;
  return OXIDE_MOE_STATUS_OK;
}

} // namespace

extern "C" OXIDE_MOE_API uint32_t oxide_moe_abi_version(void) {
  return kAbiVersion;
}

extern "C" OXIDE_MOE_API oxide_moe_status_t
oxide_moe_query_build(oxide_moe_build_info* out) try {
  if (!out)
    return OXIDE_MOE_STATUS_INVALID_ARGUMENT;
  if (!valid_header(out->struct_size, out->abi_version, sizeof(*out)))
    return out->abi_version == kAbiVersion ? OXIDE_MOE_STATUS_INVALID_ARGUMENT
                                           : OXIDE_MOE_STATUS_UNSUPPORTED_ABI;
  out->arch = kBuildArch;
  out->default_kernel = kHasHopper ? OXIDE_MOE_KERNEL_HOPPER : OXIDE_MOE_KERNEL_PORTABLE;
  out->kernel_mask = kernel_mask();
  out->model_mask = model_mask();
  out->flags = build_flags();
  return OXIDE_MOE_STATUS_OK;
} catch (...) { return OXIDE_MOE_STATUS_INTERNAL_ERROR; }

extern "C" OXIDE_MOE_API oxide_moe_status_t
oxide_moe_query_model(oxide_moe_model_kind model, oxide_moe_model_info* out) try {
  if (!out)
    return OXIDE_MOE_STATUS_INVALID_ARGUMENT;
  if (!valid_header(out->struct_size, out->abi_version, sizeof(*out)))
    return out->abi_version == kAbiVersion ? OXIDE_MOE_STATUS_INVALID_ARGUMENT
                                           : OXIDE_MOE_STATUS_UNSUPPORTED_ABI;
  ModelDesc desc{};
  if (!model_desc(model, &desc))
    return OXIDE_MOE_STATUS_UNSUPPORTED_MODEL;
  out->model = model;
  out->experts = desc.experts;
  out->top_k = desc.top_k;
  out->hidden = desc.hidden;
  out->intermediate = desc.intermediate;
  out->requires_expert_scales = desc.requires_scales ? 1 : 0;
  out->supports_renormalize = desc.supports_renormalize ? 1 : 0;
  out->kernel_mask = kernel_mask();
  return OXIDE_MOE_STATUS_OK;
} catch (...) { return OXIDE_MOE_STATUS_INTERNAL_ERROR; }

extern "C" OXIDE_MOE_API oxide_moe_status_t
oxide_moe_query_workspace(const oxide_moe_shape* shape,
                          oxide_moe_workspace_info* out) try {
  if (!out)
    return OXIDE_MOE_STATUS_INVALID_ARGUMENT;
  if (!valid_header(out->struct_size, out->abi_version, sizeof(*out)))
    return out->abi_version == kAbiVersion ? OXIDE_MOE_STATUS_INVALID_ARGUMENT
                                           : OXIDE_MOE_STATUS_UNSUPPORTED_ABI;
  ModelDesc desc{};
  oxide_moe_kernel_kind resolved = OXIDE_MOE_KERNEL_AUTO;
  const oxide_moe_status_t status = validate_shape(shape, &desc, &resolved);
  if (status != OXIDE_MOE_STATUS_OK)
    return status;
  const oxide_moe_status_t logical_status = checked_logical_bytes(shape);
  if (logical_status != OXIDE_MOE_STATUS_OK)
    return logical_status;
  const size_t required =
      raw_workspace_bytes(shape->model, resolved, shape->tokens, shape->hidden,
                          shape->intermediate);
  if (required == 0)
    return OXIDE_MOE_STATUS_OVERFLOW;
  out->arch = kBuildArch;
  out->kernel = resolved;
  out->workspace_bytes = required;
  out->flags = 0;
  return OXIDE_MOE_STATUS_OK;
} catch (...) { return OXIDE_MOE_STATUS_INTERNAL_ERROR; }

extern "C" OXIDE_MOE_API oxide_moe_status_t
oxide_moe_prepare(const oxide_moe_shape* shape, oxide_moe_plan* out) try {
  if (!out)
    return OXIDE_MOE_STATUS_INVALID_ARGUMENT;
  if (!valid_header(out->struct_size, out->abi_version, sizeof(*out)))
    return out->abi_version == kAbiVersion ? OXIDE_MOE_STATUS_INVALID_ARGUMENT
                                           : OXIDE_MOE_STATUS_UNSUPPORTED_ABI;
  ModelDesc desc{};
  oxide_moe_kernel_kind resolved = OXIDE_MOE_KERNEL_AUTO;
  const oxide_moe_status_t status = validate_shape(shape, &desc, &resolved);
  if (status != OXIDE_MOE_STATUS_OK)
    return status;
  const oxide_moe_status_t logical_status = checked_logical_bytes(shape);
  if (logical_status != OXIDE_MOE_STATUS_OK)
    return logical_status;
  const size_t required =
      raw_workspace_bytes(shape->model, resolved, shape->tokens, shape->hidden,
                          shape->intermediate);
  if (required == 0)
    return OXIDE_MOE_STATUS_OVERFLOW;

  out->arch = kBuildArch;
  out->kernel = resolved;
  out->model = shape->model;
  out->tokens = shape->tokens;
  out->hidden = shape->hidden;
  out->intermediate = shape->intermediate;
  out->experts = desc.experts;
  out->top_k = desc.top_k;
  out->renormalize = desc.supports_renormalize ? shape->renormalize : 1;
  out->workspace_bytes = required;
  out->flags =
      (out->renormalize ? static_cast<uint32_t>(OXIDE_MOE_PLAN_FLAG_RENORMALIZE) : 0u) |
      (desc.requires_scales ? static_cast<uint32_t>(OXIDE_MOE_PLAN_FLAG_EXPERT_SCALES) : 0u);
  return OXIDE_MOE_STATUS_OK;
} catch (...) { return OXIDE_MOE_STATUS_INTERNAL_ERROR; }

extern "C" OXIDE_MOE_API oxide_moe_status_t
oxide_moe_forward_bf16(const oxide_moe_plan* plan,
                       const oxide_moe_buffers* buffers,
                       int32_t* out_kernel_status) {
  try {
    ModelDesc desc{};
    oxide_moe_status_t status = validate_plan(plan, &desc);
    if (status != OXIDE_MOE_STATUS_OK)
      return status;
    status = validate_buffers(plan, desc, buffers);
    if (status != OXIDE_MOE_STATUS_OK)
      return status;

    const int32_t tokens = plan->tokens;
    const int32_t hidden = plan->hidden;
    const int32_t intermediate = plan->intermediate;
    const int32_t renormalize = plan->renormalize;
    void* gate_up = const_cast<void*>(buffers->gate_up_weight_bf16);
    void* down = const_cast<void*>(buffers->down_weight_bf16);
    void* scratch = buffers->workspace;
    const size_t scratch_bytes = buffers->workspace_bytes;
    RawStream stream = static_cast<RawStream>(buffers->stream);
    int32_t kernel_status = 0;

    if (plan->model == OXIDE_MOE_MODEL_GEMMA4_E128_TOP8) {
#if OXIDE_MOE_HAS_HOPPER
      if (plan->kernel == OXIDE_MOE_KERNEL_HOPPER) {
        kernel_status = hopper_gemma4_moe_forward_bf16(
            buffers->logits, buffers->expert_scales, buffers->input_bf16, gate_up,
            down, buffers->output_bf16, tokens, hidden, intermediate, scratch,
            scratch_bytes, stream);
      } else
#endif
      {
        kernel_status = gemma4_moe_forward_bf16(
            buffers->logits, buffers->expert_scales, buffers->input_bf16, gate_up,
            down, buffers->output_bf16, tokens, hidden, intermediate, scratch,
            scratch_bytes, stream);
      }
    } else if (plan->model == OXIDE_MOE_MODEL_QWEN35_E256_TOP8) {
#if OXIDE_MOE_HAS_HOPPER
      if (plan->kernel == OXIDE_MOE_KERNEL_HOPPER) {
        kernel_status = hopper_qwen35_moe_forward_bf16(
            buffers->logits, buffers->input_bf16, gate_up, down,
            buffers->output_bf16, tokens, hidden, intermediate, renormalize,
            scratch, scratch_bytes, stream);
      } else
#endif
      {
        kernel_status = qwen35_moe_forward_bf16(
            buffers->logits, buffers->input_bf16, gate_up, down,
            buffers->output_bf16, tokens, hidden, intermediate, renormalize,
            scratch, scratch_bytes, stream);
      }
    } else {
#if OXIDE_MOE_HAS_HOPPER
      if (plan->kernel == OXIDE_MOE_KERNEL_HOPPER) {
        kernel_status = hopper_qwen3_moe_forward_bf16(
            buffers->logits, buffers->input_bf16, gate_up, down,
            buffers->output_bf16, tokens, hidden, intermediate, renormalize,
            scratch, scratch_bytes, stream);
      } else
#endif
      {
        kernel_status = qwen3_moe_forward_bf16(
            buffers->logits, buffers->input_bf16, gate_up, down,
            buffers->output_bf16, tokens, hidden, intermediate, renormalize,
            scratch, scratch_bytes, stream);
      }
    }

    if (out_kernel_status)
      *out_kernel_status = kernel_status;
    // Existing raw units report zero on success. Preserve their integer
    // failure value through the optional out parameter above.
    return kernel_status == 0 ? OXIDE_MOE_STATUS_OK : OXIDE_MOE_STATUS_KERNEL_ERROR;
  } catch (...) {
    return OXIDE_MOE_STATUS_INTERNAL_ERROR;
  }
}

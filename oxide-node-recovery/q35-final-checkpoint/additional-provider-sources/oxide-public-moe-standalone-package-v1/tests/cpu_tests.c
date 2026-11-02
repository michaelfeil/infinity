// SPDX-License-Identifier: MIT
// CPU-only ABI tests. No CUDA header, CUDA call, GPU execution or network use.
#define _POSIX_C_SOURCE 200809L

#include "oxide_moe.h"

#include <dlfcn.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define MAX_LIBRARIES 8

typedef uint32_t (*abi_version_fn)(void);
typedef oxide_moe_status_t (*query_build_fn)(oxide_moe_build_info*);
typedef oxide_moe_status_t (*query_model_fn)(oxide_moe_model_kind, oxide_moe_model_info*);
typedef oxide_moe_status_t (*query_workspace_fn)(const oxide_moe_shape*,
                                                 oxide_moe_workspace_info*);
typedef oxide_moe_status_t (*prepare_fn)(const oxide_moe_shape*, oxide_moe_plan*);
typedef oxide_moe_status_t (*forward_fn)(const oxide_moe_plan*,
                                         const oxide_moe_buffers*,
                                         int32_t*);

typedef struct api {
  void* handle;
  const char* path;
  int arch;
  abi_version_fn abi_version;
  query_build_fn query_build;
  query_model_fn query_model;
  query_workspace_fn query_workspace;
  prepare_fn prepare;
  forward_fn forward;
} api;

static int failures;
static const char* current_lib = "<none>";

#define CHECK(cond, message)                                                       \
  do {                                                                             \
    if (!(cond)) {                                                                 \
      fprintf(stderr, "FAIL [%s] %s:%d: %s\n", current_lib, __FILE__, __LINE__,     \
              (message));                                                          \
      ++failures;                                                                  \
    }                                                                              \
  } while (0)

static void* must_symbol(void* handle, const char* name) {
  void* symbol;
  const char* error;

  dlerror();
  symbol = dlsym(handle, name);
  error = dlerror();
  if (error != NULL) {
    fprintf(stderr, "FAIL [%s]: missing symbol %s: %s\n", current_lib, name, error);
    ++failures;
    return NULL;
  }
  return symbol;
}

#define LOAD_FN(api_ptr, field, fn_type, symbol_name)                              \
  do {                                                                             \
    void* raw_symbol = must_symbol((api_ptr)->handle, (symbol_name));              \
    if (raw_symbol != NULL) {                                                      \
      fn_type loaded;                                                              \
      memcpy(&loaded, &raw_symbol, sizeof(loaded));                                \
      (api_ptr)->field = loaded;                                                   \
    }                                                                              \
  } while (0)

static void expect_hidden(void* handle, const char* name) {
  const char* error;
  void* symbol;

  dlerror();
  symbol = dlsym(handle, name);
  error = dlerror();
  if (symbol != NULL || error == NULL) {
    fprintf(stderr, "FAIL [%s]: raw symbol is exported: %s\n", current_lib, name);
    ++failures;
  }
}

static int arch_from_path(const char* path) {
  const char* base = strrchr(path, '/');
  base = base == NULL ? path : base + 1;
  if (strstr(base, "sm90") != NULL)
    return 90;
  if (strstr(base, "sm100") != NULL)
    return 100;
  if (strstr(base, "sm120") != NULL)
    return 120;
  return 0;
}

static oxide_moe_shape make_shape(oxide_moe_model_kind model,
                                  int32_t tokens,
                                  int32_t hidden,
                                  int32_t intermediate,
                                  int32_t experts,
                                  int32_t top_k,
                                  oxide_moe_kernel_kind kernel,
                                  int32_t renormalize) {
  oxide_moe_shape shape;
  memset(&shape, 0, sizeof(shape));
  shape.struct_size = (uint32_t)sizeof(shape);
  shape.abi_version = OXIDE_MOE_ABI_VERSION;
  shape.model = model;
  shape.kernel = kernel;
  shape.tokens = tokens;
  shape.hidden = hidden;
  shape.intermediate = intermediate;
  shape.experts = experts;
  shape.top_k = top_k;
  shape.renormalize = renormalize;
  return shape;
}

static void test_build(const api* loaded) {
  oxide_moe_build_info info;
  oxide_moe_status_t status;

  CHECK(loaded->abi_version != NULL, "abi_version symbol");
  if (loaded->abi_version != NULL)
    CHECK(loaded->abi_version() == OXIDE_MOE_ABI_VERSION, "ABI version");

  CHECK(loaded->query_build != NULL, "query_build symbol");
  if (loaded->query_build == NULL)
    return;

  memset(&info, 0, sizeof(info));
  info.struct_size = (uint32_t)sizeof(info);
  info.abi_version = OXIDE_MOE_ABI_VERSION;
  status = loaded->query_build(&info);
  CHECK(status == OXIDE_MOE_STATUS_OK, "query_build status");
  CHECK(info.arch == loaded->arch, "query_build arch");
  CHECK((info.kernel_mask & OXIDE_MOE_KERNEL_MASK_PORTABLE) != 0, "portable kernel mask");
  CHECK(((info.kernel_mask & OXIDE_MOE_KERNEL_MASK_HOPPER) != 0) == (loaded->arch == 90),
        "hopper kernel mask matches arch");
  CHECK((info.model_mask & OXIDE_MOE_MODEL_MASK_QWEN3) != 0, "qwen3 model mask");
  CHECK((info.model_mask & OXIDE_MOE_MODEL_MASK_QWEN35) != 0, "qwen35 model mask");
  CHECK((info.model_mask & OXIDE_MOE_MODEL_MASK_GEMMA4) != 0, "gemma4 model mask");
  CHECK(info.default_kernel ==
            (loaded->arch == 90 ? OXIDE_MOE_KERNEL_HOPPER : OXIDE_MOE_KERNEL_PORTABLE),
        "default kernel");
  CHECK(((info.flags & OXIDE_MOE_BUILD_FLAG_COMPILE_ONLY) != 0) == (loaded->arch != 90),
        "compile-only policy for non-SM90");

  info.struct_size = (uint32_t)sizeof(info) - 1u;
  info.abi_version = OXIDE_MOE_ABI_VERSION;
  CHECK(loaded->query_build(&info) == OXIDE_MOE_STATUS_INVALID_ARGUMENT,
        "bad build struct size rejected");
  info.struct_size = (uint32_t)sizeof(info);
  info.abi_version = OXIDE_MOE_ABI_VERSION + 100u;
  CHECK(loaded->query_build(&info) == OXIDE_MOE_STATUS_UNSUPPORTED_ABI,
        "bad build ABI rejected");
  CHECK(loaded->query_build(NULL) == OXIDE_MOE_STATUS_INVALID_ARGUMENT,
        "null build output rejected");
}

static void check_model(const api* loaded,
                        oxide_moe_model_kind model,
                        int32_t experts,
                        int32_t top_k,
                        int32_t hidden,
                        int32_t intermediate,
                        int32_t requires_scales,
                        int32_t supports_renormalize) {
  oxide_moe_model_info info;
  oxide_moe_status_t status;

  memset(&info, 0, sizeof(info));
  info.struct_size = (uint32_t)sizeof(info);
  info.abi_version = OXIDE_MOE_ABI_VERSION;
  status = loaded->query_model(model, &info);
  CHECK(status == OXIDE_MOE_STATUS_OK, "query_model status");
  CHECK(info.model == model, "query_model kind");
  CHECK(info.experts == experts, "query_model experts");
  CHECK(info.top_k == top_k, "query_model top_k");
  CHECK(info.hidden == hidden, "query_model hidden");
  CHECK(info.intermediate == intermediate, "query_model intermediate");
  CHECK(info.requires_expert_scales == requires_scales, "query_model scales");
  CHECK(info.supports_renormalize == supports_renormalize, "query_model renormalize");
  CHECK(info.kernel_mask ==
            (OXIDE_MOE_KERNEL_MASK_PORTABLE |
             (loaded->arch == 90 ? OXIDE_MOE_KERNEL_MASK_HOPPER : 0u)),
        "query_model kernel mask");
}

static void test_models(const api* loaded) {
  oxide_moe_model_info info;

  CHECK(loaded->query_model != NULL, "query_model symbol");
  if (loaded->query_model == NULL)
    return;

  check_model(loaded, OXIDE_MOE_MODEL_QWEN3_E128_TOP8, 128, 8, 0, 0, 0, 1);
  check_model(loaded, OXIDE_MOE_MODEL_QWEN35_E256_TOP8, 256, 8, 0, 0, 0, 1);
  check_model(loaded, OXIDE_MOE_MODEL_GEMMA4_E128_TOP8, 128, 8, 2816, 704, 1, 0);

  memset(&info, 0, sizeof(info));
  info.struct_size = (uint32_t)sizeof(info);
  info.abi_version = OXIDE_MOE_ABI_VERSION;
  CHECK(loaded->query_model(99, &info) == OXIDE_MOE_STATUS_UNSUPPORTED_MODEL,
        "unknown model rejected");
  CHECK(loaded->query_model(OXIDE_MOE_MODEL_QWEN3_E128_TOP8, NULL) ==
            OXIDE_MOE_STATUS_INVALID_ARGUMENT,
        "null model output rejected");
}

static oxide_moe_status_t query_ws(const api* loaded,
                                   const oxide_moe_shape* shape,
                                   oxide_moe_workspace_info* info) {
  memset(info, 0, sizeof(*info));
  info->struct_size = (uint32_t)sizeof(*info);
  info->abi_version = OXIDE_MOE_ABI_VERSION;
  if (loaded->query_workspace == NULL)
    return OXIDE_MOE_STATUS_INVALID_ARGUMENT;
  return loaded->query_workspace(shape, info);
}

static void check_workspace_bounds(const oxide_moe_workspace_info* info,
                                   int32_t tokens,
                                   int32_t hidden,
                                   int32_t intermediate) {
  const size_t slots = (size_t)tokens * 8u;
  const size_t row_bytes = (size_t)hidden * sizeof(uint16_t);
  const size_t gate_up_bytes = slots * 2u * (size_t)intermediate * sizeof(uint16_t);
  const size_t activated_bytes = slots * (size_t)intermediate * sizeof(uint16_t);
  const size_t expert_output_bytes = slots * (size_t)hidden * sizeof(uint16_t);

  CHECK(info->workspace_bytes >= slots * row_bytes, "workspace holds packed input");
  CHECK(info->workspace_bytes >= gate_up_bytes, "workspace holds gate-up output");
  CHECK(info->workspace_bytes >= activated_bytes, "workspace holds activation output");
  CHECK(info->workspace_bytes >= expert_output_bytes, "workspace holds expert output");
  CHECK(info->workspace_bytes % 256u == 0, "workspace is 256-byte rounded");
}

static void test_workspace(const api* loaded) {
  oxide_moe_workspace_info info;
  oxide_moe_shape shape;
  oxide_moe_kernel_kind expected_kernel =
      loaded->arch == 90 ? OXIDE_MOE_KERNEL_HOPPER : OXIDE_MOE_KERNEL_PORTABLE;
  oxide_moe_status_t status;

  CHECK(loaded->query_workspace != NULL, "query_workspace symbol");
  if (loaded->query_workspace == NULL)
    return;

  shape = make_shape(OXIDE_MOE_MODEL_QWEN3_E128_TOP8, 3, 256, 128, 128, 8,
                     OXIDE_MOE_KERNEL_AUTO, 1);
  status = query_ws(loaded, &shape, &info);
  CHECK(status == OXIDE_MOE_STATUS_OK, "qwen3 workspace status");
  CHECK(info.arch == loaded->arch, "qwen3 workspace arch");
  CHECK(info.kernel == expected_kernel, "qwen3 auto kernel selection");
  CHECK(info.workspace_bytes > 0, "qwen3 workspace positive");
  check_workspace_bounds(&info, 3, 256, 128);

  shape = make_shape(OXIDE_MOE_MODEL_QWEN3_E128_TOP8, 3, 256, 128, 128, 8,
                     OXIDE_MOE_KERNEL_PORTABLE, 1);
  status = query_ws(loaded, &shape, &info);
  CHECK(status == OXIDE_MOE_STATUS_OK, "qwen3 portable workspace status");
  CHECK(info.kernel == OXIDE_MOE_KERNEL_PORTABLE, "qwen3 portable kernel");
  CHECK(info.workspace_bytes > 0, "qwen3 portable workspace positive");
  check_workspace_bounds(&info, 3, 256, 128);

  shape = make_shape(OXIDE_MOE_MODEL_QWEN3_E128_TOP8, 1, 256, 128, 128, 8,
                     OXIDE_MOE_KERNEL_AUTO, 1);
  status = query_ws(loaded, &shape, &info);
  CHECK(status == OXIDE_MOE_STATUS_OK, "qwen3 small-token workspace");
  {
    size_t small_bytes = info.workspace_bytes;
    shape.tokens = 2;
    status = query_ws(loaded, &shape, &info);
    CHECK(status == OXIDE_MOE_STATUS_OK, "qwen3 larger-token workspace");
    CHECK(info.workspace_bytes > small_bytes, "workspace grows with token count");
  }

  shape = make_shape(OXIDE_MOE_MODEL_QWEN35_E256_TOP8, 2, 512, 256, 256, 8,
                     OXIDE_MOE_KERNEL_AUTO, 0);
  status = query_ws(loaded, &shape, &info);
  CHECK(status == OXIDE_MOE_STATUS_OK, "qwen35 workspace status");
  CHECK(info.workspace_bytes > 0, "qwen35 workspace positive");
  check_workspace_bounds(&info, 2, 512, 256);

  shape = make_shape(OXIDE_MOE_MODEL_GEMMA4_E128_TOP8, 1, 2816, 704, 128, 8,
                     OXIDE_MOE_KERNEL_AUTO, 0);
  status = query_ws(loaded, &shape, &info);
  CHECK(status == OXIDE_MOE_STATUS_OK, "gemma4 workspace status");
  CHECK(info.workspace_bytes > 0, "gemma4 workspace positive");
  check_workspace_bounds(&info, 1, 2816, 704);

  shape.kernel = OXIDE_MOE_KERNEL_HOPPER;
  status = query_ws(loaded, &shape, &info);
  if (loaded->arch == 90) {
    CHECK(status == OXIDE_MOE_STATUS_OK, "sm90 hopper workspace status");
    CHECK(info.kernel == OXIDE_MOE_KERNEL_HOPPER, "sm90 hopper kernel");
  } else {
    CHECK(status == OXIDE_MOE_STATUS_UNSUPPORTED_KERNEL,
          "hopper rejected on non-sm90");
  }

  shape = make_shape(OXIDE_MOE_MODEL_QWEN3_E128_TOP8, 1, 255, 128, 128, 8,
                     OXIDE_MOE_KERNEL_AUTO, 1);
  CHECK(query_ws(loaded, &shape, &info) == OXIDE_MOE_STATUS_SHAPE_MISMATCH,
        "hidden multiple-of-8 required");
  shape = make_shape(OXIDE_MOE_MODEL_QWEN3_E128_TOP8, 0, 256, 128, 128, 8,
                     OXIDE_MOE_KERNEL_AUTO, 1);
  CHECK(query_ws(loaded, &shape, &info) == OXIDE_MOE_STATUS_INVALID_ARGUMENT,
        "zero tokens rejected");
  shape = make_shape(OXIDE_MOE_MODEL_QWEN3_E128_TOP8, 1, 256, 128, 256, 8,
                     OXIDE_MOE_KERNEL_AUTO, 1);
  CHECK(query_ws(loaded, &shape, &info) == OXIDE_MOE_STATUS_SHAPE_MISMATCH,
        "expert count mismatch rejected");
  shape = make_shape(OXIDE_MOE_MODEL_QWEN3_E128_TOP8, 1, 256, 128, 128, 7,
                     OXIDE_MOE_KERNEL_AUTO, 1);
  CHECK(query_ws(loaded, &shape, &info) == OXIDE_MOE_STATUS_SHAPE_MISMATCH,
        "top-k mismatch rejected");
  shape = make_shape(OXIDE_MOE_MODEL_GEMMA4_E128_TOP8, 1, 2816, 512, 128, 8,
                     OXIDE_MOE_KERNEL_PORTABLE, 0);
  CHECK(query_ws(loaded, &shape, &info) == OXIDE_MOE_STATUS_SHAPE_MISMATCH,
        "gemma4 fixed intermediate required");
  shape = make_shape(OXIDE_MOE_MODEL_QWEN3_E128_TOP8, INT_MAX, 256, 128, 128, 8,
                     OXIDE_MOE_KERNEL_AUTO, 1);
  CHECK(query_ws(loaded, &shape, &info) == OXIDE_MOE_STATUS_OVERFLOW,
        "token overflow rejected");
  shape = make_shape(OXIDE_MOE_MODEL_QWEN3_E128_TOP8, 1, 256, INT_MAX - 7, 128, 8,
                     OXIDE_MOE_KERNEL_AUTO, 1);
  CHECK(query_ws(loaded, &shape, &info) == OXIDE_MOE_STATUS_OVERFLOW,
        "intermediate overflow rejected");

  shape = make_shape(OXIDE_MOE_MODEL_QWEN3_E128_TOP8, 1, 256, 128, 128, 8, 99, 1);
  CHECK(query_ws(loaded, &shape, &info) == OXIDE_MOE_STATUS_INVALID_ARGUMENT,
        "invalid kernel rejected");
  shape = make_shape(OXIDE_MOE_MODEL_QWEN3_E128_TOP8, 1, 256, 128, 128, 8,
                     OXIDE_MOE_KERNEL_AUTO, 2);
  CHECK(query_ws(loaded, &shape, &info) == OXIDE_MOE_STATUS_INVALID_ARGUMENT,
        "invalid renormalize rejected");

  info.struct_size = (uint32_t)sizeof(info) - 1u;
  info.abi_version = OXIDE_MOE_ABI_VERSION;
  shape = make_shape(OXIDE_MOE_MODEL_QWEN3_E128_TOP8, 1, 256, 128, 128, 8,
                     OXIDE_MOE_KERNEL_AUTO, 1);
  CHECK(loaded->query_workspace(&shape, &info) == OXIDE_MOE_STATUS_INVALID_ARGUMENT,
        "bad workspace output struct rejected");

  shape.struct_size = (uint32_t)sizeof(shape) - 1u;
  CHECK(query_ws(loaded, &shape, &info) == OXIDE_MOE_STATUS_INVALID_ARGUMENT,
        "bad shape struct rejected");
  shape.struct_size = (uint32_t)sizeof(shape);
  shape.abi_version = OXIDE_MOE_ABI_VERSION + 1u;
  CHECK(query_ws(loaded, &shape, &info) == OXIDE_MOE_STATUS_UNSUPPORTED_ABI,
        "bad shape ABI rejected");
}

static void test_prepare(const api* loaded) {
  oxide_moe_shape shape;
  oxide_moe_workspace_info ws;
  oxide_moe_plan plan;
  oxide_moe_status_t status;

  CHECK(loaded->prepare != NULL, "prepare symbol");
  CHECK(loaded->query_workspace != NULL, "query_workspace symbol for prepare");
  if (loaded->prepare == NULL || loaded->query_workspace == NULL)
    return;

  shape = make_shape(OXIDE_MOE_MODEL_QWEN3_E128_TOP8, 3, 256, 128, 128, 8,
                     OXIDE_MOE_KERNEL_AUTO, 1);
  status = query_ws(loaded, &shape, &ws);
  CHECK(status == OXIDE_MOE_STATUS_OK, "prepare prerequisite workspace");

  memset(&plan, 0, sizeof(plan));
  plan.struct_size = (uint32_t)sizeof(plan);
  plan.abi_version = OXIDE_MOE_ABI_VERSION;
  status = loaded->prepare(&shape, &plan);
  CHECK(status == OXIDE_MOE_STATUS_OK, "prepare qwen3 status");
  CHECK(plan.arch == loaded->arch, "prepare plan arch");
  CHECK(plan.kernel == ws.kernel, "prepare plan kernel");
  CHECK(plan.model == shape.model, "prepare plan model");
  CHECK(plan.tokens == shape.tokens, "prepare plan tokens");
  CHECK(plan.hidden == shape.hidden, "prepare plan hidden");
  CHECK(plan.intermediate == shape.intermediate, "prepare plan intermediate");
  CHECK(plan.experts == 128, "prepare plan experts");
  CHECK(plan.top_k == 8, "prepare plan top_k");
  CHECK(plan.renormalize == 1, "prepare plan renormalize");
  CHECK(plan.workspace_bytes == ws.workspace_bytes, "prepare plan workspace");
  CHECK(plan.flags == OXIDE_MOE_PLAN_FLAG_RENORMALIZE, "prepare qwen flags");

  shape = make_shape(OXIDE_MOE_MODEL_QWEN3_E128_TOP8, 3, 256, 128, 128, 8,
                     OXIDE_MOE_KERNEL_PORTABLE, 0);
  memset(&plan, 0, sizeof(plan));
  plan.struct_size = (uint32_t)sizeof(plan);
  plan.abi_version = OXIDE_MOE_ABI_VERSION;
  status = loaded->prepare(&shape, &plan);
  CHECK(status == OXIDE_MOE_STATUS_OK, "prepare qwen portable status");
  CHECK(plan.kernel == OXIDE_MOE_KERNEL_PORTABLE, "prepare qwen portable kernel");
  CHECK(plan.renormalize == 0, "prepare qwen no-renorm");
  CHECK(plan.flags == 0, "prepare qwen no-renorm flags");

  shape = make_shape(OXIDE_MOE_MODEL_GEMMA4_E128_TOP8, 1, 2816, 704, 128, 8,
                     OXIDE_MOE_KERNEL_AUTO, 0);
  memset(&plan, 0, sizeof(plan));
  plan.struct_size = (uint32_t)sizeof(plan);
  plan.abi_version = OXIDE_MOE_ABI_VERSION;
  status = loaded->prepare(&shape, &plan);
  CHECK(status == OXIDE_MOE_STATUS_OK, "prepare gemma status");
  CHECK(plan.renormalize == 1, "gemma always renormalizes");
  CHECK(plan.flags == (OXIDE_MOE_PLAN_FLAG_RENORMALIZE | OXIDE_MOE_PLAN_FLAG_EXPERT_SCALES),
        "gemma plan flags");

  shape.kernel = OXIDE_MOE_KERNEL_HOPPER;
  memset(&plan, 0, sizeof(plan));
  plan.struct_size = (uint32_t)sizeof(plan);
  plan.abi_version = OXIDE_MOE_ABI_VERSION;
  status = loaded->prepare(&shape, &plan);
  if (loaded->arch == 90)
    CHECK(status == OXIDE_MOE_STATUS_OK, "sm90 prepare hopper");
  else
    CHECK(status == OXIDE_MOE_STATUS_UNSUPPORTED_KERNEL,
          "non-sm90 prepare hopper rejected");

  plan.struct_size = (uint32_t)sizeof(plan) - 1u;
  plan.abi_version = OXIDE_MOE_ABI_VERSION;
  CHECK(loaded->prepare(&shape, &plan) == OXIDE_MOE_STATUS_INVALID_ARGUMENT,
        "bad plan output struct rejected");
  CHECK(loaded->prepare(&shape, NULL) == OXIDE_MOE_STATUS_INVALID_ARGUMENT,
        "null plan output rejected");
}

static void test_forward_validation(const api* loaded) {
  oxide_moe_shape shape;
  oxide_moe_plan plan;
  oxide_moe_plan saved_plan;
  oxide_moe_buffers buffers;
  oxide_moe_status_t status;
  int32_t kernel_status;

  CHECK(loaded->forward != NULL, "forward symbol");
  CHECK(loaded->prepare != NULL, "prepare symbol for forward validation");
  if (loaded->forward == NULL || loaded->prepare == NULL)
    return;

  shape = make_shape(OXIDE_MOE_MODEL_QWEN3_E128_TOP8, 1, 256, 128, 128, 8,
                     OXIDE_MOE_KERNEL_AUTO, 1);
  memset(&plan, 0, sizeof(plan));
  plan.struct_size = (uint32_t)sizeof(plan);
  plan.abi_version = OXIDE_MOE_ABI_VERSION;
  status = loaded->prepare(&shape, &plan);
  CHECK(status == OXIDE_MOE_STATUS_OK, "forward prerequisite prepare");
  if (status != OXIDE_MOE_STATUS_OK)
    return;
  saved_plan = plan;

  memset(&buffers, 0, sizeof(buffers));
  buffers.struct_size = (uint32_t)sizeof(buffers);
  buffers.abi_version = OXIDE_MOE_ABI_VERSION;
  kernel_status = -777;
  CHECK(loaded->forward(&plan, &buffers, &kernel_status) ==
            OXIDE_MOE_STATUS_INVALID_ARGUMENT,
        "null buffers rejected");
  CHECK(kernel_status == -777, "kernel status untouched on validation failure");

  buffers.logits = (const float*)(uintptr_t)0x1000;
  buffers.input_bf16 = (const void*)(uintptr_t)0x2000;
  buffers.gate_up_weight_bf16 = (const void*)(uintptr_t)0x3000;
  buffers.down_weight_bf16 = (const void*)(uintptr_t)0x4000;
  buffers.output_bf16 = (void*)(uintptr_t)0x5000;
  buffers.workspace = (void*)(uintptr_t)0x6000;
  buffers.workspace_bytes = plan.workspace_bytes > 0 ? plan.workspace_bytes - 1u : 0u;
  CHECK(loaded->forward(&plan, &buffers, &kernel_status) ==
            OXIDE_MOE_STATUS_WORKSPACE_TOO_SMALL,
        "short workspace rejected before launch");

  buffers.workspace_bytes = plan.workspace_bytes;
  plan.arch = 999;
  CHECK(loaded->forward(&plan, &buffers, &kernel_status) ==
            OXIDE_MOE_STATUS_UNSUPPORTED_ARCH,
        "wrong plan arch rejected");
  plan = saved_plan;

  plan.kernel = OXIDE_MOE_KERNEL_AUTO;
  CHECK(loaded->forward(&plan, &buffers, &kernel_status) ==
            OXIDE_MOE_STATUS_INVALID_ARGUMENT,
        "auto kernel in plan rejected");
  plan = saved_plan;

  plan.workspace_bytes = saved_plan.workspace_bytes > 0 ? saved_plan.workspace_bytes - 1u : 0u;
  CHECK(loaded->forward(&plan, &buffers, &kernel_status) ==
            OXIDE_MOE_STATUS_INVALID_ARGUMENT,
        "stale plan workspace rejected");
  plan = saved_plan;

  plan.flags = 0xffffu;
  CHECK(loaded->forward(&plan, &buffers, &kernel_status) ==
            OXIDE_MOE_STATUS_INVALID_ARGUMENT,
        "unknown plan flags rejected");
  plan = saved_plan;

  plan.struct_size = (uint32_t)sizeof(plan) - 1u;
  CHECK(loaded->forward(&plan, &buffers, &kernel_status) ==
            OXIDE_MOE_STATUS_INVALID_ARGUMENT,
        "bad plan struct rejected");
  plan = saved_plan;

  plan.abi_version = OXIDE_MOE_ABI_VERSION + 3u;
  CHECK(loaded->forward(&plan, &buffers, &kernel_status) ==
            OXIDE_MOE_STATUS_UNSUPPORTED_ABI,
        "bad plan ABI rejected");
  plan = saved_plan;

  buffers.struct_size = (uint32_t)sizeof(buffers) - 1u;
  CHECK(loaded->forward(&plan, &buffers, &kernel_status) ==
            OXIDE_MOE_STATUS_INVALID_ARGUMENT,
        "bad buffers struct rejected");
  buffers.struct_size = (uint32_t)sizeof(buffers);

  buffers.flags = 1u;
  CHECK(loaded->forward(&plan, &buffers, &kernel_status) ==
            OXIDE_MOE_STATUS_INVALID_ARGUMENT,
        "unknown buffer flags rejected");
  buffers.flags = 0u;

  buffers.workspace = (void*)(uintptr_t)0x6001;
  CHECK(loaded->forward(&plan, &buffers, &kernel_status) ==
            OXIDE_MOE_STATUS_INVALID_ARGUMENT,
        "unaligned workspace rejected");
  buffers.workspace = (void*)(uintptr_t)0x6000;

  buffers.expert_scales = (const float*)(uintptr_t)0x7000;
  CHECK(loaded->forward(&plan, &buffers, &kernel_status) ==
            OXIDE_MOE_STATUS_INVALID_ARGUMENT,
        "qwen scales rejected");
  buffers.expert_scales = NULL;

  buffers.logits = NULL;
  CHECK(loaded->forward(&plan, &buffers, &kernel_status) ==
            OXIDE_MOE_STATUS_INVALID_ARGUMENT,
        "null logits rejected");
  buffers.logits = (const float*)(uintptr_t)0x1000;

  shape = make_shape(OXIDE_MOE_MODEL_GEMMA4_E128_TOP8, 1, 2816, 704, 128, 8,
                     OXIDE_MOE_KERNEL_AUTO, 0);
  memset(&plan, 0, sizeof(plan));
  plan.struct_size = (uint32_t)sizeof(plan);
  plan.abi_version = OXIDE_MOE_ABI_VERSION;
  status = loaded->prepare(&shape, &plan);
  CHECK(status == OXIDE_MOE_STATUS_OK, "gemma forward prerequisite prepare");
  if (status == OXIDE_MOE_STATUS_OK) {
    buffers.workspace_bytes = plan.workspace_bytes - 1u;
    buffers.expert_scales = NULL;
    CHECK(loaded->forward(&plan, &buffers, &kernel_status) ==
              OXIDE_MOE_STATUS_INVALID_ARGUMENT,
          "gemma missing scales rejected");
  }
}

static void test_hidden_symbols(const api* loaded) {
  expect_hidden(loaded->handle, "gemma4_moe_forward_bf16");
  expect_hidden(loaded->handle, "qwen3_moe_forward_bf16");
  expect_hidden(loaded->handle, "qwen35_moe_forward_bf16");
  expect_hidden(loaded->handle, "gemma4_moe_workspace_bytes");
  expect_hidden(loaded->handle, "qwen3_moe_workspace_bytes");
  expect_hidden(loaded->handle, "qwen35_moe_workspace_bytes");
  expect_hidden(loaded->handle, "gemma4_grouped_gemm_bf16");
  expect_hidden(loaded->handle, "tei_gemma4_moe_forward_bf16");
  expect_hidden(loaded->handle, "tei_qwen3_moe_forward_bf16");
  expect_hidden(loaded->handle, "tei_qwen35_moe_forward_bf16");
  expect_hidden(loaded->handle, "tei_gemma4_moe_workspace_bytes");
  expect_hidden(loaded->handle, "tei_qwen3_moe_workspace_bytes");
  expect_hidden(loaded->handle, "tei_qwen35_moe_workspace_bytes");
  expect_hidden(loaded->handle, "tei_hopper_qwen3_moe_forward_bf16");
  expect_hidden(loaded->handle, "tei_hopper_qwen35_moe_forward_bf16");
  expect_hidden(loaded->handle, "tei_hopper_gemma4_moe_forward_bf16");
}

static int load_api(const char* path, api* loaded) {
  memset(loaded, 0, sizeof(*loaded));
  loaded->path = path;
  loaded->arch = arch_from_path(path);
  current_lib = path;
  CHECK(loaded->arch == 90 || loaded->arch == 100 || loaded->arch == 120,
        "library filename identifies arch");
  loaded->handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (loaded->handle == NULL) {
    fprintf(stderr, "FAIL [%s]: dlopen: %s\n", path, dlerror());
    ++failures;
    return 0;
  }
  LOAD_FN(loaded, abi_version, abi_version_fn, "oxide_moe_abi_version");
  LOAD_FN(loaded, query_build, query_build_fn, "oxide_moe_query_build");
  LOAD_FN(loaded, query_model, query_model_fn, "oxide_moe_query_model");
  LOAD_FN(loaded, query_workspace, query_workspace_fn, "oxide_moe_query_workspace");
  LOAD_FN(loaded, prepare, prepare_fn, "oxide_moe_prepare");
  LOAD_FN(loaded, forward, forward_fn, "oxide_moe_forward_bf16");
  return loaded->handle != NULL;
}

int main(int argc, char** argv) {
  api libraries[MAX_LIBRARIES];
  int count = 0;
  int i;

  if (argc < 2) {
    fprintf(stderr, "usage: %s <liboxide_moe_sm90.so> [...]\n", argv[0]);
    return 2;
  }
  if (argc - 1 > MAX_LIBRARIES) {
    fprintf(stderr, "too many libraries (max %d)\n", MAX_LIBRARIES);
    return 2;
  }
  for (i = 1; i < argc; ++i) {
    if (load_api(argv[i], &libraries[count]))
      ++count;
  }
  for (i = 0; i < count; ++i) {
    current_lib = libraries[i].path;
    test_build(&libraries[i]);
    test_models(&libraries[i]);
    test_workspace(&libraries[i]);
    test_prepare(&libraries[i]);
    test_forward_validation(&libraries[i]);
    test_hidden_symbols(&libraries[i]);
  }
  for (i = 0; i < count; ++i) {
    oxide_moe_build_info info;
    memset(&info, 0, sizeof(info));
    info.struct_size = (uint32_t)sizeof(info);
    info.abi_version = OXIDE_MOE_ABI_VERSION;
    current_lib = libraries[i].path;
    if (libraries[i].query_build != NULL)
      CHECK(libraries[i].query_build(&info) == OXIDE_MOE_STATUS_OK,
            "library still queries while all libraries are loaded");
  }
  for (i = 0; i < count; ++i)
    dlclose(libraries[i].handle);

  if (failures != 0) {
    fprintf(stderr, "%d CPU ABI test failure(s)\n", failures);
    return 1;
  }
  printf("CPU ABI tests passed for %d library/libraries\n", count);
  return 0;
}

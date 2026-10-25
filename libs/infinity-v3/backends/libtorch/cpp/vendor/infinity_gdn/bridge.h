// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// Native ABI for fresh, packed FlashInfer GDN prefill. State is FP32 [B,H,V,K].
#pragma once
#include <cstdint>
extern "C" {
int infinity_gdn_supported_v1(int major);
const char* infinity_gdn_error_v1();
int infinity_gdn_forward_v1(int dtype, int major, int device, void* stream,
                            int64_t tokens, int64_t batch, int64_t kh, int64_t vh, int sm_count,
                            void* q, void* k, void* v, void* alpha, void* beta, void* output,
                            void* cumulative, void* state, void* workspace, int64_t workspace_bytes);
}

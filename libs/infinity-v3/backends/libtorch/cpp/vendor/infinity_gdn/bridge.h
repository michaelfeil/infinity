// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// Native ABI v2 for packed FlashInfer GDN. State is FP32 [B,H,V,K].
#pragma once
#include <cstdint>
extern "C" {
int infinity_gdn_supported_v1(int major);
const char* infinity_gdn_error_v1();
// A null initial pointer starts from zero; otherwise one immutable state per sequence.
int infinity_gdn_forward_v2(int dtype,
                            int major,
                            int device,
                            void* stream,
                            int64_t tokens,
                            int64_t batch,
                            int64_t kh,
                            int64_t vh,
                            int sm_count,
                            void* q,
                            void* k,
                            void* v,
                            void* alpha,
                            void* beta,
                            void* output,
                            void* cumulative,
                            void* state,
                            void* workspace,
                            int64_t workspace_bytes,
                            void* initial);
}

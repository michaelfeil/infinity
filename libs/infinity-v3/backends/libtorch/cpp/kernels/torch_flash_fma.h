#pragma once
#include <ATen/ATen.h>

namespace tei {
// Exact-token native Torch Flash attention with fused softmax arithmetic.
// The supported head dimensions require no token or head-dimension padding.
bool torch_flash_fma_supported(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v);
bool torch_flash_fma_supported(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
                               bool causal,int64_t window_left,int64_t window_right);
at::Tensor torch_flash_fma(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
                          const at::Tensor& cumulative,int64_t max_sequence,
                          double scale,bool causal=false,int64_t window_left=-1,
                          int64_t window_right=-1);
// Scoped noncausal FP16 head32 ALiBi, compiled independently so ALiBi remains
// disabled for the existing qualified private Flash specializations.
bool torch_flash_fma_alibi32_supported(c10::Device,at::ScalarType,int64_t head_dim);
at::Tensor torch_flash_fma_alibi32(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
                          const at::Tensor& cumulative,const at::Tensor& slopes,
                          int64_t max_sequence,double scale);
// Different real query/key spans, used by Gemma4 image-query correction.
// Separate cumulative arrays preserve original token rows without padding.
at::Tensor torch_flash_fma_varlen(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
                          const at::Tensor& cumulative_q,const at::Tensor& cumulative_k,
                          int64_t max_queries,int64_t max_keys,double scale,
                          bool causal=false,int64_t window_left=-1,int64_t window_right=-1);
// Preserve default Torch softmax arithmetic for single global head64/128 queries without its
// single-query GQA layout change. Cold execution keeps the original provider.
at::Tensor torch_flash_unfused_varlen(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
                          const at::Tensor& cumulative_q,const at::Tensor& cumulative_k,
                          int64_t max_queries,int64_t max_keys,double scale);
}

#pragma once
#include "batching/packed_input.h"
namespace tei {
bool qwen35_recurrent_supported(c10::Device,int64_t,int64_t);
void qwen35_recurrent_cuda(const at::Tensor&,const at::Tensor&,const at::Tensor&,
    const PackedInput&,at::Tensor&,int64_t,int64_t,bool);
void qwen35_cached_conv_cuda(const at::Tensor&,const at::Tensor&,const PackedInput&,
 const at::Tensor&,const at::Tensor&,at::Tensor&,bool);
void qwen35_cached_recurrent_cuda(const at::Tensor&,const at::Tensor&,const at::Tensor&,
 const PackedInput&,at::Tensor&,int64_t,int64_t,bool,const at::Tensor&,const at::Tensor&,
 const at::Tensor&,at::Tensor&);
}

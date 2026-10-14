#pragma once
#include "batching/packed_input.h"
namespace tei {
at::Tensor qwen35_reference_delta_cuda(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const PackedInput&,int64_t,int64_t,double);
at::Tensor qwen35_reference_norm_cuda(const at::Tensor&,const at::Tensor&,double);
}

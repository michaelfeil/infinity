#pragma once
#include "batching/packed_input.h"
namespace tei {
bool qwen35_recurrent_supported(c10::Device,int64_t,int64_t);
void qwen35_recurrent_cuda(const at::Tensor&,const at::Tensor&,const at::Tensor&,
    const PackedInput&,at::Tensor&,int64_t,int64_t,bool);
}

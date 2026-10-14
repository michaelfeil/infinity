// Newly authored CUDA packed classifier row selection and bit-preserving gather.
#include "kernels/decoder_packed_readout.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>
namespace tei {
__global__ void select_last_real_row(const uint16_t* hidden,const int64_t* ids,const int32_t* cu,uint16_t* out,int64_t total,int64_t width,int64_t pad) {
 __shared__ int32_t selected;
 const auto row=blockIdx.x;
 if(threadIdx.x==0){int32_t begin=cu[row],end=cu[row+1];CUDA_KERNEL_ASSERT(begin>=0&&end>begin&&end<=total);selected=begin;for(int32_t i=end-1;i>=begin;--i){if(ids[i]!=pad){selected=i;break;}}}
 __syncthreads();for(int64_t col=threadIdx.x;col<width;col+=blockDim.x)out[int64_t(row)*width+col]=hidden[int64_t(selected)*width+col];
}
at::Tensor qwen2_packed_classifier_rows(const at::Tensor& hidden,const at::Tensor& ids,const at::Tensor& cu,int64_t batch,int64_t pad){
 TORCH_CHECK(hidden.is_cuda()&&(hidden.scalar_type()==at::kBFloat16||hidden.scalar_type()==at::kHalf)&&hidden.dim()==2&&hidden.is_contiguous()&&hidden.size(1)>0,"Packed score readout requires contiguous CUDA BF16 or FP16 hidden rows");
 TORCH_CHECK(ids.device()==hidden.device()&&ids.dim()==1&&ids.scalar_type()==at::kLong&&ids.is_contiguous()&&ids.numel()==hidden.size(0),"Invalid packed score IDs");
 TORCH_CHECK(cu.device()==hidden.device()&&cu.dim()==1&&cu.scalar_type()==at::kInt&&cu.is_contiguous()&&batch>0&&cu.numel()==batch+1,"Invalid packed score cumulative offsets");
 c10::cuda::CUDAGuard guard(hidden.device());auto out=at::empty({batch,hidden.size(1)},hidden.options());
 select_last_real_row<<<batch,128,0,at::cuda::getCurrentCUDAStream()>>>(static_cast<const uint16_t*>(hidden.const_data_ptr()),ids.const_data_ptr<int64_t>(),cu.const_data_ptr<int32_t>(),static_cast<uint16_t*>(out.mutable_data_ptr()),hidden.size(0),hidden.size(1),pad);C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
}

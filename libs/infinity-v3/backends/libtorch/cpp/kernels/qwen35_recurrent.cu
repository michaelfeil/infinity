// SPDX-License-Identifier: MIT
// Adapted from original Candle qwen35_gdn.cu and Candle fast_sum at
// 036f39e44015e78ba18d716d73dfdc2e78730c44. Original licenses retained.
// Explicit operations preserve both original FMA and separate-rounding modes.
/*
Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
*/

#include "kernels/qwen35_recurrent.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
namespace tei {
namespace {
template<bool FMA> __device__ float accumulate(float sum,float a,float b) {
 if constexpr(FMA)return __fmaf_rn(a,b,sum);
 else return __fadd_rn(sum,__fmul_rn(a,b));
}
__device__ float sum_warp(float x) {
 for(int d=16;d;d>>=1)x=__fadd_rn(x,__shfl_down_sync(0xffffffff,x,d));
 return __shfl_sync(0xffffffff,x,0);
}
template<int COLUMNS,bool FMA>
__global__ void recurrent(const __nv_bfloat16* values,const float* qk,
 const float* gates,const int32_t* cu,__nv_bfloat16* output,int kh,int vh) {
 constexpr int WARPS=8/COLUMNS;
 const int sequence=blockIdx.x,head=blockIdx.y,lane=threadIdx.x%32,warp=threadIdx.x/32;
 const int column=blockIdx.z*(COLUMNS*WARPS)+warp*COLUMNS;
 const int qhead=head/(vh/kh),channels=(2*kh+vh)*128;
 float state[COLUMNS][4]={};
 for(int t=cu[sequence];t<cu[sequence+1];++t) {
   float decay=gates[int64_t(t)*2*vh+head];
   float beta=gates[int64_t(t)*2*vh+vh+head];
   float q[4],k[4],memory[COLUMNS]={};
#pragma unroll
   for(int i=0;i<4;++i) {
    q[i]=qk[(int64_t(t)*kh+qhead)*256+lane+i*32];
    k[i]=qk[(int64_t(t)*kh+qhead)*256+128+lane+i*32];
#pragma unroll
    for(int col=0;col<COLUMNS;++col) {
     state[col][i]=__fmul_rn(state[col][i],decay);
     memory[col]=accumulate<FMA>(memory[col],state[col][i],k[i]);
    }
   }
#pragma unroll
   for(int col=0;col<COLUMNS;++col)memory[col]=sum_warp(memory[col]);
   float delta[COLUMNS],sum[COLUMNS]={};
#pragma unroll
   for(int col=0;col<COLUMNS;++col) {
    float v=__bfloat162float(values[int64_t(t)*channels+2*kh*128+head*128+column+col]);
    delta[col]=__fmul_rn(__fsub_rn(v,memory[col]),beta);
   }
#pragma unroll
   for(int i=0;i<4;++i) {
#pragma unroll
    for(int col=0;col<COLUMNS;++col) {
     state[col][i]=accumulate<FMA>(state[col][i],k[i],delta[col]);
     sum[col]=accumulate<FMA>(sum[col],state[col][i],q[i]);
    }
   }
#pragma unroll
   for(int col=0;col<COLUMNS;++col)sum[col]=sum_warp(sum[col]);
   if(lane==0) {
#pragma unroll
    for(int col=0;col<COLUMNS;++col)
     output[(int64_t(t)*vh+head)*128+column+col]=__float2bfloat16_rn(sum[col]);
   }
 }
}
template<int COLUMNS,bool FMA> void launch(const at::Tensor& v,const at::Tensor& qk,
 const at::Tensor& gates,const PackedInput& input,at::Tensor& out,int kh,int vh) {
 recurrent<COLUMNS,FMA><<<dim3(input.batch,vh,16),(8/COLUMNS)*32,0,at::cuda::getCurrentCUDAStream()>>>(
   reinterpret_cast<const __nv_bfloat16*>(v.const_data_ptr()),qk.const_data_ptr<float>(),
   gates.const_data_ptr<float>(),input.cumulative.const_data_ptr<int32_t>(),
   reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),kh,vh);
}

}
bool qwen35_recurrent_supported(c10::Device device,int64_t kh,int64_t vh) {
 if(!device.is_cuda()||device.index()<0||kh!=16||vh!=48)return false;
 auto properties=at::cuda::getDeviceProperties(device.index());
 return properties->major==9&&properties->minor==0;
}
void qwen35_recurrent_cuda(const at::Tensor& v,const at::Tensor& qk,
 const at::Tensor& gates,const PackedInput& input,at::Tensor& out,
 int64_t kh,int64_t vh,bool fma) {
 TORCH_CHECK(qwen35_recurrent_supported(v.device(),kh,vh),"Unsupported precise recurrence profile");
 TORCH_CHECK(v.scalar_type()==at::kBFloat16&&v.dim()==2&&v.is_contiguous(),"Recurrence values must be packed BF16 rows");
 const auto tokens=v.size(0);
 TORCH_CHECK(tokens>0&&tokens<=INT32_MAX&&v.size(1)==(2*kh+vh)*128,"Invalid recurrence value dimensions");
 TORCH_CHECK(qk.device()==v.device()&&gates.device()==v.device()&&out.device()==v.device()&&
   qk.scalar_type()==at::kFloat&&gates.scalar_type()==at::kFloat&&out.scalar_type()==at::kBFloat16&&
   qk.is_contiguous()&&gates.is_contiguous()&&out.is_contiguous()&&
   qk.sizes()==at::IntArrayRef({tokens*kh,256})&&gates.sizes()==at::IntArrayRef({tokens,2*vh})&&
   out.sizes()==at::IntArrayRef({tokens,vh,128}),"Invalid prepared recurrence tensors");
 TORCH_CHECK(input.batch>0&&input.batch<=INT32_MAX&&input.cumulative.device()==v.device()&&
   input.cumulative.scalar_type()==at::kInt&&input.cumulative.is_contiguous()&&
   input.cumulative.dim()==1&&input.cumulative.numel()==input.batch+1,"Invalid recurrence sequence metadata");
 TORCH_CHECK(input.offsets&&input.offsets[0]==0&&input.offsets[input.batch]==tokens,
   "Recurrence offsets must span actual token rows");
 for(int64_t row=0;row<input.batch;++row)
   TORCH_CHECK(input.offsets[row+1]>input.offsets[row],"Recurrence sequences must be nonempty and ordered");
 c10::cuda::CUDAGuard guard(v.device());
 if(fma)launch<4,true>(v,qk,gates,input,out,kh,vh);
 else launch<2,false>(v,qk,gates,input,out,kh,vh);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}

// SPDX-License-Identifier: MIT
// Adapted from original Candle qwen35_gdn.cu and Candle fast_sum at
// 036f39e44015e78ba18d716d73dfdc2e78730c44. Original licenses retained.
// This translation unit alone uses the original --fmad=false contract.
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
#include "kernels/qwen35_reference.h"
#include "kernels/qwen35_recurrent.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#define gdn_sum tei_reference_gdn_sum
#define qwen35_conv_bf16 tei_reference_qwen35_conv_bf16
#define qwen35_gdn_prepare tei_reference_qwen35_gdn_prepare
#define qwen35_gdn_recurrent tei_reference_qwen35_gdn_recurrent
#define qwen35_gdn_norm tei_reference_qwen35_gdn_norm
#include "../../../candle/src/kernels/qwen35_gdn.cu"
#undef gdn_sum
#undef qwen35_conv_bf16
#undef qwen35_gdn_prepare
#undef qwen35_gdn_recurrent
#undef qwen35_gdn_norm

namespace tei {
namespace {
__global__ void norm_tree(const __nv_bfloat16* x,const __nv_bfloat16* gamma,__nv_bfloat16* out,long rows,long width,long heads,long stride0,long stride1,float eps) {
  __shared__ float sums[1024];
  long row=blockIdx.x,tid=threadIdx.x;
  long base=(row/heads)*stride0+(row%heads)*stride1;
  float sum=0.f;
  for(long d=tid;d<width;d+=blockDim.x){float z=__bfloat162float(x[base+d]);sum=__fadd_rn(sum,__fmul_rn(z,z));}
  sums[tid]=sum;
  for(int s=blockDim.x/2;s;s>>=1){__syncthreads();if(tid<s)sums[tid]=__fadd_rn(sums[tid],sums[tid+s]);}
  __syncthreads();
  float mean=__fmul_rn(sums[0],1.f/width);
  float inv=__fdiv_rn(1.f,__fsqrt_rn(__fadd_rn(mean,eps)));
  for(long d=tid;d<width;d+=blockDim.x){float z=__bfloat162float(x[base+d]);float weight=__fadd_rn(__bfloat162float(gamma[d]),1.f);out[row*width+d]=__float2bfloat16_rn(__fmul_rn(__fmul_rn(z,inv),weight));}
}
}
at::Tensor qwen35_reference_norm_cuda(const at::Tensor& x,const at::Tensor& gamma,double eps){
  TORCH_CHECK(x.is_cuda() && gamma.device()==x.device() && x.scalar_type()==at::kBFloat16 && gamma.scalar_type()==at::kBFloat16 && gamma.dim()==1 && gamma.is_contiguous());
  TORCH_CHECK((x.dim()==2||x.dim()==3)&&x.stride(-1)==1);
  long width=x.size(-1);
  TORCH_CHECK(width>0 && width<=8192 && gamma.numel()==width);
  long heads=x.dim()==3?x.size(1):1,rows=x.numel()/width;
  TORCH_CHECK(rows>0 && rows<=INT32_MAX);
  int threads=1;while(threads<width && threads<1024)threads*=2;
  c10::cuda::CUDAGuard guard(x.device());auto out=at::empty(x.sizes(),x.options());
  norm_tree<<<rows,threads,0,at::cuda::getCurrentCUDAStream()>>>(reinterpret_cast<const __nv_bfloat16*>(x.const_data_ptr()),reinterpret_cast<const __nv_bfloat16*>(gamma.const_data_ptr()),reinterpret_cast<__nv_bfloat16*>(out.mutable_data_ptr()),rows,width,heads,x.stride(0),x.dim()==3?x.stride(1):0,static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

at::Tensor qwen35_reference_delta_cuda(const at::Tensor& qkv, const at::Tensor& z,
  const at::Tensor& ab, const at::Tensor& conv, const at::Tensor& a_log,
  const at::Tensor& dt_bias, const at::Tensor& norm, const PackedInput& input,
  int64_t key_heads, int64_t value_heads, double epsilon) {
  c10::cuda::CUDAGuard guard(qkv.device());
  TORCH_CHECK(qkv.scalar_type()==at::kBFloat16 && z.scalar_type()==at::kBFloat16 &&
    ab.scalar_type()==at::kBFloat16 && conv.scalar_type()==at::kBFloat16 &&
    norm.scalar_type()==at::kBFloat16, "Packed DeltaNet CUDA kernels require BF16");
  TORCH_CHECK(key_heads>0 && value_heads>0 && value_heads<=128 && value_heads%key_heads==0,
    "Unsupported packed DeltaNet head geometry");
  auto channels=(2*key_heads+value_heads)*128, tokens=qkv.size(0);
  TORCH_CHECK(qkv.sizes()==at::IntArrayRef({tokens,channels}) &&
    z.sizes()==at::IntArrayRef({tokens,value_heads*128}) &&
    ab.sizes()==at::IntArrayRef({tokens,2*value_heads}) &&
    conv.size(0)==channels && conv.size(-1)>0 && conv.size(-1)<=16,
    "Invalid packed DeltaNet tensors");
  TORCH_CHECK(tokens<=INT32_MAX && tokens*channels<=UINT32_MAX, "DeltaNet token buffer exceeds kernel range");
  auto stream=at::cuda::getCurrentCUDAStream(qkv.device().index());
  auto convolution=at::empty_like(qkv), qk=at::empty({tokens*key_heads,256},qkv.options().dtype(at::kFloat));
  auto decay=at::empty({tokens,2*value_heads},qkv.options().dtype(at::kFloat));
  auto recurrence=at::empty({tokens,value_heads,128},qkv.options()), output=at::empty_like(recurrence);
  auto bf=[](const at::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.const_data_ptr()); };
  auto mut=[](at::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.mutable_data_ptr()); };
  auto cu=reinterpret_cast<const uint32_t*>(input.cumulative.const_data_ptr<int32_t>());
  tei_reference_qwen35_conv_bf16<<<(tokens*channels+255)/256,256,0,stream>>>(bf(qkv),bf(conv),cu,mut(convolution),tokens,channels,conv.size(-1),input.batch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  tei_reference_qwen35_gdn_prepare<<<tokens*key_heads,128,0,stream>>>(bf(convolution),bf(ab),a_log.const_data_ptr<float>(),dt_bias.const_data_ptr<float>(),qk.mutable_data_ptr<float>(),decay.mutable_data_ptr<float>(),tokens,key_heads,value_heads);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if(qwen35_recurrent_supported(qkv.device(),key_heads,value_heads))
    qwen35_recurrent_cuda(convolution,qk,decay,input,recurrence,key_heads,value_heads,false);
  else {
    tei_reference_qwen35_gdn_recurrent<<<dim3(input.batch,value_heads,16),256,0,stream>>>(bf(convolution),qk.const_data_ptr<float>(),decay.const_data_ptr<float>(),cu,mut(recurrence),key_heads,value_heads);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  tei_reference_qwen35_gdn_norm<<<tokens*value_heads,128,0,stream>>>(bf(recurrence),bf(z),bf(norm),mut(output),epsilon);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output.view({tokens,value_heads*128});
}
}

// SPDX-License-Identifier: Apache-2.0
// RMS arithmetic adapted from the Apache-2.0 fast_kernels.cu implementation.
// Preserve its reduction order and BF16 rounding before row quantization.
#include "kernels/fp8_fusions.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <climits>
#include <cstdint>
#include <cmath>

namespace tei {
namespace {
__device__ float sum256(float value) {
  __shared__ float warps[8];
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  for(int d=16;d;d/=2)value+=__shfl_down_sync(0xffffffff,value,d);
  if(!lane)warps[warp]=value;
  __syncthreads();
  value=threadIdx.x<8?warps[lane]:0.f;
  if(warp==0)for(int d=16;d;d/=2)value+=__shfl_down_sync(0xffffffff,value,d);
  if(!threadIdx.x)warps[0]=value;
  __syncthreads();
  float total=warps[0];
  __syncthreads();
  return total;
}
__device__ float row_scale(float value) {
  __shared__ float warps[8];
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  for(int d=16;d;d/=2)value=fmaxf(value,__shfl_down_sync(0xffffffff,value,d));
  if(!lane)warps[warp]=value;
  __syncthreads();
  value=threadIdx.x<8?warps[lane]:0.f;
  if(warp==0)for(int d=16;d;d/=2)value=fmaxf(value,__shfl_down_sync(0xffffffff,value,d));
  if(!threadIdx.x)warps[0]=__fmul_rn(fmaxf(value,1e-12f),1.f/448.f);
  __syncthreads();
  return warps[0];
}
template<class T,int Columns>
__global__ void swiglu_quant_kernel(const T* input,__nv_fp8_e4m3* out,float* scales,int width) {
  const int64_t row=blockIdx.x;
  float values[Columns],amax=0;
#pragma unroll
  for(int j=0;j<Columns;++j) {
    int col=threadIdx.x+j*256;
    float value=0;
    if(col<width) {
      T gate=input[row*2*width+col],up=input[row*2*width+width+col];
      T exponential=hexp(T(-gate));
      T denominator=T(T(1)+exponential);
      T activated=T(gate/denominator);
      value=float(T(float(activated)*float(up)));
    }
    values[j]=value;amax=fmaxf(amax,fabsf(value));
  }
  float scale=row_scale(amax);
  if(!threadIdx.x)scales[row]=scale;
#pragma unroll
  for(int j=0;j<Columns;++j) {
    int col=threadIdx.x+j*256;
    if(col<width)out[row*width+col]=__nv_fp8_e4m3(__fdiv_rn(values[j],scale));
  }
}
template<class T,int N> struct alignas(sizeof(T)*N) Packed {T value[N];};
template<class T,int Groups>
__global__ void swiglu_quant_vector_kernel(const T* input,__nv_fp8_e4m3* out,float* scales,int width) {
  const int64_t row=blockIdx.x;
  float values[Groups][8],amax=0;
  auto source=reinterpret_cast<const Packed<T,8>*>(input);
  int stride=width/8;
#pragma unroll
  for(int j=0;j<Groups;++j) {
    int col=threadIdx.x+j*256;
    Packed<T,8> gate{},up{};
    if(col<stride) {gate=source[row*2*stride+col];up=source[row*2*stride+stride+col];}
#pragma unroll
    for(int lane=0;lane<8;++lane) {
      T exponential=hexp(T(-gate.value[lane]));
      T denominator=T(T(1)+exponential);
      T activated=T(gate.value[lane]/denominator);
      float value=float(T(float(activated)*float(up.value[lane])));
      values[j][lane]=value;amax=fmaxf(amax,fabsf(value));
    }
  }
  float scale=row_scale(amax);
  if(!threadIdx.x)scales[row]=scale;
#pragma unroll
  for(int j=0;j<Groups;++j) {
    int col=threadIdx.x+j*256;
    if(col<stride) {
      Packed<__nv_fp8_e4m3,8> result;
#pragma unroll
      for(int lane=0;lane<8;++lane)result.value[lane]=__nv_fp8_e4m3(__fdiv_rn(values[j][lane],scale));
      reinterpret_cast<Packed<__nv_fp8_e4m3,8>*>(out)[row*stride+col]=result;
    }
  }
}
template<int Columns>
__global__ void rms_quant_kernel(const __nv_bfloat16* x,const __nv_bfloat16* residual,
 const __nv_bfloat16* weight,__nv_fp8_e4m3* out,__nv_bfloat16* summed,float* scales,int width,float epsilon) {
  const int64_t row=blockIdx.x;float variance=0,values[Columns];
#pragma unroll
  for(int j=0;j<Columns;++j) {
    int col=threadIdx.x+j*256;
    float value=col<width?float(x[row*width+col]):0.f;
    if(col<width&&residual)value+=float(residual[row*width+col]);
    values[j]=value;variance+=value*value;
    if(col<width&&residual)summed[row*width+col]=__nv_bfloat16(value);
  }
  float inverse=rsqrtf(sum256(variance)/width+epsilon),amax=0;
#pragma unroll
  for(int j=0;j<Columns;++j) {
    int col=threadIdx.x+j*256;
    float value=col<width?float(__nv_bfloat16((values[j]*inverse)*float(weight[col]))):0.f;
    values[j]=value;amax=fmaxf(amax,fabsf(value));
  }
  float scale=row_scale(amax);
  if(!threadIdx.x)scales[row]=scale;
#pragma unroll
  for(int j=0;j<Columns;++j) {
    int col=threadIdx.x+j*256;
    if(col<width)out[row*width+col]=__nv_fp8_e4m3(__fdiv_rn(values[j],scale));
  }
}
// Transpose through shared memory to keep the old scalar reduction order while
// issuing vector global loads/stores. The reduction contract remains unchanged.
__global__ void rms_quant_vector4096(const __nv_bfloat16* x,const __nv_bfloat16* residual,
 const __nv_bfloat16* weight,__nv_fp8_e4m3* out,__nv_bfloat16* summed,float* scales,float epsilon) {
  __shared__ float values[4096];
  const int64_t row=blockIdx.x;
#pragma unroll
  for(int j=0;j<2;++j) {
    int group=threadIdx.x+j*256;
    auto input=reinterpret_cast<const Packed<__nv_bfloat16,8>*>(x)[row*512+group];
    Packed<__nv_bfloat16,8> previous{},result;
    if(residual)previous=reinterpret_cast<const Packed<__nv_bfloat16,8>*>(residual)[row*512+group];
#pragma unroll
    for(int lane=0;lane<8;++lane) {
      float value=float(input.value[lane]);
      if(residual)value+=float(previous.value[lane]);
      values[group*8+lane]=value;
      result.value[lane]=__nv_bfloat16(value);
    }
    if(residual)reinterpret_cast<Packed<__nv_bfloat16,8>*>(summed)[row*512+group]=result;
  }
  __syncthreads();
  float variance=0;
#pragma unroll
  for(int j=0;j<16;++j) {
    float value=values[threadIdx.x+j*256];variance+=value*value;
  }
  float inverse=rsqrtf(sum256(variance)/4096+epsilon),amax=0;
#pragma unroll
  for(int j=0;j<16;++j) {
    int col=threadIdx.x+j*256;
    float value=float(__nv_bfloat16((values[col]*inverse)*float(weight[col])));
    values[col]=value;amax=fmaxf(amax,fabsf(value));
  }
  float scale=row_scale(amax);
  if(!threadIdx.x)scales[row]=scale;
#pragma unroll
  for(int j=0;j<2;++j) {
    int group=threadIdx.x+j*256;
    Packed<__nv_fp8_e4m3,8> result;
#pragma unroll
    for(int lane=0;lane<8;++lane)result.value[lane]=__nv_fp8_e4m3(__fdiv_rn(values[group*8+lane],scale));
    reinterpret_cast<Packed<__nv_fp8_e4m3,8>*>(out)[row*512+group]=result;
  }
}
void validate_matrix(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda()&&x.dim()==2&&x.is_contiguous()&&x.size(0)<=INT_MAX
    &&x.size(1)>0&&x.size(1)<=65536,"Expected contiguous CUDA matrix with bounded positive width");
}
template<class T> void launch_swiglu(const at::Tensor& input,at::Tensor& output,at::Tensor& scales) {
  int width=input.size(1)/2;
  auto stream=at::cuda::getCurrentCUDAStream(input.get_device());
  auto x=reinterpret_cast<const T*>(input.const_data_ptr());
  auto y=reinterpret_cast<__nv_fp8_e4m3*>(output.mutable_data_ptr());
  if(width%8==0 && reinterpret_cast<uintptr_t>(x)%alignof(Packed<T,8>)==0) {
#define VECTOR(C) swiglu_quant_vector_kernel<T,C><<<input.size(0),256,0,stream>>>(x,y,scales.data_ptr<float>(),width)
    if(width<=2048){VECTOR(1);}else if(width<=4096){VECTOR(2);}else if(width<=12288){VECTOR(6);}
    else if(width<=14336){VECTOR(7);}else if(width<=16384){VECTOR(8);}else if(width<=28672){VECTOR(14);}else{VECTOR(16);}
#undef VECTOR
    return;
  }
#define RUN(C) swiglu_quant_kernel<T,C><<<input.size(0),256,0,stream>>>(x,y,scales.data_ptr<float>(),width)
  if(width<=256){RUN(1);}else if(width<=1024){RUN(4);}else if(width<=4096){RUN(16);}
  else if(width<=12288){RUN(48);}else if(width<=16384){RUN(64);}else{RUN(128);}
#undef RUN
}
}
std::pair<at::Tensor,at::Tensor> fp8_swiglu_quantize(const at::Tensor& input) {
  validate_matrix(input);
  TORCH_CHECK(input.size(1)%2==0&&(input.scalar_type()==at::kHalf||input.scalar_type()==at::kBFloat16),
    "SwiGLU requires even gate/up width and FP16/BF16 input");
  c10::cuda::CUDAGuard guard(input.device());
  auto output=at::empty({input.size(0),input.size(1)/2},input.options().dtype(at::kFloat8_e4m3fn));
  auto scales=at::empty({input.size(0),1},input.options().dtype(at::kFloat));
  if(input.size(0)) {
    if(input.scalar_type()==at::kHalf)launch_swiglu<__half>(input,output,scales);
    else launch_swiglu<__nv_bfloat16>(input,output,scales);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output,scales};
}
std::tuple<at::Tensor,at::Tensor,at::Tensor> fp8_add_rms_quantize(
 const at::Tensor& x,const at::Tensor& residual,const at::Tensor& weight,double epsilon) {
  validate_matrix(x);
  TORCH_CHECK(x.scalar_type()==at::kBFloat16&&x.size(1)<=8192,"Fused FP8 RMS requires BF16 width <=8192");
  TORCH_CHECK(weight.is_contiguous()&&weight.dim()==1&&weight.size(0)==x.size(1)
    &&weight.device()==x.device()&&weight.scalar_type()==x.scalar_type(),"Invalid RMS weight");
  TORCH_CHECK(!residual.defined()||(residual.sizes()==x.sizes()&&residual.is_contiguous()
    &&residual.device()==x.device()&&residual.scalar_type()==x.scalar_type()),"Invalid RMS residual");
  TORCH_CHECK(std::isfinite(epsilon)&&epsilon>=0,"Invalid RMS epsilon");
  c10::cuda::CUDAGuard guard(x.device());
  auto output=at::empty_like(x,x.options().dtype(at::kFloat8_e4m3fn));
  auto summed=residual.defined()?at::empty_like(x):x;
  auto scales=at::empty({x.size(0),1},x.options().dtype(at::kFloat));
  if(x.size(0)) {
    auto stream=at::cuda::getCurrentCUDAStream(x.get_device());
    auto ptr=[](const at::Tensor& t){return reinterpret_cast<const __nv_bfloat16*>(t.const_data_ptr());};
    if(x.size(1)==4096 && reinterpret_cast<uintptr_t>(ptr(x))%16==0
      &&(!residual.defined()||reinterpret_cast<uintptr_t>(ptr(residual))%16==0)) {
      rms_quant_vector4096<<<x.size(0),256,0,stream>>>(ptr(x),residual.defined()?ptr(residual):nullptr,
        ptr(weight),reinterpret_cast<__nv_fp8_e4m3*>(output.mutable_data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(summed.mutable_data_ptr()),scales.data_ptr<float>(),epsilon);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return {output,summed,scales};
    }
#define RUN(C) rms_quant_kernel<C><<<x.size(0),256,0,stream>>>(ptr(x),residual.defined()?ptr(residual):nullptr,ptr(weight),reinterpret_cast<__nv_fp8_e4m3*>(output.mutable_data_ptr()),reinterpret_cast<__nv_bfloat16*>(summed.mutable_data_ptr()),scales.data_ptr<float>(),x.size(1),epsilon)
    auto width=x.size(1);
    if(width<=256){RUN(1);}else if(width<=512){RUN(2);}else if(width<=768){RUN(3);}
    else if(width<=1024){RUN(4);}else if(width<=2048){RUN(8);}else if(width<=4096){RUN(16);}else{RUN(32);}
#undef RUN
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return {output,summed,scales};
}
}

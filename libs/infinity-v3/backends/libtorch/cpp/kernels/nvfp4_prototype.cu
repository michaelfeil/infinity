// Copyright 2026 Infinity Core Contributors.
// Covered by Infinity Restricted MIT License 1.0; see LICENSE.
#include "kernels/nvfp4_prototype.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
namespace tei {
namespace {
__device__ unsigned fp4_code(float x) {
  const auto v=fabsf(x);
  return unsigned(v>.25f)+unsigned(v>=.75f)+unsigned(v>1.25f)+unsigned(v>=1.75f)+
         unsigned(v>2.5f)+unsigned(v>=3.5f)+unsigned(v>5.f)+(x<0?8:0);
}
__device__ float fp4_value(unsigned code) {
  constexpr float values[]={0,.5f,1,1.5f,2,3,4,6};
  return (code&8)?-values[code&7]:values[code];
}
template<bool Dequantize>
__global__ void quantize(const __nv_bfloat16* input,unsigned char* packed,
                         __nv_fp8_e4m3* scales,__nv_bfloat16* output,
                         const float* global,int64_t blocks,int64_t kblocks,int64_t padded_k) {
  const int64_t pair=int64_t(blockIdx.x)*blockDim.x+threadIdx.x,block=pair/8;
  if(block>=blocks)return;
  float a=float(input[pair*2]),b=float(input[pair*2+1]);
  float maximum=fmaxf(fabsf(a),fabsf(b));
  const unsigned mask=__activemask();
  for(int d=4;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(mask,maximum,d,8));
  const float g=global[0];
  float s=__fdiv_rn(maximum,__fmul_rn(6.f,g));
  if(s==0)s=1;
  const __nv_fp8_e4m3 scale(fminf(448.f,s));
  const float combined=__fmul_rn(float(scale),g);
  a=combined>0?__fdiv_rn(a,combined):0;
  b=combined>0?__fdiv_rn(b,combined):0;
  const auto qa=fp4_code(a),qb=fp4_code(b);
  packed[pair]=qa|(qb<<4);
  if((pair%8)==0) {
    const auto row=block/kblocks,k=block%kblocks;
    const auto index=((((row/128)*(padded_k/4)+k/4)*32+row%32)*4+(row/32)%4)*4+k%4;
    scales[index]=scale;
  }
  if constexpr(Dequantize) {
    output[pair*2]=__float2bfloat16_rn(__fmul_rn(fp4_value(qa),combined));
    output[pair*2+1]=__float2bfloat16_rn(__fmul_rn(fp4_value(qb),combined));
  }
}
}
Nvfp4PrototypePacked nvfp4_prototype_quantize_cuda(const at::Tensor& input,
                                                 const at::Tensor& global,bool dequantize) {
  TORCH_CHECK(input.is_cuda() && input.dim()==2 && input.is_contiguous() &&
    input.scalar_type()==at::kBFloat16 && input.size(1)>0 && input.size(1)%16==0,
    "NVFP4 quantization requires contiguous CUDA BF16 matrices, K divisible by 16");
  TORCH_CHECK(global.device()==input.device() && global.scalar_type()==at::kFloat &&
    global.numel()==1 && global.is_contiguous(),"Invalid NVFP4 global scale");
  c10::cuda::CUDAGuard guard(input.device());
  const auto m=input.size(0),kblocks=input.size(1)/16,padded_k=(kblocks+3)/4*4;
  Nvfp4PrototypePacked result{at::empty({m,input.size(1)/2},input.options().dtype(at::kByte)),
    at::zeros({((m+127)/128*128)*padded_k},input.options().dtype(at::kFloat8_e4m3fn)),
    dequantize?at::empty_like(input):at::Tensor()};
  if(m) {
    const auto grid=(m*kblocks*8+255)/256;
    auto stream=at::cuda::getCurrentCUDAStream(input.get_device());
    auto launch=[&]<bool D>() {quantize<D><<<grid,256,0,stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.const_data_ptr()),result.packed.data_ptr<unsigned char>(),
      reinterpret_cast<__nv_fp8_e4m3*>(result.scales.mutable_data_ptr()),
      D?reinterpret_cast<__nv_bfloat16*>(result.dequantized.mutable_data_ptr()):nullptr,
      global.const_data_ptr<float>(),m*kblocks,kblocks,padded_k);};
    if(dequantize)launch.template operator()<true>();else launch.template operator()<false>();
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  result.packed=result.packed.view(at::kFloat4_e2m1fn_x2);
  return result;
}
}

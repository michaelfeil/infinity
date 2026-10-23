// SPDX-License-Identifier: MIT
#include "kernels/moe_kernels.h"
#ifdef TEI_TORCH_MOE_KERNELS
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
#include "kernels/moe_symbols.h"
struct __nv_bfloat16;
extern "C" void gemma4_moe_combine_vec8(const float*,const uint32_t*,const float*,__nv_bfloat16*,uint64_t,uint32_t);
extern "C" size_t qwen3_moe_workspace_bytes(int,int,int);
extern "C" size_t qwen35_moe_workspace_bytes(int,int,int);
extern "C" size_t gemma4_moe_workspace_bytes(int,int,int);
extern "C" int qwen3_moe_forward_bf16(const float*,const void*,void*,void*,void*,int,int,int,int,void*,size_t,cudaStream_t);
extern "C" int qwen35_moe_forward_bf16(const float*,const void*,void*,void*,void*,int,int,int,int,void*,size_t,cudaStream_t);
extern "C" int gemma4_moe_forward_bf16(const float*,const float*,const void*,void*,void*,void*,int,int,int,void*,size_t,cudaStream_t);
#ifdef TEI_TORCH_MOE_HOPPER
extern "C" size_t hopper_qwen3_moe_workspace_bytes(int,int,int);
extern "C" size_t hopper_qwen35_moe_workspace_bytes(int,int,int);
extern "C" size_t hopper_gemma4_moe_workspace_bytes(int,int,int);
extern "C" int hopper_qwen3_moe_forward_bf16(const float*,const void*,void*,void*,void*,int,int,int,int,void*,size_t,cudaStream_t);
extern "C" int hopper_qwen35_moe_forward_bf16(const float*,const void*,void*,void*,void*,int,int,int,int,void*,size_t,cudaStream_t);
extern "C" int hopper_gemma4_moe_forward_bf16(const float*,const float*,const void*,void*,void*,void*,int,int,int,void*,size_t,cudaStream_t);
#endif
#endif
namespace tei {
bool native_hopper_moe_available(const c10::Device& device) {
#if defined(TEI_TORCH_MOE_KERNELS) && defined(TEI_TORCH_MOE_HOPPER)
 if(!device.is_cuda())return false;
 auto props=device.has_index()?at::cuda::getDeviceProperties(device.index()):at::cuda::getCurrentDeviceProperties();
 return props->major==9;
#else
 return false;
#endif
}
at::Tensor routed_moe_cuda(const at::Tensor& x,const at::Tensor& logits,
 const at::Tensor& gu,const at::Tensor& down,bool renormalize,const at::Tensor& scales) {
#ifdef TEI_TORCH_MOE_KERNELS
 if(!x.is_cuda()||x.scalar_type()!=at::kBFloat16||x.dim()!=2||x.size(0)==0
   ||gu.dim()!=3||down.dim()!=3||logits.dim()!=2) return {};
 auto e=gu.size(0),h=x.size(1),i=down.size(2),t=x.size(0);
 if((e!=128&&e!=256)||h%8||i%8||h>INT_MAX||i>INT_MAX/2||t>INT_MAX/8) return {};
 if(scales.defined()&&(e!=128||h!=2816||i!=704))return {};
 const c10::cuda::CUDAGuard guard(x.device());
 auto props=at::cuda::getCurrentDeviceProperties();
 if(props->major<8)return {};
 TORCH_CHECK(gu.sizes()==at::IntArrayRef({e,2*i,h})&&down.sizes()==at::IntArrayRef({e,h,i})
  &&logits.sizes()==at::IntArrayRef({t,e}),"Invalid routed MoE dimensions");
 for(const auto& w:{gu,down})TORCH_CHECK(w.device()==x.device()&&w.scalar_type()==at::kBFloat16&&w.is_contiguous(),"MoE weights must be contiguous BF16 on input device");
 TORCH_CHECK(x.is_contiguous()&&logits.is_contiguous()&&logits.device()==x.device()&&logits.scalar_type()==at::kFloat,"MoE input/logits must be contiguous BF16/FP32 on same device");
 if(scales.defined())TORCH_CHECK(scales.device()==x.device()&&scales.scalar_type()==at::kFloat&&scales.is_contiguous()&&scales.numel()==e,"Invalid MoE expert scales");
 auto bytes=scales.defined()?gemma4_moe_workspace_bytes(t,h,i):(e==128?qwen3_moe_workspace_bytes(t,h,i):qwen35_moe_workspace_bytes(t,h,i));
 bool hopper=false;
#ifdef TEI_TORCH_MOE_HOPPER
 hopper=props->major==9;
 if(hopper)bytes=scales.defined()?hopper_gemma4_moe_workspace_bytes(t,h,i):(e==128?hopper_qwen3_moe_workspace_bytes(t,h,i):hopper_qwen35_moe_workspace_bytes(t,h,i));
#endif
 TORCH_CHECK(bytes,"Unsupported routed MoE workspace dimensions");
 auto scratch=at::empty({static_cast<int64_t>(bytes)},x.options().dtype(at::kByte));
 auto out=at::empty_like(x);
 auto stream=at::cuda::getCurrentCUDAStream(x.get_device()).stream();
 int status;
#ifdef TEI_TORCH_MOE_HOPPER
 if(hopper) {
  if(scales.defined())status=hopper_gemma4_moe_forward_bf16(logits.data_ptr<float>(),scales.data_ptr<float>(),x.data_ptr(),gu.data_ptr(),down.data_ptr(),out.data_ptr(),t,h,i,scratch.data_ptr(),bytes,stream);
  else {auto fn=e==128?hopper_qwen3_moe_forward_bf16:hopper_qwen35_moe_forward_bf16;status=fn(logits.data_ptr<float>(),x.data_ptr(),gu.data_ptr(),down.data_ptr(),out.data_ptr(),t,h,i,renormalize,scratch.data_ptr(),bytes,stream);}
 } else
#endif
 {
  if(scales.defined())status=gemma4_moe_forward_bf16(logits.data_ptr<float>(),scales.data_ptr<float>(),x.data_ptr(),gu.data_ptr(),down.data_ptr(),out.data_ptr(),t,h,i,scratch.data_ptr(),bytes,stream);
  else {auto fn=e==128?qwen3_moe_forward_bf16:qwen35_moe_forward_bf16;status=fn(logits.data_ptr<float>(),x.data_ptr(),gu.data_ptr(),down.data_ptr(),out.data_ptr(),t,h,i,renormalize,scratch.data_ptr(),bytes,stream);}
 }
 TORCH_CHECK(status==0,"Native routed MoE kernel failed: ",status);
 return out;
#else
 return {};
#endif
}
at::Tensor routed_moe_combine_cuda(const at::Tensor& expert,const at::Tensor& order,const at::Tensor& routing) {
#ifdef TEI_TORCH_MOE_KERNELS
 if(!expert.is_cuda() || expert.scalar_type()!=at::kBFloat16 || expert.dim()!=2
   || expert.size(0)==0 || expert.size(1)==0 || routing.dim()!=2 || routing.size(1)!=8 || expert.size(1)%8)return {};
 TORCH_CHECK(expert.is_contiguous() && routing.is_contiguous() && routing.scalar_type()==at::kFloat
   && order.is_contiguous() && order.dim()==1 && order.scalar_type()==at::kLong
   && order.numel()==expert.size(0) && routing.numel()==order.numel()
   && expert.size(0)<=INT_MAX && expert.size(1)<=INT_MAX
   && order.device()==expert.device() && routing.device()==expert.device(),"Invalid routed MoE combine inputs");
 const c10::cuda::CUDAGuard guard(expert.device());
 auto mapping=at::empty_like(order,order.options().dtype(at::kInt));
 mapping.scatter_(0,order,at::arange(order.numel(),mapping.options()));
 auto values=expert.to(at::kFloat),output=at::empty({routing.size(0),expert.size(1)},expert.options());
 const auto* x=values.const_data_ptr<float>();
 const auto* indices=reinterpret_cast<const uint32_t*>(mapping.const_data_ptr<int32_t>());
 const auto* weights=routing.const_data_ptr<float>();auto* out=output.data_ptr();
 uint64_t tokens=routing.size(0);uint32_t hidden=expert.size(1);
 void* args[]={&x,&indices,&weights,&out,&tokens,&hidden};
 C10_CUDA_CHECK(cudaLaunchKernel(reinterpret_cast<const void*>(gemma4_moe_combine_vec8),
   dim3((tokens*(hidden/8)+255)/256),dim3(256),args,0,at::cuda::getCurrentCUDAStream(expert.get_device()).stream()));
 return output;
#else
 return {};
#endif
}

}

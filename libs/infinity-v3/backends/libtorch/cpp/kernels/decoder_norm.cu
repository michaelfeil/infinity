/*
BSD 3-Clause License

Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// This adapter retains and invokes the fork's BSD-3-Clause Candle kernel.
// Private kernel names prevent preemption of Candle's own CUDA registration.
#include "kernels/decoder_norm.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#define layer_norm tei_decoder_exact_norm
#include "../../../candle/extensions/candle-layer-norm/kernels/ln_fwd_kernels.cuh"
#undef layer_norm

namespace tei {
namespace {
template<int Width,int WarpsM,int WarpsN,class T=nv_bfloat16,bool Even=true>
std::pair<at::Tensor,at::Tensor> exact_norm(const at::Tensor& x,const at::Tensor& residual,
 const at::Tensor& w,double epsilon) {
 const c10::cuda::CUDAGuard guard(x.device());
 auto out=at::empty_like(x),sum=residual.defined()?at::empty_like(x):x;
 if(x.size(0)==0)return {out,sum};
 tei_decoder_exact_norm::FwdParams p;
 p.rows=x.size(0);p.cols=Even?Width:x.size(1);p.x0=x.data_ptr();p.x1=nullptr;p.residual=residual.defined()?residual.data_ptr():nullptr;p.x=residual.defined()?sum.data_ptr():nullptr;p.z=out.data_ptr();
 p.dmask=nullptr;p.dmask1=nullptr;p.x0_subset=nullptr;p.z_subset=nullptr;p.gamma=w.data_ptr();p.epsilon=epsilon;p.inverse_cols=Even?1.f/Width:1.f/x.size(1);p.dropout_keep_p=1.f;p.dropout_scale=1.f;p.rowscale_const=1.f;p.round_residual=false;p.is_rms_norm=true;
 using Traits=tei_decoder_exact_norm::Kernel_traits<T,T,T,T,float,int,Width,1,WarpsM,WarpsN,16>;
 auto kernel=tei_decoder_exact_norm::ln_fwd_kernel<Traits,false,false,false,Even>;
 // Each registered width uses its original Candle warp geometry. Cache
 // occupancy separately per specialization and device, before graph capture.
 static thread_local std::vector<int> cache;
 const int device=x.get_device();if(cache.size()<=static_cast<size_t>(device))cache.resize(device+1,-1);
 auto& blocks=cache[device];if(blocks<0)C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,Traits::THREADS_PER_CTA,Traits::SMEM_BYTES_FWD));
 p.ctas_per_col=at::cuda::getCurrentDeviceProperties()->multiProcessorCount*blocks;
 kernel<<<p.ctas_per_col,Traits::THREADS_PER_CTA,Traits::SMEM_BYTES_FWD,at::cuda::getCurrentCUDAStream()>>>(p);C10_CUDA_KERNEL_LAUNCH_CHECK();return {out,sum};
}
}
std::pair<at::Tensor,at::Tensor> decoder_exact_rms_norm(const at::Tensor& x,
 const at::Tensor& residual,const at::Tensor& w,double epsilon) {
 TORCH_CHECK(x.is_cuda()&&x.dim()==2&&x.is_contiguous()
   &&((x.scalar_type()==at::kBFloat16&&(x.size(1)==896||x.size(1)==1024||x.size(1)==2048||x.size(1)==3072||x.size(1)==4096))
     ||(x.scalar_type()==at::kHalf&&(x.size(1)==896||x.size(1)==1024||x.size(1)==2048))),
   "Exact decoder RMS expects CUDA BF16 width 896, 1024, 2048, 3072 or 4096, or FP16 width 896, 1024 or 2048");
 TORCH_CHECK(w.device()==x.device()&&w.scalar_type()==x.scalar_type()&&w.dim()==1&&w.numel()==x.size(1)&&w.is_contiguous(),"Invalid exact decoder RMS weight");
 TORCH_CHECK(!residual.defined()||(residual.device()==x.device()&&residual.scalar_type()==x.scalar_type()&&residual.sizes()==x.sizes()&&residual.is_contiguous()),"Invalid exact decoder RMS residual");
 // Retain Candle's registered 4x1 warp geometry and FP16 rounding.
 if(x.scalar_type()==at::kHalf&&x.size(1)==896)return exact_norm<1024,4,1,__half,false>(x,residual,w,epsilon);
 if(x.scalar_type()==at::kHalf&&x.size(1)==2048)return exact_norm<2048,4,1,__half>(x,residual,w,epsilon);
 if(x.scalar_type()==at::kHalf)return exact_norm<1024,4,1,__half>(x,residual,w,epsilon);
 // Candle masks actual BF16 width896 within its original1024 registration.
 if(x.size(1)==896)return exact_norm<1024,4,1,nv_bfloat16,false>(x,residual,w,epsilon);
 if(x.size(1)==1024)return exact_norm<1024,4,1>(x,residual,w,epsilon);
 if(x.size(1)==2048)return exact_norm<2048,4,1>(x,residual,w,epsilon);
 if(x.size(1)==3072)return exact_norm<3072,1,4>(x,residual,w,epsilon);
 return exact_norm<4096,1,4>(x,residual,w,epsilon);
}
}

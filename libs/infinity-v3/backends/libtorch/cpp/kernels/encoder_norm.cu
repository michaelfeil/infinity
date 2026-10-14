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
// The included CUDA kernel is distributed under the BSD-3-Clause license in
// backends/candle/extensions/candle-layer-norm/LICENSE; retained in this source tree.
// This adapter invokes that implementation rather than recreating its reductions.
#include "kernels/encoder_norm.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>
#include "../../../candle/extensions/candle-layer-norm/kernels/ln_fwd_kernels.cuh"
#include "kernels/encoder_embedding_norm.cuh"

namespace tei {
namespace {
template<class T, int Width, int Rows=4> void launch(layer_norm::FwdParams& p, cudaStream_t stream) {
  using Traits = layer_norm::Kernel_traits<T,T,T,T,float,int,Width,1,Rows,1,16>;
  // Occupancy depends only on this template, column geometry, and device.
  // Avoid a CUDA driver query at every one of a deep encoder's norms.
  struct Occupancy {int full=-1,tail=-1;};
  static thread_local std::vector<Occupancy> cache;
  const int device=at::cuda::current_device();
  if(cache.size()<=static_cast<size_t>(device))cache.resize(device+1);
  int& blocks=p.cols==Width?cache[device].full:cache[device].tail;
  const int processors = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  if (p.cols == Width) {
    auto kernel = layer_norm::ln_fwd_kernel<Traits,false,false,false,true>;
    if(blocks<0)C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,Traits::THREADS_PER_CTA,Traits::SMEM_BYTES_FWD));
    p.ctas_per_col=processors*blocks;
    kernel<<<p.ctas_per_col,Traits::THREADS_PER_CTA,Traits::SMEM_BYTES_FWD,stream>>>(p);
  } else {
    auto kernel = layer_norm::ln_fwd_kernel<Traits,false,false,false,false>;
    if(blocks<0)C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,Traits::THREADS_PER_CTA,Traits::SMEM_BYTES_FWD));
    p.ctas_per_col=processors*blocks;
    kernel<<<p.ctas_per_col,Traits::THREADS_PER_CTA,Traits::SMEM_BYTES_FWD,stream>>>(p);
  }
}
template<class T> void dispatch(layer_norm::FwdParams& p,cudaStream_t stream) {
  const int rounded=((p.cols+255)/256)*256;
  switch(rounded) {
    case 256:launch<T,256>(p,stream);break;
    case 512:launch<T,512>(p,stream);break;
    case 768:launch<T,768>(p,stream);break;
    case 1024:
      if(p.rows>=4096)launch<T,1024,8>(p,stream);
      else launch<T,1024>(p,stream);
      break;
    case 1280:launch<T,1280>(p,stream);break;
    case 1536:launch<T,1536>(p,stream);break;
    default:TORCH_CHECK(false,"Unsupported exact encoder normalization width");
  }
}
}
static std::pair<at::Tensor,at::Tensor> encoder_layer_norm_impl(const at::Tensor& x,const at::Tensor& weight,double epsilon,const at::Tensor& residual,const at::Tensor& bias,bool round_residual,bool save_residual,bool update_residual=false,const at::Tensor& normalized_scratch=at::Tensor()) {
  TORCH_CHECK(x.is_cuda()&&x.is_contiguous()&&x.dim()>=2&&x.size(-1)>0,"Exact encoder norm requires contiguous CUDA token rows");
  TORCH_CHECK(weight.device()==x.device()&&weight.scalar_type()==x.scalar_type()&&weight.is_contiguous()&&weight.numel()==x.size(-1),"Encoder normalization weight mismatch");
  if(residual.defined())TORCH_CHECK(residual.device()==x.device()&&residual.scalar_type()==x.scalar_type()&&residual.is_contiguous()&&residual.sizes()==x.sizes(),"Encoder normalization residual mismatch");
  if(bias.defined())TORCH_CHECK(bias.device()==x.device()&&bias.scalar_type()==x.scalar_type()&&bias.is_contiguous()&&bias.numel()==x.size(-1),"Encoder normalization bias mismatch");
  c10::cuda::CUDAGuard guard(x.device());
  const bool reuse=update_residual&&residual.defined()&&x.size(-1)<=1536;
  auto normalized=normalized_scratch.defined()?normalized_scratch:at::empty_like(x);
  auto sum=residual.defined()&&save_residual?
    (reuse?residual:at::empty_like(x)):at::Tensor();
  if(x.size(-1)>1536) {
    if(residual.defined()&&save_residual)sum=x+residual;
    if(residual.defined()&&!round_residual) {
      auto stats=x.to(at::kFloat)+residual.to(at::kFloat);
      return {at::layer_norm(stats,{x.size(-1)},weight.to(at::kFloat),bias.defined()?bias.to(at::kFloat):at::Tensor(),epsilon).to(x.scalar_type()),sum};
    }
    auto added=residual.defined()?(sum.defined()?sum:x+residual):x;
    return {at::layer_norm(added,{x.size(-1)},weight,bias,epsilon),sum};
  }
  layer_norm::FwdParams p;
  p.rows=x.numel()/x.size(-1);p.cols=x.size(-1);
  p.x0=const_cast<void*>(x.const_data_ptr());p.x1=nullptr;
  p.residual=residual.defined()?const_cast<void*>(residual.const_data_ptr()):nullptr;
  p.x=sum.defined()?sum.mutable_data_ptr():nullptr;p.z=normalized.mutable_data_ptr();
  p.dmask=nullptr;p.dmask1=nullptr;p.x0_subset=nullptr;p.z_subset=nullptr;
  p.gamma=const_cast<void*>(weight.const_data_ptr());p.beta=bias.defined()?const_cast<void*>(bias.const_data_ptr()):nullptr;p.epsilon=epsilon;
  p.inverse_cols=1.f/p.cols;p.dropout_keep_p=1.f;p.dropout_scale=1.f;p.rowscale_const=1.f;p.round_residual=round_residual;
  auto stream=at::cuda::getCurrentCUDAStream(x.device().index());
  if(x.scalar_type()==at::kHalf)dispatch<half>(p,stream);
  else if(x.scalar_type()==at::kBFloat16)dispatch<nv_bfloat16>(p,stream);
  else if(x.scalar_type()==at::kFloat)dispatch<float>(p,stream);
  else TORCH_CHECK(false,"Unsupported encoder normalization dtype");
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {normalized,sum};
}
std::pair<at::Tensor,at::Tensor> encoder_layer_norm_update_residual(const at::Tensor& x,
    const at::Tensor& weight,double epsilon,at::Tensor& residual) {
  // Each vector is loaded into registers before the same disjoint vector is
  // saved. Statistics and normalization use those registers, never reread sum.
  // CTAS_PER_ROW==1 and the existing row geometry prevent cross-CTA aliasing.
  return encoder_layer_norm_impl(x,weight,epsilon,residual,{},true,true,true);
}
std::pair<at::Tensor,at::Tensor> encoder_layer_norm_update_residual_scratch(
    const at::Tensor& x,const at::Tensor& weight,double epsilon,
    at::Tensor& residual,const at::Tensor& scratch) {
  const bool reusable=scratch.defined()&&x.size(-1)<=1536&&
    scratch.device()==x.device()&&scratch.scalar_type()==x.scalar_type()&&
    scratch.sizes()==x.sizes()&&scratch.is_contiguous()&&
    !scratch.is_alias_of(x)&&!scratch.is_alias_of(weight)&&
    (!residual.defined()||!scratch.is_alias_of(residual));
  // Invalid destinations are never written, including aliases of residual or
  // affine storage. Keep both original allocating outputs in this fallback.
  return encoder_layer_norm_impl(x,weight,epsilon,residual,{},true,true,
    reusable,reusable?scratch:at::Tensor());
}
std::pair<at::Tensor,at::Tensor> encoder_layer_norm(const at::Tensor& x,const at::Tensor& weight,double epsilon,const at::Tensor& residual,const at::Tensor& bias,bool round_residual) {
  return encoder_layer_norm_impl(x,weight,epsilon,residual,bias,round_residual,true);
}
at::Tensor encoder_layer_norm_only(const at::Tensor& x,const at::Tensor& weight,double epsilon,const at::Tensor& residual,const at::Tensor& bias,bool round_residual) {
  return encoder_layer_norm_impl(x,weight,epsilon,residual,bias,round_residual,false).first;
}
at::Tensor encoder_embedding_layer_norm(const at::Tensor& word,const at::Tensor& token_type,const at::Tensor& position,const at::Tensor& weight,double epsilon,const at::Tensor& bias) {
  TORCH_CHECK(word.is_cuda()&&word.scalar_type()==at::kHalf&&word.is_contiguous()&&word.dim()>=2&&word.size(-1)==768,"Embedding norm requires contiguous CUDA Half width-768 rows");
  for(const auto* input:{&token_type,&position})
    TORCH_CHECK(input->device()==word.device()&&input->scalar_type()==at::kHalf&&input->is_contiguous()&&input->sizes()==word.sizes(),"Embedding norm input mismatch");
  for(const auto* affine:{&weight,&bias})
    TORCH_CHECK(affine->device()==word.device()&&affine->scalar_type()==at::kHalf&&affine->is_contiguous()&&affine->numel()==768,"Embedding norm affine mismatch");
  c10::cuda::CUDAGuard guard(word.device());
  auto normalized=at::empty_like(word);
  layer_norm::FwdParams p;
  p.rows=word.numel()/768;p.cols=768;
  p.x0=const_cast<void*>(word.const_data_ptr());p.x1=const_cast<void*>(token_type.const_data_ptr());
  p.residual=const_cast<void*>(position.const_data_ptr());p.z=normalized.mutable_data_ptr();
  p.dmask=nullptr;p.dmask1=nullptr;p.x0_subset=nullptr;p.z_subset=nullptr;
  p.gamma=const_cast<void*>(weight.const_data_ptr());p.beta=const_cast<void*>(bias.const_data_ptr());
  p.epsilon=epsilon;p.inverse_cols=1.f/768;p.dropout_keep_p=1.f;p.dropout_scale=1.f;p.rowscale_const=1.f;p.round_residual=false;
  using Traits=layer_norm::Kernel_traits<half,half,half,half,float,int,768,1,4,1,16>;
  auto kernel=layer_norm::embedding_ln_fwd_kernel<Traits,false,false,false,true>;
  static thread_local std::vector<int> occupancy;
  const int device=word.device().index();
  if(occupancy.size()<=static_cast<size_t>(device))occupancy.resize(device+1,-1);
  int& blocks=occupancy[device];
  if(blocks<0)C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,Traits::THREADS_PER_CTA,Traits::SMEM_BYTES_FWD));
  p.ctas_per_col=at::cuda::getCurrentDeviceProperties()->multiProcessorCount*blocks;
  auto stream=at::cuda::getCurrentCUDAStream(device);
  kernel<<<p.ctas_per_col,Traits::THREADS_PER_CTA,Traits::SMEM_BYTES_FWD,stream>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();return normalized;
}
}

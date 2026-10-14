// Independent exact-length reference for the private native Torch Flash adapter.
#include "kernels/torch_flash_fma.h"
#include "kernels/torch_flash_exact32.h"
#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <vector>
#include <cmath>


// Media attention retains true head width72 and distinct query/key lengths.
static void media_spans(const c10::Device& device) {
  auto options=at::TensorOptions().device(device).dtype(at::kBFloat16);
  auto probe=at::zeros({3,4,72},options);
  const auto* properties=at::cuda::getDeviceProperties(device.index());
  bool supported=properties->major==9 && properties->minor==0 && properties->sharedMemPerBlockOptin>=48*1024;
  TORCH_CHECK(tei::torch_flash_fma_supported(probe,probe,probe)==supported,"Head72 architecture guard");
  TORCH_CHECK(!tei::torch_flash_fma_supported(probe,probe,probe,true,-1,-1),"Head72 causal fallback");
  TORCH_CHECK(!tei::torch_flash_fma_supported(probe,probe,probe,false,2,2),"Head72 local fallback");
  auto half=probe.to(at::kHalf),cpu=probe.cpu();
  TORCH_CHECK(!tei::torch_flash_fma_supported(half,half,half),"Head72 FP16 fallback");
  TORCH_CHECK(!tei::torch_flash_fma_supported(cpu,cpu,cpu),"Head72 CPU fallback");
  if(!supported)return;
  for(int64_t dim:{72,256})for(bool causal:{false,true}) {
    if(dim==72 && causal)continue;
    std::vector<int32_t> qo{0,1,4,11},ko{0,3,11,24};
    auto cuq=at::tensor(qo,at::TensorOptions().dtype(at::kInt)).to(device),cuk=at::tensor(ko,at::TensorOptions().dtype(at::kInt)).to(device);
    auto qstore=at::randn({11,6,dim},options)*.1,kstore=at::randn({24,4,dim},options)*.1;
    auto q=qstore.narrow(1,0,4),k=kstore.narrow(1,0,2),v=kstore.narrow(1,2,2);
    auto result=tei::torch_flash_fma_varlen(q,k,v,cuq,cuk,7,13,1.,causal);
    TORCH_CHECK(result.sizes()==q.sizes(),"Media attention allocated synthetic rows/head width");
    double worst=0;
    for(size_t b=0;b<3;++b) {
      int64_t nq=qo[b+1]-qo[b],nk=ko[b+1]-ko[b];
      auto qr=q.narrow(0,qo[b],nq).cpu().to(at::kDouble).transpose(0,1);
      auto kr=k.narrow(0,ko[b],nk).cpu().to(at::kDouble).transpose(0,1).repeat_interleave(2,0);
      auto vr=v.narrow(0,ko[b],nk).cpu().to(at::kDouble).transpose(0,1).repeat_interleave(2,0);
      auto scores=at::matmul(qr,kr.transpose(1,2));
      if(causal) {
        auto delta=at::arange(nk,at::kLong).unsqueeze(0)-at::arange(nq,at::kLong).unsqueeze(1)-(nk-nq);
        scores=scores.masked_fill(delta>0,-INFINITY);
      }
      auto expected=at::matmul(at::softmax(scores,-1),vr).transpose(0,1);
      worst=std::max(worst,(result.narrow(0,qo[b],nq).cpu().to(at::kDouble)-expected).abs().max().item<double>());
    }
    TORCH_CHECK(worst<.003,"Unequal media spans FP64 reference",worst);
    auto changed=v.clone();changed.narrow(0,0,3).add_(1);
    auto isolated=tei::torch_flash_fma_varlen(q,k,changed,cuq,cuk,7,13,1.,causal);
    TORCH_CHECK(at::equal(result.narrow(0,1,10),isolated.narrow(0,1,10)),"Unequal media span sequence isolation");
  }
}

// Independent reference computes the bias from real per-sequence positions;
// it does not share the Flash kernel's pre-scaled ALiBi implementation.
static void alibi32_spans(const c10::Device& device) {
  const bool supported=at::cuda::getDeviceProperties(device.index())->major>=8;
  TORCH_CHECK(tei::torch_flash_fma_alibi32_supported(device,at::kHalf,32)==supported,
    "Head32 ALiBi architecture guard mismatch");
  TORCH_CHECK(!tei::torch_flash_fma_alibi32_supported(c10::Device(at::kCPU),at::kHalf,32)
    && !tei::torch_flash_fma_alibi32_supported(device,at::kBFloat16,32)
    && !tei::torch_flash_fma_alibi32_supported(device,at::kFloat,32)
    && !tei::torch_flash_fma_alibi32_supported(device,at::kHalf,64),
    "Head32 ALiBi unsupported geometry/dtype must retain fallback");
  if(!supported)return;
  const int64_t heads=12,dim=32;
  // Original non-power-of-two Jina slope order, with a zero-slope case too.
  std::vector<float> values{.5f,.25f,.125f,.0625f,.03125f,.015625f,.0078125f,.00390625f,
    static_cast<float>(std::sqrt(.5)),static_cast<float>(std::pow(.5,1.5)),
    static_cast<float>(std::pow(.5,2.5)),static_cast<float>(std::pow(.5,3.5))};
  auto options=at::TensorOptions().device(device).dtype(at::kHalf);
  for(bool zero_slope:{false,true})for(const auto& lengths:{std::vector<int32_t>{3,1,17},std::vector<int32_t>{32,129,7}}) {
    auto slopes=at::tensor(values,at::TensorOptions().dtype(at::kFloat)).to(device);
    if(zero_slope)slopes.zero_();
    std::vector<int32_t> offsets{0};int64_t total=0,maximum=0;
    for(auto length:lengths){total+=length;maximum=std::max<int64_t>(maximum,length);offsets.push_back(total);}
    auto combined=at::randn({total,3*heads,dim},options);
    auto q=combined.narrow(1,0,heads),k=combined.narrow(1,heads,heads),v=combined.narrow(1,2*heads,heads);
    auto cumulative=at::tensor(offsets,at::TensorOptions().dtype(at::kInt)).to(device);
    const double scale=1./std::sqrt(static_cast<double>(dim));
    auto result=tei::torch_flash_fma_alibi32(q,k,v,cumulative,slopes,maximum,scale);
    TORCH_CHECK(result.sizes()==q.sizes() && result.numel()==total*heads*dim
      && at::isfinite(result).all().item<bool>(),"Head32 ALiBi changed real token/head geometry or finiteness");
    double worst=0;
    auto cpu_slopes=slopes.cpu().to(at::kDouble).view({heads,1,1});
    for(size_t b=0;b<lengths.size();++b) {
      auto cpu=[&](const at::Tensor& t){return t.narrow(0,offsets[b],lengths[b]).cpu().to(at::kDouble).transpose(0,1);};
      auto qr=cpu(q),kr=cpu(k),vr=cpu(v),positions=at::arange(lengths[b],at::kDouble);
      auto distance=(positions.unsqueeze(0)-positions.unsqueeze(1)).abs();
      auto scores=at::matmul(qr,kr.transpose(1,2))*scale-cpu_slopes*distance;
      auto expected=at::matmul(at::softmax(scores,-1),vr).transpose(0,1);
      worst=std::max(worst,(result.narrow(0,offsets[b],lengths[b]).cpu().to(at::kDouble)-expected).abs().max().item<double>());
    }
    TORCH_CHECK(worst<.004,"Head32 ALiBi independent FP64 bias/attention mismatch: ",worst);
    auto changed=v.clone();changed.select(0,lengths[0]-1).add_(5.);
    auto isolated=tei::torch_flash_fma_alibi32(q,k,changed,cumulative,slopes,maximum,scale);
    TORCH_CHECK(at::equal(result.narrow(0,lengths[0],total-lengths[0]),isolated.narrow(0,lengths[0],total-lengths[0])),
      "Head32 ALiBi packed sequence isolation failed");
    TORCH_CHECK(!at::equal(result[0],isolated[0]),"Head32 ALiBi unexpectedly masked future tokens");
    std::cout<<"FP16 head32 ALiBi total="<<total<<" zero_slope="<<zero_slope<<" maxabs="<<worst<<'\n';
  }
}

int main(int argc,char** argv) {
  c10::InferenceMode guard;
  const c10::Device device(argc>1 ? argv[1] : "cuda:0");
  media_spans(device);
  at::manual_seed(9917);at::set_num_threads(1);
  alibi32_spans(device);
  for(auto dtype:{at::kHalf,at::kBFloat16})for(int64_t dim:{32,64,128,256})
  for(bool grouped:{false,true})for(bool causal:{false,true})for(int64_t window:{-1,2})
  for(auto lengths:{std::vector<int32_t>{3,13,7},std::vector<int32_t>{32,129,17}}) {
    // Dedicated head32 profile keeps the generic adapter's guards unchanged.
    if(dim==32 && (grouped || causal || window>=0 || at::cuda::getDeviceProperties(device.index())->major!=9))continue;
    const int64_t heads=dim==32?12:4,key_heads=grouped?2:heads;
    std::vector<int32_t> offsets{0};int64_t total=0,max_sequence=0;
    for(auto n:lengths){total+=n;max_sequence=std::max<int64_t>(max_sequence,n);offsets.push_back(total);}
    auto options=at::TensorOptions().device(device).dtype(dtype);
    // Fresh fused QKV storage tests noncontiguous token/head views, including GQA.
    auto combined=at::randn({total,heads+2*key_heads,dim},options);
    auto q=combined.narrow(1,0,heads),k=combined.narrow(1,heads,key_heads),v=combined.narrow(1,heads+key_heads,key_heads);
    auto cu=at::tensor(offsets,at::TensorOptions().dtype(at::kInt)).to(device);
    const double scale=1./std::sqrt(dim);
    auto output=dim==32?tei::torch_flash_exact32(q,k,v,cu,max_sequence,scale):tei::torch_flash_fma(q,k,v,cu,max_sequence,scale,causal,window,window);
    TORCH_CHECK(at::isfinite(output).all().item<bool>(),"Nonfinite native Torch Flash output");
    double worst=0;
    for(size_t row=0;row+1<offsets.size();++row) {
      const int64_t start=offsets[row],n=offsets[row+1]-start;
      auto cpu=[&](const at::Tensor& x){return x.narrow(0,start,n).to(at::TensorOptions().device(at::kCPU).dtype(at::kDouble)).transpose(0,1);};
      auto qr=cpu(q),kr=cpu(k),vr=cpu(v);
      if(grouped){kr=kr.repeat_interleave(heads/key_heads,0);vr=vr.repeat_interleave(heads/key_heads,0);}
      auto index=at::arange(n,at::kLong),delta=index.unsqueeze(0)-index.unsqueeze(1);
      auto allowed=at::ones({n,n},at::kBool);
      if(causal)allowed.logical_and_(delta<=0);
      if(window>=0)allowed.logical_and_(delta>=-window).logical_and_(delta<=window);
      auto probabilities=at::softmax((at::matmul(qr,kr.transpose(1,2))*scale).masked_fill(allowed.logical_not(),-INFINITY),-1);
      auto expected=at::matmul(probabilities,vr).transpose(0,1);
      worst=std::max(worst,(output.narrow(0,start,n).to(at::TensorOptions().device(at::kCPU).dtype(at::kDouble))-expected).abs().max().item<double>());
    }
    TORCH_CHECK(worst<(dtype==at::kHalf?.004:.04),"Native Torch Flash reference mismatch: ",worst);
    auto changed=v.clone();changed.narrow(0,0,lengths[0]).add_(20);
    auto isolated=dim==32?tei::torch_flash_exact32(q,k,changed,cu,max_sequence,scale):tei::torch_flash_fma(q,k,changed,cu,max_sequence,scale,causal,window,window);
    TORCH_CHECK(at::isfinite(isolated).all().item<bool>() && at::equal(output.narrow(0,lengths[0],total-lengths[0]),isolated.narrow(0,lengths[0],total-lengths[0])),"Packed sequence isolation failed");
    std::cout<<"dtype="<<dtype<<" total="<<total<<" dim="<<dim<<" grouped="<<grouped<<" causal="<<causal<<" window="<<window<<" maxabs="<<worst<<'\n';
  }
  // The exact-width512 launch is intentionally restricted; callers retain
  // efficient attention for unsupported devices, masks and activation types.
  auto wide_options=at::TensorOptions().device(device).dtype(at::kBFloat16);
  auto probe=at::zeros({3,1,512},wide_options);
  const auto* properties=at::cuda::getDeviceProperties(device.index());
  const bool hopper=properties->major==9 && properties->minor==0 && properties->sharedMemPerBlockOptin>=128*1024;
  TORCH_CHECK(tei::torch_flash_fma_supported(probe,probe,probe,false,-1,-1)==hopper,"Wide-head architecture guard mismatch");
  TORCH_CHECK(tei::torch_flash_fma_supported(probe,probe,probe,true,-1,-1)==hopper,"Wide-head causal architecture guard mismatch");
  TORCH_CHECK(!tei::torch_flash_fma_supported(probe,probe,probe,false,2,2),"Wide-head local mask must retain fallback");
  auto half_probe=probe.to(at::kHalf),cpu_probe=probe.cpu();
  TORCH_CHECK(!tei::torch_flash_fma_supported(half_probe,half_probe,half_probe,false,-1,-1),"Wide-head FP16 must retain fallback");
  TORCH_CHECK(!tei::torch_flash_fma_supported(cpu_probe,cpu_probe,cpu_probe,false,-1,-1),"Wide-head CPU must retain fallback");
  if(hopper)for(bool causal:{false,true})for(bool grouped:{false,true})for(auto lengths:{std::vector<int32_t>{3,17,9},std::vector<int32_t>{32,129,17}}) {
    const int64_t heads=4,key_heads=grouped?1:4;std::vector<int32_t> offsets{0};int64_t total=0,maximum=0;
    for(auto n:lengths){total+=n;maximum=std::max<int64_t>(maximum,n);offsets.push_back(total);}
    auto combined=at::randn({total,heads+2*key_heads,512},wide_options)*.1;
    auto q=combined.narrow(1,0,heads),k=combined.narrow(1,heads,key_heads),v=combined.narrow(1,heads+key_heads,key_heads);
    auto cu=at::tensor(offsets,at::TensorOptions().dtype(at::kInt)).to(device);
    auto output=tei::torch_flash_fma(q,k,v,cu,maximum,1.,causal,-1,-1);
    TORCH_CHECK(output.sizes()==q.sizes()&&at::isfinite(output).all().item<bool>(),"Wide-head packed result geometry/finiteness mismatch");
    double worst=0;
    for(size_t row=0;row+1<offsets.size();++row) {
      int64_t start=offsets[row],n=offsets[row+1]-start;
      auto cpu=[&](const at::Tensor& x){return x.narrow(0,start,n).to(at::TensorOptions().device(at::kCPU).dtype(at::kDouble)).transpose(0,1);};
      auto qr=cpu(q),kr=cpu(k),vr=cpu(v);
      if(grouped){kr=kr.repeat_interleave(heads/key_heads,0);vr=vr.repeat_interleave(heads/key_heads,0);}
      auto scores=at::matmul(qr,kr.transpose(1,2));
      if(causal)scores=scores.masked_fill(at::ones({n,n},at::TensorOptions().dtype(at::kBool)).triu(1),-INFINITY);
      auto expected=at::matmul(at::softmax(scores,-1),vr).transpose(0,1);
      worst=std::max(worst,(output.narrow(0,start,n).cpu().to(at::kDouble)-expected).abs().max().item<double>());
    }
    TORCH_CHECK(worst<.003,"Wide-head independent FP64 ragged/GQA mismatch: ",worst);
    auto changed=v.clone();changed.narrow(0,0,lengths[0]).add_(1.);
    auto isolated=tei::torch_flash_fma(q,k,changed,cu,maximum,1.,causal,-1,-1);
    TORCH_CHECK(at::equal(output.narrow(0,lengths[0],total-lengths[0]),isolated.narrow(0,lengths[0],total-lengths[0])),"Wide-head packed sequence isolation failed");
    if(causal) {
      auto one=at::tensor(std::vector<int32_t>{0,1},at::TensorOptions().dtype(at::kInt)).to(device);
      auto prefix=tei::torch_flash_fma(q.narrow(0,0,1),k.narrow(0,0,1),v.narrow(0,0,1),one,1,1.,true,-1,-1);
      TORCH_CHECK(at::equal(prefix,output.narrow(0,0,1)),"Wide-head causal prefix preservation failed");
    }
    std::cout<<"BF16 head512 ragged/GQA total="<<total<<" grouped="<<grouped<<" causal="<<causal<<" maxabs="<<worst<<'\n';
  }
}

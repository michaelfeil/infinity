#include "kernels/qwen35_reference.h"
#include "kernels/qwen35_recurrent.h"
#include <array>
#include "kernels/qwen35_kernels.h"
#include "batching/prefix_cache.h"
#include <ATen/ATen.h>
#include <cmath>
#include <iostream>
#include <vector>

void resumed_delta(c10::Device device,int64_t kh,int64_t vh,bool fma) {
  constexpr int64_t tokens=35;
  auto opts=at::TensorOptions().device(device).dtype(at::kBFloat16);
  auto qkv=(at::randn({tokens,(2*kh+vh)*128},opts)*.2).contiguous();
  auto z=(at::randn({tokens,vh*128},opts)*.2).contiguous();
  auto ab=(at::randn({tokens,2*vh},opts)*.2).contiguous();
  auto conv=(at::randn({(2*kh+vh)*128,4},opts)*.2).contiguous();
  auto alog=at::zeros({vh},opts.dtype(at::kFloat)),dt=at::zeros_like(alog);
  auto norm=at::ones({128},opts);
  int32_t offsets[]={0,tokens};tei::PackedInput full;
  full.offsets=offsets;full.batch=1;full.max_sequence=tokens;
  full.cumulative=at::tensor(std::vector<int32_t>{0,tokens},opts.dtype(at::kInt));
  auto run=[&](const tei::PackedInput& input,int64_t start) {
    auto count=tokens-start;
    if(fma)return tei::qwen35_delta_cuda(qkv.narrow(0,start,count),z.narrow(0,start,count),ab.narrow(0,start,count),
      conv,alog,dt,norm,input,kh,vh,1e-6,{},nullptr,{},{},false,0);
    return tei::qwen35_reference_delta_cuda(qkv.narrow(0,start,count),z.narrow(0,start,count),ab.narrow(0,start,count),
      conv,alog,dt,norm,input,kh,vh,1e-6,0);
  };
  auto expected=run(full,0);
  for(int64_t boundary:{1,13,34}) {
    if(fma) {
      auto prefill=full;int32_t prefix_offsets[]={0,int32_t(boundary)};
      prefill.offsets=prefix_offsets;prefill.max_sequence=boundary;
      prefill.cumulative=at::tensor(std::vector<int32_t>{0,int32_t(boundary)},opts.dtype(at::kInt));
      at::Tensor state;
      auto prefix=tei::qwen35_delta_cuda(qkv.narrow(0,0,boundary),z.narrow(0,0,boundary),ab.narrow(0,0,boundary),
        conv,alog,dt,norm,prefill,kh,vh,1e-6,{},&state,{},{},false);
      TORCH_CHECK(at::equal(prefix,expected.narrow(0,0,boundary)),"Block prefill changed recurrent arithmetic");
      auto immutable=state.clone();auto suffix=full;int32_t suffix_offsets[]={0,int32_t(tokens-boundary)};
      suffix.offsets=suffix_offsets;suffix.max_sequence=tokens-boundary;
      suffix.cumulative=at::tensor(std::vector<int32_t>{0,int32_t(tokens-boundary)},opts.dtype(at::kInt));
      suffix.block_prefix_sources={0};suffix.block_cache_sources=at::zeros({1},opts.dtype(at::kLong));
      auto history=qkv.narrow(0,std::max(int64_t(0),boundary-3),std::min(boundary,int64_t(3)));
      auto history_cu=at::tensor(std::vector<int32_t>{0,int32_t(history.size(0))},opts.dtype(at::kInt));
      at::Tensor final_state;
      auto resumed=tei::qwen35_delta_cuda(qkv.narrow(0,boundary,tokens-boundary),z.narrow(0,boundary,tokens-boundary),
        ab.narrow(0,boundary,tokens-boundary),conv,alog,dt,norm,suffix,kh,vh,1e-6,state,&final_state,history,history_cu,false);
      TORCH_CHECK(at::equal(resumed,expected.narrow(0,boundary,tokens-boundary))&&at::equal(state,immutable),
        "Block resume changed outputs or mutated shared state");
      auto batched=suffix;std::vector<int32_t> batch_offsets;
      for(int i=0;i<=8;++i)batch_offsets.push_back(i*(tokens-boundary));
      batched.batch=8;batched.offsets=batch_offsets.data();
      batched.cumulative=at::tensor(batch_offsets,opts.dtype(at::kInt));
      batched.block_prefix_sources.assign(8,0);batched.block_cache_sources=at::zeros({8},opts.dtype(at::kLong));
      at::Tensor batch_state;
      auto repeated=tei::qwen35_delta_cuda(qkv.narrow(0,boundary,tokens-boundary).repeat({8,1}),
        z.narrow(0,boundary,tokens-boundary).repeat({8,1}),ab.narrow(0,boundary,tokens-boundary).repeat({8,1}),
        conv,alog,dt,norm,batched,kh,vh,1e-6,state,&batch_state,history,history_cu,false);
      TORCH_CHECK(at::equal(repeated,resumed.repeat({8,1}))&&at::equal(batch_state,final_state.repeat({8,1,1,1}))
        &&at::equal(state,immutable),"Batched recurrence changed outputs, final states or shared prefix");
    }
    auto budget=std::make_shared<tei::PrefixBudget>();
    auto saved=std::make_shared<tei::PrefixEntry>();saved->budget=budget;
    // This kernel-level fixture supplies the admission reservation normally
    // established by PrefixCache before requesting any checkpoint buffers.
    saved->reserved_bytes=budget->reserved_bytes=64<<20;
    saved->identity.tokens.resize(boundary);
    tei::PrefixContext capture{full,{0,tokens},{0},{nullptr},{saved},64<<20,budget};
    auto cold=full;cold.prefix=&capture;
    TORCH_CHECK(at::equal(run(cold,0),expected),"Cache capture changed DeltaNet arithmetic kh=",kh," vh=",vh," fma=",fma);
    auto state=saved->layers.find(0);
    TORCH_CHECK(state!=saved->layers.end()&&state->second.recurrent.defined(),"DeltaNet checkpoint was not exported");
    int32_t suffix_offsets[]={0,int32_t(tokens-boundary)};
    tei::PrefixContext resume{full,{0,int32_t(tokens-boundary)},{boundary},{saved},{nullptr},64<<20,budget};
    auto suffix=full;suffix.offsets=suffix_offsets;suffix.max_sequence=tokens-boundary;suffix.prefix=&resume;
    suffix.cumulative=at::tensor(std::vector<int32_t>{0,int32_t(tokens-boundary)},opts.dtype(at::kInt));
    auto actual=run(suffix,boundary);
    TORCH_CHECK(at::equal(actual,expected.narrow(0,boundary,tokens-boundary)),
      "Resumed DeltaNet changed outputs kh=",kh," vh=",vh," fma=",fma," boundary=",boundary,
      " maxabs=",(actual-expected.narrow(0,boundary,tokens-boundary)).abs().max().item<float>());
  }
}

// CPU oracle for the original 32-lane recurrence; does not call a CUDA dispatcher.
void independent_recurrent(c10::Device device,int kh,int vh,bool fma) {
  if(!tei::qwen35_recurrent_supported(device,kh,vh))return;
  constexpr int tokens=13;
  auto values=(at::randn({tokens,(2*kh+vh)*128})*.2).to(at::kBFloat16);
  auto qk=at::randn({tokens*kh,256})*.05,gates=at::rand({tokens,2*vh});
  auto expected=at::empty({tokens,vh,128},values.options());
  auto v=values.const_data_ptr<at::BFloat16>();
  auto q=qk.const_data_ptr<float>(),g=gates.const_data_ptr<float>();
  auto out=expected.mutable_data_ptr<at::BFloat16>();
  auto accumulate=[&](float sum,float a,float b) {
    if(fma)return std::fma(a,b,sum);
    volatile float product=a*b;return sum+product;
  };
  auto reduce=[](std::array<float,32> partial) {
    for(int d=16;d;d/=2)for(int lane=0;lane<d;++lane)partial[lane]+=partial[lane+d];
    return partial[0];
  };
  for(int head=0;head<vh;++head)for(int column=0;column<128;++column) {
    std::array<float,128> state{};
    for(int t=0;t<tokens;++t) {
      const auto* query=q+(t*kh+head/(vh/kh))*256;
      std::array<float,32> partial{};
      for(int lane=0;lane<32;++lane)for(int i=0;i<4;++i) {
        int k=lane+i*32;state[k]*=g[t*2*vh+head];
        partial[lane]=accumulate(partial[lane],state[k],query[128+k]);
      }
      volatile float difference=float(v[t*(2*kh+vh)*128+2*kh*128+head*128+column])-reduce(partial);
      float delta=difference*g[t*2*vh+vh+head];partial.fill(0.f);
      for(int lane=0;lane<32;++lane)for(int i=0;i<4;++i) {
        int k=lane+i*32;state[k]=accumulate(state[k],query[128+k],delta);
        partial[lane]=accumulate(partial[lane],state[k],query[k]);
      }
      out[(t*vh+head)*128+column]=at::BFloat16(reduce(partial));
    }
  }
  int32_t offsets[]={0,tokens};tei::PackedInput input;
  input.offsets=offsets;input.batch=1;input.max_sequence=tokens;
  input.cumulative=at::tensor(std::vector<int32_t>{0,tokens},at::TensorOptions().dtype(at::kInt).device(device));
  auto actual=at::empty_like(expected,expected.options().device(device));
  tei::qwen35_recurrent_cuda(values.to(device),qk.to(device),gates.to(device),input,actual,kh,vh,fma);
  TORCH_CHECK(at::equal(actual.cpu(),expected),"Independent recurrence oracle failed kh=",kh," vh=",vh," fma=",fma);
}

at::Tensor independent_norm(const at::Tensor& x,const at::Tensor& gamma,float eps) {
  auto f=x.to(at::kFloat).contiguous(),g=gamma.to(at::kFloat).contiguous();
  auto result=at::empty_like(f);auto width=x.size(-1),rows=x.numel()/width;
  auto input=f.const_data_ptr<float>(),weight=g.const_data_ptr<float>();auto out=result.mutable_data_ptr<float>();
  int threads=1;while(threads<width && threads<1024)threads*=2;
  for(int64_t row=0;row<rows;++row) {
    std::vector<float> sums(threads,0.f);
    for(int lane=0;lane<threads;++lane)for(int64_t col=lane;col<width;col+=threads) {
      volatile float square=input[row*width+col]*input[row*width+col];
      volatile float add=sums[lane]+square;sums[lane]=add;
    }
    for(int stride=threads/2;stride;stride/=2)for(int lane=0;lane<stride;++lane) {
      volatile float add=sums[lane]+sums[lane+stride];sums[lane]=add;
    }
    volatile float mean=sums[0]*static_cast<float>(1./static_cast<double>(width));
    volatile float variance=mean+eps;volatile float inv=1.f/std::sqrt(variance);
    for(int64_t col=0;col<width;++col) {
      volatile float normalized=input[row*width+col]*inv;
      volatile float scale=weight[col]+1.f;volatile float value=normalized*scale;
      out[row*width+col]=value;
    }
  }
  return result.to(at::kBFloat16);
}

int main(int argc,char** argv) {
  c10::Device device(argc>1?argv[1]:"cuda:0");at::manual_seed(717);
  for(int width:{1,7,32,33,256,384,2048,4096,5120,8192})for(bool strided:{false,true}) {
    auto base=(at::randn({2,3,width*(strided?2:1)})*3).to(at::kBFloat16);
    if(strided)base.narrow(-1,width,width).fill_(NAN);
    auto x=base.narrow(-1,0,width),gamma=(at::randn({width})*.2).to(at::kBFloat16);
    auto cuda_base=base.to(device),cuda_x=cuda_base.narrow(-1,0,width);
    auto actual=tei::qwen35_reference_norm_cuda(cuda_x,gamma.to(device),1e-6).cpu();
    TORCH_CHECK(at::equal(actual,independent_norm(x,gamma,1e-6)),"Independent original binary-tree norm oracle failed width=",width," strided=",strided);
  }
  auto rejects=[&](const at::Tensor& x,const at::Tensor& gamma) {
    bool rejected=false;
    try { (void)tei::qwen35_reference_norm_cuda(x,gamma,1e-6); }
    catch(const c10::Error&) { rejected=true; }
    TORCH_CHECK(rejected,"Invalid norm operands must reject before launch");
  };
  auto opts=at::TensorOptions().device(device).dtype(at::kBFloat16);
  rejects(at::empty({2,0},opts),at::empty({0},opts));
  rejects(at::ones({2,7},opts),at::ones({7},opts.device(at::kCPU)));
  rejects(at::ones({2,7},opts),at::ones({1,7},opts));
  // Deterministic synthetic operands exercise packed recurrence isolation.
  auto q=at::randn({32,384},opts)*.2,z=at::randn({32,128},opts)*.2;
  auto ab=at::randn({32,2},opts)*.2,conv=at::randn({384,4},opts)*.2;
  auto alog=at::zeros({1},opts.dtype(at::kFloat));
  auto dt=at::zeros({1},opts.dtype(at::kFloat)),weight=at::ones({128},opts);
  int32_t offsets[3]={0,32,64};tei::PackedInput input;input.offsets=offsets;input.batch=1;input.max_sequence=32;
  input.cumulative=at::from_blob(offsets,{2},at::TensorOptions().dtype(at::kInt)).clone().to(device);
  auto expected=tei::qwen35_reference_delta_cuda(q,z,ab,conv,alog,dt,weight,input,1,1,1e-6);
  TORCH_CHECK(at::isfinite(expected).all().item<bool>(),"Synthetic recurrence must be finite");
  input.batch=2;input.cumulative=at::from_blob(offsets,{3},at::TensorOptions().dtype(at::kInt)).clone().to(device);
  auto twice=tei::qwen35_reference_delta_cuda(at::cat({q,q},0),at::cat({z,z},0),at::cat({ab,ab},0),conv,alog,dt,weight,input,1,1,1e-6);
  TORCH_CHECK(at::equal(twice,at::cat({expected,expected},0)),"Packed sequence recurrence state leaked");
  auto poisoned=tei::qwen35_reference_delta_cuda(at::cat({q,at::full_like(q,NAN)},0),at::cat({z,z},0),at::cat({ab,ab},0),conv,alog,dt,weight,input,1,1,1e-6);
  TORCH_CHECK(at::equal(poisoned.narrow(0,0,32),expected),"Independent sequence contaminated by later NaNs");
  for(bool fma:{false,true})for(auto geometry:{std::pair<int64_t,int64_t>{1,1},{2,4},{4,8},{16,32},{16,48},{32,64},{32,96},{16,128}})
    { resumed_delta(device,geometry.first,geometry.second,fma); independent_recurrent(device,geometry.first,geometry.second,fma); }
  std::cout<<"Independent original norms and synthetic packed sequence isolation passed\n";
  std::cout<<"DeltaNet cache capture and split/resume preserve both arithmetic modes exactly\n";
}

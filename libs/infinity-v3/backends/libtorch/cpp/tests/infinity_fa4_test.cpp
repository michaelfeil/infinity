// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#include "kernels/infinity_fa4.h"
#include "models/encoder_models.h"
#include "helpers/admission_error.h"
#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/core/InferenceMode.h>
#include <cmath>
#include <iostream>
#include <barrier>
#include <future>

namespace {
void close(const at::Tensor& actual,const at::Tensor& expected,const char* label) {
  TORCH_CHECK(actual.defined()&&actual.sizes()==expected.sizes(),label,": shape mismatch");
  auto a=actual.to(at::kFloat),b=expected.to(at::kFloat);
  auto error=at::sqrt((a-b).square().sum()/b.square().sum().clamp_min(1e-12)).item<float>();
  TORCH_CHECK(at::isfinite(a).all().item<bool>()&&error<.015,label,": relative L2=",error);
}
at::Tensor reference(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
    const std::vector<int64_t>& qends,const std::vector<int64_t>& kends,
    double scale,bool causal,int64_t window=-1) {
  std::vector<at::Tensor> result;
  for(size_t row=0;row+1<qends.size();++row) {
    auto nq=qends[row+1]-qends[row],nk=kends[row+1]-kends[row];
    auto qs=q.narrow(0,qends[row],nq).to(at::kFloat).transpose(0,1);
    auto ks=k.narrow(0,kends[row],nk).to(at::kFloat).repeat_interleave(q.size(1)/k.size(1),1).transpose(0,1);
    auto vs=v.narrow(0,kends[row],nk).to(at::kFloat).repeat_interleave(q.size(1)/k.size(1),1).transpose(0,1);
    auto scores=at::matmul(qs,ks.transpose(1,2))*scale;
    auto qi=at::arange(nq,q.options().dtype(at::kLong)).unsqueeze(1);
    auto ki=at::arange(nk,q.options().dtype(at::kLong)).unsqueeze(0);
    auto delta=qi+nk-nq-ki;
    if(causal)scores.masked_fill_(delta<0,-INFINITY);
    if(window>=0)scores.masked_fill_(delta.abs()>window,-INFINITY);
    auto probs=at::nan_to_num(at::softmax(scores,-1),0.);
    result.push_back(at::matmul(probs,vs).transpose(0,1));
  }
  return at::cat(result,0);
}
at::Tensor offsets(const std::vector<int64_t>& ends,const at::TensorOptions& options) {
  return at::tensor(ends,at::TensorOptions().dtype(at::kLong)).to(options.dtype(at::kInt));
}
void concurrent_first_use() {
  constexpr int workers=4;
  std::barrier start(workers);
  std::vector<std::future<void>> futures;
  const int devices=std::min<int>(2,c10::cuda::device_count());
  for(int worker=0;worker<workers;++worker)futures.push_back(std::async(std::launch::async,[&,worker] {
    c10::InferenceMode inference;
    c10::cuda::CUDAGuard device(worker%devices);
    auto stream=c10::cuda::getStreamFromPool();
    c10::cuda::CUDAStreamGuard guard(stream);
    auto opts=at::TensorOptions().device(at::Device(at::kCUDA,worker%devices)).dtype(at::kBFloat16);
    auto q=at::randn({17,6,64},opts),k=at::randn_like(q),v=at::randn_like(q);
    auto cu=offsets({0,7,17},opts);
    stream.synchronize();
    start.arrive_and_wait();
    for(int repeat=0;repeat<3;++repeat)
      close(tei::infinity_fa4(q,k,v,cu,cu,10,.125),
        reference(q,k,v,{0,7,17},{0,7,17},.125,false),"concurrent/device launch");
    stream.synchronize();
  }));
  for(auto& future:futures)future.get();
}
void standard(at::ScalarType dtype,int mode,bool cross) {
  const int64_t d=mode<2?64:128,h=mode<2?6:mode==2?12:6,hk=mode<2?h:3;
  std::vector<int64_t> qe={0,7,136,649},ke=cross?std::vector<int64_t>{0,11,140,700}:qe;
  auto opts=at::TensorOptions().device(at::kCUDA).dtype(dtype);
  // Interleaved projections and a nonzero aligned storage offset exercise
  // real model views rather than only contiguous export-example tensors.
  auto qstore=at::randn({qe.back()+1,3*h,d},opts)*.2;
  auto q=qstore.narrow(0,1,qe.back()).narrow(1,0,h);
  auto k=at::randn({ke.back(),hk,d},opts)*.2,v=at::randn_like(k);
  auto cq=offsets(qe,opts),ck=offsets(ke,opts);
  auto window=mode==1?37:-1;
  auto forward=[&]{return tei::infinity_fa4(q,k,v,cq,ck,513,.7/std::sqrt(double(d)),
    mode==2,window,window);};
  auto expected=[&]{return reference(q,k,v,qe,ke,.7/std::sqrt(double(d)),mode==2,window);};
  auto output=forward();close(output,expected(),"packed attention");
  auto stream=c10::cuda::getCurrentCUDAStream();
  stream.synchronize();
  at::cuda::CUDAGraph graph;
  graph.capture_begin();auto captured=forward();graph.capture_end();
  q.mul_(.8);v.add_(.125);
  graph.replay();close(captured,expected(),"graph replay with changed tensors");
  // Same storage/shape, different per-sequence boundaries on graph replay.
  qe={0,9,140,649};ke=cross?std::vector<int64_t>{0,13,144,700}:qe;
  cq.copy_(offsets(qe,opts));ck.copy_(offsets(ke,opts));
  graph.replay();close(captured,expected(),"graph replay with changed boundaries");
  auto first=captured.narrow(0,0,qe[1]).clone();
  q.narrow(0,qe[1],qe.back()-qe[1]).fill_(3);
  v.narrow(0,ke[1],ke.back()-ke[1]).fill_(4);
  graph.replay();
  TORCH_CHECK(at::equal(first,captured.narrow(0,0,qe[1])),"InfinityFA4 mixed independent sequences");
}
void deberta(at::ScalarType dtype) {
  auto opts=at::TensorOptions().device(at::kCUDA).dtype(dtype);
  constexpr int64_t h=3,n=649,span=16,length=513;
  std::vector<int64_t> ends={0,7,136,n};
  auto q=at::randn({n,h,64},opts)*.2,k=at::randn_like(q)*.2,v=at::randn_like(q);
  auto a=at::randn({h,n,2*span},opts)*.05,b=at::randn_like(a)*.05;
  auto cu=offsets(ends,opts);
  auto buckets=(at::arange(1-length,length,opts.dtype(at::kInt))+span).clamp(0,2*span-1);
  auto forward=[&]{return tei::infinity_fa4_deberta(q,k,v,cu,a,b,buckets,length);};
  auto expected=[&] {
    std::vector<at::Tensor> rows;
    for(size_t row=0;row+1<ends.size();++row) {
      auto start=ends[row],count=ends[row+1]-start;
      auto qs=q.narrow(0,start,count).to(at::kFloat).transpose(0,1);
      auto ks=k.narrow(0,start,count).to(at::kFloat).transpose(0,1);
      auto vs=v.narrow(0,start,count).to(at::kFloat).transpose(0,1);
      auto ix=at::arange(count,opts.dtype(at::kLong));
      auto idx=(ix.unsqueeze(1)-ix.unsqueeze(0)+span).clamp(0,2*span-1);
      auto bias=a.narrow(1,start,count).gather(2,idx.unsqueeze(0).expand({h,count,count}))
        +b.narrow(1,start,count).gather(2,idx.t().unsqueeze(0).expand({h,count,count})).transpose(1,2);
      auto scores=(at::matmul(qs,ks.transpose(1,2)).to(dtype)+bias).to(at::kFloat);
      rows.push_back(at::matmul(at::softmax(scores,-1),vs).transpose(0,1));
    }
    return at::cat(rows,0);
  };
  close(forward(),expected(),"DeBERTa relative attention");
  at::cuda::CUDAGraph graph;
  graph.capture_begin();auto captured=forward();graph.capture_end();
  a.mul_(1.1);q.mul_(.7);graph.replay();
  close(captured,expected(),"DeBERTa graph replay");
}
void deberta_model(at::ScalarType dtype,bool shared,const std::string& terms) {
  tei::Options options;
  options.values={{"model_type","deberta-v2"},{"hidden_size","128"},{"num_attention_heads","2"},
    {"num_hidden_layers","1"},{"intermediate_size","256"},{"max_position_embeddings","1024"},
    {"position_buckets","8"},{"relative_attention","true"},{"position_biased_input","false"},
    {"share_att_key",shared?"true":"false"},{"pos_att_type",terms},{"norm_rel_ebd","layer_norm"},
    {"layer_norm_eps","0.00001"},{"_attention_bias_max_bytes","2097152"}};
  tei::Weights weights,cpu_weights;
  auto opts=at::TensorOptions().device(at::kCUDA).dtype(dtype);
  auto add=[&](const std::string& name,std::vector<int64_t> shape,bool ones=false) {
    weights[name]=(ones?at::ones(shape,opts):at::randn(shape,opts)*.02);
  };
  auto norm=[&](const std::string& name) {add(name+".weight",{128},true);add(name+".bias",{128});};
  auto linear=[&](const std::string& name,int64_t out,int64_t in) {
    add(name+".weight",{out,in});add(name+".bias",{out});
  };
  add("embeddings.word_embeddings.weight",{32,128});norm("embeddings.LayerNorm");
  add("encoder.rel_embeddings.weight",{16,128});norm("encoder.LayerNorm");
  const std::string prefix="encoder.layer.0.";
  for(auto name:{"query_proj","key_proj","value_proj","pos_key_proj","pos_query_proj"})
    linear(prefix+"attention.self."+name,128,128);
  linear(prefix+"attention.output.dense",128,128);norm(prefix+"attention.output.LayerNorm");
  linear(prefix+"intermediate.dense",256,128);linear(prefix+"output.dense",128,256);
  norm(prefix+"output.LayerNorm");
  auto model=tei::create_encoder(options,weights,at::Device(at::kCUDA),dtype);model->ready();
  for(const auto& [name,value]:weights)cpu_weights[name]=value.to(at::kCPU).to(at::kFloat);
  auto reference_options=options;
  reference_options.values["_attention_bias_max_bytes"]="67108864";
  auto cpu=tei::create_encoder(reference_options,cpu_weights,at::Device(at::kCPU),at::kFloat);cpu->ready();
  const std::vector<int32_t> ends={0,7,136,649};
  auto ids=at::arange(649,opts.dtype(at::kLong))%32;
  auto positions=at::cat({at::arange(7,ids.options()),at::arange(129,ids.options()),at::arange(513,ids.options())});
  auto types=at::zeros_like(ids),cu=offsets({0,7,136,649},opts);
  tei::PackedInput input{ids,types,positions,cu,ends.data(),3,513};
  auto ci=ids.cpu(),ct=types.cpu(),cp=positions.cpu(),cc=cu.cpu();
  tei::PackedInput cpu_input{ci,ct,cp,cc,ends.data(),3,513};
  // The old quadratic bucket table alone exceeds 32 MiB for length 513.
  // This complete model must run within a 2 MiB attention auxiliary budget.
  auto expected=cpu->forward(cpu_input).to(at::kCUDA);
  close(model->forward(input),expected,"DeBERTa model and logarithmic buckets");
  at::cuda::CUDAGraph graph;
  graph.capture_begin();auto captured=model->forward(input);graph.capture_end();
  ids.copy_((ids+3)%32);ci.copy_(ids.cpu());graph.replay();
  close(captured,cpu->forward(cpu_input).to(at::kCUDA),"DeBERTa model graph replay");
  options.values["_attention_bias_max_bytes"]="1024";
  auto limited=tei::create_encoder(options,weights,at::Device(at::kCUDA),dtype);limited->ready();
  bool rejected=false;
  try {limited->forward(input);}catch(const tei::MemoryAdmissionError&){rejected=true;}
  TORCH_CHECK(rejected,"DeBERTa FA4 ignored the auxiliary memory budget");
}
}
int main(int argc,char** argv) {
  c10::InferenceMode inference;
  at::set_num_threads(2);
  at::manual_seed(724);
  if(argc>1&&std::string(argv[1])=="--disabled") {
    auto q=at::zeros({7,2,64},at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
    auto cu=offsets({0,7},q.options());
    TORCH_CHECK(!tei::infinity_fa4(q,q,q,cu,cu,7,.125).defined()&&
      !tei::infinity_fa4_deberta_supported(q),"Disabled InfinityFA4 intercepted attention");
    return 0;
  }
  concurrent_first_use();
  // Every reference and launch uses a non-default stream.
  auto stream=c10::cuda::getStreamFromPool();
  c10::cuda::CUDAStreamGuard guard(stream);
  for(auto dtype:{at::kHalf,at::kBFloat16}) {
    for(int mode=0;mode<4;++mode)for(bool cross:{false,true})standard(dtype,mode,cross);
    deberta(dtype);
    for(bool shared:{false,true})
      for(const auto& terms:{"c2p","p2c","c2p|p2c"})
        deberta_model(dtype,shared,terms);
  }
  auto unsupported=at::zeros({7,2,80},at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
  auto cu=offsets({0,7},unsupported.options());
  TORCH_CHECK(!tei::infinity_fa4(unsupported,unsupported,unsupported,cu,cu,7,1).defined(),
    "Unsupported geometry did not retain the original attention path");
  std::cout<<"InfinityFA4 packed/cross attention, DeBERTa model/budget, concurrent devices and graph replay passed\n";
}

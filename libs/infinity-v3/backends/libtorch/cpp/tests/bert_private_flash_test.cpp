#include "bridge.h"
#include "models/bert_model.h"
#include "batching/cuda_graph.h"
#include <ATen/ATen.h>
#include <cmath>
#include <iostream>
#include <map>
#include <vector>
static void check(int32_t code) { TORCH_CHECK(code==0,tei_error()); }
static void classifier_workspace_recovery(const TeiConfig& cfg,const std::map<std::string,at::Tensor>& source,const c10::Device& device) {
  if(at::cuda::getDeviceProperties(device.index())->major!=9) return;
  struct RestoreBackend {
    at::BlasBackend prior=at::globalContext().blasPreferredBackend();
    ~RestoreBackend(){at::globalContext().setBlasPreferredBackend(prior);}
  } restore;
  at::globalContext().setBlasPreferredBackend(at::BlasBackend::Cublaslt);
  auto state=tei::workspace_admission_state(device.index());
  auto retained=[&]{std::lock_guard lock(state->mutex);return state->retained;};
  const auto baseline=retained();
  auto wide_cfg=cfg;wide_cfg.intermediate=3072;
  auto gpu_options=at::TensorOptions().device(device).dtype(at::kHalf);
  auto intermediate=at::randn({3072,cfg.hidden},gpu_options)*.03;
  auto intermediate_bias=at::zeros({3072},gpu_options);
  auto down=at::randn({cfg.hidden,3072},gpu_options)*.03;
  auto build=[&]{
    const auto before_ready=retained();
    auto model=std::make_unique<Bert>(wide_cfg,device,at::kHalf);
    model->options.values["model_type"]="bert";
    for(const auto& [name,tensor]:source)model->weights[name]=tensor.to(device);
    model->weights["encoder.layer.0.intermediate.dense.weight"]=intermediate;
    model->weights["encoder.layer.0.intermediate.dense.bias"]=intermediate_bias;
    model->weights["encoder.layer.0.output.dense.weight"]=down;
    model->weights["classifier.weight"]=at::randn({16,cfg.hidden},model->weights.begin()->second.options())*.03;
    model->weights["classifier.bias"]=at::zeros({16},model->weights.begin()->second.options());
    model->ready();TORCH_CHECK(model->cached_classifier_gemm,"Eligible BERT classifier fixture did not select cached GEMM");TORCH_CHECK(retained()==before_ready,"BERT ready allocated retained classifier workspace");return model;
  };
  bool exercised=false;
  for(const bool gelu:{false,true}) {
  for(int64_t rows:{32,128,512}) {
    auto x=at::randn({rows,gelu?cfg.hidden:int64_t(3072)},at::TensorOptions().device(device).dtype(at::kHalf));
    auto reference_model=build();auto matrix=reference_model->w(gelu?"encoder.layer.0.intermediate.dense.weight":"encoder.layer.0.output.dense.weight");
    auto bias=reference_model->w(gelu?"encoder.layer.0.intermediate.dense.bias":"encoder.layer.0.output.dense.bias");
    auto reference=reference_model->classifier_projection(x,matrix,bias,gelu).cpu();
    for(const auto& [key,plan]:reference_model->classifier_plans)
      std::cout<<"BERT classifier natural plan rows="<<rows<<" input="<<x.size(1)<<" output="<<matrix.size(0)<<" valid="<<plan->valid<<" bytes="<<plan->bytes<<"\n";
    exercised=retained()>baseline;reference_model.reset();
    TORCH_CHECK(retained()==baseline,"BERT destruction retained classifier workspace credit");
    if(!exercised)continue;
    size_t available;{std::lock_guard lock(state->mutex);available=(state->limit?state->limit:size_t(512)<<20)-state->retained;}
    auto pressure=tei::reserve_workspace_memory(device.index(),available);TORCH_CHECK(pressure,"Unable to saturate BERT classifier workspace credit");
    auto model=build();auto fallback=model->classifier_projection(x,matrix,bias,gelu).cpu();
    TORCH_CHECK(at::isfinite(fallback).all().item<bool>()&&retained()==baseline+available,
      "BERT classifier fallback allocated an uncredited workspace");
    pressure.reset();auto recovered=model->classifier_projection(x,matrix,bias,gelu).cpu();
    TORCH_CHECK(retained()>baseline&&at::equal(reference,recovered)&&at::equal(recovered,model->classifier_projection(x,matrix,bias,gelu).cpu()),
      "BERT same-shape workspace retry changed full projection outputs");
    std::array<int32_t,2> offsets{0,32};auto ids=at::arange(32,at::TensorOptions().device(device).dtype(at::kLong));
    auto types=at::zeros_like(ids),positions=ids;auto cumulative=at::tensor({0,32},at::TensorOptions().device(device).dtype(at::kInt));
    tei::PackedInput input{ids,types,positions,cumulative,offsets.data(),1,32};
    auto expected=model->forward(input).cpu();
    {tei::CudaGraphCache graphs(1,4096,1);
      for(int repeat=0;repeat<2;++repeat)TORCH_CHECK(at::equal(expected,graphs.forward(*model,input).cpu()),"Captured BERT classifier workspace changed full outputs");
      TORCH_CHECK(graphs.size()==1,"BERT classifier workspace control did not capture");}
    model.reset();TORCH_CHECK(retained()==baseline,"BERT graph/model teardown retained classifier workspace credit");
    std::cout<<"BERT classifier workspace credit/retry/full-output/capture recovery rows="<<rows<<" passed\n";break;
  }
  if(exercised)break;
  }
  TORCH_CHECK(exercised,"BERT classifier workspace regression did not exercise a supported workspace plan");
}
int main(int argc,char** argv) {
  at::manual_seed(321);const char* device=argc>1?argv[1]:"cuda:0";
  TeiConfig cfg{768,12,1,64,64,256,2,1e-5,0};
  std::map<std::string,at::Tensor> weights;
  auto add=[&](std::string name,std::vector<int64_t> shape,bool unit=false,bool zero=false) {
    weights[name]=(unit?at::ones(shape):zero?at::zeros(shape):at::randn(shape)*.03).to(at::kHalf);
  };
  add("embeddings.word_embeddings.weight",{64,768});add("embeddings.position_embeddings.weight",{256,768});add("embeddings.token_type_embeddings.weight",{2,768});
  auto ln=[&](const std::string& p){add(p+".weight",{768},true);add(p+".bias",{768},false,true);};
  auto fc=[&](const std::string& p,int64_t out,int64_t in){add(p+".weight",{out,in});add(p+".bias",{out},false,true);};
  const std::string layer="encoder.layer.0.";
  ln("embeddings.LayerNorm");for(auto part:{"query","key","value"})fc(layer+"attention.self."+part,768,768);
  fc(layer+"attention.output.dense",768,768);ln(layer+"attention.output.LayerNorm");
  fc(layer+"intermediate.dense",64,768);fc(layer+"output.dense",768,64);ln(layer+"output.LayerNorm");
  void* engine=tei_create(&cfg,device,1);TORCH_CHECK(engine,tei_error());
  check(tei_option(engine,"model_type","bert"));check(tei_option(engine,"_pooling","embedding"));
  check(tei_option(engine,"_cuda_graphs","true"));check(tei_option(engine,"_cudnn_varlen","true"));
  check(tei_option(engine,"_cuda_graph_reuse","1"));
  for(const auto& [name,t]:weights)check(tei_weight(engine,name.c_str(),reinterpret_cast<const uint8_t*>(t.const_data_ptr()),t.sizes().data(),t.dim(),1));
  check(tei_ready(engine));
  std::vector<int64_t> ids,types,positions;std::vector<int32_t> cu{0,5,134,143};
  for(int row=0;row<3;++row)for(int i=0;i<cu[row+1]-cu[row];++i){ids.push_back((7*i+11*row)%64);types.push_back(row%2);positions.push_back(i);}
  const int64_t pooled[]={2,0,2},raw[]={1,0};const int64_t outrows=3+129+5;
  auto run=[&]() {
    auto out=at::empty({outrows,768},at::kFloat);
    check(tei_forward(engine,ids.data(),types.data(),positions.data(),cu.data(),3,129,0,pooled,3,raw,2,out.mutable_data_ptr<float>(),out.numel(),nullptr,nullptr,0,nullptr,0));return out;
  };
  // Independent FP32 CPU reference uses explicit attention probabilities at each
  // exact sequence length; no native model/helper or packed attention is reused.
  auto reference=[&]() {
    auto vector=[&](const std::vector<int64_t>& v){return at::from_blob(const_cast<int64_t*>(v.data()),{int64_t(v.size())},at::kLong).clone();};
    auto w=[&](const std::string& n){return weights.at(n).to(at::kFloat);};
    auto norm=[&](const at::Tensor& x,const std::string& n){return at::layer_norm(x,{768},w(n+".weight"),w(n+".bias"),1e-5);};
    auto linear=[&](const at::Tensor& x,const std::string& n){return at::matmul(x,w(n+".weight").t())+w(n+".bias");};
    auto h=norm(at::embedding(w("embeddings.word_embeddings.weight"),vector(ids))+at::embedding(w("embeddings.token_type_embeddings.weight"),vector(types))+at::embedding(w("embeddings.position_embeddings.weight"),vector(positions)),"embeddings.LayerNorm");
    auto q=linear(h,layer+"attention.self.query"),k=linear(h,layer+"attention.self.key"),v=linear(h,layer+"attention.self.value");
    std::vector<at::Tensor> parts;
    for(int row=0;row<3;++row){int length=cu[row+1]-cu[row];auto shape=[&](const at::Tensor& x){return x.narrow(0,cu[row],length).view({length,12,64}).transpose(0,1);};
      auto probabilities=at::softmax(at::matmul(shape(q),shape(k).transpose(-1,-2))/8.,-1);
      parts.push_back(at::matmul(probabilities,shape(v)).transpose(0,1).contiguous().view({length,768}));}
    h=norm(linear(at::cat(parts,0),layer+"attention.output.dense")+h,layer+"attention.output.LayerNorm");
    h=norm(linear(at::gelu(linear(h,layer+"intermediate.dense"),"tanh"),layer+"output.dense")+h,layer+"output.LayerNorm");
    return at::cat({h.narrow(0,134,1),h.narrow(0,0,1),h.narrow(0,134,1),h.narrow(0,5,129),h.narrow(0,0,5)},0);
  };
  auto agreement=[&](const at::Tensor& actual,const at::Tensor& expected){
    TORCH_CHECK(at::isfinite(actual).all().item<bool>());
    auto cosine=(actual*expected).sum(1)/(actual.norm(2,1)*expected.norm(2,1));
    TORCH_CHECK(cosine.min().item<float>()>=.999,"Independent exact-length BERT reference mismatch");
  };
  auto first=run();agreement(first,reference());TORCH_CHECK(tei_graph_count(engine)==1,"BERT private attention was not captured");
  auto original=ids;for(int i=134;i<143;++i)ids[i]=(ids[i]+3)%64;
  auto changed=run();agreement(changed,reference());TORCH_CHECK(tei_graph_count(engine)==1,"Same packed shape did not replay graph");
  TORCH_CHECK(at::equal(first.narrow(0,1,1),changed.narrow(0,1,1))&&at::equal(first.narrow(0,3,134),changed.narrow(0,3,134)),"Changed sequence contaminated another packed span");
  TORCH_CHECK(!at::equal(first.narrow(0,0,1),changed.narrow(0,0,1)),"Graph replay ignored changed input");
  TORCH_CHECK(at::equal(changed.narrow(0,0,1),changed.narrow(0,2,1)),"Repeated pooled request changed output");
  ids=original;TORCH_CHECK(at::equal(first,run()),"Restored inputs changed graph output");
  tei_destroy(engine);classifier_workspace_recovery(cfg,weights,c10::Device(device));std::cout<<"BERT head64 mixed pooled/raw independent reference, packed isolation and changed-input graph replay passed\n";
}

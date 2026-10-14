#include "bridge.h"
#include <ATen/ATen.h>
#include <cmath>
#include <iostream>
#include <map>
#include <vector>
static void check(int32_t code) { TORCH_CHECK(code==0,tei_error()); }
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
  tei_destroy(engine);std::cout<<"BERT head64 mixed pooled/raw independent reference, packed isolation and changed-input graph replay passed\n";
}

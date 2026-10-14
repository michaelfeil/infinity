#include "models/qwen35_models.h"
#include "models/decision_models.h"
#include "batching/radix.h"
#include "helpers/workspace_admission.h"
#include "batching/cuda_graph.h"
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/cuda/CUDAContext.h>
#endif
#include <iostream>
using namespace tei;
void radix_workspace_recovery_test(const c10::Device& device) {
#ifdef TEI_TORCH_CUDA_KERNELS
  const auto* properties=at::cuda::getDeviceProperties(device.index());
  if(properties->major!=9 || properties->minor!=0) {
    std::cout<<"Qwen3.5 shared workspace fixture skipped: production path requires SM 9.0\n";
    return;
  }
  struct RestoreBlasBackend {
    at::BlasBackend prior=at::globalContext().blasPreferredBackend();
    ~RestoreBlasBackend(){at::globalContext().setBlasPreferredBackend(prior);}
  } restore;
  at::globalContext().setBlasPreferredBackend(at::BlasBackend::Cublaslt);
#else
  return;
#endif
  Options cfg;cfg.values={{"model_type","qwen3_5_moe"},{"hidden_size","2048"},
    {"num_attention_heads","32"},{"num_key_value_heads","2"},{"head_dim","128"},
    {"num_hidden_layers","1"},{"max_position_embeddings","1024"},
    {"rope_parameters.partial_rotary_factor",".5"},{"layer_types.0","full_attention"},
    {"num_experts","128"},{"num_experts_per_tok","8"}};
  auto options=at::TensorOptions().device(device).dtype(at::kBFloat16);Weights checkpoint;
  auto put=[&](const std::string& name,std::initializer_list<int64_t> dimensions,bool norm=false) {
    checkpoint["model."+name]=norm?at::ones(dimensions,options):
      (at::randn(dimensions,options.dtype(at::kFloat))*.003).to(at::kBFloat16);
  };
  put("embed_tokens.weight",{64,2048});put("norm.weight",{2048},true);
  const std::string layer="layers.0.";
  put(layer+"input_layernorm.weight",{2048},true);put(layer+"post_attention_layernorm.weight",{2048},true);
  put(layer+"self_attn.q_proj.weight",{8192,2048});put(layer+"self_attn.k_proj.weight",{256,2048});
  put(layer+"self_attn.v_proj.weight",{256,2048});put(layer+"self_attn.o_proj.weight",{2048,4096});
  put(layer+"self_attn.q_norm.weight",{128},true);put(layer+"self_attn.k_norm.weight",{128},true);
  put(layer+"mlp.gate.weight",{128,2048});put(layer+"mlp.experts.gate_up_proj",{128,128,2048});
  put(layer+"mlp.experts.down_proj",{128,2048,64});
  put(layer+"mlp.shared_expert.gate_proj.weight",{12,2048});put(layer+"mlp.shared_expert.up_proj.weight",{12,2048});
  put(layer+"mlp.shared_expert.down_proj.weight",{2048,12});put(layer+"mlp.shared_expert_gate.weight",{1,2048});
  auto build=[&]{auto weights=checkpoint;auto model=create_qwen35(cfg,weights,device,at::kBFloat16);model->ready();return model;};
  auto state=workspace_admission_state(device.index());
  auto retained=[&]{std::lock_guard lock(state->mutex);return state->retained;};
  const auto baseline=retained();bool exercised=false;
  for(int32_t length:{32,128,512}) {
    std::array<int32_t,2> offsets{0,length};auto ids=at::arange(length,at::kLong).remainder(64);
    auto positions=at::arange(length,at::kLong);auto types=at::zeros_like(ids);
    auto cumulative=at::tensor({0,length},at::kInt);
    PackedInput input{ids,types,positions,cumulative,offsets.data(),1,length};
    input.compact_ids=ids;input.compact_positions=positions;input.compact_types=types;
    input.scatter_unfold=positions;input.fold_gather=positions;validate_radix(input,true);
    for(auto* tensor:{&input.ids,&input.types,&input.positions,&input.cumulative,&input.compact_ids,
        &input.compact_positions,&input.compact_types,&input.scatter_unfold,&input.fold_gather})
      *tensor=tensor->to(device);
    at::Tensor reference;
    {auto model=build();reference=model->forward(input).cpu();exercised=retained()>baseline;}
    TORCH_CHECK(retained()==baseline,"Qwen3.5 model destruction retained workspace credit");
    if(!exercised)continue;
    size_t available;{std::lock_guard lock(state->mutex);available=(state->limit?state->limit:size_t(512)<<20)-state->retained;}
    auto pressure=reserve_workspace_memory(device.index(),available);TORCH_CHECK(pressure,"Unable to hold Qwen3.5 workspace credit");
    auto model=build();auto fallback=model->forward(input).cpu();
    TORCH_CHECK(at::isfinite(fallback).all().item<bool>()&&retained()==baseline+available,
      "Qwen3.5 fallback allocated an uncredited custom workspace");
    pressure.reset();auto recovered=model->forward(input).cpu();
    TORCH_CHECK(retained()>baseline,"Released credit did not retry the cached Qwen3.5 supported plan");
    TORCH_CHECK(at::equal(reference,recovered)&&at::equal(recovered,model->forward(input).cpu()),
      "Qwen3.5 workspace recovery changed full supported-plan output coordinates");
    {
      CudaGraphCache graphs(1,4096,1);
      for(int repeat=0;repeat<2;++repeat)TORCH_CHECK(at::equal(graphs.forward(*model,input).cpu(),reference),
        "Qwen3.5 captured workspace generation changed full outputs");
      TORCH_CHECK(graphs.size()==1,"Qwen3.5 workspace recovery control did not capture");
    }
    model.reset();TORCH_CHECK(retained()==baseline,"Qwen3.5 graph/model teardown retained workspace credit");
    std::cout<<"Qwen3.5 shared workspace fallback/release/full recovery tokens="<<length<<" passed\n";
    break;
  }
  TORCH_CHECK(exercised,"Qwen3.5 workspace regression did not exercise a workspace-requiring supported plan");
}
int main(int argc,char** argv) {
  at::manual_seed(1451);
  Options cfg;
  cfg.values={{"model_type","qwen3_5"},{"hidden_size","16"},{"num_attention_heads","2"},
    {"num_key_value_heads","1"},{"head_dim","8"},{"num_hidden_layers","2"},
    {"max_position_embeddings","32"},{"rope_parameters.partial_rotary_factor",".5"},
    {"linear_num_key_heads","1"},{"linear_num_value_heads","1"},
    {"layer_types.0","linear_attention"},{"layer_types.1","full_attention"}};
  Weights weights;
  auto put=[&](const std::string& name,std::initializer_list<int64_t> shape,bool zero=false) {
    weights[name]=zero?at::zeros(shape):at::randn(shape)*.05;
  };
  put("model.embed_tokens.weight",{32,16});put("model.norm.weight",{16},true);
  for(int i=0;i<2;++i) {
    auto p="model.layers."+std::to_string(i)+".";
    put(p+"input_layernorm.weight",{16},true);put(p+"post_attention_layernorm.weight",{16},true);
    put(p+"mlp.gate_proj.weight",{24,16});put(p+"mlp.up_proj.weight",{24,16});put(p+"mlp.down_proj.weight",{16,24});
    if(i==0) {
      p+="linear_attn.";
      put(p+"in_proj_qkv.weight",{384,16});put(p+"in_proj_z.weight",{128,16});
      put(p+"in_proj_a.weight",{1,16});put(p+"in_proj_b.weight",{1,16});
      put(p+"conv1d.weight",{384,1,4});put(p+"A_log",{1});put(p+"dt_bias",{1});
      weights[p+"norm.weight"]=at::ones({128});put(p+"out_proj.weight",{16,128});
    } else {
      p+="self_attn.";
      put(p+"q_proj.weight",{32,16});put(p+"k_proj.weight",{8,16});put(p+"v_proj.weight",{8,16});put(p+"o_proj.weight",{16,16});
      put(p+"q_norm.weight",{8},true);put(p+"k_norm.weight",{8},true);
    }
  }
  auto build=[](const Options& options,const Weights& source,c10::Device device,at::ScalarType dtype) {
    auto loader=source;return create_qwen35(options,loader,device,dtype);
  };
  auto model=build(cfg,weights,c10::Device("cpu"),at::kFloat);model->ready();
  TORCH_CHECK(model->transient_memory_profile()->recurrent_heads==1,
    "Mixed Qwen3.5 attention must retain its actual recurrent state estimate");
  auto full_cfg=cfg;full_cfg.values["layer_types.0"]="full_attention";
  auto full_weights=weights;
  for(const auto& [name,value]:weights) {
    const std::string from="model.layers.1.self_attn.";
    if(name.rfind(from,0)==0)
      full_weights["model.layers.0.self_attn."+name.substr(from.size())]=value.clone();
  }
  auto full_model=build(full_cfg,full_weights,c10::Device("cpu"),at::kFloat);
  TORCH_CHECK(full_model->transient_memory_profile()->recurrent_heads==0,
    "Full-attention Qwen3.5 must not charge unused DeltaNet metadata before ready");
  full_model->ready();
  TORCH_CHECK(full_model->transient_memory_profile()->recurrent_heads==0,
    "Full-attention Qwen3.5 must not charge unused DeltaNet metadata after ready");
  std::vector<int32_t> offsets={0,3,5},one={0,3},two={0,2};
  auto make=[](const std::vector<int64_t>& ids,const std::vector<int32_t>& ends) {
    auto tokens=at::from_blob(const_cast<int64_t*>(ids.data()),{static_cast<int64_t>(ids.size())},at::TensorOptions().dtype(at::kLong)).clone();
    std::vector<int64_t> positions;
    int64_t max=0;
    for(size_t i=1;i<ends.size();++i) { auto n=ends[i]-ends[i-1];max=std::max<int64_t>(max,n);for(int j=0;j<n;++j)positions.push_back(j); }
    auto pos=at::from_blob(positions.data(),{static_cast<int64_t>(positions.size())},at::TensorOptions().dtype(at::kLong)).clone();
    auto cu=at::from_blob(const_cast<int32_t*>(ends.data()),{static_cast<int64_t>(ends.size())},at::TensorOptions().dtype(at::kInt)).clone();
    return PackedInput{tokens,at::zeros_like(tokens),pos,cu,ends.data(),static_cast<int64_t>(ends.size()-1),max};
  };
  TORCH_CHECK(at::isfinite(full_model->forward(make({1,2,3,4,5},offsets))).all().item<bool>(),
    "Full-attention Qwen3.5 profile must preserve valid packed inference");
  auto output=model->forward(make({1,2,3,4,5},offsets));
  auto independent=at::cat({model->forward(make({1,2,3},one)),model->forward(make({4,5},two))},0);
  Weights owned_loader;
  for(const auto& [name,value]:weights)owned_loader[name]=value.clone();
  c10::weak_intrusive_ptr<c10::TensorImpl> original_gate(owned_loader.at("model.layers.0.mlp.gate_proj.weight").getIntrusivePtr());
  auto tied_cfg=cfg;tied_cfg.values["tie_word_embeddings"]="true";
  auto tied=create_decision_head("option_tokens",{},tied_cfg,{},owned_loader);
  auto owned=create_qwen35(cfg,owned_loader,c10::Device("cpu"),at::kFloat);
  TORCH_CHECK(owned_loader.empty() && !original_gate.expired(),"Qwen3.5 must own loaded tensors before initialization");
  owned->ready();
  TORCH_CHECK(original_gate.expired(),"Consumed Qwen3.5 FFN projection still retained after initialization");
  auto owned_hidden=owned->forward(make({1,2,3,4,5},offsets));
  TORCH_CHECK(at::equal(output,owned_hidden),"Qwen3.5 ownership/cache pruning changed full outputs");
  DecisionRequest options{DecisionRequest::Kind::OptionTokens};options.token_ids={1,2};
  auto baseline_head=create_decision_head("option_tokens",{},tied_cfg,{},weights);
  auto before=baseline_head->forward(output,make({1,2,3,4,5},offsets),{options,options});
  auto after=tied->forward(owned_hidden,make({1,2,3,4,5},offsets),{options,options});
  TORCH_CHECK(at::equal(before[0].logits,after[0].logits) && at::equal(before[1].logits,after[1].logits),"Tied option head lost owned embeddings");
  std::cout<<"Qwen3.5 loader ownership, consumed FFN lifetime and tied decision outputs passed\n";

  TORCH_CHECK(at::allclose(output,independent,1e-5,1e-6),"DeltaNet packed recurrence leaked sequence state");
  auto changed=model->forward(make({31,2,3,4,5},offsets));
  TORCH_CHECK(at::allclose(output.narrow(0,3,2),changed.narrow(0,3,2),1e-5,1e-6),"DeltaNet state not reset at sequence boundary");
  std::cout<<"Qwen3.5 packed/single parity and recurrence reset passed\n";
  auto shared=make({1,2,3,1,2},offsets),folded=shared;
  folded.compact_ids=at::tensor({1,2,3},at::kLong);
  folded.compact_positions=at::tensor({0,1,2},at::kLong);
  folded.compact_types=at::zeros({3},at::kLong);
  folded.scatter_unfold=at::tensor({0,1,2,0,1},at::kLong);
  folded.fold_gather=at::tensor({0,1,2},at::kLong);
  validate_radix(folded,true);
  TORCH_CHECK(model->supports_radix(),"Causal Qwen3.5 folding capability missing");
  auto shared_output=model->forward(shared);
  TORCH_CHECK(at::allclose(shared_output,model->forward(folded),1e-5,1e-6),"Qwen3.5 folded Delta/full attention lost logical rows");
  auto reordered=folded;
  reordered.compact_ids=folded.compact_ids.flip({0});reordered.compact_positions=folded.compact_positions.flip({0});
  reordered.fold_gather=folded.fold_gather.flip({0});reordered.scatter_unfold=2-folded.scatter_unfold;
  validate_radix(reordered,true);
  TORCH_CHECK(at::allclose(shared_output,model->forward(reordered),1e-5,1e-6),"Qwen3.5 reordered compact rows changed readout");

  auto bidirectional_cfg=cfg;bidirectional_cfg.values["decision_attention_mode"]="noncausal_full_attention";
  auto bidirectional=build(bidirectional_cfg,weights,c10::Device("cpu"),at::kFloat);bidirectional->ready();
  TORCH_CHECK(!bidirectional->supports_radix(),"Noncausal Qwen3.5 must reject prefix folding");
  bool rejected=false;try{bidirectional->forward(folded);}catch(const c10::Error&){rejected=true;}
  TORCH_CHECK(rejected,"Noncausal Qwen3.5 accepted folded prefixes");
  auto bi=bidirectional->forward(make({1,2,3,4,5},offsets));
  auto future=bidirectional->forward(make({1,2,31,4,5},offsets));
  TORCH_CHECK((bi.select(0,0)-future.select(0,0)).abs().max().item<float>()>1e-5,"Qwen3.5 noncausal full attention ignored future tokens");
  TORCH_CHECK(at::allclose(bi.narrow(0,3,2),future.narrow(0,3,2),1e-5,1e-6),"Noncausal Qwen3.5 crossed packed sequence boundaries");
  std::cout<<"Qwen3.5 noncausal-full attention decision mode and sequence isolation passed\n";
  auto moe_cfg=cfg;moe_cfg.values["num_experts"]="3";moe_cfg.values["num_experts_per_tok"]="2";
  auto moe_weights=weights;
  for(int i=0;i<2;++i) {
    auto p="model.layers."+std::to_string(i)+".mlp.";
    moe_weights[p+"gate.weight"]=at::zeros({3,16});
    auto gu=at::cat({weights.at(p+"gate_proj.weight"),weights.at(p+"up_proj.weight")},0);
    moe_weights[p+"experts.gate_up_proj"]=at::stack({gu,gu*1.5,gu*2.});
    auto down=weights.at(p+"down_proj.weight");moe_weights[p+"experts.down_proj"]=at::stack({down,down,down});
    for(auto name:{"gate_proj","up_proj","down_proj"})moe_weights[p+"shared_expert."+name+".weight"]=weights.at(p+name+".weight");
    moe_weights[p+"shared_expert_gate.weight"]=at::zeros({1,16});
  }
  auto moe=build(moe_cfg,moe_weights,c10::Device("cpu"),at::kFloat);moe->ready();
  auto routed=moe->forward(make({1,2,3,4,5},offsets));
  auto routed_single=at::cat({moe->forward(make({1,2,3},one)),moe->forward(make({4,5},two))},0);
  TORCH_CHECK(at::allclose(routed,routed_single,1e-5,1e-6),"Qwen3.5 routed experts leak packed sequences");
  for(int i=0;i<2;++i)moe_weights["model.layers."+std::to_string(i)+".mlp.experts.gate_up_proj"].select(0,2).fill_(100.);
  TORCH_CHECK(at::allclose(routed,moe->forward(make({1,2,3,4,5},offsets)),1e-5,1e-6),"Qwen3.5 router ties must prefer lower expert IDs");
  std::cout<<"Qwen3.5 MoE/shared expert, stable routing ties and packed isolation passed\n";
  if(argc>1) {
    c10::Device device(argv[1]);
    for(auto& [key,value]:weights)value=value.to(at::kBFloat16);
    auto cpu=build(cfg,weights,c10::Device("cpu"),at::kBFloat16);cpu->ready();
    Weights cuda_weights;
    for(const auto& [key,value]:weights)cuda_weights[key]=value.to(device);
    c10::weak_intrusive_ptr<c10::TensorImpl> cuda_gate(cuda_weights.at("model.layers.0.mlp.gate_proj.weight").getIntrusivePtr());
    auto cuda=create_qwen35(cfg,cuda_weights,device,at::kBFloat16);cuda->ready();
    TORCH_CHECK(cuda_weights.empty() && cuda_gate.expired(),"CUDA Qwen3.5 retained consumed loader/FFN ownership");
    auto host=make({1,2,3,4,5},offsets),dev=host;
    dev.ids=host.ids.to(device);dev.types=host.types.to(device);dev.positions=host.positions.to(device);dev.cumulative=host.cumulative.to(device);
    auto reference=cpu->forward(host).to(at::kFloat),actual=cuda->forward(dev).cpu().to(at::kFloat);
    TORCH_CHECK(at::allclose(reference,actual,.025,.025),"Qwen3.5 native CUDA DeltaNet disagrees with CPU recurrence, maxabs=",(reference-actual).abs().max().item<float>());
    std::cout<<"Qwen3.5 CUDA DeltaNet/CPU reference parity maxabs="<<(reference-actual).abs().max().item<float>()<<"\n";
    const auto prior_backend=at::globalContext().blasPreferredBackend();
    radix_workspace_recovery_test(device);
    TORCH_CHECK(at::globalContext().blasPreferredBackend()==prior_backend,
      "Qwen3.5 workspace fixture did not restore the prior BLAS backend");
  }
}

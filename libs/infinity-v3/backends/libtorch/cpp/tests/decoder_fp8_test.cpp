// Dynamic FP8 decoder integration checks: actual rows, causal isolation and
// weight ownership. This fixture measures functional synthetic-model drift;
// it does not qualify trained task accuracy or serving performance.
#include "models/decoder_models.h"
#include "batching/radix.h"
#include "kernels/fast_kernels.h"
#include "kernels/decoder_norm.h"
#include "kernels/torch_flash_fma.h"
#include "helpers/fp8_linear.h"
#include "tests/fp8_reference_test.h"
#include <cmath>
#include <cstdlib>
#include <ATen/cuda/CUDAContext.h>
#include <c10/core/InferenceMode.h>
#include <iostream>

namespace {
tei::Options config(const std::string& family) {
  tei::Options options;
  options.values={{"model_type",family},{"hidden_size","32"},
    {"num_attention_heads","4"},{"num_key_value_heads","2"},{"head_dim","8"},
    {"intermediate_size","64"},{"num_hidden_layers","2"},{"vocab_size","32"},
    {"max_position_embeddings","32"},{"hidden_act","silu"},{"rms_norm_eps","0.000001"}};
  return options;
}
tei::Weights weights(const tei::Options& options,c10::Device device,at::ScalarType dtype) {
  tei::Weights tensors;
  auto add=[&](const std::string& name,std::initializer_list<int64_t> shape,bool norm=false) {
    int64_t count=1;float offset=0;
    for(auto n:shape)count*=n;
    for(auto c:name)offset+=c;
    auto value=at::sin(at::arange(count,at::TensorOptions().dtype(at::kFloat))*.131+offset*.019)*.065;
    if(norm)value=1+value;
    tensors[name]=value.view(at::IntArrayRef(shape)).to(device,dtype);
  };
  add("model.embed_tokens.weight",{32,32});add("model.norm.weight",{32},true);
  for(int i=0;i<2;++i) {
    const auto p="model.layers."+std::to_string(i)+".";
    add(p+"input_layernorm.weight",{32},true);add(p+"post_attention_layernorm.weight",{32},true);
    for(auto name:{"q_proj","k_proj","v_proj"}) {
      auto width=std::string(name)=="q_proj"?32:16;
      add(p+"self_attn."+name+".weight",{width,32});
      if(options.string("model_type")=="qwen2")add(p+"self_attn."+name+".bias",{width});
    }
    add(p+"self_attn.o_proj.weight",{32,32});
    if(options.string("model_type")=="qwen3") {
      add(p+"self_attn.q_norm.weight",{8},true);add(p+"self_attn.k_norm.weight",{8},true);
    }
    add(p+"mlp.gate_proj.weight",{64,32});add(p+"mlp.up_proj.weight",{64,32});
    add(p+"mlp.down_proj.weight",{32,64});
  }
  return tensors;
}
void close(const at::Tensor& a,const at::Tensor& b) {
  auto x=a.to(at::kFloat),y=b.to(at::kFloat);
  auto cosine=(x*y).sum(-1)/(x.norm(2,-1)*y.norm(2,-1)).clamp_min(1e-12);
  TORCH_CHECK(cosine.min().item<float>()>.999,"FP8 causal/folding isolation failed");
}
void check_family(const std::string& family,c10::Device device,at::ScalarType dtype) {
  auto options=config(family),enabled=options;enabled.values["_fp8_dynamic"]="true";
  auto quantized_weights=weights(options,device,dtype);
  auto original_down=quantized_weights.at("model.layers.0.mlp.down_proj.weight").storage().getWeakStorageImpl();
  auto original_attention=quantized_weights.at("model.layers.0.self_attn.o_proj.weight").storage().getWeakStorageImpl();
  auto model=tei::create_decoder(enabled,quantized_weights,device,dtype);model->ready();
  TORCH_CHECK(original_down.expired(),"FP8 cache retained original model-dtype down-projection storage");
  TORCH_CHECK(!original_attention.expired(),"FP8 cache released an unchanged attention projection");
  auto dense_weights=weights(options,device,dtype);
  auto dense_down=dense_weights.at("model.layers.0.mlp.down_proj.weight").storage().getWeakStorageImpl();
  auto dense=tei::create_decoder(options,dense_weights,device,dtype);dense->ready();
  TORCH_CHECK(!dense_down.expired(),"Disabled FP8 unexpectedly replaced dense projection weights");

  auto index=at::TensorOptions().device(device).dtype(at::kLong);
  std::vector<int32_t> offsets={0,3,7};
  tei::PackedInput input{at::tensor({3,4,5,3,4,6,7},index),at::zeros({7},index),
    at::tensor({0,1,2,0,1,2,3},index),at::tensor({0,3,7},index.dtype(at::kInt)),offsets.data(),2,4};
  auto full=model->forward(input),reference=dense->forward(input);
  TORCH_CHECK(full.sizes()==at::IntArrayRef({7,32})&&full.scalar_type()==dtype
    &&at::isfinite(full).all().item<bool>(),"FP8 decoder changed actual-row count, dtype or finiteness");
  auto error=(full.to(at::kFloat)-reference.to(at::kFloat)).square().mean();
  auto energy=reference.to(at::kFloat).square().mean().clamp_min(1e-12);
  TORCH_CHECK(at::sqrt(error/energy).item<float>()<.10,
    "Synthetic FP8 decoder drift exceeds the functional fixture bound");
  TORCH_CHECK(!at::equal(full,reference),"Requested FP8 was silently ignored");

  // Seven actual rows compact to five: neither dimension is tile-aligned.
  input.compact_ids=at::tensor({3,4,5,6,7},index);
  input.compact_positions=at::tensor({0,1,2,2,3},index);
  input.compact_types=at::zeros({5},index);
  input.scatter_unfold=at::tensor({0,1,2,0,1,3,4},index);
  input.fold_gather=at::tensor({0,1,2,5,6},index);
  auto folded=model->forward(input);close(folded,full);
  TORCH_CHECK(folded.size(0)==7&&tei::radix_ids(input).numel()==5,
    "FP8 decoder introduced padded compact rows");
  input.ids.index_put_({6},8);input.compact_ids.index_put_({4},8);
  auto changed=model->forward(input);
  TORCH_CHECK(at::equal(changed.narrow(0,0,3),folded.narrow(0,0,3)),
    "FP8 suffix change leaked into another sequence");
  TORCH_CHECK(!at::equal(changed.narrow(0,3,4),folded.narrow(0,3,4)),
    "FP8 fixture did not consume changed suffix IDs");
  std::cout<<family<<" FP8 actual7/compact5, isolation, weight ownership passed\n";
}
// This exercises the registered BF16 width2048 causal contract, rather than
// reproducing a particular checkpoint's layer count, vocabulary or MLP width.
// The FP8 projection oracle translates the original Candle cuBLASLt recipe;
// norm/rotary/attention primitives have independent exact-reference tests.
void check_registered_causal_contract(c10::Device device) {
  constexpr int64_t hidden=2048,heads=32,kvheads=4,head=64,intermediate=64;
  tei::Options options=config("llama");
  options.values["hidden_size"]="2048";options.values["num_attention_heads"]="32";
  options.values["num_key_value_heads"]="4";options.values["head_dim"]="64";
  options.values["intermediate_size"]="64";options.values["num_hidden_layers"]="1";
  options.values["rms_norm_eps"]="0.00001";
  tei::Weights checkpoint;
  auto add=[&](const std::string& name,std::initializer_list<int64_t> shape,bool norm=false) {
    int64_t count=1;float offset=0;for(auto n:shape)count*=n;for(auto c:name)offset+=c;
    auto values=at::sin(at::arange(count,at::TensorOptions().dtype(at::kFloat))*.131+offset*.019)*.021;
    if(norm)values=1+values;
    checkpoint[name]=values.view(at::IntArrayRef(shape)).to(device,at::kBFloat16);
  };
  add("model.embed_tokens.weight",{32,hidden});add("model.norm.weight",{hidden},true);
  const std::string p="model.layers.0.";
  add(p+"input_layernorm.weight",{hidden},true);add(p+"post_attention_layernorm.weight",{hidden},true);
  add(p+"self_attn.q_proj.weight",{heads*head,hidden});
  add(p+"self_attn.k_proj.weight",{kvheads*head,hidden});
  add(p+"self_attn.v_proj.weight",{kvheads*head,hidden});
  add(p+"self_attn.o_proj.weight",{hidden,heads*head});
  add(p+"mlp.gate_proj.weight",{intermediate,hidden});
  add(p+"mlp.up_proj.weight",{intermediate,hidden});
  add(p+"mlp.down_proj.weight",{hidden,intermediate});
  std::vector<int32_t> offsets={0,9,17};
  auto index=at::TensorOptions().device(device).dtype(at::kLong);
  tei::PackedInput input{at::tensor({1,2,3,4,5,6,7,8,9,1,2,3,10,11,12,13,14},index),
    at::zeros({17},index),at::tensor({0,1,2,3,4,5,6,7,8,0,1,2,3,4,5,6,7},index),
    at::tensor({0,9,17},index.dtype(at::kInt)),offsets.data(),2,9};
  auto embedding=at::embedding(checkpoint.at("model.embed_tokens.weight"),input.ids);
  auto first=tei::decoder_exact_rms_norm(embedding,{},checkpoint.at(p+"input_layernorm.weight"),1e-5);
  auto qkv_weight=at::cat({checkpoint.at(p+"self_attn.q_proj.weight"),
    checkpoint.at(p+"self_attn.k_proj.weight"),checkpoint.at(p+"self_attn.v_proj.weight")},0);
  auto packed=at::linear(first.first,qkv_weight).view({17,heads+2*kvheads,head});
  std::vector<float> inverse(head/2);
  for(int64_t i=0;i<head/2;++i)inverse[i]=1.f/std::pow(10000.f,float(2*i)/head);
  auto frequency=at::from_blob(inverse.data(),{head/2},at::TensorOptions().dtype(at::kFloat)).clone().to(device);
  auto angles=input.positions.to(at::kFloat).unsqueeze(1)*frequency;
  auto rotated=tei::fused_rotary(packed.narrow(1,0,heads),packed.narrow(1,heads,kvheads),
    at::cos(angles).to(at::kBFloat16).unsqueeze(1),at::sin(angles).to(at::kBFloat16).unsqueeze(1),true);
  auto attended=tei::torch_flash_fma(rotated.first,rotated.second,packed.narrow(1,heads+kvheads,kvheads),
    input.cumulative,9,.125,true).contiguous().view({17,heads*head});
  auto projected=at::linear(attended,checkpoint.at(p+"self_attn.o_proj.weight"));
  auto post=tei::decoder_exact_rms_norm(projected,first.second,checkpoint.at(p+"post_attention_layernorm.weight"),1e-5);
  auto gate_weight=at::cat({checkpoint.at(p+"mlp.gate_proj.weight"),checkpoint.at(p+"mlp.up_proj.weight")},0);
  for(bool fp8:{false,true}) {
    auto enabled=options;enabled.values["_fp8_dynamic"]=fp8?"true":"false";
    auto owned=checkpoint;auto model=tei::create_decoder(enabled,owned,device,at::kBFloat16);model->ready();
    auto original=[&](const at::Tensor& x,const at::Tensor& weight) {
      if(!fp8)return at::linear(x,weight);
      auto activations=tei::fp8_quantize_rows(x),weights=tei::fp8_quantize_rows(weight);
      return fp8_original_recipe(activations.first,weights.first,activations.second,weights.second,at::kBFloat16);
    };
    auto gated=tei::fused_gated_activation(original(post.first,gate_weight),5);
    auto down=original(gated,checkpoint.at(p+"mlp.down_proj.weight"));
    auto expected=tei::decoder_exact_rms_norm(down,post.second,checkpoint.at("model.norm.weight"),1e-5).first;
    auto actual=model->forward(input);
    TORCH_CHECK(actual.sizes()==at::IntArrayRef({17,hidden})&&at::equal(actual,expected),
      "Registered causal BF16 decoder contract differs from original norm/attention/FP8 recipe, fp8=",fp8);
  }
  std::cout<<"Registered causal BF16 width2048 reference: dense and FP8 bitwise,17 actual rows passed\n";
}

void rejects(tei::Options options,c10::Device device,at::ScalarType dtype) {
  options.values["_fp8_dynamic"]="true";tei::Weights empty;
  auto model=tei::create_decoder(options,empty,device,dtype);
  bool rejected=false;try{model->ready();}catch(const c10::Error&){rejected=true;}
  TORCH_CHECK(rejected,"Unsupported requested decoder FP8 was silently accepted");
}
}
int main(int argc,char** argv) {
  // Match the independent original recipe without relying on the test runner.
  setenv("CUBLASLT_WORKSPACE_SIZE","32768",1);
  at::globalContext().setBlasPreferredBackend(at::BlasBackend::Cublaslt);
  at::globalContext().setAllowTF32CuBLAS(false);
  c10::InferenceMode inference;
  auto device=c10::Device(argc>1?argv[1]:"cuda:0");
  rejects(config("llama"),c10::Device("cpu"),at::kFloat);
  rejects(config("qwen3_moe"),device,at::kBFloat16);
  auto quantized=config("qwen3");quantized.values["quantization_config.quant_method"]="fp8";
  rejects(quantized,device,at::kBFloat16);
  auto properties=at::cuda::getDeviceProperties(device.index());
  auto capability=properties->major*10+properties->minor;
  if(capability!=89&&capability!=90&&capability!=100&&capability!=120) {
    std::cout<<"FP8 CUDA fixture skipped unsupported capability "<<capability<<"\n";return 0;
  }
  check_registered_causal_contract(device);
  for(auto dtype:{at::kHalf,at::kBFloat16})
    for(auto family:{"llama","mistral","qwen2","qwen3"})check_family(family,device,dtype);
}

// Native packed decoder regression tests; no Python runtime or padded tokens.
#include "models/decoder_models.h"
#include "batching/radix.h"
#include "batching/input_validation.h"
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <ATen/Parallel.h>
using namespace tei;
namespace {
void input_descriptor_test() {
 int64_t data[64]{};int32_t offsets[]={0,32,64};float payload=0;uint8_t mask=1;
 TeiImage image{};image.pixels=&payload;image.rows=32;image.patch_dim=1536;
 image.grid[0]=2;image.grid[1]=image.grid[2]=4;image.merge_size=2;image.token_start=4;image.token_count=8;
 TeiAudio audio{&payload,&mask,4,128,2,1};
 auto validate=[&](const TeiImage* images,size_t ni,const TeiAudio* audios,size_t na,int64_t maximum=32) {
  return validate_input_descriptors(data,data,data,offsets,2,maximum,nullptr,images,ni,audios,na);
 };
 TORCH_CHECK(validate(&image,1,&audio,1)==64,"Valid temporal image/audio descriptors rejected");
 // max_sequence is an upper bound, not a padded input or required exact length.
 TORCH_CHECK(validate(nullptr,0,nullptr,0,64)==64,"Valid maximum sequence upper bound rejected");
 auto rejects=[&](auto call,const char* message) {
  bool failed=false;try{call();}catch(const c10::Error&){failed=true;}
  TORCH_CHECK(failed,message);
 };
 rejects([&]{validate(nullptr,1,nullptr,0);},"Null counted image array accepted");
 rejects([&]{validate(nullptr,0,nullptr,1);},"Null counted audio array accepted");
 rejects([&]{validate(nullptr,0,nullptr,0,31);},"Undersized max_sequence accepted");
 rejects([&]{validate(nullptr,0,nullptr,0,65);},"Maximum sequence beyond packed token count accepted");
 int64_t selection[]={1,0,1};
 TORCH_CHECK(validate_output_indices(offsets,2,selection,3,selection,3)==96,
   "Repeated output selection rejected");
 TORCH_CHECK(expected_output_elements(3,96,16,16)==1584,"Output element count changed");
 rejects([&]{validate_output_indices(offsets,2,nullptr,1,nullptr,0);},"Null pooled indices accepted");
 rejects([&]{validate_output_indices(offsets,2,nullptr,0,nullptr,1);},"Null raw indices accepted");
 int64_t invalid=-1;
 rejects([&]{validate_output_indices(offsets,2,&invalid,1,nullptr,0);},"Negative output index accepted");
 invalid=2;
 rejects([&]{validate_output_indices(offsets,2,nullptr,0,&invalid,1);},"Out-of-bounds output index accepted");
 rejects([&]{expected_output_elements(std::numeric_limits<size_t>::max(),0,16,16);},
   "Overflowing output element count accepted");
 auto bad=image;bad.rows=std::numeric_limits<int64_t>::max();bad.grid[0]=bad.rows;
 rejects([&]{validate(&bad,1,nullptr,0);},"Overflowing image grid accepted");
 bad=image;bad.patch_dim=std::numeric_limits<int64_t>::max();
 rejects([&]{validate(&bad,1,nullptr,0);},"Overflowing image payload accepted");
 bad=image;bad.token_start=30;
 rejects([&]{validate(&bad,1,nullptr,0);},"Cross-sequence image span accepted");
 bad=image;bad.pixels=nullptr;
 rejects([&]{validate(&bad,1,nullptr,0);},"Null image pixels accepted");
 auto bad_audio=audio;bad_audio.frames=-1;
 rejects([&]{validate(nullptr,0,&bad_audio,1);},"Negative audio frames accepted");
 bad_audio=audio;bad_audio.mask=nullptr;
 rejects([&]{validate(nullptr,0,&bad_audio,1);},"Null audio mask accepted");
 TeiImage gemma=image;gemma.rows=36;gemma.patch_dim=768;gemma.grid[0]=1;gemma.grid[1]=gemma.grid[2]=6;
 gemma.merge_size=3;gemma.token_start=40;gemma.token_count=4;gemma.sequence_start=32;
 TORCH_CHECK(validate_input_descriptors(data,data,data,offsets,2,32,nullptr,&gemma,1,nullptr,0,true)==64,
   "Valid Gemma pooled image rejected");
 auto wrong_sequence=gemma;wrong_sequence.sequence_start=0;
 rejects([&]{validate_input_descriptors(data,data,data,offsets,2,32,nullptr,&wrong_sequence,1,nullptr,0,true);},
   "Gemma image admitted keys from another packed sequence");
 wrong_sequence.sequence_start=33;
 rejects([&]{validate_input_descriptors(data,data,data,offsets,2,32,nullptr,&wrong_sequence,1,nullptr,0,true);},
   "Gemma image accepted a non-boundary sequence start");
 auto second_row=image;second_row.token_start=40;second_row.sequence_start=32;
 TORCH_CHECK(validate(&second_row,1,nullptr,0)==64,"Valid second-row temporal image rejected");
 second_row.sequence_start=0;
 rejects([&]{validate(&second_row,1,nullptr,0);},"Temporal image accepted another sequence start");
 rejects([&]{validate_input_descriptors(data,data,data,offsets,2,32,nullptr,&image,1,nullptr,0,true);},
   "Multiple-frame Gemma image accepted");
 // Exercise the actual native bridge before model creation/any tensor upload.
 TeiConfig cfg{16,4,1,24,32,64,2,1e-5,0};auto handle=tei_create(&cfg,"cpu",0);TORCH_CHECK(handle,tei_error());
 auto code=tei_forward(handle,data,data,data,offsets,2,32,0,nullptr,0,nullptr,0,nullptr,0,nullptr,nullptr,1,nullptr,0);
 TORCH_CHECK(code!=0&&std::string(tei_error()).find("counted image")!=std::string::npos,
   "Native bridge did not reject malformed media before upload/model access");
 auto pending=tei_forward_submit(handle,data,data,data,offsets,2,32,0,nullptr,0,nullptr,0,
   std::numeric_limits<size_t>::max(),nullptr,nullptr,1,nullptr,0,nullptr);
 TORCH_CHECK(!pending&&std::string(tei_error()).find("counted image")!=std::string::npos,
   "Deferred bridge did not validate descriptors before output allocation");
 code=tei_forward(handle,data,data,data,offsets,2,32,0,nullptr,1,nullptr,0,nullptr,0,nullptr,nullptr,0,nullptr,0);
 TORCH_CHECK(code!=0&&std::string(tei_error()).find("counted pooled")!=std::string::npos,
   "Native bridge did not reject missing output indices before model access");
 pending=tei_forward_submit(handle,data,data,data,offsets,2,32,0,nullptr,0,&invalid,1,
   std::numeric_limits<size_t>::max(),nullptr,nullptr,0,nullptr,0,nullptr);
 TORCH_CHECK(!pending&&std::string(tei_error()).find("raw output index")!=std::string::npos,
   "Deferred bridge did not reject output indices before output allocation");
 tei_destroy(handle);
}
Options config(const std::string& family, bool moe = false) {
  Options cfg;
  cfg.values = {{"model_type",family},{"hidden_size","16"},{"num_attention_heads","4"},
    {"num_key_value_heads","2"},{"head_dim","4"},{"intermediate_size","24"},
    {"num_hidden_layers","2"},{"vocab_size","32"},{"max_position_embeddings","32"},
    {"hidden_act","silu"},{"rms_norm_eps","0.000001"}};
  if (moe) { cfg.values["num_experts"]="3";cfg.values["num_experts_per_tok"]="2";cfg.values["moe_intermediate_size"]="12"; }
  return cfg;
}
Weights weights(const Options& cfg, bool fused = false) {
  Weights w;
  auto add = [&](const std::string& name, std::initializer_list<int64_t> shape, bool norm = false) {
    int64_t count=1; for(auto n:shape)count*=n;
    float offset=0;for(auto c:name)offset+=c;
    auto data = at::sin(at::arange(count,at::TensorOptions().dtype(at::kFloat))*.13 + offset*.01)*.15;
    if(norm)data+=1;
    w[name]=data.view(at::IntArrayRef(shape));
  };
  add("model.embed_tokens.weight",{32,16});add("model.norm.weight",{16},true);
  for(int layer=0;layer<2;++layer){
    auto p="model.layers."+std::to_string(layer)+".";
    add(p+"input_layernorm.weight",{16},true);add(p+"post_attention_layernorm.weight",{16},true);
    add(p+"self_attn.q_proj.weight",{16,16});add(p+"self_attn.k_proj.weight",{8,16});add(p+"self_attn.v_proj.weight",{8,16});add(p+"self_attn.o_proj.weight",{16,16});
    if(cfg.string("model_type")=="qwen2")for(auto name:{"q_proj","k_proj","v_proj"})add(p+"self_attn."+name+".bias",{std::string(name)=="q_proj"?16:8});
    if(cfg.string("model_type")=="qwen3" || cfg.string("model_type")=="qwen3_moe") {add(p+"self_attn.q_norm.weight",{4},true);add(p+"self_attn.k_norm.weight",{4},true);}
    if(!cfg.integer("num_experts")){add(p+"mlp.gate_proj.weight",{24,16});add(p+"mlp.up_proj.weight",{24,16});add(p+"mlp.down_proj.weight",{16,24});}
    else {
      add(p+"mlp.gate.weight",{3,16});std::vector<at::Tensor> gu,down;
      for(int e=0;e<3;++e){auto ep=p+"mlp.experts."+std::to_string(e)+".";add(ep+"gate_proj.weight",{12,16});add(ep+"up_proj.weight",{12,16});add(ep+"down_proj.weight",{16,12});gu.push_back(at::cat({w.at(ep+"gate_proj.weight"),w.at(ep+"up_proj.weight")},0));down.push_back(w.at(ep+"down_proj.weight"));}
      if(fused){w[p+"mlp.experts.gate_up_proj"]=at::stack(gu);w[p+"mlp.experts.down_proj"]=at::stack(down);}
    }
  }
  return w;
}
void output_capacity_bridge_test() {
 auto cfg=config("qwen3");auto w=weights(cfg);
 TeiConfig native_cfg{16,4,2,24,32,32,1,1e-6,0};
 auto handle=tei_create(&native_cfg,"cpu",0);TORCH_CHECK(handle,tei_error());
 for(const auto& [key,value]:cfg.values)TORCH_CHECK(tei_option(handle,key.c_str(),value.c_str())==0,tei_error());
 for(const auto& [key,value]:w)
  TORCH_CHECK(tei_weight(handle,key.c_str(),reinterpret_cast<const uint8_t*>(value.const_data_ptr()),
    value.sizes().data(),value.dim(),0)==0,tei_error());
 TORCH_CHECK(tei_ready(handle)==0,tei_error());
 int64_t ids[]={3,4,5,6},types[4]{},positions[]={0,1,0,1},pooled=0;int32_t cumulative[]={0,2,4};
 for(const auto capacity:{size_t(17),std::numeric_limits<size_t>::max()}) {
  auto pending=tei_forward_submit(handle,ids,types,positions,cumulative,2,2,0,&pooled,1,nullptr,0,
    capacity,nullptr,nullptr,0,nullptr,0,nullptr);
  TORCH_CHECK(!pending&&std::string(tei_error()).find("Deferred output capacity must exactly")!=std::string::npos,
    "Deferred spare capacity reached CUDA/output allocation");
 }
 auto pending=tei_forward_submit(handle,ids,types,positions,cumulative,2,2,0,&pooled,1,nullptr,0,
   16,nullptr,nullptr,0,nullptr,0,nullptr);
 TORCH_CHECK(!pending&&tei_last_error_kind()==1&&std::string(tei_error()).find("Deferred output requires CUDA")!=std::string::npos,
   "Exact deferred capacity failed metadata validation");
 float output[20];std::fill_n(output,20,std::numeric_limits<float>::quiet_NaN());
 TORCH_CHECK(tei_forward(handle,ids,types,positions,cumulative,2,2,0,&pooled,1,nullptr,0,
   output,20,nullptr,nullptr,0,nullptr,0)==0,tei_error());
 for(int i=0;i<16;++i)TORCH_CHECK(std::isfinite(output[i]),"Valid response failed after capacity rejection");
 for(int i=16;i<20;++i)TORCH_CHECK(std::isnan(output[i]),"Synchronous spare capacity was overwritten");
 // Oversized finite payload descriptors must be rejected without reading them.
 const auto fake_float=reinterpret_cast<const float*>(uintptr_t(1));
 const auto fake_mask=reinterpret_cast<const uint8_t*>(uintptr_t(1));
 TeiImage huge{};huge.pixels=fake_float;huge.rows=1;huge.patch_dim=(512LL<<20)/sizeof(float)+1;
 huge.grid[0]=huge.grid[1]=huge.grid[2]=huge.merge_size=huge.token_count=1;
 auto budget_error=[&](int code) {
  TORCH_CHECK(code!=0&&tei_last_error_kind()==2&&std::string(tei_error()).find("Native input upload exceeds")!=std::string::npos,
    "Input budget did not reject before payload read/allocation");
 };
 budget_error(tei_forward(handle,ids,types,positions,cumulative,2,2,0,&pooled,1,nullptr,0,
   output,20,nullptr,&huge,1,nullptr,0));
 pending=tei_forward_submit(handle,ids,types,positions,cumulative,2,2,0,&pooled,1,nullptr,0,
   16,nullptr,&huge,1,nullptr,0,nullptr);budget_error(pending?0:-1);
 budget_error(tei_decide(handle,ids,types,positions,cumulative,2,2,nullptr,nullptr,0,nullptr,
   nullptr,&huge,1,nullptr,0));
 TeiAudio huge_audio{fake_float,fake_mask,1,(512LL<<20)/sizeof(float)+1,0,1};
 budget_error(tei_forward(handle,ids,types,positions,cumulative,2,2,0,&pooled,1,nullptr,0,
   output,20,nullptr,nullptr,0,&huge_audio,1));
 TORCH_CHECK(tei_option(handle,"_input_upload_max_bytes","108")==0,tei_error());
 auto fake_indices=reinterpret_cast<const int64_t*>(uintptr_t(1));
 TeiRadix fake_radix{fake_indices,fake_indices,fake_indices,fake_indices,fake_indices,4};
 budget_error(tei_forward_radix(handle,ids,types,positions,cumulative,2,2,0,&pooled,1,nullptr,0,
   output,20,nullptr,nullptr,0,nullptr,0,&fake_radix));
 budget_error(tei_forward(handle,ids,types,positions,cumulative,2,2,0,&pooled,1,nullptr,0,
   output,20,fake_indices,nullptr,0,nullptr,0));
 std::fill_n(output,20,std::numeric_limits<float>::quiet_NaN());
 TORCH_CHECK(tei_forward(handle,ids,types,positions,cumulative,2,2,0,&pooled,1,nullptr,0,
   output,20,nullptr,nullptr,0,nullptr,0)==0,tei_error());
 TORCH_CHECK(tei_last_error_kind()==0&&std::string(tei_error()).empty(),"Successful inference retained an admission error kind");
 for(int i=0;i<16;++i)TORCH_CHECK(std::isfinite(output[i]),"Valid response failed after input budget rejection");
 TeiDecision huge_metadata{};huge_metadata.markers=fake_indices;huge_metadata.marker_count=std::numeric_limits<size_t>::max();
 bool metadata_rejected=false;
 try{validate_decision_metadata_budget(&huge_metadata,1,512<<20);}catch(const MemoryAdmissionError&){metadata_rejected=true;}
 TORCH_CHECK(metadata_rejected,"Decision metadata overflow was copied before admission");
 TORCH_CHECK(tei_option(handle,"_input_upload_max_bytes","0")==0,tei_error());
 TORCH_CHECK(tei_forward(handle,ids,types,positions,cumulative,2,2,0,&pooled,1,nullptr,0,
   output,20,nullptr,nullptr,0,nullptr,0)!=0,"Zero input budget accepted");
 tei_destroy(handle);
}
at::Tensor run(Model& model, std::vector<int64_t> ids, std::vector<int32_t> offsets) {
  std::vector<int64_t> positions;
  int64_t maximum=0;
  for(size_t row=0;row+1<offsets.size();++row){auto len=offsets[row+1]-offsets[row];maximum=std::max(maximum,int64_t(len));for(int n=0;n<len;++n)positions.push_back(n);}
  auto opts=at::TensorOptions().dtype(at::kLong);
  PackedInput input{at::from_blob(ids.data(),{int64_t(ids.size())},opts),at::zeros({int64_t(ids.size())},opts),at::from_blob(positions.data(),{int64_t(positions.size())},opts),at::from_blob(offsets.data(),{int64_t(offsets.size())},opts.dtype(at::kInt)),offsets.data(),int64_t(offsets.size()-1),maximum};
  return model.forward(input);
}
void check_close(const at::Tensor& a,const at::Tensor& b){auto error=(a-b).abs().max().item<float>();TORCH_CHECK(error<2e-5,"Decoder parity error ",error);}
void radix_decoder_test(Model& model) {
 auto options=at::TensorOptions().dtype(at::kLong);
 std::vector<int32_t> offsets={0,3,7,10};
 PackedInput input{at::tensor({3,4,5,3,4,6,7,3,4,5},options),at::zeros({10},options),
   at::tensor({0,1,2,0,1,2,3,0,1,2},options),at::tensor({0,3,7,10},options.dtype(at::kInt)),offsets.data(),3,4};
 auto expected=model.forward(input);
 input.compact_ids=at::tensor({3,4,5,6,7},options);input.compact_positions=at::tensor({0,1,2,2,3},options);
 input.compact_types=at::zeros({5},options);input.scatter_unfold=at::tensor({0,1,2,0,1,3,4,0,1,2},options);
 input.fold_gather=at::tensor({0,1,2,5,6},options);
 validate_radix(input,true);
 if(!model.supports_radix()) {
   bool rejected=false;try{model.forward(input);}catch(const c10::Error&){rejected=true;}
   TORCH_CHECK(rejected,"Unsupported bidirectional folding was silently ignored");return;
 }
 check_close(model.forward(input),expected);
 TORCH_CHECK(radix_ids(input).numel()==5&&input.ids.numel()==10,"Radix retained duplicate token rows");
}
void multimodal_test(){
 auto cfg=config("qwen3");auto w=weights(cfg);auto model=create_decoder(cfg,w,c10::Device("cpu"),at::kFloat);model->ready();
 auto opts=at::TensorOptions().dtype(at::kLong);
 std::vector<int64_t> ids={3,4,5,6,7},pos={0,1,2,0,1};std::vector<int32_t> offsets={0,3,5};
 PackedInput input{at::from_blob(ids.data(),{5},opts),at::zeros({5},opts),at::from_blob(pos.data(),{5},opts),at::from_blob(offsets.data(),{3},opts.dtype(at::kInt)),offsets.data(),2,3};
 auto frequencies=at::tensor({1.f,.01f});auto angles=input.positions.to(at::kFloat).unsqueeze(1)*frequencies;
 auto cosine=at::cos(angles),sine=at::sin(angles);
 check_close(model->forward(input),decoder_multimodal_forward(*model,input,{}, {},cosine,sine));
 auto indices=at::tensor({int64_t(1)},opts),visual=at::ones({1,16})*.25;
 auto injected=decoder_multimodal_forward(*model,input,indices,visual,cosine,sine);
 auto zero=at::zeros_like(visual);
 check_close(injected,decoder_multimodal_forward(*model,input,indices,visual,cosine,sine,{zero,zero}));
 auto deep=decoder_multimodal_forward(*model,input,indices,visual,cosine,sine,{at::arange(16).to(at::kFloat).view({1,16})*.07});
 TORCH_CHECK((deep-injected).abs().max().item<float>()>1e-4,"Deepstack feature injection did not affect text states");
 check_close(deep.narrow(0,3,2),model->forward(input).narrow(0,3,2));
 check_close(deep.narrow(0,0,1),injected.narrow(0,0,1));
 std::cout<<"Qwen3 external embeddings, multimodal RoPE and deepstack packed isolation passed\n";
}
}
int main(){c10::InferenceMode inference;at::set_num_threads(1);
 {
  c10::AcceleratorError allocation({__func__,__FILE__,__LINE__},2,"Resource allocation failed");
  c10::AcceleratorError illegal({__func__,__FILE__,__LINE__},700,"Illegal device address");
  c10::AcceleratorError unknown({__func__,__FILE__,__LINE__},999,"Unknown device failure");
  TORCH_CHECK(accelerator_error_kind(allocation)==3,"CUDA allocation failure must remain recoverable");
  TORCH_CHECK(accelerator_error_kind(illegal)==1&&accelerator_error_kind(unknown)==1,
    "Fatal accelerator failures must not be classified as overload");
  for(int32_t kind:{1,2,3}) {
    try { throw NativeCallError(kind); }
    catch(const NativeCallError& e) {
      TORCH_CHECK(e.kind==kind,
        "Nested native failure lost its structured classification");
    }
  }
 }
 output_capacity_bridge_test();
 input_descriptor_test();
 {
  // Preselected, non-normalized FP32 routes preserve learned expert scales
  // and their accumulation dtype, including a different activation callback.
  auto opts=at::TensorOptions().dtype(at::kBFloat16);
  auto x=at::arange(16,at::TensorOptions().dtype(at::kFloat)).view({2,8}).to(opts);
  auto gate_up=at::zeros({3,16,8},opts),down=at::zeros({3,8,8},opts);
  for(int64_t e=0;e<3;++e){gate_up[e].narrow(0,0,8).copy_(at::eye(8,opts));gate_up[e].narrow(0,8,8).fill_((e+1)/8.);down[e].copy_(at::eye(8,opts));}
  auto ids=at::tensor({2,0,1,2},at::TensorOptions().dtype(at::kLong)).view({2,2});
  auto routing=at::tensor({.123f,1.501f,.379f,.941f},at::TensorOptions().dtype(at::kFloat)).view({2,2});
  auto activation=[](const at::Tensor& both){return at::relu(both.narrow(-1,0,8))*both.narrow(-1,8,8);};
  auto actual=decoder_moe_dispatch(x,gate_up,down,ids,routing,activation);
  auto expected=at::zeros({2,8},at::TensorOptions().dtype(at::kFloat));
  for(int64_t row=0;row<2;++row)for(int64_t slot=0;slot<2;++slot){auto e=ids[row][slot].item<int64_t>();auto y=at::linear(activation(at::linear(x[row].unsqueeze(0),gate_up[e])),down[e]);expected[row].add_(y.squeeze(0).to(at::kFloat)*routing[row][slot]);}
  TORCH_CHECK(actual.scalar_type()==at::kFloat,"Preselected MoE lost FP32 accumulation");check_close(actual,expected);
 }

 for(auto family:{"llama","mistral","qwen2","qwen3","qwen3_moe"}){
  auto cfg=config(family,std::string(family)=="qwen3_moe");auto w=weights(cfg);auto initial=w;auto model=create_decoder(cfg,initial,c10::Device("cpu"),at::kFloat);model->ready();
  radix_decoder_test(*model);
  auto packed=run(*model,{3,4,5,6,7,8,9},{0,3,7});check_close(packed,at::cat({run(*model,{3,4,5},{0,3}),run(*model,{6,7,8,9},{0,4})},0));
  check_close(run(*model,{3,4},{0,2}),run(*model,{3,4,5},{0,3}).narrow(0,0,2));
  if(std::string(family)=="qwen2"||std::string(family)=="qwen3"){
   auto scaledcfg=cfg;scaledcfg.values["rope_scaling.rope_type"]="llama3";scaledcfg.values["rope_scaling.factor"]="4";scaledcfg.values["rope_scaling.high_freq_factor"]="4";
   auto scaledweights=weights(scaledcfg);auto scaled=create_decoder(scaledcfg,scaledweights,c10::Device("cpu"),at::kFloat);scaled->ready();
   check_close(packed,run(*scaled,{3,4,5,6,7,8,9},{0,3,7}));
  }
  if(std::string(family)=="qwen3_moe"){
   auto unsupportedcfg=cfg;unsupportedcfg.values["rope_scaling.factor"]="4";auto unsupportedweights=weights(unsupportedcfg);auto unsupported=create_decoder(unsupportedcfg,unsupportedweights,c10::Device("cpu"),at::kFloat);
   bool rejected=false;try{unsupported->ready();}catch(const c10::Error&){rejected=true;}TORCH_CHECK(rejected,"Scaled Qwen3 MoE configuration was silently accepted");
  }
  if(std::string(family)=="qwen3_moe"){auto fused=weights(cfg,true);auto other=create_decoder(cfg,fused,c10::Device("cpu"),at::kFloat);other->ready();check_close(packed,run(*other,{3,4,5,6,7,8,9},{0,3,7}));}
  cfg.values["use_bidirectional_attention"]="true";if(std::string(family)=="qwen2")cfg.values["is_causal"]="false";
  auto bidi=create_decoder(cfg,w,c10::Device("cpu"),at::kFloat);bidi->ready();auto b=run(*bidi,{3,4,5},{0,3});TORCH_CHECK((b.narrow(0,0,2)-run(*bidi,{3,4},{0,2})).abs().max().item<float>()>1e-5,"Bidirectional attention was accidentally causal");
  radix_decoder_test(*bidi);
  TORCH_CHECK(!bidi->supports_radix(),"Bidirectional decoder incorrectly advertises prefix folding");
  check_close(run(*bidi,{3,4,5,6,7},{0,3,5}),at::cat({b,run(*bidi,{6,7},{0,2})},0));
  std::cout<<family<<" packed boundaries, causality and bidirectional attention passed\n";
 }
 multimodal_test();
}

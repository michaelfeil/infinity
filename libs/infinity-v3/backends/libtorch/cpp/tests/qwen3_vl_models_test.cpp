// Native actual-length image/deepstack regression; no Python or token padding.
#include "models/qwen3_vl_models.h"
#include "models/qwen35_models.h"
#include "bridge.h"
#include <c10/core/InferenceMode.h>
#include <ATen/Parallel.h>
#include <iostream>
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif
using namespace tei;
#ifdef TEI_TORCH_CUDA_KERNELS
// Exercise the bridge policy: a multimodal family can capture text without
// opting image/audio requests into capture, including when those requests are
// interleaved with a resident text graph.
void text_graph_policy_test(const Options& options,const Weights& weights,
                            const PackedInput& media,const std::string& device) {
  TeiConfig cfg{16,2,2,24,32,32,1,1e-6,0};
  auto load=[&](bool capture) {
    const auto dtype=options.string("model_type")=="qwen3_5"?2:1;
    std::unique_ptr<void,decltype(&tei_destroy)> handle(tei_create(&cfg,device.c_str(),dtype),tei_destroy);
    TORCH_CHECK(handle,tei_error());
    for(const auto& [key,value]:options.values)
      TORCH_CHECK(tei_option(handle.get(),key.c_str(),value.c_str())==0,tei_error());
    TORCH_CHECK(tei_option(handle.get(),"_cuda_graphs",capture?"true":"false")==0,tei_error());
    TORCH_CHECK(tei_option(handle.get(),"_media_cuda_graphs","false")==0,tei_error());
    TORCH_CHECK(tei_option(handle.get(),"_cuda_graph_reuse","1")==0,tei_error());
    for(const auto& [name,value]:weights) {
      auto host=value.to(at::kCPU,at::kFloat).contiguous();
      TORCH_CHECK(tei_weight(handle.get(),name.c_str(),reinterpret_cast<const uint8_t*>(host.const_data_ptr()),
        host.sizes().data(),host.dim(),0)==0,tei_error());
    }
    TORCH_CHECK(tei_ready(handle.get())==0,tei_error());return handle;
  };
  auto eager=load(false),graph=load(true);
  auto ids=media.ids.clone(),types=media.types.contiguous(),positions=media.positions.clone();
  auto text_rotary=positions.unsqueeze(0).repeat({3,1});
  int64_t rows[]={0,1};
  auto run=[&](void* handle,bool with_image) {
    const auto& source=media.images.at(0);auto pixels=source.pixels.to(at::kCPU,at::kFloat).contiguous();
    TeiImage image{};image.pixels=pixels.const_data_ptr<float>();image.rows=pixels.size(0);image.patch_dim=pixels.size(1);
    for(int i=0;i<3;++i)image.grid[i]=source.grid_thw[i];
    image.merge_size=source.merge_size;image.token_start=source.token_start;
    image.token_count=source.token_count;image.sequence_start=source.sequence_start;
    auto output=at::empty({ids.numel(),16},at::TensorOptions().dtype(at::kFloat));
    // Text conversations also carry three-axis rotary positions through the
    // router; position metadata alone must not opt the request out of capture.
    auto rotary=with_image?media.multimodal_positions:text_rotary;
    TORCH_CHECK(tei_forward(handle,ids.const_data_ptr<int64_t>(),types.const_data_ptr<int64_t>(),positions.const_data_ptr<int64_t>(),
      media.offsets,2,4,0,nullptr,0,rows,2,output.data_ptr<float>(),output.numel(),
      rotary.const_data_ptr<int64_t>(),with_image?&image:nullptr,with_image?1:0,nullptr,0)==0,tei_error());
    return output;
  };
  auto compare=[&](bool image) {
    auto expected=run(eager.get(),image),actual=run(graph.get(),image);
    TORCH_CHECK(at::allclose(expected,actual,.003,.003),"Text-only graph policy changed eager outputs");return actual;
  };
  auto first=compare(false);ids.add_(3);auto changed=compare(false);
  TORCH_CHECK(!at::equal(first,changed),"Text graph retained stale token IDs");
  text_rotary.select(0,1).select(0,1).add_(5);
  text_rotary.select(0,2).select(0,2).add_(7);
  auto moved=compare(false);
  TORCH_CHECK(!at::equal(changed,moved),"Rotary position fixture did not affect text output");
  uint64_t before[7]{},after[7]{};
  TORCH_CHECK(tei_graph_stats(graph.get(),before,7)==0,tei_error());
  TORCH_CHECK(before[2]==1&&before[0]>=1,"Text-only multimodal family did not capture and replay");
  compare(true);
  TORCH_CHECK(tei_graph_stats(graph.get(),after,7)==0,tei_error());
  TORCH_CHECK(std::equal(std::begin(before),std::end(before),std::begin(after)),"Image request entered text-only graph cache");
  compare(false);
  TORCH_CHECK(tei_graph_stats(graph.get(),after,7)==0&&after[0]>before[0]&&after[2]==1,
    "Image request disturbed the resident text graph");
  std::cout<<options.string("model_type")<<" "<<options.string("text_config.layer_types.0","full_attention")
    <<" bridge text-only capture/replay, changed IDs/rotary positions, eager image parity and interleaving passed\n";
}
#endif
int main(int argc,char** argv) {
  c10::InferenceMode inference;at::set_num_threads(1);at::manual_seed(712);
  Options cfg;
  cfg.values={{"model_type","qwen3_vl"},{"text_config.hidden_size","16"},{"text_config.num_attention_heads","2"},
    {"text_config.num_key_value_heads","1"},{"text_config.head_dim","8"},{"text_config.num_hidden_layers","2"},
    {"text_config.vocab_size","32"},{"text_config.intermediate_size","24"},
    {"text_config.max_position_embeddings","32"},{"text_config.rope_theta","10000"},
    {"text_config.rope_scaling.mrope_interleaved","true"},{"text_config.rope_scaling.mrope_section.0","2"},
    {"text_config.rope_scaling.mrope_section.1","1"},{"text_config.rope_scaling.mrope_section.2","1"},
    {"vision_config.hidden_size","16"},{"vision_config.num_heads","2"},{"vision_config.depth","2"},
    {"vision_config.spatial_merge_size","2"},{"vision_config.hidden_act","gelu"},{"vision_config.deepstack_visual_indexes.0","0"}};
  Weights weights;
  auto add=[&](const std::string& name,std::initializer_list<int64_t> shape,bool norm=false) {
    weights[name]=norm?at::ones(shape):at::randn(shape)*.1;
  };
  auto tp=std::string("model.language_model.");
  add(tp+"embed_tokens.weight",{32,16});add(tp+"norm.weight",{16},true);
  for(int i=0;i<2;++i) {
    auto p=tp+"layers."+std::to_string(i)+".";
    add(p+"input_layernorm.weight",{16},true);add(p+"post_attention_layernorm.weight",{16},true);
    add(p+"self_attn.q_proj.weight",{16,16});add(p+"self_attn.k_proj.weight",{8,16});add(p+"self_attn.v_proj.weight",{8,16});add(p+"self_attn.o_proj.weight",{16,16});
    add(p+"self_attn.q_norm.weight",{8},true);add(p+"self_attn.k_norm.weight",{8},true);
    add(p+"mlp.gate_proj.weight",{24,16});add(p+"mlp.up_proj.weight",{24,16});add(p+"mlp.down_proj.weight",{16,24});
  }
  auto vp=std::string("model.visual.");
  auto ln=[&](const std::string& p,int64_t width){add(vp+p+".weight",{width},true);weights[vp+p+".bias"]=at::zeros({width});};
  auto fc=[&](const std::string& p,int64_t out,int64_t in){add(vp+p+".weight",{out,in});add(vp+p+".bias",{out});};
  add(vp+"patch_embed.proj.weight",{16,3,2,1,1});add(vp+"patch_embed.proj.bias",{16});add(vp+"pos_embed.weight",{4,16});
  for(int i=0;i<2;++i){auto p="blocks."+std::to_string(i)+".";ln(p+"norm1",16);ln(p+"norm2",16);fc(p+"attn.qkv",48,16);fc(p+"attn.proj",16,16);fc(p+"mlp.linear_fc1",24,16);fc(p+"mlp.linear_fc2",16,24);}
  for(const auto& p:{std::string("merger"),std::string("deepstack_merger_list.0")}){ln(p+".norm",p=="merger"?16:64);fc(p+".linear_fc1",64,64);fc(p+".linear_fc2",16,64);}
  auto model=create_qwen3_vl(cfg,weights,c10::Device("cpu"),at::kFloat);model->ready();
  TORCH_CHECK(model->supports_images(),"Qwen3-VL image capability unavailable");
  std::vector<int64_t> ids={1,2,3,4,5,6},pos={0,1,2,3,0,1};std::vector<int32_t> offsets={0,4,6};
  auto options=at::TensorOptions().dtype(at::kLong);
  PackedInput input{at::from_blob(ids.data(),{6},options),at::zeros({6},options),at::from_blob(pos.data(),{6},options),
    at::from_blob(offsets.data(),{3},options.dtype(at::kInt)),offsets.data(),2,4};
  auto plain=model->forward(input);
  input.images.push_back({at::rand({4,6}),{1,2,2},2,1,1,0});
  input.multimodal_positions=input.positions.unsqueeze(0).repeat({3,1});
  auto visual=model->forward(input);
  TORCH_CHECK((visual-plain).abs().max().item<float>()>1e-4,"Image features not injected");
  TORCH_CHECK(at::allclose(visual.narrow(0,4,2),plain.narrow(0,4,2),1e-5,1e-6),"Images leaked between packed sequences");
  TORCH_CHECK(at::allclose(visual.narrow(0,0,1),plain.narrow(0,0,1),1e-5,1e-6),"Image affected causal prefix");
  input.images[0].pixels=at::zeros({4,6});auto changed=model->forward(input);
  TORCH_CHECK((changed.narrow(0,1,3)-visual.narrow(0,1,3)).abs().max().item<float>()>1e-4,"Image pixels ignored");
  TORCH_CHECK(at::allclose(changed.narrow(0,4,2),visual.narrow(0,4,2),1e-5,1e-6),"Changed image leaked between sequences");
  std::cout<<"Qwen3-VL natural patches, image injection, deepstack and packed isolation passed\n";
  auto q35cfg=cfg;q35cfg.values["model_type"]="qwen3_5";
  q35cfg.values.erase("vision_config.deepstack_visual_indexes.0");
  q35cfg.values["text_config.layer_types.0"]="full_attention";q35cfg.values["text_config.layer_types.1"]="full_attention";
  q35cfg.values["text_config.rope_parameters.mrope_interleaved"]="true";
  q35cfg.values["text_config.rope_parameters.partial_rotary_factor"]="1";
  for(int i=0;i<3;++i)q35cfg.values["text_config.rope_parameters.mrope_section."+std::to_string(i)]=cfg.values.at("text_config.rope_scaling.mrope_section."+std::to_string(i));
  auto q35weights=weights;
  for(int i=0;i<2;++i) {
    auto key=tp+"layers."+std::to_string(i)+".self_attn.q_proj.weight";
    // Qwen3.5 interleaves each head's query with its attention output gate.
    auto query=q35weights.at(key).view({2,8,16});
    q35weights[key]=at::cat({query,at::zeros_like(query)},1).reshape({32,16});
  }
  auto q35loader=q35weights;auto q35=create_qwen35(q35cfg,q35loader,c10::Device("cpu"),at::kFloat);q35->ready();
  auto q35plain=input;q35plain.images.clear();q35plain.multimodal_positions=at::Tensor();
  auto q35text=q35->forward(q35plain),q35image=q35->forward(input);
  TORCH_CHECK(q35->supports_images() && (q35image-q35text).abs().max().item<float>()>1e-4,"Qwen3.5 vision features not injected");
  TORCH_CHECK(at::allclose(q35image.narrow(0,4,2),q35text.narrow(0,4,2),1e-5,1e-6),"Qwen3.5 vision leaked between packed sequences");
  Weights published;
  for(const auto& [name,value]:q35weights)
    published[name.starts_with("model.")?name.substr(6):name]=value;
  auto published_loader=published;auto unprefixed=create_qwen35(q35cfg,published_loader,c10::Device("cpu"),at::kFloat);unprefixed->ready();
  TORCH_CHECK(unprefixed->supports_images() && at::equal(q35text,unprefixed->forward(q35plain)) &&
    at::equal(q35image,unprefixed->forward(input)),"Published Qwen3.5 visual/language prefix changed full text or image outputs");
  auto missing=published;missing.erase("visual.patch_embed.proj.weight");bool rejected=false;
  try {auto incomplete=create_qwen35(q35cfg,missing,c10::Device("cpu"),at::kFloat);incomplete->ready();}
  catch(const c10::Error& error) {rejected=std::string(error.what()).find("patch_embed.proj.weight")!=std::string::npos;}
  TORCH_CHECK(rejected,"Incomplete published visual tower silently accepted");
  auto duplicate=published;duplicate["model.visual.patch_embed.proj.weight"]=published.at("visual.patch_embed.proj.weight");rejected=false;
  try {auto ambiguous=create_qwen35(q35cfg,duplicate,c10::Device("cpu"),at::kFloat);ambiguous->ready();}
  catch(const c10::Error& error) {rejected=std::string(error.what()).find("Duplicate Qwen visual weight")!=std::string::npos;}
  TORCH_CHECK(rejected,"Duplicate canonical/published visual weight silently accepted");
  std::cout<<"Qwen3.5 published prefixes preserve full outputs; missing/ambiguous tower rejected\n";
  std::cout<<"Qwen3.5 FP16 vision, BF16-compatible injection and mRoPE packed isolation passed\n";
  if(argc>1) {
    c10::Device device(argv[1]);Weights hw,cw;
    for(const auto& [name,value]:weights){hw[name]=value.to(at::kHalf);cw[name]=hw[name].to(device);}
    auto reference=create_qwen3_vl(cfg,hw,c10::Device("cpu"),at::kHalf);reference->ready();
    auto native=create_qwen3_vl(cfg,cw,device,at::kHalf);native->ready();
    auto host=input,dev=input;host.images[0].pixels=host.images[0].pixels.to(at::kHalf);
    dev.ids=host.ids.to(device);dev.types=host.types.to(device);dev.positions=host.positions.to(device);dev.cumulative=host.cumulative.to(device);
    dev.multimodal_positions=host.multimodal_positions.to(device);dev.images[0].pixels=host.images[0].pixels.to(device);
    auto expected=reference->forward(host).to(at::kFloat),actual=native->forward(dev).cpu().to(at::kFloat);
    TORCH_CHECK(at::allclose(expected,actual,.03,.03),"Qwen3-VL CUDA/CPU image parity mismatch: ",(expected-actual).abs().max().item<float>());
    std::cout<<"Qwen3-VL CUDA/CPU float16 parity maxabs="<<(expected-actual).abs().max().item<float>()<<"\n";
#ifdef TEI_TORCH_CUDA_KERNELS
    text_graph_policy_test(cfg,weights,input,device.str());
    text_graph_policy_test(q35cfg,q35weights,input,device.str());
    auto mixed_cfg=q35cfg;auto mixed_weights=q35weights;
    mixed_cfg.values["text_config.layer_types.0"]="linear_attention";
    mixed_cfg.values["text_config.linear_num_key_heads"]="1";
    mixed_cfg.values["text_config.linear_num_value_heads"]="1";
    auto linear=tp+"layers.0.linear_attn.";
    auto delta_weight=[&](const char* name,std::initializer_list<int64_t> shape) {
      mixed_weights[linear+name]=at::randn(shape)*.05;
    };
    delta_weight("in_proj_qkv.weight",{384,16});delta_weight("in_proj_z.weight",{128,16});
    delta_weight("in_proj_a.weight",{1,16});delta_weight("in_proj_b.weight",{1,16});
    delta_weight("conv1d.weight",{384,1,4});delta_weight("A_log",{1});delta_weight("dt_bias",{1});
    mixed_weights[linear+"norm.weight"]=at::ones({128});delta_weight("out_proj.weight",{16,128});
    text_graph_policy_test(mixed_cfg,mixed_weights,input,device.str());
    mixed_cfg.values["decision_attention_mode"]="noncausal_full_attention";
    text_graph_policy_test(mixed_cfg,mixed_weights,input,device.str());
    // Actual 64x64 patch grid: two heads make each original F32 score
    // tensor 128 MiB, exercising production query chunking rather than the
    // unchanged small-image branch. These are synthetic functional weights,
    // not a replacement for the separate original learned-image oracle.
    auto bounded_cfg=q35cfg;
    bounded_cfg.values["text_config.max_position_embeddings"]="2048";
    auto bounded_host_weights=q35weights;
    bounded_host_weights[vp+"patch_embed.proj.weight"]=at::randn({16,3,2,16,16})*.002;
    Weights bounded_weights;
    for(const auto& [name,value]:bounded_host_weights)bounded_weights[name]=value.to(device,at::kBFloat16);
    auto bounded_loader=bounded_weights;
    auto bounded=create_qwen35(bounded_cfg,bounded_loader,device,at::kBFloat16);bounded->ready();
    constexpr int64_t patches=4096,merged=1024,total=1027;
    std::vector<int32_t> large_offsets={0,1025,1027};
    auto long_options=at::TensorOptions().device(device).dtype(at::kLong);
    auto positions=at::cat({at::arange(1025,long_options),at::arange(2,long_options)},0);
    PackedInput large{at::arange(total,long_options).remainder(30)+1,at::zeros({total},long_options),positions,
      at::from_blob(large_offsets.data(),{3},at::TensorOptions().dtype(at::kInt)).clone().to(device),large_offsets.data(),2,1025};
    large.multimodal_positions=positions.unsqueeze(0).repeat({3,1});
    auto without_image=bounded->forward(large);
    large.images.push_back({at::rand({patches,1536},long_options.dtype(at::kHalf)),{1,64,64},2,1,merged,0});
    Weights bounded_cpu_weights;
    for(const auto& [name,value]:bounded_host_weights)bounded_cpu_weights[name]=value.to(at::kBFloat16).to(at::kHalf);
    Qwen3VL cpu_vision(bounded_cfg,bounded_cpu_weights,c10::Device("cpu"),at::kHalf,true);cpu_vision.ready();
    auto cpu_image=large.images[0];cpu_image.pixels=cpu_image.pixels.cpu();
    auto original_cpu_features=cpu_vision.image_features(cpu_image).to(at::kFloat);
    // Initialize vision workspaces before recording live allocation peak.
    Qwen3VL vision_only(bounded_cfg,bounded_weights,device,at::kHalf,true);vision_only.ready();
    ImageInput warm_image{large.images[0].pixels.narrow(0,0,4),{1,2,2},2,1,1,0};
    vision_only.image_features(warm_image);
    auto chunked_features=vision_only.image_features(large.images[0]).cpu().to(at::kFloat);
    TORCH_CHECK(at::allclose(original_cpu_features,chunked_features,.03,.03),"Large chunked Qwen3.5 vision differs from independent original CPU path: ",(original_cpu_features-chunked_features).abs().max().item<float>());
    at::cuda::getCurrentCUDAStream(device.index()).synchronize();
    const auto allocated_before=c10::cuda::CUDACachingAllocator::getDeviceStats(device.index()).allocated_bytes[0].current;
    c10::cuda::CUDACachingAllocator::resetPeakStats(device.index());
    auto large_image=bounded->forward(large);
    at::cuda::getCurrentCUDAStream(device.index()).synchronize();
    const auto peak_extra=c10::cuda::CUDACachingAllocator::getDeviceStats(device.index()).allocated_bytes[0].peak-allocated_before;
    TORCH_CHECK(peak_extra<128LL*1024*1024,"Large vision request recreated full square score/probability storage: ",peak_extra);
    TORCH_CHECK(at::isfinite(large_image).all().item<bool>(),"Chunked large-image output is nonfinite");
    TORCH_CHECK((large_image.narrow(0,1,merged)-without_image.narrow(0,1,merged)).abs().max().item<float>()>1e-5,"Large merged image span was ignored");
    TORCH_CHECK(at::equal(large_image.narrow(0,1025,2),without_image.narrow(0,1025,2)),"Large image leaked into another packed request");
    large.images[0].pixels=at::zeros_like(large.images[0].pixels);
    auto changed_image=bounded->forward(large);
    TORCH_CHECK(at::isfinite(changed_image).all().item<bool>() &&
      (changed_image.narrow(0,1,merged)-large_image.narrow(0,1,merged)).abs().max().item<float>()>1e-5,"Large chunked image pixels ignored");
    TORCH_CHECK(at::equal(changed_image.narrow(0,1025,2),large_image.narrow(0,1025,2)),"Changed large image leaked between requests");
    std::cout<<"Qwen3.5 actual 4096 patches / 1024 merged image tokens, chunked F32 attention, changed pixels and packed isolation passed; peak extra bytes="<<peak_extra<<"\n";
#endif

  }
}

#include "helpers/attention.h"
// Native adaptation of Candle Qwen3.5. Upstream notices retained in LICENSE/NOTICE.
#include "models/qwen3_vl_models.h"
#include "models/decoder_models.h"
#include "kernels/qwen35_kernels.h"
#include "kernels/qwen35_reference.h"
#include "kernels/qwen35_vision.h"
#include "kernels/encoder_activation.h"
#include "kernels/torch_flash_fma.h"
#include "kernels/moe_kernels.h"
#include <ATen/ops/_efficient_attention_forward.h>
#include <ATen/ops/_flash_attention_forward.h>
#include <cmath>
#include <algorithm>

namespace tei {
// Adapted from Candle qwen3_vl/vision.rs; upstream Apache-2.0 notices retained.
const at::Tensor& Qwen3VL::w(const std::string& name) const { return weight(visual_weights,"model.visual."+name); }

at::Tensor Qwen3VL::linear(const at::Tensor& x,const std::string& name) const {
    // Original Linear rounds the model-dtype GEMM before adding its bias.
    if(exact_vision)return at::linear(x,w(name+".weight"))+w(name+".bias");
    return at::linear(x,w(name+".weight"),w(name+".bias"));
  }

at::Tensor Qwen3VL::ln(const at::Tensor& x,const std::string& name) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    if(exact_vision)return qwen35_vision_layer_norm(x.contiguous(),w(name+".weight"),w(name+".bias"),1e-6f);
#endif
    return at::layer_norm(x,{x.size(-1)},w(name+".weight"),w(name+".bias"),1e-6);
  }

at::Tensor Qwen3VL::merger(const at::Tensor& x,const std::string& p,bool post) const {
    auto grouped=x.view({-1,hidden*merge*merge});
    auto normalized=post?ln(grouped,p+".norm"):ln(x,p+".norm").view_as(grouped);
    // Candle Tensor::gelu is the tanh approximation; merger has no activation config.
    auto mid=linear(normalized,p+".linear_fc1");auto act=at::Tensor();
#ifdef TEI_TORCH_CUDA_KERNELS
    if(exact_vision)act=encoder_gelu(mid);else
#endif
    act=at::gelu(mid,"tanh");auto out=linear(act,p+".linear_fc2");return out;
  }

std::pair<at::Tensor,std::vector<at::Tensor>> Qwen3VL::vision(const ImageInput& image) const {
    const auto [frames,height,width]=image.grid_thw;
    auto patch_dim=w("patch_embed.proj.weight").numel()/hidden;
    TORCH_CHECK(frames>0 && height>0 && width>0 && height%merge==0 && width%merge==0 &&
      image.merge_size==merge && image.pixels.sizes()==at::IntArrayRef({frames*height*width,patch_dim}),"Invalid Qwen3-VL packed image grid");
    // Each preprocessed row is one complete temporal/spatial patch; Conv3D's
    // nonoverlapping kernel is exactly this flattened projection.
    at::Tensor states;
    if(exact_vision) {
      // Candle temporal-2 projection: two actual K768 GEMMs, rounded FP16
      // sum, then separate FP16 bias. No temporal or token padding.
      auto patch=image.pixels.to(device,dtype).view({-1,3,2,16,16});
      auto first=at::linear(patch.select(2,0).contiguous().view({-1,768}),temporal_first);
      auto second=at::linear(patch.select(2,1).contiguous().view({-1,768}),temporal_second);
      states=(first+second)+w("patch_embed.proj.bias");
    } else {
      states=at::linear(image.pixels.to(dtype),w("patch_embed.proj.weight").reshape({hidden,patch_dim}),w("patch_embed.proj.bias"));
    }
    auto side=static_cast<int64_t>(std::llround(std::sqrt(double(w("pos_embed.weight").size(0)))));
    TORCH_CHECK(side*side==w("pos_embed.weight").size(0),"Qwen3-VL position table must be square");
    auto ix=at::arange(frames*height*width,states.options().dtype(at::kLong));
    auto c=ix.remainder(merge)+at::floor_divide(ix,merge*merge).remainder(width/merge)*merge;
    auto r=at::floor_divide(ix,merge).remainder(merge)+at::floor_divide(ix,merge*width).remainder(height/merge)*merge;
    auto rf=r.to(at::kFloat)*(height==1?0.f:float(side-1)/float(height-1));
    auto cf=c.to(at::kFloat)*(width==1?0.f:float(side-1)/float(width-1));
    auto r0=rf.floor().to(at::kLong),r1=rf.ceil().clamp_max(side-1).to(at::kLong);
    auto c0=cf.floor().to(at::kLong),c1=cf.ceil().clamp_max(side-1).to(at::kLong);
    auto dr=rf-r0.to(at::kFloat),dc=cf-c0.to(at::kFloat);
    auto table=w("pos_embed.weight");
    auto interpolate=[&](const at::Tensor& rr,const at::Tensor& cc,const at::Tensor& scale) {
      return table.index_select(0,rr*side+cc)*scale.to(dtype).unsqueeze(-1);
    };
    // Sum in table dtype, matching the four-corner interpolation stages.
    auto corners=at::stack({interpolate(r0,c0,(1-dr)*(1-dc)),interpolate(r0,c1,(1-dr)*dc),
      interpolate(r1,c0,dr*(1-dc)),interpolate(r1,c1,dr*dc)},0);
    auto positional=exact_vision?(corners.select(0,0)+corners.select(0,2))+(corners.select(0,1)+corners.select(0,3)):corners.sum(0);
    states=states+positional;

    auto angles=at::cat({r.to(at::kFloat).unsqueeze(1)*vision_frequency,c.to(at::kFloat).unsqueeze(1)*vision_frequency},-1);
    auto cos=angles.cos().unsqueeze(1),sin=angles.sin().unsqueeze(1);
    auto rotate=[&](const at::Tensor& value) {
      auto x=value.to(at::kFloat),a=x.narrow(-1,0,dim/2),b=x.narrow(-1,dim/2,dim/2);
      return at::cat({a*cos-b*sin,b*cos+a*sin},-1);
    };
    auto cu=at::arange(frames+1,states.options().dtype(at::kInt))*(height*width);
    std::vector<at::Tensor> deep;
    for(int64_t layer=0;layer<depth;++layer) {
      auto p="blocks."+std::to_string(layer)+".";
      auto normalized=ln(states,p+"norm1");
      auto qkv=linear(normalized,p+"attn.qkv").view({states.size(0),3,heads,dim});
      auto q=rotate(qkv.select(1,0)),k=rotate(qkv.select(1,1)),v=qkv.select(1,2).to(at::kFloat);

      at::Tensor attention;
      // Round rotary Q/K back to BF16 before attention, matching the
      // model dtype, and keep frames independent through cu_seqlens.
      if(device.is_cuda() && dtype==at::kBFloat16) {
        attention=std::get<0>(at::_flash_attention_forward(q.to(dtype).contiguous(),k.to(dtype).contiguous(),v.to(dtype).contiguous(),
          cu,cu,height*width,height*width,0.,false,false,1./std::sqrt(double(dim))));
      } else if(device.is_cuda()) {
        attention=std::get<0>(at::_efficient_attention_forward(q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0),
          std::nullopt,cu,cu,height*width,height*width,0.,0,false,1./std::sqrt(double(dim)))).squeeze(0);
      } else {
        std::vector<at::Tensor> outputs;
        for(int64_t t=0;t<frames;++t) {
          auto slice=[&](const at::Tensor& x){return x.narrow(0,t*height*width,height*width).transpose(0,1).unsqueeze(0);};
          outputs.push_back(at::scaled_dot_product_attention(slice(q),slice(k),slice(v),std::nullopt,0.,false,1./std::sqrt(double(dim))).squeeze(0).transpose(0,1));
        }
        attention=at::cat(outputs,0);
      }

      auto projected=linear(attention.reshape({states.size(0),hidden}).to(dtype),p+"attn.proj");
      states=states+projected;
      auto normalized2=ln(states,p+"norm2");
      auto ff=linear(normalized2,p+"mlp.linear_fc1");
      auto act=cfg.string("vision_config.hidden_act","gelu_pytorch_tanh");
#ifdef TEI_TORCH_CUDA_KERNELS
      if(exact_vision && act!="silu" && act!="gelu")ff=encoder_gelu(ff);else
#endif
      ff=act=="silu"?at::silu(ff):at::gelu(ff,act=="gelu"?"none":"tanh");
      auto down=linear(ff,p+"mlp.linear_fc2");
      states=states+down;
      for(int64_t i=0;cfg.values.count("vision_config.deepstack_visual_indexes."+std::to_string(i));++i)
        if(cfg.integer("vision_config.deepstack_visual_indexes."+std::to_string(i))==layer)
          deep.push_back(merger(states,"deepstack_merger_list."+std::to_string(i),true));
    }
    return {merger(states,"merger",false),std::move(deep)};
  }

Qwen3VL::Qwen3VL(const Options& options,Weights& weights,c10::Device dev,at::ScalarType type,bool only_vision):cfg(options),device(dev),dtype(type),tower_only(only_vision) {
    // Published full checkpoints use both model.visual.* and visual.*. Keep
    // one canonical local name and reject ambiguous towers instead of mixing
    // weights from two namespaces. Complete tower validation remains in ready.
    for(const auto& [name,value]:weights) {
      std::string canonical;
      if(name.starts_with("model.visual."))canonical=name;
      else if(name.starts_with("visual."))canonical="model."+name;
      else continue;
      TORCH_CHECK(!visual_weights.count(canonical),"Duplicate Qwen visual weight: ",canonical);
      visual_weights.emplace(std::move(canonical),value.to(dtype));
    }
    Options tc=cfg;
    for(const auto& [name,value]:cfg.values)if(name.starts_with("text_config."))tc.values[name.substr(12)]=value;
    tc.values["model_type"]="qwen3";tc.values["use_sliding_window"]="false";
    text_dim=tc.integer("head_dim",tc.integer("hidden_size")/tc.integer("num_attention_heads",1));
    hidden=cfg.integer("vision_config.hidden_size");heads=cfg.integer("vision_config.num_heads",1);
    dim=hidden/heads;merge=cfg.integer("vision_config.spatial_merge_size",2);depth=cfg.integer("vision_config.depth");
    if(tower_only)return;
    Weights text_weights;
    for(const auto& [name,value]:weights)
      if(name.starts_with("model.language_model."))text_weights[name.substr(21)]=value;
      else if(name.starts_with("language_model."))text_weights[name.substr(15)]=value;
    TORCH_CHECK(!text_weights.empty(),"Qwen3-VL language model weights missing");
    text=create_decoder(tc,text_weights,device,dtype);
  }

void Qwen3VL::ready() {
    TORCH_CHECK(hidden>0 && heads>0 && hidden%heads==0 && dim%4==0 && merge>0 && depth>0,"Invalid Qwen3-VL architecture");
    const auto& patch_kernel=w("patch_embed.proj.weight");w("pos_embed.weight");
#ifdef TEI_TORCH_CUDA_KERNELS
    exact_vision=device.is_cuda() && dtype==at::kHalf && cfg.string("model_type").starts_with("qwen3_5") &&
      patch_kernel.dim()==5 && patch_kernel.size(0)==hidden && patch_kernel.size(1)==3 &&
      patch_kernel.size(2)==2 && patch_kernel.size(3)==16 && patch_kernel.size(4)==16;
    if(exact_vision) {
      temporal_first=patch_kernel.select(2,0).contiguous().view({hidden,768});
      temporal_second=patch_kernel.select(2,1).contiguous().view({hidden,768});
    }
#endif
    std::vector<float> frequencies;
    for(int64_t j=0;j<dim/4;++j)frequencies.push_back(1.f/std::pow(10000.f,float(2*j)/float(dim/2)));
    vision_frequency=at::from_blob(frequencies.data(),{dim/4},at::TensorOptions().dtype(at::kFloat)).clone().to(device);
    if(tower_only)return;
    TORCH_CHECK(text,"Qwen3-VL text decoder missing");
    TORCH_CHECK(cfg.boolean("text_config.rope_scaling.mrope_interleaved",false),"Only interleaved Qwen3-VL mRoPE is supported");
    int64_t sections=0;for(int i=0;i<3;++i)sections+=cfg.integer("text_config.rope_scaling.mrope_section."+std::to_string(i));
    TORCH_CHECK(sections==text_dim/2,"Invalid Qwen3-VL mRoPE sections");
    std::vector<float> freqs;std::vector<int64_t> axes;
    auto theta=cfg.real("text_config.rope_theta",1000000.);
    auto s1=cfg.integer("text_config.rope_scaling.mrope_section.1"),s2=cfg.integer("text_config.rope_scaling.mrope_section.2");
    for(int64_t j=0;j<text_dim/2;++j) {
      freqs.push_back(float(1./std::pow(theta,double(2*j)/text_dim)));
      axes.push_back(j%3==1 && j<s1*3?1:j%3==2 && j<s2*3?2:0);
    }
    text_frequency=at::from_blob(freqs.data(),{text_dim/2},at::TensorOptions().dtype(at::kFloat)).clone().to(device);
    text_axes=at::from_blob(axes.data(),{text_dim/2},at::TensorOptions().dtype(at::kLong)).clone().to(device);
    w("patch_embed.proj.weight");w("pos_embed.weight");text->ready();
  }

bool Qwen3VL::supports_images() const {return true;}

at::Tensor Qwen3VL::image_features(const ImageInput& image) const {return vision(image).first;}

TowerMemoryProfile Qwen3VL::vision_memory_profile() const {
  int64_t saved=0;
  for(int64_t i=0;cfg.values.count("vision_config.deepstack_visual_indexes."+std::to_string(i));++i) {
    const auto layer=cfg.integer("vision_config.deepstack_visual_indexes."+std::to_string(i));
    TORCH_CHECK(layer>=0&&layer<depth,"Invalid vision deepstack layer index");++saved;
  }
  TowerMemoryProfile p{hidden,heads,std::max(hidden,transient_linear_width(visual_weights)),saved,0};
  p.per_frame_attention=true;
  p.efficient_scores=device.is_cuda();return p;
}
std::optional<TransientMemoryProfile> Qwen3VL::transient_memory_profile() const {
  if(!text)return std::nullopt;
  auto p=text->transient_memory_profile();if(p)p->vision=vision_memory_profile();return p;
}
int64_t Qwen3VL::output_width() const {return text->output_width();}

at::Tensor Qwen3VL::forward(const PackedInput& input) const {
    auto positions=input.multimodal_positions.defined()?input.multimodal_positions:input.positions.unsqueeze(0).expand({3,-1});
    auto angles=positions.index_select(0,text_axes).transpose(0,1).to(at::kFloat)*text_frequency;
    std::vector<at::Tensor> indices,features;std::vector<std::vector<at::Tensor>> deep;
    for(const auto& image:input.images) {
      auto [feat,ds]=vision(image);
      TORCH_CHECK(feat.size(0)==image.token_count,"Qwen3-VL merged visual token count mismatch");
      indices.push_back(at::arange(image.token_start,image.token_start+image.token_count,input.ids.options()));features.push_back(feat);
      if(deep.empty())deep.resize(ds.size());
      TORCH_CHECK(deep.size()==ds.size(),"Qwen3-VL deepstack count mismatch");
      for(size_t i=0;i<ds.size();++i)deep[i].push_back(ds[i]);
    }
    std::vector<at::Tensor> stacked;for(auto& rows:deep)stacked.push_back(at::cat(rows,0));
    return decoder_multimodal_forward(*text,input,indices.empty()?at::Tensor():at::cat(indices,0),
      features.empty()?at::Tensor():at::cat(features,0),angles.cos().to(dtype).contiguous(),angles.sin().to(dtype).contiguous(),stacked);
  }
std::unique_ptr<Model> create_qwen3_vl(const Options& cfg,Weights& weights,c10::Device device,at::ScalarType dtype) {
  if(cfg.string("model_type")=="qwen3_vl")return std::make_unique<Qwen3VL>(cfg,weights,device,dtype);
  return nullptr;
}
}

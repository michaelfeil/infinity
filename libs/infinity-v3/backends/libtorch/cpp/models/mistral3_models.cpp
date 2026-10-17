// Pixtral/Mistral3 adaptation of Hugging Face Transformers 5.12.1 (Apache-2.0).
// See vendor/transformers_pixtral/NOTICE; fork modifications follow repository LICENSE.
#include "models/mistral3_models.h"
#include "models/decoder_models.h"
#include <ATen/ops/scaled_dot_product_attention.h>
#include <cmath>
#include <algorithm>

namespace tei { namespace {
class Mistral3 final : public Model {
  Options cfg; Weights weights; c10::Device device; at::ScalarType dtype;
  std::unique_ptr<Model> text;
  int64_t hidden,heads,dim,depth,patch,merge,feature_layer;
  const at::Tensor& w(const std::string& name) const {return weight(weights,name);}
  at::Tensor rms(const at::Tensor& x,const at::Tensor& scale,double eps) const {
    auto f=x.to(at::kFloat);
    return (f*at::rsqrt(f.square().mean(-1,true)+eps)).to(x.scalar_type())*scale;
  }
  at::Tensor linear(const at::Tensor& x,const std::string& name) const {
    auto it=weights.find(name+".bias");
    return at::linear(x,w(name+".weight"),it==weights.end()?at::Tensor():it->second);
  }
  at::Tensor vision(const ImageInput& image) const {
    auto h=image.grid_thw[1],width=image.grid_thw[2];
    TORCH_CHECK(image.grid_thw[0]==1&&h>0&&width>0&&h%merge==0&&width%merge==0&&
      image.merge_size==merge,"Invalid Pixtral patch grid");
    TORCH_CHECK(h<=cfg.integer("vision_config.image_size",1540)/patch&&width<=cfg.integer("vision_config.image_size",1540)/patch,"Pixtral image exceeds its position grid");
    auto n=h*width;
    TORCH_CHECK(image.pixels.sizes()==at::IntArrayRef({n,3*patch*patch}),"Invalid Pixtral pixel matrix");
    auto x=at::linear(image.pixels.to(device,dtype),w("vision_tower.patch_conv.weight").reshape({hidden,-1}));
    x=rms(x,w("vision_tower.ln_pre.weight"),1e-5);
    auto index=at::arange(n,x.options().dtype(at::kLong));
    auto rows=at::floor_divide(index,width).to(at::kFloat).unsqueeze(1);
    auto cols=index.remainder(width).to(at::kFloat).unsqueeze(1);
    auto theta=cfg.real("vision_config.rope_parameters.rope_theta",cfg.real("vision_config.rope_theta",10000.));
    auto freq=at::arange(dim/4,x.options().dtype(at::kFloat));
    auto angles=at::cat({rows*at::pow(theta,-4*freq/dim),cols*at::pow(theta,-(4*freq+2)/dim)},-1);
    auto cos=at::cat({angles,angles},-1).cos().to(dtype).unsqueeze(1);
    auto sin=at::cat({angles,angles},-1).sin().to(dtype).unsqueeze(1);
    auto rotate=[&](const at::Tensor& t) {return t*cos+at::cat({-t.narrow(-1,dim/2,dim/2),t.narrow(-1,0,dim/2)},-1)*sin;};
    at::Tensor selected;
    if(feature_layer==0)selected=x;
    for(int64_t i=0;i<depth;++i) {
      auto p="vision_tower.transformer.layers."+std::to_string(i)+".";
      auto norm=rms(x,w(p+"attention_norm.weight"),1e-5);
      auto q=rotate(linear(norm,p+"attention.q_proj").view({n,heads,dim}));
      auto k=rotate(linear(norm,p+"attention.k_proj").view({n,heads,dim}));
      auto v=linear(norm,p+"attention.v_proj").view({n,heads,dim});
      auto a=at::scaled_dot_product_attention(q.transpose(0,1).unsqueeze(0),k.transpose(0,1).unsqueeze(0),v.transpose(0,1).unsqueeze(0));
      x=x+linear(a.squeeze(0).transpose(0,1).contiguous().view({n,hidden}),p+"attention.o_proj");
      norm=rms(x,w(p+"ffn_norm.weight"),1e-5);
      auto gate=linear(norm,p+"feed_forward.gate_proj");
      auto activation=cfg.string("vision_config.hidden_act","silu");
      TORCH_CHECK(activation=="silu"||activation=="gelu","Unsupported Pixtral MLP activation");
      auto act=activation=="silu"?at::silu(gate):at::gelu(gate);
      x=x+linear(act*linear(norm,p+"feed_forward.up_proj"),p+"feed_forward.down_proj");
      if(feature_layer==i+1)selected=x;
    }
    TORCH_CHECK(selected.defined(),"Invalid Pixtral feature layer");
    x=rms(selected,w("multi_modal_projector.norm.weight"),cfg.real("text_config.rms_norm_eps",1e-5));
    // Match F.unfold's channel, patch-row, patch-column order (not flattened pixels).
    x=x.view({h/merge,merge,width/merge,merge,hidden}).permute({0,2,4,1,3}).contiguous().view({-1,hidden*merge*merge});
    x=linear(x,"multi_modal_projector.patch_merger.merging_layer");
    x=linear(x,"multi_modal_projector.linear_1");
    auto act=cfg.string("projector_hidden_act","gelu");
    TORCH_CHECK(act=="gelu"||act=="gelu_pytorch_tanh"||act=="silu","Unsupported Mistral3 projector activation");
    x=act=="silu"?at::silu(x):at::gelu(x,act=="gelu_pytorch_tanh"?"tanh":"none");
    return linear(x,"multi_modal_projector.linear_2");
  }
public:
  Mistral3(const Options& c,Weights& source,c10::Device dev,at::ScalarType dt):cfg(c),device(dev),dtype(dt) {
    // Consume the loader map: retaining it would keep unfused text parameters
    // alive alongside the decoder's packed matrices.
    Weights tw;
    for(auto& [name,value]:source){
      auto key=name.starts_with("model.")?name.substr(6):name;
      if(key.starts_with("language_model."))tw[key.substr(15)]=std::move(value);
      else if(key.starts_with("vision_tower.")||key.starts_with("multi_modal_projector."))weights[key]=std::move(value);
    }
    source.clear();
    Options tc;for(const auto& [k,v]:c.values)if(k.starts_with("text_config."))tc.values[k.substr(12)]=v;
    TORCH_CHECK(tc.string("model_type")=="ministral3"||tc.string("model_type")=="mistral","Mistral3 requires a Mistral/Ministral3 text model");
    // This wrapper follows Transformers rounding boundaries, independently
    // of the legacy Candle-compatible Mistral kernel contracts.
    tc.values["_transformers_math"]="true";
    text=create_decoder(tc,tw,device,dtype);
    hidden=cfg.integer("vision_config.hidden_size");heads=cfg.integer("vision_config.num_attention_heads");
    dim=heads>0?hidden/heads:0;depth=cfg.integer("vision_config.num_hidden_layers");
    patch=cfg.integer("vision_config.patch_size");merge=cfg.integer("spatial_merge_size",2);
    feature_layer=cfg.integer("vision_feature_layer",-1);if(feature_layer<0)feature_layer+=depth+1;
  }
  void ready() override {
    TORCH_CHECK(text&&hidden>0&&heads>0&&hidden%heads==0&&dim%4==0&&depth>0&&patch>0&&merge>0,"Invalid Mistral3 architecture");
    TORCH_CHECK(cfg.string("vision_config.model_type")=="pixtral","Mistral3 requires a Pixtral vision encoder");
    TORCH_CHECK(cfg.string("vision_config.rope_parameters.rope_type","default")=="default","Unsupported Pixtral rotary configuration");
    TORCH_CHECK(cfg.integer("vision_config.num_channels",3)==3&&feature_layer>=0&&feature_layer<=depth,"Unsupported Pixtral feature configuration");
    TORCH_CHECK(!cfg.values.count("vision_feature_layer.0"),"Multiple Pixtral feature layers are not supported");
    TORCH_CHECK(patch<=64&&merge<=8&&hidden<=32768&&depth<=256&&cfg.integer("vision_config.image_size",1540)>0&&cfg.integer("vision_config.image_size",1540)<=4096,"Unsupported Pixtral dimensions");
    const auto intermediate=cfg.integer("vision_config.intermediate_size"),out=text->output_width();
    const auto theta=cfg.real("vision_config.rope_parameters.rope_theta",cfg.real("vision_config.rope_theta",10000.));
    TORCH_CHECK(intermediate>0&&std::isfinite(theta)&&theta>1,"Invalid Pixtral feed-forward/rotary dimensions");
    TORCH_CHECK(cfg.string("vision_config.hidden_act","silu")=="silu"||cfg.string("vision_config.hidden_act")=="gelu","Unsupported Pixtral activation");
    const auto act=cfg.string("projector_hidden_act","gelu");
    TORCH_CHECK(act=="gelu"||act=="gelu_pytorch_tanh"||act=="silu","Unsupported Mistral3 projector activation");
    auto require=[&](const std::string& name,std::initializer_list<int64_t> shape){TORCH_CHECK(w(name).sizes()==at::IntArrayRef(shape),"Invalid Mistral3 weight dimensions: ",name);};
    require("vision_tower.patch_conv.weight",{hidden,3,patch,patch});require("vision_tower.ln_pre.weight",{hidden});
    for(int64_t i=0;i<depth;++i){
      auto p="vision_tower.transformer.layers."+std::to_string(i)+".";
      require(p+"attention_norm.weight",{hidden});require(p+"ffn_norm.weight",{hidden});
      for(const auto* proj:{"q_proj","k_proj","v_proj","o_proj"})require(p+"attention."+proj+".weight",{hidden,hidden});
      require(p+"feed_forward.gate_proj.weight",{intermediate,hidden});require(p+"feed_forward.up_proj.weight",{intermediate,hidden});require(p+"feed_forward.down_proj.weight",{hidden,intermediate});
    }
    require("multi_modal_projector.norm.weight",{hidden});
    require("multi_modal_projector.patch_merger.merging_layer.weight",{hidden,hidden*merge*merge});
    require("multi_modal_projector.linear_1.weight",{out,hidden});require("multi_modal_projector.linear_2.weight",{out,out});
    for(const auto* name:{"multi_modal_projector.linear_1.bias","multi_modal_projector.linear_2.bias"}){
      if(cfg.boolean("multimodal_projector_bias",true))require(name,{out});
      else TORCH_CHECK(!weights.count(name),"Unexpected projector bias");
    }
    text->ready();
  }
  bool supports_images() const override {return true;}
  bool model_dtype_mean_pool() const override {return true;}
  at::Tensor mean_pool(const at::Tensor& value) const override {
    // Match a masked sum followed by division in the hidden-state dtype.
    // Packed sequences contain only real tokens, including image markers.
    return (value.sum(0,true)/value.size(0)).to(at::kFloat);
  }
  int64_t output_width() const override {return text->output_width();}
  at::Tensor forward(const PackedInput& input) const override {
    if(input.images.empty())return text->forward(input);
    std::vector<at::Tensor> indices,features;
    for(const auto& image:input.images) {
      auto nh=image.grid_thw[1]/merge,nw=image.grid_thw[2]/merge;
      TORCH_CHECK(image.token_count==nh*nw&&image.token_start>=image.sequence_start,"Invalid Pixtral image token count");
      auto id=at::arange(nh*nw,input.ids.options());
      id=image.token_start+at::floor_divide(id,nw)*(nw+1)+id.remainder(nw);
      TORCH_CHECK(image.sequence_start>=0&&image.sequence_start<input.ids.numel(),"Invalid Pixtral sequence start");
      int64_t end=-1;for(int64_t r=0;r<input.batch;++r)if(input.offsets[r]==image.sequence_start)end=input.offsets[r+1];
      TORCH_CHECK(end>=0&&image.token_start<end&&nh*(nw+1)<=end-image.token_start,"Pixtral image tokens cross a sequence boundary");
      TORCH_CHECK(at::all(input.ids.index_select(0,id)==cfg.integer("image_token_index",10)).item<bool>(),"Pixtral image placeholders do not match");
      indices.push_back(id);features.push_back(vision(image));
    }
    return decoder_visual_forward(*text,input,at::cat(indices),at::cat(features));
  }
};
}
std::unique_ptr<Model> create_mistral3(const Options& cfg,Weights& weights,c10::Device device,at::ScalarType dtype) {
  if(cfg.string("model_type")!="mistral3")return nullptr;
  return std::make_unique<Mistral3>(cfg,weights,device,dtype);
}
}

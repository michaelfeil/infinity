#include "batching/radix.h"
#include "batching/prefix_cache.h"
#include "models/decoder_models.h"
#include "kernels/moe_kernels.h"
#include "kernels/infinity_fa4.h"
#ifdef TEI_TORCH_CUDA_KERNELS
#include "kernels/fast_kernels.h"
#include "helpers/fp8_linear.h"
#include "kernels/fp8_fusions.h"
#include "kernels/decoder_norm.h"
#include "kernels/torch_flash_fma.h"
#include "kernels/cached_decoder_linear.h"
#include "kernels/decoder_packed_readout.h"
#include "kernels/transformers_norm.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/detail/CUDAHooksInterface.h>
#endif
#include <ATen/ops/_flash_attention_forward.h>
#include <ATen/ops/_cudnn_attention_forward.h>
#include <ATen/ops/_fused_rms_norm.h>
#include <ATen/ops/_grouped_mm.h>
#include <cmath>
#include <algorithm>
#include <sstream>
#include <unordered_set>
#include <map>
#include <cctype>

namespace tei {
namespace {
struct DecoderLayer {
  int64_t rotary_index=-1,left_window=-1,right_window=-1;
  at::Tensor qkv, qkv_bias, out, out_bias, input_norm, post_norm, q_norm, k_norm;
  at::Tensor gate_up, gate_up_bias, down, down_bias, router;
#ifdef TEI_TORCH_CUDA_KERNELS
  std::unique_ptr<Fp8Linear> fp8_gate_up, fp8_down, fp8_qkv, fp8_out;
#endif
  std::vector<at::Tensor> expert_gate_up, expert_down;
  at::Tensor grouped_gate_up, grouped_down;
};
class Decoder final : public Model {
#ifdef TEI_TORCH_CUDA_KERNELS
  mutable CachedNoBiasLt no_bias_cache;
#endif
  at::Tensor cached_linear(const at::Tensor& x,const at::Tensor& w,const at::Tensor& b={}) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    if(!b.defined()&&!fp8_dynamic&&!transformers_math()&&dtype==at::kBFloat16
      &&hidden==4096&&(family=="qwen3"||family=="llama")&&w.size(1)==4096
      &&(w.size(0)==24576||w.size(0)==28672))return no_bias_cache.forward(x,w,true);
    if(!b.defined()&&!fp8_dynamic&&dtype==at::kBFloat16&&family=="qwen2"&&hidden==896&&head_dim==64)
      return no_bias_cache.forward(x,w);
#endif
    return at::linear(x,w,b);
  }
  at::Tensor cached_prefix_linear(const at::Tensor& x,const at::Tensor& w,const at::Tensor& b,
    const PackedInput& input) const {
    if(input.prefix&&radix_ids(input.prefix->original).numel()>x.size(0))
      return prefix_linear(x,w,b,input);
    return cached_linear(x,w,b);
  }
  Options options;
  Weights weights;
  c10::Device device;
  at::ScalarType dtype;
  std::string family, prefix, activation;
  int64_t hidden, heads, kvheads, head_dim, intermediate, layer_count;
  double epsilon;
  bool causal, renormalize;
  bool token_classifier=false;
  std::vector<std::pair<at::Tensor,at::Tensor>> rotary_caches;
  std::vector<int64_t> rotary_indices,attention_windows;
  bool cudnn_varlen = false;
  bool transformers_fused_norm = false;
  bool fp8_dynamic = false;
  bool fp8_all_linear = false;
  bool qwen_half_exact = false;
  int64_t left_window = -1, right_window = -1, top_k;
  at::Tensor embedding, final_norm, projection, projection_bias, score, score_bias, cos_cache, sin_cache, query_scale;
  std::vector<DecoderLayer> layers;
  const at::Tensor& weight(const std::string& name) const {
    auto it = weights.find(prefix + name);
    TORCH_CHECK(it != weights.end(), "Missing decoder weight: ", prefix, name);
    return it->second;
  }
  at::Tensor optional(const std::string& name) const {
    auto it = weights.find(prefix + name);
    return it == weights.end() ? at::Tensor() : it->second;
  }
  at::Tensor checked_weight(const std::string& name, std::initializer_list<int64_t> shape) const {
    auto tensor = weight(name);
    TORCH_CHECK(tensor.sizes() == at::IntArrayRef(shape), "Invalid decoder tensor dimensions: ", prefix, name);
    return tensor;
  }
  at::Tensor separate_linear(const at::Tensor& x,const at::Tensor& weight,const at::Tensor& bias,std::initializer_list<int64_t> widths) const {
    std::vector<at::Tensor> parts;int64_t start=0;
    for(auto width:widths){parts.push_back(at::linear(x,weight.narrow(0,start,width),bias.defined()?bias.narrow(0,start,width):at::Tensor()));start+=width;}
    return at::cat(parts,-1);
  }
  bool transformers_math() const {return options.boolean("_transformers_math",false);}
  at::Tensor norm(const at::Tensor& x, const at::Tensor& w) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    if(transformers_fused_norm)return transformers_rms_norm(x,{},w,epsilon).first;
#endif
    if(transformers_math()) {auto f=x.to(at::kFloat);return (f*at::rsqrt(f.square().mean(-1,true)+epsilon)).to(x.scalar_type())*w;}
    if (x.is_cuda()) return std::get<0>(at::_fused_rms_norm(x.contiguous(), {w.numel()}, w, epsilon));
    return at::rms_norm(x, {w.numel()}, w, epsilon);
  }
#ifdef TEI_TORCH_CUDA_KERNELS
  bool exact_mistral_math() const {
    // The original Candle RMS launcher registers BF16 width 3072 with a
    // 1x4 warp layout. Learned causal Mistral operands independently prove
    // its residual statistics, rotary HFMA and fused softmax contracts.
    return !transformers_math()&&dtype==at::kBFloat16&&family=="mistral"&&causal&&hidden==3072;
  }
  bool exact_decoder_math() const {
    // Registered BF16 width 2048 Llama uses the same Candle RMS and fused
    // softmax contracts for causal and bidirectional attention. FP8 makes
    // upstream one-ULP differences observable at quantization boundaries.
    // Widths are registered kernel geometries, not checkpoint signatures.
    // BF16 Llama width4096 retains Candle's registered 1x4 RMS layout
    // and fused softmax contract.
    // Qwen3 width1024 uses its registered 4x1 RMS layout and fused
    // attention arithmetic; both are needed for trained BF16 score parity.
    // FP16 Qwen3 needs the same registered RMS and fused softmax boundaries.
    return qwen_half_exact
      ||exact_mistral_math()||(dtype==at::kBFloat16
      &&((hidden==2048&&family=="qwen3_moe"&&options.integer("num_experts",0)==128&&top_k==8)
        ||(family=="llama"&&(hidden==2048||hidden==4096))
        ||(family=="qwen3"&&hidden==1024&&head_dim==128)
        ||(family=="qwen2"&&hidden==896&&head_dim==64)));
  }
  bool allow_cudnn_attention() const {
    // The explicit cuDNN option permits its different softmax reduction for
    // BF16 Llama-8B as well as all-linear FP8. Normalization stays unchanged.
    return fp8_all_linear||!exact_decoder_math()
      ||(!fp8_dynamic&&dtype==at::kBFloat16&&family=="llama"&&hidden==4096);
  }
  std::pair<at::Tensor,at::Tensor> residual_norm(const at::Tensor& x,
    const at::Tensor& residual,const at::Tensor& weight) const {
    if(transformers_fused_norm)return transformers_rms_norm(x,residual,weight,epsilon);
    if(exact_decoder_math())return decoder_exact_rms_norm(x,residual,weight,epsilon);
    return fused_add_rms_norm(x,residual,weight,epsilon);
  }
#endif
  at::Tensor activate(const at::Tensor& x) const {
    if (activation == "silu" || activation == "swish") return at::silu(x);
    if (activation == "relu") return at::relu(x);
    if (activation == "gelu") return at::gelu(x, "tanh");
    if (activation == "gelu_new" || activation == "gelu_pytorch_tanh" || activation == "gelu_fast") return at::gelu(x, "tanh");
    TORCH_CHECK(false, "Unsupported decoder activation: ", activation);
  }
  at::Tensor rotate(const at::Tensor& x, const at::Tensor& cos, const at::Tensor& sin) const {
    const auto half = head_dim / 2;
    auto a = x.narrow(-1, 0, half), b = x.narrow(-1, half, half);
    return at::cat({a * cos - b * sin, b * cos + a * sin}, -1);
  }
  std::vector<std::string> array_items(const std::string& key) const {
    auto value=options.string(key);
    value.erase(std::remove_if(value.begin(),value.end(),[](unsigned char c){return std::isspace(c);}),value.end());
    TORCH_CHECK(value.size()>=2&&value.front()=='['&&value.back()==']',"UltraBERT requires per-layer ",key);
    std::vector<std::string> items;std::istringstream stream(value.substr(1,value.size()-2));std::string item;
    while(std::getline(stream,item,',')) {TORCH_CHECK(!item.empty(),"Invalid UltraBERT ",key);items.push_back(item);}
    TORCH_CHECK(items.size()==static_cast<size_t>(layer_count)&&value[value.size()-2]!=',',"UltraBERT ",key," must have one entry per layer");
    return items;
  }
  void load_ultrabert_attention() {
    TORCH_CHECK(!options.boolean("attention_bias",false)&&!options.boolean("mlp_bias",false),"UltraBERT requires bias-free projections");
    TORCH_CHECK(!options.boolean("is_causal",false),"UltraBERT requires bidirectional attention");
    auto bases=array_items("rope_theta"),windows=array_items("attention_windows");
    auto maximum=options.integer("max_position_embeddings",32768);
    TORCH_CHECK(maximum>0,"UltraBERT position limit must be positive");
    std::map<double,int64_t> indices;
    auto cpu=at::TensorOptions().dtype(at::kFloat);
    for(int64_t i=0;i<layer_count;++i) {
      size_t used=0;auto theta=std::stod(bases[i],&used);
      TORCH_CHECK(used==bases[i].size()&&std::isfinite(theta)&&theta>0,"Invalid UltraBERT rotary base");
      auto it=indices.find(theta);
      if(it==indices.end()) {
        std::vector<float> inverse(head_dim/2);
        for(int64_t j=0;j<head_dim/2;++j)inverse[j]=1.f/std::pow(static_cast<float>(theta),static_cast<float>(2*j)/head_dim);
        auto inv=at::from_blob(inverse.data(),{head_dim/2},cpu).clone().to(device);
        auto angles=at::arange(maximum,inv.options()).unsqueeze(1)*inv.unsqueeze(0);
        it=indices.emplace(theta,rotary_caches.size()).first;
        rotary_caches.emplace_back(angles.cos().to(dtype),angles.sin().to(dtype));
      }
      rotary_indices.push_back(it->second);
      int64_t window=-1;
      if(windows[i]!="null") {
        used=0;window=std::stoll(windows[i],&used);
        TORCH_CHECK(used==windows[i].size()&&window>0,"UltraBERT attention windows must be positive or null");
      }
      attention_windows.push_back(window);
    }
  }
  void load_rope() {
    auto theta = options.real("rope_theta", 10000.0);
    auto scale = options.real("rope_scaling.factor", 1.0);
    auto kind = options.string("rope_scaling.rope_type", options.string("rope_scaling.type", ""));
    // Candle's dense Qwen2/Qwen3 layers use unscaled RoPE even when a
    // rope_scaling field is present in the checkpoint configuration.
    const bool qwen=family=="qwen2"||family=="qwen3"||family=="qwen3_moe";
    if(qwen){kind.clear();scale=1.;}
    const auto yarn = options.string("rope_parameters.rope_type", "") == "yarn";
    if (yarn) theta = options.real("rope_parameters.rope_theta", theta);
    TORCH_CHECK(theta > 1 && scale >= 1, "Invalid decoder RoPE parameters");
    std::vector<float> inverse(head_dim / 2);
    double attention_scale = 1.;
    for (int64_t i = 0; i < head_dim / 2; ++i) {
      float inv = 1.f / std::pow(static_cast<float>(theta), static_cast<float>(2 * i) / head_dim);
      if (kind == "llama3" || (!qwen&&options.values.count("rope_scaling.high_freq_factor"))) {
        // Candle computes Llama3 scaling in F32, including PI and each
        // smoothing stage; F64 changes long-context BF16 rotary tables.
        const float original = options.real("rope_scaling.original_max_position_embeddings", 8192);
        const float low = options.real("rope_scaling.low_freq_factor", 1), high = options.real("rope_scaling.high_freq_factor", 4);
        TORCH_CHECK(high > low && original > 0, "Invalid Llama3 RoPE scaling");
        const float wavelength = 2.f * static_cast<float>(M_PI) / inv;
        if (wavelength > original / low) inv /= static_cast<float>(scale);
        else if (wavelength >= original / high) {
          const float smooth = (original / wavelength - low) / (high - low);
          const float inv_scaled = inv / static_cast<float>(scale);
          inv = (1.f - smooth) * inv_scaled + smooth * inv;
        }
      } else if (!kind.empty()) {
        // Match Candle's two-parameter NTK RoPE implementation.
        inv = 1.f / std::pow(static_cast<float>(theta * scale), static_cast<float>(2 * i) / head_dim)
          / std::pow(scale, 2.0 / head_dim);
      }
      if (yarn) {
        auto factor = options.real("rope_parameters.factor", 1), original = options.real("rope_parameters.original_max_position_embeddings", 0);
        auto fast = options.real("rope_parameters.beta_fast", 32), slow = options.real("rope_parameters.beta_slow", 1);
        TORCH_CHECK(factor >= 1 && original > 0 && fast > 0 && slow > 0, "Invalid Ministral YaRN parameters");
        auto correction = [&](double r) { return head_dim * std::log(original / (r * 2 * M_PI)) / (2 * std::log(theta)); };
        auto low = std::max(0., std::floor(correction(fast))), high = std::min(double(head_dim - 1), std::ceil(correction(slow)));
        if (low == high) high += 0.001;
        auto extrapolation = 1. - std::clamp((i - low) / (high - low), 0., 1.);
        inv = inv * ((1 - extrapolation) / factor + extrapolation);
        auto m = options.real("rope_parameters.mscale", 0), all = options.real("rope_parameters.mscale_all_dim", 0);
        auto mscale = [&](double m) { return 1. + 0.1 * m * std::log(factor); };
        attention_scale = m != 0 && all != 0 ? mscale(m) / mscale(all) : mscale(1.);
      }
      inverse[i] = inv;
    }
    auto cpu = at::TensorOptions().dtype(at::kFloat);
    auto inv = at::from_blob(inverse.data(), {head_dim / 2}, cpu).clone().to(device);
    auto maximum = options.integer("max_position_embeddings", 32768);
    TORCH_CHECK(maximum > 0, "max_position_embeddings must be positive");
    auto angles = at::arange(maximum, inv.options()).unsqueeze(1) * inv.unsqueeze(0);
    cos_cache = (at::cos(angles) * attention_scale).to(dtype);
    sin_cache = (at::sin(angles) * attention_scale).to(dtype);
    if (yarn) {
      auto beta = options.real("rope_parameters.llama_4_scaling_beta", 0);
      auto original = options.real("rope_parameters.original_max_position_embeddings", 1);
      // Candle uses integer position/original division in the query scaling cache.
      query_scale = (1 + beta * at::log1p(at::floor(at::arange(maximum, inv.options()) / original))).to(dtype);
    }
  }
  at::Tensor attend_full(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const PackedInput& input,int64_t left_window,int64_t right_window) const {
    if (device.is_cuda()) {
      if(!cudnn_varlen) {
        auto fa4=infinity_fa4(q,k,v,input.cumulative,input.cumulative,input.max_sequence,
          1./std::sqrt(static_cast<double>(head_dim)),causal,left_window,right_window);
        if(fa4.defined())return fa4;
      }
      // cuDNN packed attention cannot express the local window here. It is
      // equivalent only when every sequence fits inside the configured window.
      const bool window_inactive=(left_window<0||left_window>=input.max_sequence-1)
        &&(right_window<0||(causal&&right_window==0)||right_window>=input.max_sequence-1);
      bool use_cudnn=cudnn_varlen&&input.max_sequence>=512&&window_inactive;
#ifdef TEI_TORCH_CUDA_KERNELS
      use_cudnn=use_cudnn&&allow_cudnn_attention();
      if(!use_cudnn&&exact_decoder_math()&&torch_flash_fma_supported(q,k,v,causal,left_window,right_window))
        return torch_flash_fma(q,k,v,input.cumulative,input.max_sequence,
          1./std::sqrt(static_cast<double>(head_dim)),causal,left_window,right_window);
#endif
      if(use_cudnn)
        return std::get<0>(at::_cudnn_attention_forward(q,k,v,std::nullopt,
          input.cumulative,input.cumulative,input.max_sequence,input.max_sequence,
          true,0.,causal,false));
      return std::get<0>(at::_flash_attention_forward(q, k, v,
        input.cumulative, input.cumulative, input.max_sequence, input.max_sequence,
        0., causal, false, std::nullopt, left_window, right_window));
    }
    std::vector<at::Tensor> outputs;
    for (int64_t row = 0; row < input.batch; ++row) {
      auto length = input.offsets[row + 1] - input.offsets[row];
      auto slice = [&](const at::Tensor& t) { return t.narrow(0, input.offsets[row], length).transpose(0, 1).unsqueeze(0); };
      auto qs = slice(q), ks = slice(k), vs = slice(v);
      if (kvheads != heads) { ks = at::repeat_interleave(ks, heads / kvheads, 1); vs = at::repeat_interleave(vs, heads / kvheads, 1); }
      std::optional<at::Tensor> mask;
      if (left_window >= 0 || right_window >= 0) {
        auto indices = at::arange(length, q.options().dtype(at::kLong));
        auto delta = indices.unsqueeze(1) - indices.unsqueeze(0);
        auto allowed = at::ones({length, length}, q.options().dtype(at::kBool));
        if (left_window >= 0) allowed = allowed & (delta <= left_window);
        if (right_window >= 0) allowed = allowed & (delta >= -right_window);
        if (causal) allowed = allowed & (delta >= 0);
        mask = allowed;
      }
      outputs.push_back(at::scaled_dot_product_attention(qs, ks, vs, mask, 0., causal && !mask.has_value()).squeeze(0).transpose(0, 1));
    }
    return at::cat(outputs, 0);
  }
  at::Tensor attend_cached(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,const PackedInput& input,int64_t layer_index,const DecoderLayer& layer) const {
    auto rows=radix_unfold_qkv(q,k,v,input);
    if(std::none_of(input.prefix->hits.begin(),input.prefix->hits.end(),[](auto hit){return hit!=0;})) {
      // Cold capture must retain the model's original attention provider and
      // arithmetic, including configured cuDNN varlen execution.
      for(int64_t row=0;row<input.batch;++row)if(input.prefix->writers[row]) {
        auto saved=int64_t(input.prefix->writers[row]->identity.tokens.size());auto start=input.offsets[row];
        input.prefix->store(row,layer_index,PrefixLayer{rows[1].narrow(0,start,saved),rows[2].narrow(0,start,saved),{}, {}});
      }
      return radix_fold(attend_full(rows[0],rows[1],rows[2],input,layer.left_window,layer.right_window),input);
    }
    bool exact=false;
    bool allow_cudnn=true;
#ifdef TEI_TORCH_CUDA_KERNELS
    exact=exact_decoder_math();
    allow_cudnn=allow_cudnn_attention();
#endif
    auto full_length=input.prefix->original.max_sequence;
    const bool window_inactive=(layer.left_window<0||layer.left_window>=full_length-1)
      &&(layer.right_window<0||layer.right_window==0||layer.right_window>=full_length-1);
    const bool use_cudnn=allow_cudnn&&cudnn_varlen&&full_length>=512&&window_inactive;
    return radix_fold(prefix_attention(rows[0],rows[1],rows[2],input,layer_index,
      1./std::sqrt(static_cast<double>(head_dim)),layer.left_window,layer.right_window,exact,true,use_cudnn),input);
  }
  at::Tensor attend(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,const PackedInput& input,int64_t layer_index,const DecoderLayer& layer) const {
    if(input.prefix)return attend_cached(q,k,v,input,layer_index,layer);
    if(!radix_enabled(input)) {
      if(input.capture_prefix)prefix_capture_attention(input,layer_index,k,v);
      return attend_full(q,k,v,input,layer.left_window,layer.right_window);
    }
    auto unfolded=radix_unfold_qkv(q,k,v,input);
    if(input.capture_prefix)prefix_capture_attention(input,layer_index,unfolded[1],unfolded[2]);
    auto full=attend_full(unfolded[0],unfolded[1],unfolded[2],input,layer.left_window,layer.right_window);
    return radix_fold(full,input);
  }
  at::Tensor mlp(const at::Tensor& x, const DecoderLayer& layer,const PackedInput& input,
      const at::Tensor& row_scales={}) const {
    if (!layer.router.defined()) {
#ifdef TEI_TORCH_CUDA_KERNELS
      auto both = layer.fp8_gate_up ? (row_scales.defined()
        ? layer.fp8_gate_up->forward_quantized(x,row_scales,layer.gate_up_bias)
        : layer.fp8_gate_up->forward(x,layer.gate_up_bias))
        : transformers_math()?separate_linear(x,layer.gate_up,layer.gate_up_bias,{intermediate,intermediate})
        : cached_prefix_linear(x,layer.gate_up,layer.gate_up_bias,input);
#else
      auto both = at::linear(x,layer.gate_up,layer.gate_up_bias);
#endif
      auto width = both.size(-1) / 2;
#ifdef TEI_TORCH_CUDA_KERNELS
      if(both.is_cuda()&&!transformers_math()) {
        int code=activation=="silu"||activation=="swish"?5:activation=="relu"?3:4;
        if(layer.fp8_down && code==5 && width<=32768 && both.is_contiguous()) {
          auto quantized=fp8_swiglu_quantize(both);
          return layer.fp8_down->forward_quantized(quantized.first,quantized.second,layer.down_bias);
        }
        auto activated=fused_gated_activation(both,code);
        return layer.fp8_down ? layer.fp8_down->forward(activated,layer.down_bias)
          : cached_prefix_linear(activated,layer.down,layer.down_bias,input);
      }
#endif
      return at::linear(activate(both.narrow(-1, 0, width)) * both.narrow(-1, width, width), layer.down, layer.down_bias);
    }
    auto logits=at::linear(x, layer.router).to(at::kFloat);
    if(top_k==8&&layer.grouped_down.defined()) {
      auto native=routed_moe_cuda(x,logits,layer.grouped_gate_up.transpose(-2,-1),
        layer.grouped_down.transpose(-2,-1),renormalize);
      if(native.defined())return native;
    }
    at::Tensor probabilities;
#ifdef TEI_TORCH_CUDA_KERNELS
    if(qwen_half_exact&&family=="qwen3_moe"&&logits.is_cuda())
      probabilities=decoder_half_moe_probabilities(logits);
    else
#endif
      probabilities=at::softmax(logits,-1);
    // Stable sorting gives Candle's lower-expert-index tie break. Routing remains on-device.
    auto ids = at::argsort(probabilities, true, -1, true).narrow(-1, 0, top_k);
    auto routing = probabilities.gather(-1, ids);
#ifdef TEI_TORCH_CUDA_KERNELS
    if(qwen_half_exact&&family=="qwen3_moe"&&routing.is_cuda())
      routing=decoder_half_moe_route_weights(routing,renormalize);
    else
#endif
    {
      if(renormalize)routing=routing/routing.sum(-1,true);
      routing=routing.to(dtype);
    }
    auto result = at::zeros_like(x);
    if (x.is_cuda() && layer.grouped_down.defined()) {
      auto experts = layer.grouped_down.size(0);
      auto flattened = ids.flatten();
      auto order = at::argsort(flattened, true, 0, false);
      auto tokens = at::floor_divide(order, top_k);
      auto counts = at::zeros({experts}, x.options().dtype(at::kInt));
      counts.scatter_add_(0, flattened, at::ones_like(flattened, flattened.options().dtype(at::kInt)));
      auto ends = counts.cumsum(0, at::kInt);
      auto selected = x.index_select(0, tokens);
      auto both = at::_grouped_mm(selected, layer.grouped_gate_up, ends);
      auto width = both.size(-1) / 2;
      at::Tensor activated;
#ifdef TEI_TORCH_CUDA_KERNELS
      activated=fused_gated_activation(both,0);
#else
      activated=at::silu(both.narrow(-1,0,width))*both.narrow(-1,width,width);
#endif
      auto outputs = at::_grouped_mm(activated, layer.grouped_down, ends);
      auto route = routing.flatten().index_select(0, order).unsqueeze(1);
      result.index_add_(0, tokens, outputs * route);
      return result;
    }
    // The original Half reference visits experts in ascending order and keeps
    // token order within each expert. Copy only the real top8 indices once,
    // then upload their stable bucket order once, avoiding128 nonzero syncs.
    at::Tensor bucket_order;
    std::vector<int64_t> bucket_offsets;
    if(qwen_half_exact&&family=="qwen3_moe"&&x.is_cuda()) {
      const auto experts=static_cast<int64_t>(layer.expert_down.size());
      auto host_ids=ids.contiguous().to(at::kCPU);
      const auto* ids_ptr=host_ids.const_data_ptr<int64_t>();
      std::vector<int64_t> counts(experts,0);
      for(int64_t route=0;route<host_ids.numel();++route)++counts[ids_ptr[route]];
      bucket_offsets.resize(experts+1,0);
      for(int64_t expert=0;expert<experts;++expert)
        bucket_offsets[expert+1]=bucket_offsets[expert]+counts[expert];
      auto next=bucket_offsets;
      auto host_order=at::empty({2,host_ids.numel()},host_ids.options());
      auto* order_ptr=host_order.mutable_data_ptr<int64_t>();
      for(int64_t route=0;route<host_ids.numel();++route) {
        const auto position=next[ids_ptr[route]]++;
        order_ptr[position]=route/top_k;
        order_ptr[host_ids.numel()+position]=route%top_k;
      }
      bucket_order=host_order.to(x.device()).transpose(0,1);
    }
    for (int64_t expert = 0; expert < static_cast<int64_t>(layer.expert_down.size()); ++expert) {
      at::Tensor selected;
      if(bucket_order.defined()) {
        auto count=bucket_offsets[expert+1]-bucket_offsets[expert];
        if(!count)continue;
        selected=bucket_order.narrow(0,bucket_offsets[expert],count);
      } else selected=at::nonzero(ids==expert);
      auto tokens = selected.select(1, 0), slots = selected.select(1, 1);
      // Other reference paths retain valid empty expert GEMMs.
      auto input = x.index_select(0, tokens);
      auto both = at::linear(input, layer.expert_gate_up[expert]);
      auto width = both.size(-1) / 2;
      at::Tensor activated;
#ifdef TEI_TORCH_CUDA_KERNELS
      // Original Candle Half MoE rounds exp, denominator, SiLU and product
      // separately in model dtype, including Half exp overflow.
      if(qwen_half_exact && family=="qwen3_moe" && both.is_cuda())
        activated=fused_gated_activation(both,5);
      else
#endif
        activated=at::silu(both.narrow(-1,0,width))*both.narrow(-1,width,width);
      auto output = at::linear(activated, layer.expert_down[expert]);
      auto route = routing.index_select(0, tokens).gather(1, slots.unsqueeze(1));
      result.index_add_(0, tokens, output * route);
    }
    return result;
  }
public:
  Decoder(const Options& cfg, Weights& tensors, c10::Device dev, at::ScalarType type)
    : options(cfg), weights(std::move(tensors)), device(dev), dtype(type), family(cfg.string("model_type", "")),
      activation(cfg.string("hidden_act", "silu")), hidden(cfg.integer("hidden_size", 0)),
      heads(cfg.integer("num_attention_heads", 0)), kvheads(cfg.integer("num_key_value_heads", heads)),
      head_dim(cfg.integer("head_dim", heads ? hidden / heads : 0)), intermediate(cfg.integer("intermediate_size", 0)),
      layer_count(cfg.integer("num_hidden_layers", 0)), epsilon(cfg.real("rms_norm_eps", 1e-6)),
      causal(!cfg.boolean("use_bidirectional_attention", false)), renormalize(cfg.boolean("norm_topk_prob", true)),
      top_k(cfg.integer("num_experts_per_tok", 0)) {
    transformers_fused_norm=device.is_cuda()&&transformers_math();
    if (family == "ultrabert") causal=false;
    if (family == "ministral3") family = "mistral";
    if (family == "llama_bidirec") family = "llama";
    // An explicit saved flag applies to every decoder family, including
    // bidirectional Ministral3 embedding fine-tunes, after family normalization.
    if (family == "qwen2" || cfg.values.count("is_causal")) causal = cfg.boolean("is_causal", true);
  }
  void ready() override {
    fp8_dynamic=options.boolean("_fp8_dynamic",false);
    const auto fp8_scope=options.string("_fp8_scope",fp8_dynamic?"all":"mlp");
    TORCH_CHECK(fp8_scope=="mlp"||fp8_scope=="all","FP8 scope must be mlp or all");
    fp8_all_linear=fp8_scope=="all";
    TORCH_CHECK(!fp8_all_linear||fp8_dynamic,"All-linear FP8 requires dynamic FP8 enabled");
    if(fp8_dynamic) {
      TORCH_CHECK(!transformers_math(),"Dynamic FP8 requires native decoder arithmetic");
      TORCH_CHECK(device.is_cuda()&&(dtype==at::kHalf||dtype==at::kBFloat16),
        "Dynamic FP8 decoder MLPs require CUDA float16/bfloat16");
      TORCH_CHECK((family=="llama"||family=="mistral"||family=="qwen2"||family=="qwen3")
        &&options.integer("num_experts",0)==0,
        "Dynamic FP8 decoder MLPs support dense Llama/Mistral/Qwen2/Qwen3 only; routed experts are unsupported");
      for(const auto& [key,value]:options.values)
        TORCH_CHECK(key!="quantization_config"&&key.rfind("quantization_config.",0)!=0,
          "Dynamic FP8 requires an unquantized checkpoint without supplied quantization scales");
#ifndef TEI_TORCH_CUDA_KERNELS
      TORCH_CHECK(false,"Dynamic FP8 decoder MLPs require native CUDA kernels");
#endif
    }
    TORCH_CHECK(activation=="silu"||activation=="swish"||activation=="relu"||activation=="gelu"||activation=="gelu_new"||activation=="gelu_fast"||activation=="gelu_pytorch_tanh", "Unsupported decoder activation: ",activation);
    TORCH_CHECK(hidden > 0 && heads > 0 && kvheads > 0 && head_dim > 0 && head_dim % 2 == 0 && heads % kvheads == 0 && layer_count > 0,
      "Invalid decoder dimensions");
    TORCH_CHECK(device.is_cpu() || device.is_cuda(), "Packed decoder attention currently requires CPU or CUDA");
    if (device.is_cuda()) TORCH_CHECK((dtype == at::kHalf || dtype == at::kBFloat16) && head_dim % 8 == 0 && head_dim <= 256,
      "CUDA varlen attention requires half/bfloat16 and head dimension divisible by eight, at most 256");
    TORCH_CHECK(options.string("quantization_config.quant_method", "").empty(), "Quantized decoder checkpoints are not yet supported");
    if(family=="qwen3_moe") for(const auto& [key,value]:options.values)
      TORCH_CHECK(key!="rope_scaling"&&key.rfind("rope_scaling.",0)!=0,"Scaled Qwen3 MoE RoPE is unsupported by Candle and this backend");
    TORCH_CHECK(!options.boolean("use_sliding_window", false) || family == "mistral", "Qwen sliding-window configuration is unsupported by Candle and this backend");
#ifdef TEI_TORCH_CUDA_KERNELS
    if(device.is_cuda()) {
      const auto* properties=at::cuda::getDeviceProperties(device.index());
      auto major=properties->major;
      // Other architectures retain installed Torch arithmetic until qualified.
      qwen_half_exact=!fp8_dynamic&&dtype==at::kHalf&&major==9&&properties->minor==0
        &&((family=="qwen3"&&hidden==1024&&head_dim==128)
          ||(family=="qwen2"&&hidden==896&&head_dim==64)
          ||(family=="qwen3_moe"&&hidden==2048&&head_dim==128&&options.integer("num_experts",0)==128&&top_k==8));
      cudnn_varlen=options.boolean("_cudnn_varlen",false)&&(major==9||major==10)
        &&at::detail::getCUDAHooks().versionRuntimeCuDNN()>=91800;
    }
#endif
    prefix = family=="ultrabert"&&weights.count("ultrabert.embed_tokens.weight")?"ultrabert.":weights.count("model.embed_tokens.weight") ? "model." : "";
    if(family=="ultrabert")load_ultrabert_attention();
    embedding = checked_weight("embed_tokens.weight", {options.integer("vocab_size", 0), hidden});
    final_norm = checked_weight("norm.weight", {hidden});
    auto window = options.integer("sliding_window", -1);
    if (window > 0 && family != "qwen2" && family != "qwen3" && family != "qwen3_moe") {
      left_window = causal ? window - 1 : window; right_window = causal ? 0 : -1;
    }
    std::unordered_set<int64_t> dense_layers;
    auto list = options.string("mlp_only_layers", "");
    for (auto& c : list) if (c == '[' || c == ']' || c == ',') c = ' ';
    std::istringstream stream(list); int64_t index; while (stream >> index) dense_layers.insert(index);
    for (const auto& [key, value] : options.values)
      if (key.rfind("mlp_only_layers.", 0) == 0) dense_layers.insert(std::stoll(value));
    auto experts = options.integer("num_experts", 0), sparse_step = options.integer("decoder_sparse_step", 1);
    if (experts) TORCH_CHECK(sparse_step > 0 && top_k > 0 && top_k <= experts && options.integer("moe_intermediate_size", 0) > 0 && activation == "silu",
      "Invalid Qwen3 MoE routing dimensions or activation");
    layers.clear();
    for (int64_t i = 0; i < layer_count; ++i) {
      auto p = "layers." + std::to_string(i) + ".";
      DecoderLayer layer;
      layer.left_window=left_window;layer.right_window=right_window;
      if(family=="ultrabert") {layer.rotary_index=rotary_indices[i];layer.left_window=layer.right_window=attention_windows[i];}
      layer.input_norm = checked_weight(p + "input_layernorm.weight", {hidden});
      layer.post_norm = checked_weight(p + "post_attention_layernorm.weight", {hidden});
      std::vector<at::Tensor> qkv, bias;
      for (auto name : {"q_proj", "k_proj", "v_proj"}) {
        auto width = std::string(name) == "q_proj" ? heads * head_dim : kvheads * head_dim;
        auto name_full = p + "self_attn." + name;
        qkv.push_back(checked_weight(name_full + ".weight", {width, hidden}));
        auto b = optional(name_full + ".bias"); if (b.defined()) bias.push_back(b);
      }
      layer.qkv = at::cat(qkv, 0);
      TORCH_CHECK(bias.empty() || bias.size() == 3, "Partial QKV biases are unsupported");
      if (!bias.empty()) layer.qkv_bias = at::cat(bias, 0);
      layer.out = checked_weight(p + "self_attn.o_proj.weight", {hidden, heads * head_dim});
      layer.out_bias = optional(p + "self_attn.o_proj.bias");
      if (family == "qwen3" || family == "qwen3_moe") {
        layer.q_norm = checked_weight(p + "self_attn.q_norm.weight", {head_dim});
        layer.k_norm = checked_weight(p + "self_attn.k_norm.weight", {head_dim});
      }
      if (experts > 0 && !dense_layers.count(i) && (i + 1) % sparse_step == 0) {
        layer.router = checked_weight(p + "mlp.gate.weight", {experts, hidden});
        auto fused = optional(p + "mlp.experts.gate_up_proj");
        auto expert_width = options.integer("moe_intermediate_size", 0);
        if (fused.defined()) {
          auto downs = checked_weight(p + "mlp.experts.down_proj", {experts, hidden, expert_width});
          TORCH_CHECK(fused.sizes() == at::IntArrayRef({experts, 2 * expert_width, hidden}), "Invalid fused MoE expert dimensions");
          for (int64_t e = 0; e < experts; ++e) { layer.expert_gate_up.push_back(fused.select(0, e)); layer.expert_down.push_back(downs.select(0, e)); }
          if (device.is_cuda() && dtype==at::kBFloat16 && hidden%8==0 && expert_width%8==0) {
            layer.grouped_gate_up=fused.transpose(-2,-1);layer.grouped_down=downs.transpose(-2,-1);
          }
        } else for (int64_t e = 0; e < experts; ++e) {
          auto ep = p + "mlp.experts." + std::to_string(e) + ".";
          layer.expert_gate_up.push_back(at::cat({checked_weight(ep + "gate_proj.weight", {expert_width, hidden}), checked_weight(ep + "up_proj.weight", {expert_width, hidden})}, 0));
          layer.expert_down.push_back(checked_weight(ep + "down_proj.weight", {hidden, expert_width}));
        }
      } else {
        layer.gate_up = at::cat({checked_weight(p + "mlp.gate_proj.weight", {intermediate, hidden}), checked_weight(p + "mlp.up_proj.weight", {intermediate, hidden})}, 0);
        auto gb = optional(p + "mlp.gate_proj.bias"), ub = optional(p + "mlp.up_proj.bias");
        TORCH_CHECK(gb.defined() == ub.defined(), "Partial gated MLP bias");
        if (gb.defined()) layer.gate_up_bias = at::cat({gb, ub}, 0);
        layer.down = checked_weight(p + "mlp.down_proj.weight", {hidden, intermediate});
        layer.down_bias = optional(p + "mlp.down_proj.bias");
#ifdef TEI_TORCH_CUDA_KERNELS
        if(fp8_dynamic) {
          // Shape checks above precede one-time row quantization. Bias and the
          // gated activation retain their existing model-dtype arithmetic.
          layer.fp8_gate_up=std::make_unique<Fp8Linear>(layer.gate_up);
          layer.fp8_down=std::make_unique<Fp8Linear>(layer.down);
          if(fp8_all_linear) {
            layer.fp8_qkv=std::make_unique<Fp8Linear>(layer.qkv);
            layer.fp8_out=std::make_unique<Fp8Linear>(layer.out);
            layer.qkv=at::Tensor();layer.out=at::Tensor();
            for(auto name:{"q_proj.weight","k_proj.weight","v_proj.weight","o_proj.weight"})
              weights.erase(prefix+p+"self_attn."+name);
          }
          layer.gate_up=at::Tensor();layer.down=at::Tensor();
          // Quantized caches own all required weight storage. Release the
          // original checkpoint FFN matrices before the next layer is cached.
          for(auto name:{"gate_proj.weight","up_proj.weight","down_proj.weight"})
            weights.erase(prefix+p+"mlp."+name);
        }
#endif
      }
      if(family=="ultrabert")TORCH_CHECK(!layer.qkv_bias.defined()&&!layer.out_bias.defined()&&!layer.gate_up_bias.defined()&&!layer.down_bias.defined(),"Unexpected UltraBERT projection bias");
      if (!layer.expert_gate_up.empty() && device.is_cuda() && dtype == at::kBFloat16 && hidden%8==0 && layer.expert_down[0].size(-1)%8==0) {
        if (!layer.grouped_gate_up.defined()) {
          layer.grouped_gate_up = at::stack(layer.expert_gate_up).transpose(-2, -1);
          layer.grouped_down = at::stack(layer.expert_down).transpose(-2, -1);
        }
        layer.expert_gate_up.clear();layer.expert_down.clear();
      }
      // The decoder owns this map. Cached expert tensors above keep every
      // required storage alive; release original per-expert checkpoint tensors
      // layer by layer so a 30B MoE never needs two full copies during fusion.
      if (layer.router.defined()) {
        const auto expert_prefix = prefix + p + "mlp.experts.";
        for (auto it = weights.begin(); it != weights.end();) {
          if (it->first.rfind(expert_prefix, 0) == 0) it = weights.erase(it);
          else ++it;
        }
      }
      layers.push_back(std::move(layer));
    }
    if (!causal && weights.count("linear.weight")) projection = weights.at("linear.weight");
    else if (options.boolean("use_linear_output_projection", false)) {
      projection = checked_weight("linear_output_projection.weight", {options.integer("linear_output_size", 0), hidden});
      projection_bias = optional("linear_output_projection.bias");
    }
    if (projection.defined()) TORCH_CHECK(projection.dim() == 2 && projection.size(1) == hidden, "Invalid decoder output projection");
    auto architecture=options.string("architectures.0", "");
    auto supported=architecture=="LlamaForSequenceClassification"||architecture=="Qwen2ForSequenceClassification"||architecture=="Qwen3ForSequenceClassification"||architecture=="Qwen3MoeForSequenceClassification";
    const bool token_head=architecture=="LlamaForTokenClassification"||architecture=="Qwen2ForTokenClassification"||architecture=="Qwen3ForTokenClassification";
    if(token_head) {
      TORCH_CHECK(!options.values.count("architectures.1")&&
        architecture==(family=="llama"?"LlamaForTokenClassification":family=="qwen2"?"Qwen2ForTokenClassification":family=="qwen3"?"Qwen3ForTokenClassification":""),
        "Token classification requires an unambiguous matching Llama, Qwen2 or Qwen3 architecture");
      TORCH_CHECK(!projection.defined(),"Decoder token classification does not support embedding output projections");
      token_classifier=true;
    }
    if ((supported||token_head) && !options.values.count("architectures.1")) {
      auto it=weights.find("score.weight");
      TORCH_CHECK(it!=weights.end(),"Missing decoder classifier score.weight");
      score=it->second;
      int64_t mapped_labels=0;for(const auto& [key,value]:options.values)if(key.rfind("id2label.",0)==0)++mapped_labels;
      auto labels=mapped_labels?mapped_labels:options.integer("num_labels",2);
      TORCH_CHECK(labels>0&&(!options.values.count("num_labels")||options.integer("num_labels")==labels)&&score.sizes()==at::IntArrayRef({labels,hidden}),"Invalid decoder classifier label dimensions");
      auto bias=weights.find("score.bias");
      TORCH_CHECK(!token_head||((bias!=weights.end())==options.boolean("token_classification_bias",true)),
        "Decoder token classifier score.bias must match token_classification_bias");
      if(bias!=weights.end()){score_bias=bias->second;TORCH_CHECK(score_bias.sizes()==at::IntArrayRef({labels}),"Invalid decoder score.bias");}
    }
    if(family=="ultrabert") {
      auto arch=options.string("architectures.0");
      const bool sequence=arch=="UltrabertForSequenceClassification",token=arch=="UltrabertForTokenClassification";
      TORCH_CHECK(!options.boolean("_classifier",false)||(sequence||token),"UltraBERT classification requires a sequence or token classification checkpoint");
      TORCH_CHECK(!(sequence||token)||options.boolean("_classifier",false),"UltraBERT embedding requests require an embedding checkpoint");
      TORCH_CHECK(options.string("_pooling")!="splade","UltraBERT does not support SPLADE pooling");
      if(sequence||token) {
        TORCH_CHECK(!options.values.count("architectures.1"),"Ambiguous UltraBERT classifier architecture");
        auto w=weights.find("classifier.weight"),b=weights.find("classifier.bias");
        int64_t mapped=0;for(const auto& [key,value]:options.values)if(key.starts_with("id2label."))++mapped;
        auto labels=mapped?mapped:options.integer("num_labels",2);
        TORCH_CHECK(labels>0&&(!options.values.count("num_labels")||options.integer("num_labels")==labels),"UltraBERT num_labels and id2label must agree and be nonzero");
        TORCH_CHECK(w!=weights.end()&&b!=weights.end(),"Missing UltraBERT classifier weight or bias");
        score=w->second;score_bias=b->second;token_classifier=token;
        TORCH_CHECK(score.sizes()==at::IntArrayRef({labels,hidden})&&score_bias.sizes()==at::IntArrayRef({labels}),"Invalid UltraBERT classifier dimensions");
      }
    } else load_rope();
    // All runtime weights are captured in layers; release unfused checkpoint tensors.
    weights.clear();
  }
  bool supports_radix() const override {return causal;}
  int64_t prefix_layers() const override {return layer_count;}
  bool supports_prefix_cache(int operation) const override {return causal&&(operation==0||(operation==1&&!token_classifier&&classification_width()>0));}
  uint64_t prefix_payload_bytes(int64_t tokens) const override {
    auto bytes=prefix_storage_bytes(tokens,output_width(),embedding.element_size());
    const auto kv=prefix_storage_bytes(tokens,2*kvheads*head_dim,embedding.element_size());
    for(int64_t layer=0;layer<layer_count;++layer)bytes=prefix_storage_sum(bytes,kv);
    return bytes;
  }
  int64_t output_width() const override { return projection.defined() ? projection.size(0) : hidden; }
  int64_t classification_width() const override {return score.defined()?score.size(0):0;}
  at::Tensor predict(const PackedInput& input,bool tokenwise) const override {
    TORCH_CHECK(!tokenwise||token_classifier,"Decoder sequence classifier does not implement token classification");
    TORCH_CHECK(score.defined(),"Decoder checkpoint has no supported classification score head");
    for(int64_t row=0;row<input.batch;++row)TORCH_CHECK(input.offsets[row+1]>input.offsets[row],"Sequence classification requires nonempty inputs");
    if(tokenwise)return at::linear(forward(input),score,score_bias);
    return classify_hidden(input,forward(input));
  }
  at::Tensor classify_hidden(const PackedInput& input,const at::Tensor& hidden_states) const override {
    const auto& logical=prefix_logical_input(input);
    if(family=="ultrabert") {
      auto starts=input.cumulative.narrow(0,0,input.batch).to(at::kLong);
      return at::linear(hidden_states.index_select(0,starts),score,score_bias);
    }
#ifdef TEI_TORCH_CUDA_KERNELS
    if(hidden_states.is_cuda()&&hidden_states.dim()==2&&hidden_states.is_contiguous()
      &&(hidden_states.scalar_type()==at::kBFloat16||hidden_states.scalar_type()==at::kHalf||hidden_states.scalar_type()==at::kFloat)
      &&logical.ids.device()==hidden_states.device()&&logical.ids.scalar_type()==at::kLong
      &&logical.ids.dim()==1&&logical.ids.is_contiguous()
      &&logical.cumulative.device()==hidden_states.device()&&logical.cumulative.scalar_type()==at::kInt
      &&logical.cumulative.is_contiguous()) {
      TORCH_CHECK(hidden_states.size(-1)==score.size(1),"Classifier backbone output projection has incompatible width");
      return at::linear(decoder_packed_classifier_rows(hidden_states,logical.ids,logical.cumulative,
        logical.batch,options.integer("pad_token_id",-1),options.values.count("pad_token_id")!=0),score,score_bias);
    }
#endif
    auto starts=logical.cumulative.narrow(0,0,logical.batch).to(at::kLong);
    auto ends=logical.cumulative.narrow(0,1,logical.batch).to(at::kLong)-1;
    at::Tensor selected=ends;
    if(options.values.count("pad_token_id")) {
      auto rows=at::repeat_interleave(at::arange(logical.batch,logical.ids.options()),ends-starts+1,0,logical.ids.numel());
      auto positions=at::arange(logical.ids.numel(),logical.ids.options());
      auto candidates=at::where(logical.ids==options.integer("pad_token_id"),-1,positions);
      selected=at::full({logical.batch},-1,logical.ids.options());
      selected.scatter_reduce_(0,rows,candidates,"amax",true);
      selected=at::where(selected<0,starts,selected);
    }
    TORCH_CHECK(hidden_states.size(-1)==score.size(1),"Classifier backbone output projection has incompatible width");
    return at::linear(hidden_states.index_select(0,selected),score,score_bias);
  }
  at::Tensor forward(const PackedInput& input) const override {
    TORCH_CHECK(!radix_enabled(input)||supports_radix(),"Radix folding currently requires a causal text decoder");
    const auto& positions=radix_positions(input);
    if(family=="ultrabert")return forward_states(input,at::embedding(embedding,input.ids),{},{});
    auto compact=forward_states(input,at::embedding(embedding,radix_ids(input)),
      cos_cache.index_select(0,positions).unsqueeze(1),sin_cache.index_select(0,positions).unsqueeze(1));
    return prefix_restore(input,radix_unfold(compact,input));
  }
  at::Tensor multimodal_forward(const PackedInput& input,const at::Tensor& indices,const at::Tensor& visual,
    const at::Tensor& cosine,const at::Tensor& sine,const std::vector<at::Tensor>& deepstack) const {
    TORCH_CHECK(!radix_enabled(input),"Multimodal radix folding is not supported");
    TORCH_CHECK(family=="qwen3"||family=="mistral","Unsupported multimodal text family");
    auto h=at::embedding(embedding,input.ids);
    TORCH_CHECK(cosine.numel()==input.ids.numel()*(head_dim/2)&&sine.sizes()==cosine.sizes(),"Invalid multimodal RoPE geometry");
    TORCH_CHECK(cosine.device()==device&&sine.device()==device&&cosine.scalar_type()==dtype&&sine.scalar_type()==dtype,"Multimodal RoPE must match decoder device and dtype");
    TORCH_CHECK(deepstack.empty()||indices.defined(),"Deepstack features require visual token indices");
    TORCH_CHECK(deepstack.size()<=layers.size(),"More deepstack feature layers than text decoder layers");
    if(indices.defined()) {
      TORCH_CHECK(indices.dim()==1&&indices.scalar_type()==at::kLong&&indices.device()==device,"Visual indices must be a packed on-device INT64 tensor");
      TORCH_CHECK(visual.dim()==2&&visual.size(0)==indices.numel()&&visual.size(1)==hidden&&visual.device()==device&&visual.scalar_type()==dtype,"Invalid external visual token embeddings");
      h.index_copy_(0,indices,visual);
    }
    return forward_states(input,h,cosine.view({input.ids.numel(),1,head_dim/2}),sine.view({input.ids.numel(),1,head_dim/2}),indices,deepstack);
  }
  at::Tensor visual_forward(const PackedInput& input,const at::Tensor& indices,const at::Tensor& visual) const {
    return multimodal_forward(input,indices,visual,cos_cache.index_select(0,input.positions).unsqueeze(1),sin_cache.index_select(0,input.positions).unsqueeze(1),{});
  }
  at::Tensor forward_states(const PackedInput& input,at::Tensor h,const at::Tensor& cosine,
    const at::Tensor& sine,const at::Tensor& visual_indices=at::Tensor(),const std::vector<at::Tensor>& deepstack={}) const {
    at::Tensor residual;
    std::vector<std::pair<at::Tensor,at::Tensor>> selected_rotary;
    for(const auto& cache:rotary_caches)selected_rotary.emplace_back(
      cache.first.index_select(0,input.positions).unsqueeze(1),cache.second.index_select(0,input.positions).unsqueeze(1));
    size_t layer_index=0;
    for (const auto& layer : layers) {
      const auto& layer_cosine=layer.rotary_index<0?cosine:selected_rotary[layer.rotary_index].first;
      const auto& layer_sine=layer.rotary_index<0?sine:selected_rotary[layer.rotary_index].second;
      at::Tensor normalized,normalized_scales;
#ifdef TEI_TORCH_CUDA_KERNELS
      if (h.is_cuda()&&(!transformers_math()||transformers_fused_norm)) {
        if(layer.fp8_qkv && dtype==at::kBFloat16 && hidden==4096
          && !exact_decoder_math() && h.is_contiguous()
          && (!residual.defined()||residual.is_contiguous())) {
          auto [quantized,summed,scales]=fp8_add_rms_quantize(h,residual,layer.input_norm,epsilon);
          normalized=std::move(quantized);residual=std::move(summed);normalized_scales=std::move(scales);
        } else {
          auto pair=residual_norm(h,residual,layer.input_norm);
          normalized=pair.first;residual=pair.second;
        }
      } else
#endif
      normalized=norm(h,layer.input_norm);
      at::Tensor qkv;
#ifdef TEI_TORCH_CUDA_KERNELS
      if(layer.fp8_qkv)qkv=normalized_scales.defined()
        ? layer.fp8_qkv->forward_quantized(normalized,normalized_scales,layer.qkv_bias)
        : layer.fp8_qkv->forward(normalized,layer.qkv_bias);
      else
#endif
        qkv=transformers_math()?separate_linear(normalized,layer.qkv,layer.qkv_bias,{heads*head_dim,kvheads*head_dim,kvheads*head_dim}):at::linear(normalized,layer.qkv,layer.qkv_bias);
      qkv=qkv.view({h.size(0),heads+2*kvheads,head_dim});
      auto q = qkv.narrow(1, 0, heads), k = qkv.narrow(1, heads, kvheads), v = qkv.narrow(1, heads + kvheads, kvheads);
#ifdef TEI_TORCH_CUDA_KERNELS
      if (q.is_cuda() && layer.q_norm.defined() && head_dim==128) {
        if(input.prefix&&input.prefix->collect_graph&&!radix_enabled(input)
          &&(q.scalar_type()==at::kHalf||q.scalar_type()==at::kBFloat16)) {
          auto& packed=input.prefix->prepare_graph_layer(layer_index,k,v);
          if(input.batch==1) {
            auto key_target=packed.key.narrow(0,input.prefix->hits[0],k.size(0));
            auto value_target=packed.value.narrow(0,input.prefix->hits[0],v.size(0));
            q=fused_qkv_norm_rope_into(q,k,v,layer.q_norm,layer.k_norm,layer_cosine,layer_sine,epsilon,key_target,value_target);
            k=key_target;v=value_target;
          }else {
            const auto& rows=input.prefix->prepare_graph_rows(q.device());
            q=fused_qkv_norm_rope_into(q,k,v,layer.q_norm,layer.k_norm,layer_cosine,layer_sine,epsilon,packed.key,packed.value,rows);
            k=packed.key;v=packed.value;
          }
        }else {
          auto pair=fused_qk_norm_rope(q,k,layer.q_norm,layer.k_norm,layer_cosine,layer_sine,epsilon);
          q=pair.first;k=pair.second;
        }
      } else
#endif
      {
        if (layer.q_norm.defined()) { q = norm(q, layer.q_norm); k = norm(k, layer.k_norm); }
#ifdef TEI_TORCH_CUDA_KERNELS
        if(q.is_cuda()&&!transformers_math()) {// Candle contracts one modeldtype product into HFMA.
        // Preserve Llama's path; learned causal BF16 Mistral operands prove
        // the same contract with the registered exact RMS geometry.
        if((dtype==at::kBFloat16||qwen_half_exact)&&family=="qwen2"&&hidden==896&&head_dim==64) {
          // The fresh packed QKV projection gives disjoint Q/K/V views.
          // Preserve original model-dtype HFMA and leave V unchanged.
          encoder_rotary_inplace(q,k,layer_cosine,layer_sine);
        } else {auto pair=fused_rotary(q,k,layer_cosine,layer_sine,family=="llama"||exact_mistral_math());q=pair.first;k=pair.second;}}

        else
#endif
        {q = rotate(q, layer_cosine, layer_sine); k = rotate(k, layer_cosine, layer_sine);}
      }
      if (query_scale.defined()) q = q * query_scale.index_select(0, radix_positions(input)).view({-1, 1, 1});
      auto attended = attend(q, k, v, input, layer_index,layer).contiguous().view({h.size(0), heads * head_dim});
      at::Tensor projected;
#ifdef TEI_TORCH_CUDA_KERNELS
      if(layer.fp8_out)projected=layer.fp8_out->forward(attended,layer.out_bias);
      else
#endif
        projected=cached_prefix_linear(attended,layer.out,layer.out_bias,input);
#ifdef TEI_TORCH_CUDA_KERNELS
      if (h.is_cuda()&&(!transformers_math()||transformers_fused_norm)) {
        if(layer.fp8_gate_up && dtype==at::kBFloat16 && hidden==4096
          && !exact_decoder_math() && projected.is_contiguous()
          && (!residual.defined()||residual.is_contiguous())) {
          auto [quantized,summed,scales]=fp8_add_rms_quantize(projected,residual,layer.post_norm,epsilon);
          residual=std::move(summed);h=mlp(quantized,layer,input,scales);
        } else {
          auto pair=residual_norm(projected,residual,layer.post_norm);
          residual=pair.second;h=mlp(pair.first,layer,input);
        }
      } else
#endif
      {
        h=h+projected;
        h=h+mlp(norm(h,layer.post_norm),layer,input);
      }
      if(visual_indices.defined()&&layer_index<deepstack.size()) {
        const auto& features=deepstack[layer_index];
        TORCH_CHECK(features.dim()==2&&features.size(0)==visual_indices.numel()&&features.size(1)==hidden&&features.device()==device&&features.scalar_type()==dtype,"Invalid multimodal deepstack features");
        if(residual.defined()) {h=h+residual;residual=at::Tensor();}
        h=h.index_add(0,visual_indices,features);
      }
      ++layer_index;
    }
#ifdef TEI_TORCH_CUDA_KERNELS
    if (h.is_cuda()&&(!transformers_math()||transformers_fused_norm)) h=residual_norm(h,residual,final_norm).first;
    else
#endif
    h = norm(h, final_norm);
    if (projection.defined()) h = at::linear(h, projection, projection_bias);
    return h;
  }
};
}
at::Tensor decoder_moe_forward(const at::Tensor& x,const at::Tensor& router,
 const at::Tensor& gate_up,const at::Tensor& down,int64_t top_k,bool renormalize) {
 TORCH_CHECK(x.dim()==2&&router.dim()==2&&gate_up.dim()==3&&down.dim()==3,"Invalid MoE tensor ranks");
 auto experts=router.size(0),hidden=x.size(1),width=down.size(2);
 TORCH_CHECK(router.size(1)==hidden&&gate_up.sizes()==at::IntArrayRef({experts,2*width,hidden})&&down.size(0)==experts&&down.size(1)==hidden&&top_k>0&&top_k<=experts,"Invalid MoE expert geometry");
 auto probabilities=at::softmax(at::linear(x,router).to(at::kFloat),-1);
 auto ids=at::argsort(probabilities,true,-1,true).narrow(-1,0,top_k);
 auto routing=probabilities.gather(-1,ids);
 if(renormalize)routing=routing/routing.sum(-1,true);
 routing=routing.to(x.scalar_type());
 auto activate=[](const at::Tensor& both){
#ifdef TEI_TORCH_CUDA_KERNELS
  if(both.is_cuda())return fused_gated_activation(both,0);
#endif
  auto width=both.size(-1)/2;return at::silu(both.narrow(-1,0,width))*both.narrow(-1,width,width);
 };
 return decoder_moe_dispatch(x,gate_up,down,ids,routing,activate);
}
at::Tensor decoder_moe_dispatch(const at::Tensor& x,const at::Tensor& gate_up,
 const at::Tensor& down,const at::Tensor& ids,const at::Tensor& routing,
 const std::function<at::Tensor(const at::Tensor&)>& activate) {
 TORCH_CHECK(x.dim()==2&&gate_up.dim()==3&&down.dim()==3&&ids.dim()==2&&routing.sizes()==ids.sizes(),"Invalid preselected MoE tensor ranks");
 auto experts=gate_up.size(0),hidden=x.size(1),width=down.size(2),top_k=ids.size(1);
 TORCH_CHECK(ids.size(0)==x.size(0)&&ids.scalar_type()==at::kLong&&top_k>0&&top_k<=experts&&gate_up.sizes()==at::IntArrayRef({experts,2*width,hidden})&&down.sizes()==at::IntArrayRef({experts,hidden,width}),"Invalid preselected MoE geometry");
 TORCH_CHECK(ids.device()==x.device()&&routing.device()==x.device()&&gate_up.device()==x.device()&&down.device()==x.device(),"MoE tensors must share device");
 TORCH_CHECK(gate_up.scalar_type()==x.scalar_type()&&down.scalar_type()==x.scalar_type()&&routing.is_floating_point(),"Invalid MoE weight/routing dtype");
 auto result=at::zeros_like(x,x.options().dtype(routing.scalar_type()));
 if(x.is_cuda()&&x.scalar_type()==at::kBFloat16&&hidden%8==0&&width%8==0) {
  auto flat=ids.flatten(),order=at::argsort(flat,true,0,false),tokens=at::floor_divide(order,top_k);
  auto counts=at::zeros({experts},x.options().dtype(at::kInt));
  counts.scatter_add_(0,flat,at::ones_like(flat,flat.options().dtype(at::kInt)));
  auto ends=counts.cumsum(0,at::kInt);
  auto both=at::_grouped_mm(x.index_select(0,tokens),gate_up.transpose(-2,-1),ends);
  auto output=at::_grouped_mm(activate(both),down.transpose(-2,-1),ends);
  result.index_add_(0,tokens,output.to(routing.scalar_type())*routing.flatten().index_select(0,order).unsqueeze(1));
 }else for(int64_t e=0;e<experts;++e){
  auto selected=at::nonzero(ids==e),tokens=selected.select(1,0),slots=selected.select(1,1);
  auto output=at::linear(activate(at::linear(x.index_select(0,tokens),gate_up[e])),down[e]);
  result.index_add_(0,tokens,output.to(routing.scalar_type())*routing.index_select(0,tokens).gather(1,slots.unsqueeze(1)));
 }
 return result;
}
at::Tensor decoder_visual_forward(const Model& model,const PackedInput& input,const at::Tensor& indices,const at::Tensor& visual) {
  const auto* decoder=dynamic_cast<const Decoder*>(&model);
  TORCH_CHECK(decoder,"Visual embeddings require a dense text decoder");
  return decoder->visual_forward(input,indices,visual);
}
at::Tensor decoder_multimodal_forward(const Model& model,const PackedInput& input,
 const at::Tensor& indices,const at::Tensor& visual,const at::Tensor& cosine,const at::Tensor& sine,
 const std::vector<at::Tensor>& deepstack) {
  auto* decoder=dynamic_cast<const Decoder*>(&model);
  TORCH_CHECK(decoder,"Multimodal text backbone is not a native decoder model");
  return decoder->multimodal_forward(input,indices,visual,cosine,sine,deepstack);
}
std::unique_ptr<Model> create_decoder(const Options& cfg, Weights& weights, c10::Device device, at::ScalarType dtype) {
  auto kind = cfg.string("model_type", "");
  if (kind != "llama" && kind != "mistral" && kind != "qwen2" && kind != "qwen3" && kind != "qwen3_moe" && kind != "ministral3" && kind != "llama_bidirec" && kind != "ultrabert") return nullptr;
  return std::make_unique<Decoder>(cfg, weights, device, dtype);
}
}

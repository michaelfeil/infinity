#include "env_config.h"
#include "models/bert_model.h"
#include "bridge.h"
#include "helpers/fp8_linear.h"
#include "batching/radix.h"
#include "batching/block_prefix.h"
#include "batching/output.h"
#include "helpers/workspace_admission.h"
#include "models/model.h"
#include "helpers/options.h"
#ifdef TEI_TORCH_CUDA_KERNELS
#include "kernels/fast_kernels.h"
#include "kernels/encoder_norm.h"
#include "kernels/encoder_activation.h"
#include "kernels/torch_flash_fma.h"
#include "batching/packed_pool.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/detail/CUDAHooksInterface.h>
#endif
#include "models/decoder_models.h"
#include "models/mistral3_models.h"
#include "kernels/moe_kernels.h"
#include "models/decision_models.h"
#include "models/dense_models.h"
#include "models/multivector_model.h"
#include "models/encoder_models.h"
#include "models/gemma_models.h"
#include "models/qwen35_models.h"
#include "models/qwen3_vl_models.h"
#include "batching/cuda_graph.h"
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/ops/_flash_attention_forward.h>
#include <ATen/ops/_addmm_activation.h>
#include <ATen/ops/_cudnn_attention_forward.h>
#include <c10/core/InferenceMode.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <cstring>
#include <cstdio>
#include "helpers/vendor_error.h"
#include <cstdlib>
#include <mutex>
#include "batching/input_validation.h"
#include <atomic>
#include "batching/prefix_cache.h"
#include <memory>
#include <optional>
#include <string>
#include <set>
#include <unordered_map>

namespace {
thread_local std::string error;
thread_local int32_t error_kind=0;
thread_local const char* error_fallback=nullptr;
void set_error(int32_t kind,const char* message) noexcept {
  error_kind=kind;
  try { error=message; } catch (...) { error_fallback="Native error message unavailable under memory pressure"; }
}
std::once_flag precision_initialized;
void initialize_precision() {
  std::call_once(precision_initialized, [] {
    const auto* value = std::getenv(INFINITY_ENV("TORCH_FULL_PRECISION_GEMM"));
    if (value && (std::strcmp(value, "true") == 0 || std::strcmp(value, "1") == 0)) {
      at::globalContext().setAllowFP16ReductionCuBLAS(false);
      at::globalContext().setAllowBF16ReductionCuBLAS(false);
      at::globalContext().setAllowFP16AccumulationCuBLAS(false);
      at::globalContext().setAllowTF32CuBLAS(false);
      at::globalContext().setAllowTF32CuDNN(false);
    }
  });
}
at::ScalarType scalar(int32_t dtype) {
  switch (dtype) {
    case 0: return at::kFloat;
    case 1: return at::kHalf;
    case 2: return at::kBFloat16;
    default: throw std::runtime_error("Unsupported tensor dtype");
  }
}
c10::Device device(const char* name) {
  if (std::strcmp(name, "auto") == 0)
    return c10::Device(at::globalContext().hasCUDA() && at::getNumGPUs() > 0 ? "cuda:0" : "cpu");
  return c10::Device(name);
}
struct Engine {
  std::mutex mutex;
  TeiConfig config;
  c10::Device device;
  at::ScalarType dtype;
  tei::Options options;
  tei::Weights weights;
  std::shared_ptr<tei::WorkspaceMemoryPolicy> workspace_policy;
  std::unique_ptr<tei::Model> model;
  std::shared_ptr<tei::OutputPool> outputs = std::make_shared<tei::OutputPool>();
  std::unique_ptr<tei::DenseChain> dense;
  std::unique_ptr<tei::DecisionHead> decision;
  std::unique_ptr<tei::CudaGraphCache> graphs;
  std::unique_ptr<tei::PrefixCache> prefix;
  std::atomic<bool> prefix_probe=false;
  bool media_graphs=false;
  bool nomic_moe_graphs=false;
#ifdef TEI_TORCH_CUDA_KERNELS
  std::unique_ptr<at::cuda::CUDAEvent> submitted;
  std::optional<c10::cuda::CUDAStream> producer;
  bool submission_failed=false;
#endif
  void drain() {
#ifdef TEI_TORCH_CUDA_KERNELS
    if(submission_failed&&producer)producer->synchronize();
    else if(submitted)submitted->synchronize();
#endif
  }
  ~Engine() {try{drain();}catch(...) { /* A failed CUDA context is already reported by inference. */ }}
  bool graph_policy_ready() const {
    return !nomic_moe_graphs || tei::encoder_nomic_device_routing_ready(*model);
  }
  Engine(TeiConfig cfg, c10::Device dev, at::ScalarType type) : config(cfg), device(dev), dtype(type) {}
  void ready() {
    drain();
    // A reload can disable features as well as enable them. Retire old graph
    // buffers while their planner/model still exist, after submitted work ends.
    graphs.reset();prefix.reset();decision.reset();
    media_graphs=false;nomic_moe_graphs=false;
    TORCH_CHECK(options.integer("_input_upload_max_bytes",512LL<<20)>0,
      "Native input upload budget must be positive");
    const auto workspace_bytes=options.integer("_retained_workspace_max_bytes",512LL<<20);
    TORCH_CHECK(workspace_bytes>0&&uint64_t(workspace_bytes)<=std::numeric_limits<size_t>::max(),"Invalid native retained workspace allowance");
    if(device.is_cuda())workspace_policy=tei::configure_workspace_admission(device.index(),static_cast<size_t>(workspace_bytes));
    auto staging_bytes = options.integer("_output_staging_max_bytes", 512LL << 20);
    TORCH_CHECK(staging_bytes > 0, "Output staging memory budget must be positive");
    outputs->max_bytes = static_cast<size_t>(staging_bytes);
    dense = std::make_unique<tei::DenseChain>(options, weights);
    auto family = options.string("model_type", "bert");
    if(options.boolean("_fp8_dynamic",false)) {
      TORCH_CHECK(family=="bert" || family=="roberta" || family=="xlm-roberta"
        || family=="camembert" || family=="modernbert" || family=="llama"
        || family=="llama_bidirec" || family=="mistral" || family=="ministral3"
        || family=="qwen2" || family=="qwen3" || family=="qwen3_5" || family=="qwen3_5_text",
        "Dynamic FP8 is unsupported for this model family");
      TORCH_CHECK(!options.values.count("quantization_config") && !options.values.count("quantization_config.quant_method"),
        "Dynamic FP8 requires an unquantized checkpoint");
    }
    auto decision_kind=options.string("_decision.kind");
    if(!decision_kind.empty()) {
      tei::Options head_config;tei::Weights head_weights;
      constexpr std::string_view config_prefix="_decision.config.",weight_prefix="__tei_decision.";
      for(const auto& [key,value]:options.values)if(key.starts_with(config_prefix))head_config.values[key.substr(config_prefix.size())]=value;
      for(auto it=weights.begin();it!=weights.end();) {
        if(it->first.starts_with(weight_prefix)){head_weights[it->first.substr(weight_prefix.size())]=it->second;it=weights.erase(it);}else ++it;
      }
      decision=tei::create_decision_head(decision_kind,head_config,options,decision_kind=="laya"?weights:head_weights,weights);
      if(decision_kind=="laya") {
        TORCH_CHECK(family=="modernbert","Laya requires a ModernBERT encoder");tei::Weights encoder;
        for(const auto& [key,value]:weights)if(key.starts_with("encoder."))encoder[key.substr(8)]=value;
        TORCH_CHECK(!encoder.empty(),"Laya checkpoint contains no encoder.* weights");weights=std::move(encoder);
      }
    }
    const bool multi_vector=options.boolean("_multi_vector",false);
    at::Tensor token_projection,token_bias;
    double token_norm_eps=1e-12;
    if(multi_vector && weights.count("embedding_proj_layer.weight")) {
      // ColQwen3 uses finfo(dtype).eps rather than F.normalize's default.
      token_norm_eps=dtype==at::kBFloat16?1.0/128:dtype==at::kHalf?1.0/1024:std::numeric_limits<float>::epsilon();
      tei::Weights canonical;
      for(const auto& [name,value]:weights) {
        auto key=name.starts_with("vlm.")?name.substr(4):name;
        TORCH_CHECK(!canonical.count(key),"Duplicate ColQwen3 weight: ",key);
        canonical.emplace(std::move(key),value);
      }
      weights=std::move(canonical);
      token_projection=weights.at("embedding_proj_layer.weight");
      if(weights.count("embedding_proj_layer.bias"))token_bias=weights.at("embedding_proj_layer.bias");
    } else if(multi_vector && weights.count("linear.weight")) {
      token_projection=weights.at("linear.weight");
      if(weights.count("linear.bias"))token_bias=weights.at("linear.bias");
    }
    model = tei::create_encoder(options, weights, device, dtype);
    if (!model) model = tei::create_mistral3(options, weights, device, dtype);
    if (!model) model = tei::create_decoder(options, weights, device, dtype);
    if (!model) model = tei::create_gemma(options, weights, device, dtype);
    if (!model) model = tei::create_qwen35(options, weights, device, dtype);
    if (!model) model = tei::create_qwen3_vl(options, weights, device, dtype);
    if (!model && (family == "bert" || family == "roberta" || family == "xlm-roberta" || family == "camembert")) {
      const auto activation=options.string("hidden_act","gelu");
      TORCH_CHECK(activation=="gelu"||activation=="gelu_new"||activation=="gelu_pytorch_tanh"||activation=="relu"||activation=="silu",
        "Unsupported native BERT activation: ",activation);
      auto bert = std::make_unique<Bert>(config, device, dtype);
      bert->options = options;
      bert->weights = std::move(weights);
      model = std::move(bert);
    }
    TORCH_CHECK(model, "Unsupported native Torch model family: ", family);
    if(multi_vector) {
      TORCH_CHECK(token_projection.defined(),"Multi-vector checkpoint is missing its token projection");
      model=std::make_unique<tei::MultiVectorModel>(std::move(model),token_projection,token_bias,token_norm_eps);
    }
    model->ready();
    auto cache_bytes=options.integer("_prefix_cache_max_bytes",0);
    auto cache_tokens=options.integer("_prefix_cache_max_tokens",8192),cache_entries=options.integer("_prefix_cache_entries",64);
    TORCH_CHECK(cache_bytes>=0&&cache_tokens>0&&cache_entries>0,"Invalid prefix cache bounds");
    if(cache_bytes)prefix=std::make_unique<tei::PrefixCache>(cache_bytes,cache_tokens,cache_entries,2,
      tei::PrefixCache::StreamOrder::EngineOrdered);
    TORCH_CHECK(model->output_width() > 0, "Model must declare a positive output width");
    dense->output_width(options.string("_pooling") == "splade" ? config.vocab : model->output_width());
    media_graphs=options.boolean("_media_cuda_graphs",false);
    // These Hopper BF16 geometries use device-only routing. Other MoE
    // geometries retain eager execution rather than synchronizing in capture.
    nomic_moe_graphs=tei::encoder_nomic_device_routing_ready(*model);
    bool native_moe_graphs=nomic_moe_graphs;
#if defined(TEI_TORCH_MOE_KERNELS) && defined(TEI_TORCH_MOE_HOPPER)
    if(device.is_cuda()&&dtype==at::kBFloat16&&family=="qwen3_moe"
      &&options.integer("hidden_size",0)==2048&&options.integer("num_experts",0)==128
      &&options.integer("num_experts_per_tok",0)==8&&options.integer("moe_intermediate_size",0)==768
      &&options.integer("head_dim",0)==128&&options.integer("num_attention_heads",0)==32
      &&options.integer("num_key_value_heads",0)==4)
      native_moe_graphs=tei::native_hopper_moe_available(device);
    if(device.is_cuda()&&dtype==at::kBFloat16&&(family=="qwen3_5_moe"||family=="qwen3_5_moe_text")) {
      auto ti=[&](const char* key){return options.integer(std::string("text_config.")+key,options.integer(key,0));};
      if(ti("hidden_size")==2048&&ti("num_experts")==256&&ti("num_experts_per_tok")==8&&ti("moe_intermediate_size")==512&&ti("head_dim")==256&&ti("num_attention_heads")==16&&ti("num_key_value_heads")==2)
        native_moe_graphs=tei::native_hopper_moe_available(device);
    }
    if(device.is_cuda()&&dtype==at::kBFloat16&&(family=="gemma4"||family=="gemma4_text")) {
      auto text_integer=[&](const char* key,int64_t fallback=0) {
        return options.integer(std::string("text_config.")+key,options.integer(key,fallback));
      };
      auto act=options.string("text_config.hidden_activation",options.string("hidden_activation","gelu_pytorch_tanh"));
      if(text_integer("hidden_size")==2816&&text_integer("num_experts")==128
        &&text_integer("top_k_experts",8)==8&&text_integer("moe_intermediate_size")==704
        &&(act=="gelu_pytorch_tanh"||act=="gelu_new"))
        native_moe_graphs=tei::native_hopper_moe_available(device);
    }
#endif
    if (device.is_cuda() && !multi_vector && options.boolean("_cuda_graphs", false)
        && (native_moe_graphs || (options.integer("num_experts", options.integer("text_config.num_experts", 0)) == 0
        && options.integer("moe_every_n_layers", options.integer("text_config.moe_every_n_layers", 0)) == 0
        && !options.boolean("enable_moe_block", options.boolean("text_config.enable_moe_block", false))
        && (family == "bert" || family == "roberta"
        || family == "xlm-roberta" || family == "camembert" || family == "distilbert" || family == "modernbert"
        || family == "mpnet" || family == "deberta-v2" || family == "gte" || family == "nomic_bert" || family == "llama" || family == "mistral"
        || family == "ultrabert" || family == "qwen2" || family == "qwen3" || family == "gemma3" || family == "gemma3_text"
        || family == "gemma4" || family == "gemma4_text" || family == "embedding_gemma2"
        || (media_graphs && (family == "qwen3_vl" || family == "qwen3_vl_text" || family == "qwen3_5" || family == "qwen3_5_text")))))) {
      auto entries=options.integer("_cuda_graph_entries",4), reuse=options.integer("_cuda_graph_reuse",3);
      auto bytes=options.integer("_cuda_graph_max_bytes",512LL<<20), captures=options.integer("_cuda_graph_max_captures",8);
      TORCH_CHECK(entries>=0 && reuse>0 && bytes>0 && captures>=0,"Invalid CUDA graph admission budgets");
      auto max_tokens=options.integer("_cuda_graph_max_tokens",4096);
      // Larger Nomic batches use host-count reference routing and remain eager.
      if(nomic_moe_graphs) max_tokens=std::min<int64_t>(max_tokens,8192);
      graphs=std::make_unique<tei::CudaGraphCache>(entries,max_tokens,reuse,bytes,captures);
      graphs->retain_prefix_states(bool(prefix));
    }
  }
};
// Host serialization protects model/graph mutation. A CUDA event also orders
// queued work across callers' streams; a host mutex alone cannot do that.
class EngineCall {
  Engine& engine;
  std::unique_lock<std::mutex> lock;
  bool inference,finished=false;
#ifdef TEI_TORCH_CUDA_KERNELS
  c10::cuda::OptionalCUDAGuard device;
  std::optional<c10::cuda::CUDAStream> stream;
#endif
  static Engine& get(void* handle) {TORCH_CHECK(handle,"Missing native engine");return *static_cast<Engine*>(handle);}
public:
  explicit EngineCall(void* handle,bool submits=false):engine(get(handle)),lock(engine.mutex),inference(submits) {
#ifdef TEI_TORCH_CUDA_KERNELS
    if(inference&&engine.device.is_cuda()) {
      TORCH_CHECK(!engine.submission_failed,"Native engine requires recreation after CUDA submission failure");
      device.set_device(engine.device);
      stream=c10::cuda::getCurrentCUDAStream(engine.device.index());
      if(engine.submitted&&engine.producer!=stream)engine.submitted->block(*stream);
    }
#endif
  }
  void finish() {
    if(finished)return;
#ifdef TEI_TORCH_CUDA_KERNELS
    if(stream) {
      // Also remember the stream if recording fails after work was queued.
      engine.producer=stream;
      if(!engine.submitted)engine.submitted=std::make_unique<at::cuda::CUDAEvent>();
      engine.submitted->record(*stream);
    }
#endif
    finished=true;
  }
  ~EngineCall() {
    try{finish();}catch(...) {
#ifdef TEI_TORCH_CUDA_KERNELS
      engine.submission_failed=true;
#endif
    }
  }
};
size_t input_upload_limit(const Engine& model) {
 const auto limit=model.options.integer("_input_upload_max_bytes",512LL<<20);
 TORCH_CHECK(limit>0&&uint64_t(limit)<=std::numeric_limits<size_t>::max(),"Native input upload budget must be positive and representable");
 return static_cast<size_t>(limit);
}
bool share_image_uploads(const Engine& model) {
 return model.options.string("_decision.kind")=="d3"||model.options.string("_decision.kind")=="pplx"||
   model.options.string("_decision_attention_mode")=="block_bidirectional";
}
size_t validate_request_input(Engine& model,const int64_t* ids,const int64_t* types,
 const int64_t* positions,const int32_t* cumulative,int64_t batch,int64_t max_sequence,
 const int64_t* media_positions,const TeiImage* images,size_t image_count,const TeiAudio* audios,size_t audio_count,const TeiRadix* radix) {
 const auto family=model.options.string("model_type");
 const bool gemma=family=="gemma4"||family=="gemma4_text"||family=="embedding_gemma2";
 const auto tokens=tei::validate_input_descriptors(ids,types,positions,cumulative,batch,max_sequence,
   media_positions,images,image_count,audios,audio_count,gemma);
 const auto image_dtype=model.dtype;
 return tei::validate_input_upload_budget(tokens,batch,media_positions,images,image_count,audios,audio_count,
   radix,c10::elementSize(image_dtype),input_upload_limit(model),gemma,share_image_uploads(model));
}
void validate_forward_descriptors(Engine& model,const int64_t* ids,const int64_t* types,
 const int64_t* positions,const int32_t* cumulative,int64_t batch,int64_t max_sequence,int32_t pool,
 const int64_t* pooled,size_t pooled_count,const int64_t* raw,size_t raw_count,size_t capacity,
 const int64_t* media_positions,const TeiImage* images,size_t image_count,const TeiAudio* audios,size_t audio_count,const TeiRadix* radix,bool deferred=false) {
 validate_request_input(model,ids,types,positions,cumulative,batch,max_sequence,media_positions,
   images,image_count,audios,audio_count,radix);
 const auto raw_rows=tei::validate_output_indices(cumulative,batch,pooled,pooled_count,raw,raw_count);
 TORCH_CHECK(model.model&&model.dense,"Native model is not ready");
 TORCH_CHECK(!image_count||model.model->supports_images(),"This native model does not yet support images");
 TORCH_CHECK(!audio_count||model.model->supports_audio(),"This native model does not yet support audio");
 const auto raw_width=pool>=4?model.model->classification_width():model.model->output_width();
 const auto pooled_width=pool>=4?raw_width:model.dense->output_width(pool==3?model.config.vocab:raw_width);
 const auto expected=tei::expected_output_elements(pooled_count,raw_rows,pooled_width,raw_width);
 TORCH_CHECK(expected<=capacity,"Native output exceeds provided capacity");
 TORCH_CHECK(!deferred||expected==capacity,"Deferred output capacity must exactly match requested output");
}
// Synchronous calls must drain work queued before an exception while their
// borrowed host descriptors and temporary tensors are still alive.
struct SynchronousStreamGuard {
#ifdef TEI_TORCH_CUDA_KERNELS
 std::optional<c10::cuda::CUDAStream> stream;
#endif
 explicit SynchronousStreamGuard(Engine& model) {
#ifdef TEI_TORCH_CUDA_KERNELS
  if(model.device.is_cuda())stream=c10::cuda::getCurrentCUDAStream(model.device.index());
#endif
 }
 ~SynchronousStreamGuard() {
#ifdef TEI_TORCH_CUDA_KERNELS
  try {if(stream)stream->synchronize();}catch(...){}
#endif
 }
};
tei::PackedInput packed_input(Engine& model,const int64_t* ids,const int64_t* types,
 const int64_t* positions,const int32_t* cumulative,int64_t batch,int64_t max_sequence,
 const int64_t* media_positions,const TeiImage* images,size_t image_count,const TeiAudio* audios,size_t audio_count,tei::OutputLease* completion=nullptr) {
 const auto family=model.options.string("model_type");
 const bool gemma_image=family=="gemma4"||family=="gemma4_text"||family=="embedding_gemma2";
 const auto tokens=tei::validate_input_descriptors(ids,types,positions,cumulative,batch,max_sequence,
   media_positions,images,image_count,audios,audio_count,gemma_image);
 TORCH_CHECK(!image_count||model.model->supports_images(),"This native model does not yet support images");
 TORCH_CHECK(!audio_count||model.model->supports_audio(),"This native model does not yet support audio");
 auto options=at::TensorOptions().dtype(at::kLong).device(at::kCPU);
 auto upload=[&](const at::Tensor& host){return completion?completion->upload(host):host.to(model.device);};
 auto input=[&](const int64_t* data){TORCH_CHECK(data,"Missing native token buffer");return upload(at::from_blob(const_cast<int64_t*>(data),{tokens},options));};
 auto cu=upload(at::from_blob(const_cast<int32_t*>(cumulative),{batch+1},options.dtype(at::kInt)));
 tei::PackedInput packed{input(ids),input(types),input(positions),cu,cumulative,batch,max_sequence};
 packed.host_ids=at::from_blob(const_cast<int64_t*>(ids),{tokens},options);
 packed.host_types=at::from_blob(const_cast<int64_t*>(types),{tokens},options);
 packed.host_positions=at::from_blob(const_cast<int64_t*>(positions),{tokens},options);
 TORCH_CHECK(!image_count||model.model->supports_images(),"This native model does not yet support images");
 TORCH_CHECK(!audio_count||model.model->supports_audio(),"This native model does not yet support audio");
 if(media_positions)packed.multimodal_positions=at::from_blob(const_cast<int64_t*>(media_positions),{3,tokens},options).to(model.device);
 if(media_positions)packed.host_multimodal_positions=at::from_blob(const_cast<int64_t*>(media_positions),{3,tokens},options);
 tei::RequestImageUploads shared_images;
 for(size_t i=0;i<image_count;++i){const auto& item=images[i];
  auto host_pixels=at::from_blob(const_cast<float*>(item.pixels),{item.rows,item.patch_dim},options.dtype(at::kFloat));
  auto upload=[&]{return tei::upload_image_pixels(host_pixels,model.device,model.dtype,family);};
  auto pixels=share_image_uploads(model)?shared_images.get(host_pixels,upload):upload();
  packed.images.push_back({pixels,{item.grid[0],item.grid[1],item.grid[2]},item.merge_size,item.token_start,item.token_count,item.sequence_start});
  std::copy_n(item.feature_key,32,packed.images.back().feature_key.begin());
  packed.images.back().has_feature_key=item.has_feature_key!=0;
 }
 for(size_t i=0;i<audio_count;++i){const auto& item=audios[i];
  auto features=at::from_blob(const_cast<float*>(item.values),{item.frames,item.feature_size},options.dtype(at::kFloat)).to(model.device);
  auto validity=at::from_blob(const_cast<uint8_t*>(item.mask),{item.frames},options.dtype(at::kByte)).to(model.device);
  packed.audios.push_back({features,validity,item.token_start,item.token_count});
 }return packed;
}
void attach_radix(Engine& model,tei::PackedInput& packed,const int64_t* ids,const int64_t* types,
 const int64_t* positions,const int32_t* cumulative,int64_t batch,int64_t max_sequence,
 const int64_t* media_positions,const TeiImage* images,size_t image_count,size_t audio_count,const TeiRadix* radix,tei::OutputLease* completion=nullptr) {
  if(radix) {
      TORCH_CHECK(model.model->supports_radix(),"This native model does not support radix folding");
      TORCH_CHECK(!audio_count,"Radix audio inference is not supported");
      const int64_t total=cumulative[batch],compact=radix->compact_tokens;
      TORCH_CHECK(compact>0&&compact<=total&&radix->ids&&radix->types&&radix->positions&&radix->scatter&&radix->fold,"Invalid native radix buffers");
      const auto options=at::TensorOptions().dtype(at::kLong).device(at::kCPU);
      auto host=[&](const int64_t* data,int64_t rows){return at::from_blob(const_cast<int64_t*>(data),{rows},options);};
      tei::PackedInput metadata{host(ids,total),host(types,total),host(positions,total),
        at::from_blob(const_cast<int32_t*>(cumulative),{batch+1},options.dtype(at::kInt)),cumulative,batch,max_sequence};
      metadata.compact_ids=host(radix->ids,compact);
      metadata.compact_types=host(radix->types,compact);
      metadata.compact_positions=host(radix->positions,compact);
      metadata.scatter_unfold=host(radix->scatter,total);
      metadata.fold_gather=host(radix->fold,compact);
      packed.host_scatter_unfold=metadata.scatter_unfold;
      // Validate complete causal contexts on host before capture or map upload.
      tei::validate_radix(metadata,true);
      if(image_count||media_positions) {
        // Token IDs alone cannot establish equivalent image contexts. Compare
        // host pixel bits, placement and all rotary axes before map upload.
        std::vector<std::vector<const TeiImage*>> contexts(batch);
        std::vector<int64_t> rows(total);
        for(int64_t row=0;row<batch;++row)
          std::fill(rows.begin()+cumulative[row],rows.begin()+cumulative[row+1],row);
        for(size_t i=0;i<image_count;++i) {
          const auto& image=images[i];
          TORCH_CHECK(image.sequence_start>=0&&image.sequence_start<total,"Invalid radix image sequence");
          auto row=rows[image.sequence_start];
          TORCH_CHECK(image.sequence_start==cumulative[row]&&image.token_start>=image.sequence_start&&image.token_count>0
            && image.token_count<=cumulative[row+1]-image.token_start,"Invalid radix image span");
          TORCH_CHECK(image.rows>0&&image.patch_dim>0&&image.pixels&&
            uint64_t(image.rows)<=SIZE_MAX/sizeof(float)/uint64_t(image.patch_dim),"Invalid radix image pixels");
          contexts[row].push_back(&image);
        }
        std::set<std::pair<int64_t,int64_t>> checked_contexts;
        for(int64_t i=0;i<total;++i) {
          const auto a=rows[i],b=rows[radix->fold[radix->scatter[i]]];
          if(!checked_contexts.emplace(a,b).second)continue;
          const auto& left=contexts[a];const auto& right=contexts[b];
          TORCH_CHECK(left.size()==right.size(),"Radix aliases require identical image contexts");
          int64_t prefix=0;
          for(size_t j=0;j<left.size();++j) {
            const auto& x=*left[j];const auto& y=*right[j];
            const auto start=x.token_start-cumulative[a];
            TORCH_CHECK(start==y.token_start-cumulative[b]&&x.token_count==y.token_count&&x.rows==y.rows&&x.patch_dim==y.patch_dim
              && x.merge_size==y.merge_size&&std::equal(x.grid,x.grid+3,y.grid)
              && !std::memcmp(x.pixels,y.pixels,size_t(x.rows)*size_t(x.patch_dim)*sizeof(float)),
              "Radix aliases require identical image pixels and layout");
            prefix=std::max(prefix,start+x.token_count);
          }
          TORCH_CHECK(prefix<=cumulative[a+1]-cumulative[a]&&prefix<=cumulative[b+1]-cumulative[b],"Invalid radix image prefix");
          for(int64_t j=0;j<prefix;++j) {
            TORCH_CHECK(ids[cumulative[a]+j]==ids[cumulative[b]+j],"Radix aliases require identical image prefixes");
            if(media_positions)for(int64_t axis=0;axis<3;++axis)
              TORCH_CHECK(media_positions[axis*total+cumulative[a]+j]==media_positions[axis*total+cumulative[b]+j],
                "Radix aliases require identical image rotary positions");
          }
          // Even for text-only media positions, every shared row must agree.
        }
        if(media_positions)for(int64_t i=0;i<total;++i)for(int64_t axis=0;axis<3;++axis)
          TORCH_CHECK(media_positions[axis*total+i]==media_positions[axis*total+radix->fold[radix->scatter[i]]],
            "Radix aliases require identical rotary positions");
      }
      auto upload=[&](const at::Tensor& host){return completion?completion->upload(host):host.to(model.device);};
      packed.compact_ids=upload(metadata.compact_ids);
      packed.compact_types=upload(metadata.compact_types);
      packed.compact_positions=upload(metadata.compact_positions);
      packed.scatter_unfold=upload(metadata.scatter_unfold);
      packed.fold_gather=upload(metadata.fold_gather);
    }
}
std::vector<tei::DecisionRequest> decision_requests(const TeiDecision* requests,size_t count) {
 TORCH_CHECK(requests||!count,"Missing decision metadata buffer");std::vector<tei::DecisionRequest> result;result.reserve(count);
 for(size_t i=0;i<count;++i){const auto& request=requests[i];TORCH_CHECK(request.kind>=0&&request.kind<=5,"Invalid decision metadata kind");
  tei::DecisionRequest converted{static_cast<tei::DecisionRequest::Kind>(request.kind==4?1:request.kind)};converted.question_type=request.question_type;
  TORCH_CHECK((!request.marker_count||request.markers)&&(!request.token_count||request.token_ids)&&(!request.field_count||request.fields),"Missing decision metadata values");
  if(request.marker_count)converted.markers.assign(request.markers,request.markers+request.marker_count);
  if(request.token_count)converted.token_ids.assign(request.token_ids,request.token_ids+request.token_count);
  for(size_t j=0;j<request.field_count;++j){const auto& field=request.fields[j];TORCH_CHECK(!field.option_count||field.options,"Missing Clef option spans");tei::DecisionField f{field.kind,{field.question_start,field.question_end},{}};
   for(size_t k=0;k<field.option_count;++k)f.options.emplace_back(field.options[2*k],field.options[2*k+1]);converted.fields.push_back(std::move(f));
  }result.push_back(std::move(converted));
 }return result;
}
template<class F> int32_t checked(F&& f) {
  error_kind=0;error_fallback=nullptr;error.clear();
  try { c10::InferenceMode guard; f(); return 0; }
  catch (const tei::NativeCallError& ex) { error_kind=ex.kind;return -1; }
  catch (const tei::VendorAllocationError& ex) {
    char diagnostic[192];
    std::snprintf(diagnostic,sizeof(diagnostic),"%s allocation failed (vendor status %d)",ex.operation,ex.status);
    set_error(3,diagnostic);return -1;
  }
  catch (const tei::MemoryAdmissionError& ex) { set_error(2,ex.what());return -1; }
  catch (const c10::OutOfMemoryError& ex) { set_error(3,ex.what());return -1; }
  catch (const c10::AcceleratorError& ex) { set_error(tei::accelerator_error_kind(ex),ex.what());return -1; }
  catch (const std::bad_alloc& ex) { set_error(3,ex.what());return -1; }
  catch (const std::exception& ex) { set_error(1,ex.what()); return -1; }
  catch (...) { set_error(1,"Unknown native exception"); return -1; }
}
}
extern "C" {
const char* tei_error() { return error_fallback?error_fallback:error.c_str(); }
int32_t tei_last_error_kind() { return error_kind; }
int64_t tei_device_count(const char* name) {
  int64_t count = -1;
  checked([&] {
    auto dev = device(name);
    count = c10::impl::getDeviceGuardImpl(dev.type())->deviceCount();
  });
  return count;
}
void* tei_create(const TeiConfig* cfg, const char* name, int32_t dtype) {
  Engine* model = nullptr;
  checked([&] { initialize_precision(); model = new Engine(*cfg, device(name), scalar(dtype)); });
  return model;
}
int32_t tei_weight(void* handle, const char* name, const uint8_t* data,
                   const int64_t* shape, size_t rank, int32_t dtype) {
  return checked([&] {
    EngineCall access(handle);
    auto& model = *static_cast<Engine*>(handle);
    auto tensor = at::empty(at::IntArrayRef(shape, rank), at::TensorOptions().dtype(scalar(dtype)).device(at::kCPU));
    std::memcpy(tensor.data_ptr(), data, tensor.nbytes());
    const std::string key(name);
    // DeltaNet decay parameters are FP32 in Candle and in the checkpoint's compute contract.
    const bool fp32_audio_filter = key.find("audio_tower.") != std::string::npos
      && ((key.find("subsample_conv_projection.") != std::string::npos && key.ends_with(".conv.weight"))
        || key.ends_with(".lconv1d.depthwise_conv1d.weight"));
    auto target_dtype = key.ends_with(".A_log") || key.ends_with(".dt_bias") || fp32_audio_filter ? at::kFloat : model.dtype;
    if(key.starts_with("__tei_decision.")) {
      auto kind=model.options.string("_decision.kind");
      if(kind=="pplx")target_dtype=at::kBFloat16;
      if(kind=="d3")target_dtype=at::kFloat;
    }
    TORCH_CHECK(model.weights.emplace(name, tensor.to(model.device, target_dtype)).second,
                "Duplicate weight: ", name);
  });
}
int32_t tei_option(void* handle, const char* key, const char* value) {
  return checked([&] { EngineCall access(handle);static_cast<Engine*>(handle)->options.values[key] = value; });
}
int64_t tei_output_width(void* handle) {
  int64_t width = -1;
  checked([&] { EngineCall access(handle);width = static_cast<Engine*>(handle)->model->output_width(); });
  return width;
}
int32_t tei_supports_radix(void* handle) {
  int32_t supported=-1;
  checked([&]{EngineCall access(handle);supported=static_cast<Engine*>(handle)->model->supports_radix()?1:0;});
  return supported;
}
int32_t tei_prefix_cache_supported(void* handle,int32_t operation) {
  int32_t supported=-1;
  checked([&] {
    EngineCall access(handle);
    const auto& engine=*static_cast<Engine*>(handle);
    TORCH_CHECK(engine.model&&operation>=0&&operation<=2,"Invalid prefix-cache capability query");
    supported=engine.model->supports_prefix_cache(operation)?1:0;
  });
  return supported;
}
int32_t tei_prefix_cache_stats(void* handle,uint64_t* counters,size_t count) {
  return checked([&] {
    EngineCall access(handle);
    TORCH_CHECK(counters&&count>=7,"Prefix-cache stats require seven counters");
    const auto& engine=*static_cast<Engine*>(handle);
    auto s=engine.prefix?engine.prefix->stats():tei::PrefixCache::Stats{};
    counters[0]=s.hits;counters[1]=s.misses;counters[2]=s.reused_tokens;counters[3]=s.admissions;
    counters[4]=s.evictions;counters[5]=s.bypasses;counters[6]=engine.prefix?engine.prefix->bytes():0;
  });
}
int32_t tei_prefix_cache_clear(void* handle) {
  return checked([&] {EngineCall access(handle);auto& engine=*static_cast<Engine*>(handle);if(engine.prefix)engine.prefix->clear();});
}
int32_t tei_prefix_cache_set_probe(void* handle,int32_t enabled) {
  int32_t previous=-1;
  checked([&] {
    TORCH_CHECK(handle&&(enabled==0||enabled==1),"Invalid prefix-cache probe flag");
    EngineCall access(handle);
    auto& engine=*static_cast<Engine*>(handle);
    previous=engine.prefix_probe.exchange(enabled!=0,std::memory_order_relaxed)?1:0;
  });
  return previous;
}
int64_t tei_graph_count(void* handle) {
  int64_t count=-1;
  checked([&]{EngineCall access(handle);const auto& engine=*static_cast<Engine*>(handle);count=engine.graphs?engine.graphs->size():0;});
  return count;
}
int32_t tei_graph_stats(void* handle,uint64_t* counters,size_t count) {
  return checked([&] {
    EngineCall access(handle);
    TORCH_CHECK(counters && count>=7,"CUDA graph stats require seven counters");
    const auto& engine=*static_cast<Engine*>(handle);
    auto s=engine.graphs?engine.graphs->stats():tei::CudaGraphCache::Stats{};
    counters[0]=s.hits;counters[1]=s.misses;counters[2]=s.captures;counters[3]=s.attempts;
    counters[4]=s.evictions;counters[5]=s.live_bytes;counters[6]=s.admitted_bytes;
    if(count>=9){counters[7]=s.prefix_hits;counters[8]=s.prefix_captures;}
  });
}
int64_t tei_classification_width(void* handle) {
  int64_t width = -1;
  checked([&] { EngineCall access(handle);width = static_cast<Engine*>(handle)->model->classification_width(); });
  return width;
}
int64_t tei_pooled_width(void* handle) {
  int64_t width = -1;
  checked([&] {
    EngineCall access(handle);
    const auto& engine = *static_cast<Engine*>(handle);
    width = engine.dense->output_width(engine.options.string("_pooling") == "splade" ? engine.config.vocab : engine.model->output_width());
  });
  return width;
}
int32_t tei_ready(void* handle) { return checked([&] { EngineCall access(handle,true);static_cast<Engine*>(handle)->ready();access.finish(); }); }
int32_t tei_decision_counts(void* handle,const TeiDecision* requests,size_t count,int64_t* counts) {
 return checked([&]{EngineCall access(handle);auto& engine=*static_cast<Engine*>(handle);TORCH_CHECK(engine.decision,"Model was not loaded for typed decisions");
  TORCH_CHECK(counts||!count,"Missing decision counts buffer");
  tei::validate_decision_metadata_budget(requests,count,input_upload_limit(engine));
  auto converted=decision_requests(requests,count);
  for(size_t i=0;i<count;++i)counts[i]=engine.decision->output_count(converted[i]);
 });
}
int32_t tei_decide(void* handle,const int64_t* ids,const int64_t* types,const int64_t* positions,
 const int32_t* cumulative,int64_t batch,int64_t max_sequence,const TeiDecision* requests,
 float* logits,size_t capacity,float* actions,const int64_t* media_positions,
 const TeiImage* images,size_t image_count,const TeiAudio* audios,size_t audio_count) {
 return tei_decide_radix(handle,ids,types,positions,cumulative,batch,max_sequence,requests,logits,capacity,actions,media_positions,images,image_count,audios,audio_count,nullptr);
}
int32_t tei_decide_radix(void* handle,const int64_t* ids,const int64_t* types,const int64_t* positions,
 const int32_t* cumulative,int64_t batch,int64_t max_sequence,const TeiDecision* requests,
 float* logits,size_t capacity,float* actions,const int64_t* media_positions,
 const TeiImage* images,size_t image_count,const TeiAudio* audios,size_t audio_count,const TeiRadix* radix) {
 return checked([&]{EngineCall access(handle,true);auto& engine=*static_cast<Engine*>(handle);
  const auto input_bytes=validate_request_input(engine,ids,types,positions,cumulative,batch,max_sequence,media_positions,images,image_count,audios,audio_count,radix);
  const bool blocks=engine.options.string("_decision_attention_mode")=="block_bidirectional";
  TORCH_CHECK(!blocks||!image_count||media_positions,"OneJev block images require mRoPE coordinates");
  TORCH_CHECK(engine.decision&&batch>0,"Model was not loaded for typed decisions");
  tei::validate_decision_metadata_budget(requests,batch,input_upload_limit(engine),input_bytes);
  auto converted=decision_requests(requests,batch);size_t expected=0;
  for(const auto& request:converted){auto count=engine.decision->output_count(request);TORCH_CHECK(count>0&&static_cast<size_t>(count)<=capacity-expected,"Decision output exceeds provided capacity");expected+=count;}
  TORCH_CHECK((logits||!expected)&&actions,"Missing decision output buffer");
  SynchronousStreamGuard stream_guard(engine);
  auto packed=packed_input(engine,ids,types,positions,cumulative,batch,max_sequence,media_positions,images,image_count,audios,audio_count);
  attach_radix(engine,packed,ids,types,positions,cumulative,batch,max_sequence,media_positions,images,image_count,audio_count,radix);
  if(blocks) {
    TORCH_CHECK(!audio_count&&!radix,"OneJev block attention does not support audio or radix folding");
    for(int64_t row=0;row<batch;++row) {
      const auto length=int64_t(cumulative[row+1]-cumulative[row]);
      const auto boundary=requests[row].kind==3?length:requests[row].question_type;
      TORCH_CHECK((requests[row].kind==4||requests[row].kind==3)&&boundary>0&&boundary<=length,
        "OneJev block attention requires an explicit valid prefix boundary");
      int64_t source=row;
      for(int64_t prev=0;prev<row;++prev)if(packed.block_prefix_lengths[prev]==boundary &&
        tei::same_block_prefix(row,prev,boundary,ids,types,positions,media_positions,cumulative[batch],cumulative,images,image_count)) {source=packed.block_prefix_sources[prev];break;}
      packed.block_prefix_lengths.push_back(boundary);packed.block_prefix_sources.push_back(source);
    }
  } else for(int64_t row=0;row<batch;++row)TORCH_CHECK(requests[row].kind!=4,"Block decision metadata requires a block-attention checkpoint");
  const bool cached=!blocks&&!engine.prefix_probe.load(std::memory_order_relaxed)&&engine.prefix&&engine.model->supports_prefix_cache(0)
    &&std::none_of(converted.begin(),converted.end(),[](const auto& request){return request.kind==tei::DecisionRequest::Kind::Warmup;});
  const bool use_graph=!blocks&&engine.graphs&&engine.graph_policy_ready()&&(engine.media_graphs||(!image_count&&!audio_count&&!media_positions));
  auto prepared_decision=engine.decision->prepare(packed,converted);
  if(cached||!use_graph)engine.model->prepare_input(packed);
  auto hidden=cached?engine.prefix->run(*engine.model,packed,0,use_graph?engine.graphs.get():nullptr):use_graph?engine.graphs->forward(*engine.model,packed):engine.model->forward(packed);
  auto results=engine.decision->forward_prepared(hidden,packed,converted,prepared_decision);TORCH_CHECK(results.size()==size_t(batch),"Decision head output count mismatch");
  std::vector<at::Tensor> values;for(size_t i=0;i<results.size();++i){TORCH_CHECK(results[i].logits.numel()==engine.decision->output_count(converted[i]),"Decision head output width mismatch");values.push_back(results[i].logits.flatten());}
  std::vector<at::Tensor> action_values;for(const auto& result:results)action_values.push_back(result.action_probability);
  values.push_back(at::stack(action_values));auto output=at::cat(values).to(at::kFloat).to(at::kCPU).contiguous();
  std::memcpy(logits,output.data_ptr<float>(),expected*sizeof(float));std::memcpy(actions,output.data_ptr<float>()+expected,batch*sizeof(float));
  access.finish();
 });
}
static int32_t forward_radix(void* handle, const int64_t* ids, const int64_t* types,
                    const int64_t* positions, const int32_t* cumulative,
                    int64_t batch, int64_t max_sequence, int32_t pool,
                    const int64_t* pooled, size_t pooled_count,
                    const int64_t* raw, size_t raw_count, float* output, size_t capacity,
                    const int64_t* media_positions, const TeiImage* images, size_t image_count,
                    const TeiAudio* audios, size_t audio_count, const TeiRadix* radix, tei::OutputLease* completion);
int32_t tei_forward(void* handle, const int64_t* ids, const int64_t* types,
                    const int64_t* positions, const int32_t* cumulative,
                    int64_t batch, int64_t max_sequence, int32_t pool,
                    const int64_t* pooled, size_t pooled_count,
                    const int64_t* raw, size_t raw_count, float* output, size_t capacity,
                    const int64_t* media_positions, const TeiImage* images, size_t image_count,
                    const TeiAudio* audios, size_t audio_count) {
  return tei_forward_radix(handle,ids,types,positions,cumulative,batch,max_sequence,pool,
    pooled,pooled_count,raw,raw_count,output,capacity,media_positions,images,image_count,
    audios,audio_count,nullptr);
}
static int32_t forward_radix(void* handle, const int64_t* ids, const int64_t* types,
                    const int64_t* positions, const int32_t* cumulative,
                    int64_t batch, int64_t max_sequence, int32_t pool,
                    const int64_t* pooled, size_t pooled_count,
                    const int64_t* raw, size_t raw_count, float* output, size_t capacity,
                    const int64_t* media_positions, const TeiImage* images, size_t image_count,
                    const TeiAudio* audios, size_t audio_count, const TeiRadix* radix, tei::OutputLease* completion) {
  return checked([&] {
    EngineCall access(handle,true);
    auto& model = *static_cast<Engine*>(handle);
    validate_forward_descriptors(model,ids,types,positions,cumulative,batch,max_sequence,pool,
      pooled,pooled_count,raw,raw_count,capacity,media_positions,images,image_count,audios,audio_count,radix);
    std::optional<SynchronousStreamGuard> stream_guard;
    if(!completion)stream_guard.emplace(model);
    TORCH_CHECK(completion||output||(!pooled_count&&!raw_count),"Missing native output buffer");
    auto packed=packed_input(model,ids,types,positions,cumulative,batch,max_sequence,
      media_positions,images,image_count,audios,audio_count,completion);
    auto upload=[&](const at::Tensor& host){return completion?completion->upload(host):host.to(model.device);};
    attach_radix(model,packed,ids,types,positions,cumulative,batch,max_sequence,media_positions,images,image_count,audio_count,radix,completion);
    const int operation=pool>=4?(pool==5?2:1):0;
    const bool cached=!model.prefix_probe.load(std::memory_order_relaxed)&&model.prefix&&model.model->supports_prefix_cache(operation);
    const bool graph_eligible = model.graphs && model.graph_policy_ready() && (model.media_graphs || (!image_count && !audio_count && !media_positions));
    if(cached||!graph_eligible)model.model->prepare_input(packed);
    auto h = cached ? model.prefix->run(*model.model,packed,operation,graph_eligible?model.graphs.get():nullptr) : graph_eligible ? model.graphs->forward(*model.model, packed, pool >= 4 ? (pool == 5 ? 2 : 1) : 0)
      : pool >= 4 ? model.model->predict(packed, pool == 5) : model.model->forward(packed);
    // Project only requested real token spans once. Offsets below preserve
    // caller order, repeated requests and mixed pooled/raw output semantics.
    at::Tensor sparse_scores;
    std::vector<int64_t> sparse_starts;
    if (pool == 3 && pooled_count) {
      std::vector<at::Tensor> selected;
      int64_t cursor = 0;
      for (size_t i = 0; i < pooled_count; ++i) {
        const auto row = pooled[i];
        const auto start = cumulative[row], length = cumulative[row + 1] - start;
        sparse_starts.push_back(cursor); cursor += length;
        selected.push_back(h.narrow(0, start, length));
      }
      sparse_scores = model.model->token_scores(at::cat(selected, 0));
    }
    std::vector<at::Tensor> outputs;
    bool mean_pooled=false;
#ifdef TEI_TORCH_CUDA_KERNELS
    if(pool==1&&pooled_count&&!model.model->model_dtype_mean_pool()&&h.is_cuda()&&h.is_contiguous()
       &&(h.scalar_type()==at::kHalf||h.scalar_type()==at::kBFloat16)&&pooled_count<=65535) {
      std::vector<int32_t> spans;spans.reserve(pooled_count*2);
      for(size_t i=0;i<pooled_count;++i) {
        spans.push_back(cumulative[pooled[i]]);
        spans.push_back(cumulative[pooled[i]+1]-cumulative[pooled[i]]);
      }
      auto host=at::from_blob(spans.data(),{int64_t(pooled_count),2},at::TensorOptions().dtype(at::kInt));
      outputs.push_back(tei::packed_mean_pool(h,upload(host)));
      mean_pooled=true;
    }
#endif
    for (size_t i = 0; i < pooled_count && !mean_pooled; ++i) {
      const auto row = pooled[i];
      const auto start = cumulative[row], length = cumulative[row + 1] - start;
      if (pool == 4) outputs.push_back(h.narrow(0, row, 1));
      else if (pool == 3) outputs.push_back(std::get<0>(sparse_scores.narrow(0, sparse_starts[i], length).max(0, true)));
      else if (pool == 0) outputs.push_back(h.narrow(0, start, 1));
      else if (pool == 1) outputs.push_back(model.model->mean_pool(h.narrow(0, start, length)));
      else outputs.push_back(h.narrow(0, start + length - 1, 1));
    }
    if (pool < 4 && !model.dense->empty() && !outputs.empty()) {
      auto pooled_output = model.dense->forward(at::cat(outputs, 0));
      outputs.clear();
      outputs.push_back(pooled_output);
    }
    // Identity raw selection already occupies one packed contiguous tensor.
    // Preserve the completion path while avoiding a redundant device cat.
    bool identity_raw = !pooled_count && raw_count == static_cast<size_t>(batch)
      && pool < 4 && h.is_cuda() && h.is_contiguous()
      && h.size(0) == cumulative[batch];
    for (size_t i = 0; i < raw_count && identity_raw; ++i)
      identity_raw = raw[i] == static_cast<int64_t>(i);
    for (size_t i = 0; i < raw_count; ++i)
      outputs.push_back(h.narrow(0, cumulative[raw[i]], cumulative[raw[i] + 1] - cumulative[raw[i]]));
    if (!outputs.empty()) {
      size_t offset = 0;
      std::vector<at::Tensor> group;
      auto flush = [&] {
        if (group.empty()) return;
        auto joined = identity_raw ? h : (completion && group.size() == 1 ? group.front() : at::cat(group, 0));
        TORCH_CHECK(offset <= capacity && static_cast<size_t>(joined.numel()) <= capacity - offset,
                    "Native output exceeds provided capacity");
        if (completion) completion->copy(joined, offset);
        else {
          // Cast on the producing device before transfer.
          auto cpu = joined.to(at::kFloat).to(at::kCPU).contiguous();
          std::memcpy(output + offset, cpu.data_ptr<float>(), cpu.nbytes());
        }
        offset += joined.numel();
        group.clear();
      };
      for (const auto& tensor : outputs) {
        if (!group.empty() && group.back().size(-1) != tensor.size(-1)) flush();
        group.push_back(tensor);
      }
      flush();
      if (completion) TORCH_CHECK(offset == capacity, "Native output size mismatch");
    }
    if (completion) completion->record();
    access.finish();
  });
}
int32_t tei_forward_radix(void* handle, const int64_t* ids, const int64_t* types,
                    const int64_t* positions, const int32_t* cumulative,
                    int64_t batch, int64_t max_sequence, int32_t pool,
                    const int64_t* pooled, size_t pooled_count,
                    const int64_t* raw, size_t raw_count, float* output, size_t capacity,
                    const int64_t* media_positions, const TeiImage* images, size_t image_count,
                    const TeiAudio* audios, size_t audio_count, const TeiRadix* radix) {
  return forward_radix(handle,ids,types,positions,cumulative,batch,max_sequence,pool,pooled,pooled_count,raw,raw_count,output,capacity,media_positions,images,image_count,audios,audio_count,radix,nullptr);
}
void* tei_forward_submit(void* handle, const int64_t* ids, const int64_t* types,
                    const int64_t* positions, const int32_t* cumulative,
                    int64_t batch, int64_t max_sequence, int32_t pool,
                    const int64_t* pooled, size_t pooled_count,
                    const int64_t* raw, size_t raw_count, size_t capacity,
                    const int64_t* media_positions, const TeiImage* images, size_t image_count,
                    const TeiAudio* audios, size_t audio_count, const TeiRadix* radix) {
  std::unique_ptr<tei::OutputLease> lease;
  const auto status=checked([&] {
    std::shared_ptr<tei::OutputPool> output_pool;auto device=c10::Device(c10::kCPU);
    {
      EngineCall access(handle);auto& model=*static_cast<Engine*>(handle);
      validate_forward_descriptors(model,ids,types,positions,cumulative,batch,max_sequence,pool,
        pooled,pooled_count,raw,raw_count,capacity,media_positions,images,image_count,audios,audio_count,radix,true);
      TORCH_CHECK(model.device.is_cuda(), "Deferred output requires CUDA");
      output_pool=model.outputs;device=model.device;
    }
    // Waiting for the A/B slot must not hold the engine lock. Completion and
    // cancellation own the pool independently of the model's submission lock.
    lease=std::make_unique<tei::OutputLease>(std::move(output_pool),device,capacity);
    lease->allocate();
    if (forward_radix(handle,ids,types,positions,cumulative,batch,max_sequence,pool,pooled,pooled_count,raw,raw_count,nullptr,capacity,media_positions,images,image_count,audios,audio_count,radix,lease.get()) != 0)
      throw tei::NativeCallError(error_kind);
  });
  return status==0?lease.release():nullptr;
}
const float* tei_output_wait(void* handle) {
  const float* output=nullptr;
  checked([&] {output=static_cast<tei::OutputLease*>(handle)->wait();});
  return output;
}
void tei_output_destroy(void* handle) { delete static_cast<tei::OutputLease*>(handle); }
void tei_destroy(void* handle) { delete static_cast<Engine*>(handle); }
}

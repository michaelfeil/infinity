#include "helpers/vision_batch.h"
#include "helpers/attention.h"
// Native adaptations of backends/candle/src/models/{gemma3,gemma4,embedding_gemma2}.rs.
// Upstream Apache-2.0 notices remain in LICENSE and NOTICE.
#include "models/gemma_models.h"
#include "helpers/image_feature_cache.h"
#include "helpers/fp8_linear.h"
#include "kernels/fp8_fusions.h"
#include "helpers/workspace_admission.h"
#include "helpers/vendor_error.h"
#include "batching/radix.h"
#include "batching/prefix_cache.h"
#include "helpers/prefix_linear.h"
#include "models/decoder_models.h"
#include "kernels/qwen35_kernels.h"
#include "kernels/gemma2_norm.h"
#include "kernels/moe_kernels.h"
#include "kernels/fast_kernels.h"
#include "kernels/torch_flash_fma.h"
#include <ATen/ops/im2col.h>
#include <ATen/ops/_flash_attention_forward.h>
#include <ATen/ops/_efficient_attention_forward.h>
#include <ATen/ops/_cudnn_attention_forward.h>
#include <charconv>
#include <cmath>
#include <string_view>
#include <array>
#include <limits>
#include <map>
#include <memory>
#ifdef TEI_TORCH_CUDA_KERNELS
#include "kernels/gemma_score.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#endif

namespace tei {
namespace {
class Gemma final : public Model {
  Options cfg;
  Weights& weights;
  c10::Device device;
  at::ScalarType dtype;
  std::string prefix, family;
  int64_t hidden, heads, layers, ple_width;
  double epsilon;
  bool legacy, embedding2, causal;
  bool publisher_sequence=false;
  bool embedding2_reference_cuda = false;
  bool fused_legacy_residual=false;
  struct Layer {
    std::string p;
    at::Tensor qkv, gate_up;
    std::unique_ptr<Fp8Linear> fp8_gate_up, fp8_down;
    std::unique_ptr<Fp8GroupedLinear> fp8_expert_gate_up, fp8_expert_down;
    int64_t dim, kv, window;
    bool sliding, shared, store, k_equals_v;
    at::Tensor cos, sin;
  };
  std::vector<Layer> stack;
  std::unique_ptr<OrdinaryLinearExecutor> ordinary_ffn;
  Weights norm_scales;
  Weights router_f32;
  at::Tensor vision_frequency, legacy_embedding_scale;
  mutable ImageFeatureCache image_features;
  std::unordered_map<int64_t,at::Tensor> norm_ones;
  std::unordered_map<std::string,std::array<double,4>> audio_bounds;
  // Choose the same accumulation tree as the original logical-row GEMM,
  // using descriptors only. Execute on actual compact rows, with no padding.
#ifdef TEI_TORCH_CUDA_KERNELS
  struct RadixLinearPlan {
    cublasLtMatmulDesc_t operation{};
    cublasLtMatrixLayout_t weight{},input{},output{};
    cublasLtMatmulAlgo_t algorithm{};
    size_t workspace_bytes{};
    // Captured graphs retain this allocation address even when a later plan
    // grows the shared workspace. Keep each cached plan's generation alive.
    std::shared_ptr<WorkspaceMemoryPermit> memory_permit;
    at::Tensor workspace;
    bool valid=false;
    ~RadixLinearPlan(){if(output)cublasLtMatrixLayoutDestroy(output);if(input)cublasLtMatrixLayoutDestroy(input);if(weight)cublasLtMatrixLayoutDestroy(weight);if(operation)cublasLtMatmulDescDestroy(operation);}
  };
  // Includes device, real logical/compact dimensions, operand strides/types,
  // and the ATen workspace policy. Plans are shared by identical layer shapes.
  using RadixLinearKey=std::array<int64_t,15>;
  mutable std::map<RadixLinearKey,std::unique_ptr<RadixLinearPlan>> radix_linear_plans;
  mutable std::shared_ptr<WorkspaceMemoryPermit> radix_linear_memory_permit;
  mutable at::Tensor radix_linear_workspace;
#endif
  at::Tensor prefix_projection(const at::Tensor& x,const std::string& name,const PackedInput& input) const {
    // Keep the original projection recipe when a cache hit leaves only suffix rows.
    // Cold capture and unsupported recipes retain their original operator.
    return input.prefix?prefix_linear(x,w(name+".weight"),has(name+".bias")?w(name+".bias"):at::Tensor(),input):linear(x,name);
  }
  at::Tensor linear_radix_projection(const at::Tensor& x,const std::string& name,const PackedInput& packed,const at::Tensor& supplied=at::Tensor{}) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    const auto& matrix=supplied.defined()?supplied:w(name+".weight");
    // Cached Gemma4 routers retain the original FP32 reduction geometry. Cold
    // execution still uses its original operator and BLAS provider; warm plans
    // use ATen's stream-owned scratch rather than a model-owned workspace.
    if(packed.prefix&&packed.prefix->linear&&supplied.defined()&&!legacy&&!embedding2
      &&boolean("enable_moe_block",false)&&device.is_cuda()&&dtype==at::kBFloat16
      &&hidden==2816&&heads==16&&x.dim()==2&&matrix.dim()==2
      &&x.scalar_type()==at::kFloat&&matrix.scalar_type()==at::kFloat
      &&x.size(1)==2816&&matrix.sizes()==at::IntArrayRef({128,2816})
      &&std::any_of(packed.prefix->hits.begin(),packed.prefix->hits.end(),[](auto hit){return hit!=0;}))
      return packed.prefix->linear->forward(x,matrix,{},prefix_logical_input(packed).ids.numel(),true);
    const bool fp32_router=supplied.defined() && x.scalar_type()==at::kFloat && matrix.scalar_type()==at::kFloat &&
      !at::globalContext().allowTF32CuBLAS() && at::globalContext().blasPreferredBackend()==at::BlasBackend::Cublaslt;
    if(!legacy && !embedding2 && boolean("enable_moe_block",false) &&
       device.is_cuda() && dtype==at::kBFloat16 && hidden==2816 && heads==16 &&
       x.dim()==2 && (x.size(1)==4096 || x.size(1)==8192 || x.size(1)==2816) && x.size(0)>0 &&
       x.is_contiguous() && matrix.is_contiguous() &&
       ((x.scalar_type()==at::kBFloat16 && matrix.scalar_type()==at::kBFloat16) || fp32_router) &&
       matrix.dim()==2 && matrix.size(1)==x.size(1) &&
       ((matrix.size(0)==2816 && (x.size(1)==4096 || x.size(1)==8192)) ||
        (matrix.size(0)==1024 && x.size(1)==2816) ||
        (fp32_router && matrix.size(0)==128 && x.size(1)==2816)) && !has(name+".bias")) {
      c10::cuda::CUDAGuard guard(x.device());
      auto property=at::cuda::getDeviceProperties(device.index());
      if(property->major==9 && property->minor==0 &&
         reinterpret_cast<uintptr_t>(x.const_data_ptr())%256==0 &&
         reinterpret_cast<uintptr_t>(matrix.const_data_ptr())%256==0) {
        const auto m=x.size(0),full_m=prefix_logical_input(packed).ids.numel(),k=x.size(1),n=matrix.size(0);
        const auto limit=at::cuda::getCUDABlasLtWorkspaceSize();
        const RadixLinearKey key{{device.index(),full_m,m,k,n,x.stride(0),x.stride(1),matrix.stride(0),matrix.stride(1),n,1,static_cast<int64_t>(x.scalar_type()),static_cast<int64_t>(matrix.scalar_type()),static_cast<int64_t>(x.scalar_type()),static_cast<int64_t>(limit)}};
        auto found=radix_linear_plans.find(key);
        if(found==radix_linear_plans.end() && radix_linear_plans.size()<64) {
          auto plan=std::make_unique<RadixLinearPlan>();
          struct Planning {
            cublasLtMatrixLayout_t input{},output{};
            cublasLtMatmulPreference_t preference{};
            ~Planning(){if(preference)cublasLtMatmulPreferenceDestroy(preference);if(output)cublasLtMatrixLayoutDestroy(output);if(input)cublasLtMatrixLayoutDestroy(input);}
          } logical;
          auto transpose=CUBLAS_OP_T,no_transpose=CUBLAS_OP_N;
          const auto data_type=fp32_router?CUDA_R_32F:CUDA_R_16BF;
          bool valid=cublasLtMatmulDescCreate(&plan->operation,CUBLAS_COMPUTE_32F,CUDA_R_32F)==CUBLAS_STATUS_SUCCESS;
          if(valid)valid=cublasLtMatmulDescSetAttribute(plan->operation,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose))==CUBLAS_STATUS_SUCCESS;
          if(valid)valid=cublasLtMatmulDescSetAttribute(plan->operation,CUBLASLT_MATMUL_DESC_TRANSB,&no_transpose,sizeof(no_transpose))==CUBLAS_STATUS_SUCCESS;
          if(valid)valid=cublasLtMatrixLayoutCreate(&plan->weight,data_type,k,n,k)==CUBLAS_STATUS_SUCCESS;
          if(valid)valid=cublasLtMatrixLayoutCreate(&plan->input,data_type,k,m,k)==CUBLAS_STATUS_SUCCESS;
          if(valid)valid=cublasLtMatrixLayoutCreate(&plan->output,data_type,n,m,n)==CUBLAS_STATUS_SUCCESS;
          // Logical layouts never allocate or compute duplicated rows.
          if(valid)valid=cublasLtMatrixLayoutCreate(&logical.input,data_type,k,full_m,k)==CUBLAS_STATUS_SUCCESS;
          if(valid)valid=cublasLtMatrixLayoutCreate(&logical.output,data_type,n,full_m,n)==CUBLAS_STATUS_SUCCESS;
          if(valid)valid=cublasLtMatmulPreferenceCreate(&logical.preference)==CUBLAS_STATUS_SUCCESS;
          if(valid)valid=cublasLtMatmulPreferenceSetAttribute(logical.preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&limit,sizeof(limit))==CUBLAS_STATUS_SUCCESS;
          cublasLtMatmulHeuristicResult_t original{},check{};int count=0;
          auto handle=at::cuda::getCurrentCUDABlasLtHandle();
          if(valid)valid=cublasLtMatmulAlgoGetHeuristic(handle,plan->operation,plan->weight,logical.input,logical.output,logical.output,logical.preference,1,&original,&count)==CUBLAS_STATUS_SUCCESS && count==1 && original.state==CUBLAS_STATUS_SUCCESS;
          if(valid) {
            plan->algorithm=original.algo;
            valid=cublasLtMatmulAlgoCheck(handle,plan->operation,plan->weight,plan->input,plan->output,plan->output,&plan->algorithm,&check)==CUBLAS_STATUS_SUCCESS && check.state==CUBLAS_STATUS_SUCCESS && check.workspaceSize<=limit;
            plan->workspace_bytes=check.workspaceSize;
          }
          plan->valid=valid;
          found=radix_linear_plans.emplace(key,std::move(plan)).first;
        }
        if(found!=radix_linear_plans.end() && found->second->valid) {
          auto& plan=*found->second;
          bool admitted=true;
          if(plan.workspace_bytes>0 && !plan.workspace.defined()) {
            // Cache a supported tactic even when shared credit is temporarily
            // unavailable. Preserve existing plan generations/captured addresses.
            if(!radix_linear_workspace.defined() || static_cast<size_t>(radix_linear_workspace.numel())<plan.workspace_bytes) {
              auto permit=reserve_workspace_memory(x.get_device(),limit);
              if(!permit)admitted=false;
              else {
                radix_linear_workspace=at::empty({static_cast<int64_t>(limit)},x.options().dtype(at::kByte));
                radix_linear_memory_permit=std::move(permit);
              }
            }
            if(admitted){plan.workspace=radix_linear_workspace;plan.memory_permit=radix_linear_memory_permit;}
          }
          if(admitted) {
          auto output=at::empty({m,n},x.options());
          const float alpha=1.f,beta=0.f;
          auto status=cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(),plan.operation,&alpha,matrix.const_data_ptr(),plan.weight,x.const_data_ptr(),plan.input,&beta,output.const_data_ptr(),plan.output,output.mutable_data_ptr(),plan.output,&plan.algorithm,plan.workspace_bytes?plan.workspace.mutable_data_ptr():nullptr,plan.workspace_bytes,c10::cuda::getCurrentCUDAStream(device.index()).stream());
          if(status==CUBLAS_STATUS_ALLOC_FAILED)
            throw VendorAllocationError{"Gemma4 packed projection cublasLtMatmul",static_cast<int32_t>(status)};
          TORCH_CHECK(status==CUBLAS_STATUS_SUCCESS,"Gemma4 packed projection failed: ",static_cast<int>(status));
          return output;
          }
        }
      }
    }
#endif
    // Unsupported shapes retain original full-row arithmetic. These are the
    // request's real logical rows, never rounded-up or padded rows.
    if(supplied.defined())return radix_fold(at::linear(radix_unfold(x,packed),supplied),packed);
    if(!legacy && !embedding2 && boolean("enable_moe_block",false) && dtype==at::kBFloat16 && hidden==2816 && heads==16)
      return radix_fold(linear(radix_unfold(x,packed),name),packed);
    return prefix_projection(x,name,packed);
  }
  at::Tensor moe(const at::Tensor& residual,const at::Tensor& dense,const Layer& layer,const PackedInput& packed) const {
    const auto& p=layer.p;
    auto input=norm(residual,p+"pre_feedforward_layernorm_2");
    // Preserve the router's BF16 rounding before both root-size and learned scales.
    auto root=at::scalar_tensor(1./std::sqrt(double(hidden)),residual.options());
    auto routing=((norm(residual,"",true)*root)*w(p+"router.scale")).to(at::kFloat);
    // Preserve original FP32 accumulation with the logical-shape plan while
    // computing only actual compact router rows. Unsupported plans retain
    // the original logical-row ATen projection through the helper fallback.
    const auto& router_matrix=weight(router_f32,prefix+p+"router.proj.weight");
    auto logits=radix_enabled(packed)||packed.prefix
      ?linear_radix_projection(routing,p+"router.proj",packed,router_matrix)
      :at::linear(routing,router_matrix);
    auto count=integer("num_experts",0),top=integer("top_k_experts",8);
    TORCH_CHECK(count>0 && top>0 && top<=count,"Invalid Gemma4 MoE routing geometry");
    const auto gate_up=layer.fp8_expert_gate_up?at::Tensor():w(p+"experts.gate_up_proj");
    const auto down=layer.fp8_expert_down?at::Tensor():w(p+"experts.down_proj");
    auto act=string("hidden_activation","gelu_pytorch_tanh");
    if(!layer.fp8_expert_gate_up && device.is_cuda() && dtype==at::kBFloat16 && count==128 && top==8 && hidden==2816 && down.size(2)==704 &&
       (act=="gelu_pytorch_tanh" || act=="gelu_new")) {
      auto output=routed_moe_cuda(input,logits,gate_up,down,true,weight(router_f32,prefix+p+"router.per_expert_scale"));
      if(output.defined())return norm(dense,p+"post_feedforward_layernorm_1")+norm(output,p+"post_feedforward_layernorm_2");
    }
    // Stable descending order gives lower expert IDs precedence on tied logits.
    auto ids=std::get<1>(at::sort(logits,true,-1,true)).narrow(-1,0,top);
    auto probability=at::softmax(logits,-1).gather(-1,ids);
    probability=probability/probability.sum(-1,true);
    probability=probability*weight(router_f32,prefix+p+"router.per_expert_scale").index_select(0,ids.reshape({-1})).view_as(ids);
    auto activation=[&](const at::Tensor& gu) {
#ifdef TEI_TORCH_CUDA_KERNELS
      auto act=string("hidden_activation","gelu_pytorch_tanh");
      if(device.is_cuda()&&dtype==at::kBFloat16&&act!="relu"&&act!="gelu")return gemma_gated_cuda(gu,act!="silu"&&act!="swish");
#endif
      auto width=gu.size(-1)/2;
      return activate(gu.narrow(-1,0,width))*gu.narrow(-1,width,width);
    };
    at::Tensor output;
    if(layer.fp8_expert_gate_up) {
      const auto flat=ids.flatten(),order=at::argsort(flat,true,0,false);
      const auto tokens=at::floor_divide(order,top);
      auto counts=at::zeros({count},ids.options().dtype(at::kInt));
      counts.scatter_add_(0,flat,at::ones_like(flat,flat.options().dtype(at::kInt)));
      const auto ends=counts.cumsum(0,at::kInt);
      // Quantize each token once, before routing duplicates it across experts.
      auto quantized=fp8_quantize_rows(input.contiguous());
      auto both=layer.fp8_expert_gate_up->forward_quantized(quantized.first.index_select(0,tokens),
        quantized.second.index_select(0,tokens),ends);
      at::Tensor expert;
#ifdef TEI_TORCH_CUDA_KERNELS
      if((act=="gelu_pytorch_tanh" || act=="gelu_new") && both.size(1)<=16384) {
        auto gated=fp8_geglu_quantize(both);
        expert=layer.fp8_expert_down->forward_quantized(gated.first,gated.second,ends);
      } else
#endif
        expert=layer.fp8_expert_down->forward(activation(both),ends);
      output=routed_moe_combine_cuda(expert,order,probability);
      if(!output.defined()) {
        expert=(expert.to(at::kFloat)*probability.flatten().index_select(0,order).unsqueeze(1)).to(dtype);
        auto ordered=at::empty_like(expert);ordered.index_copy_(0,order,expert);
        output=ordered.view({input.size(0),top,hidden}).to(at::kFloat).sum(1).to(dtype);
      }
    } else output=decoder_moe_dispatch(input,gate_up,down,ids,probability,activation);
    return norm(dense,p+"post_feedforward_layernorm_1")+norm(output.to(dtype),p+"post_feedforward_layernorm_2");
  }
  at::Tensor audio(const AudioInput& media) const {
    auto channels=cfg.integer("audio_config.hidden_size",1024);
    auto heads=cfg.integer("audio_config.num_attention_heads",cfg.integer("audio_config.conf_num_attention_heads",8));
    auto layers=cfg.integer("audio_config.num_hidden_layers",cfg.integer("audio_config.conf_num_hidden_layers",12));
    auto chunk=cfg.integer("audio_config.attention_chunk_size",12);
    auto past=cfg.integer("audio_config.attention_context_left",13)-1;
    auto future=cfg.integer("audio_config.attention_context_right",0);
    auto eps=cfg.real("audio_config.rms_norm_eps",1e-6),cap=cfg.real("audio_config.attention_logit_cap",50.);
    auto clip=std::min(cfg.real("audio_config.gradient_clipping",1e10),static_cast<double>(std::numeric_limits<float>::max()));
    TORCH_CHECK(media.features.dim()==2 && media.features.size(1)==128 && media.validity.numel()==media.features.size(0),"Invalid Gemma audio features");
    TORCH_CHECK(channels>0 && heads>0 && channels%heads==0 && chunk>0 && past>=0 && future>=0 && cap>0,"Invalid Gemma audio architecture");
    auto root=weights.count("model.audio_tower.subsample_conv_projection.layer0.conv.weight")?std::string("model."):std::string();
    auto aw=[&](const std::string& name)->const at::Tensor& { return weight(weights,root+name); };
    auto al=[&](const at::Tensor& value,const std::string& name) {
      auto base=root+name;
      auto input=value;
      auto bounds=audio_bounds.find(base);
      if(bounds!=audio_bounds.end()) input=input.clamp(bounds->second[0],bounds->second[1]);
      auto key=base+".weight";
      if(!weights.count(key))key=base+".linear.weight";
      std::optional<at::Tensor> bias;
      if(weights.count(base+".bias"))bias=weight(weights,base+".bias");
      auto output=gemma_audio_linear(input.to(weight(weights,key).scalar_type()),weight(weights,key),bias.value_or(at::Tensor{}));
      if(bounds!=audio_bounds.end())output=output.clamp(bounds->second[2],bounds->second[3]);
      return output;
    };
    auto an=[&](const at::Tensor& x,const std::string& name) {
#ifdef TEI_TORCH_CUDA_KERNELS
      if(device.is_cuda()&&x.scalar_type()==at::kBFloat16)
        return gemma_audio_norm_cuda(x.contiguous(),weight(norm_scales,root+name+".weight"),eps);
#endif
      auto f=x.to(at::kFloat);
      return (f*at::pow(f.square().mean(-1,true)+eps,-.5)*weight(norm_scales,root+name+".weight")).to(x.scalar_type());
    };
    auto valid=media.validity.to(at::kBool);
    auto states=media.features.to(dtype).unsqueeze(0).unsqueeze(0);
    for(int block=0;block<2;++block) {
      auto p="audio_tower.subsample_conv_projection.layer"+std::to_string(block);
      auto filter=aw(p+".conv.weight").to(at::kFloat);
      auto stride=cfg.integer("audio_config.sscp_conv_stride_size."+std::to_string(block)+".0",2);
      // Convolution's mathematical zero boundary is intrinsic to the filter;
      // actual audio frame lengths are never rounded up or padded into batches.
      states=gemma_audio_conv2d((states*valid.view({1,1,-1,1})).to(at::kFloat),filter,stride).to(dtype);
      auto ix=at::arange(states.size(2),valid.options().dtype(at::kLong))*stride;
      valid=valid.index_select(0,ix.clamp_max(valid.size(0)-1));
      auto x=states.permute({0,2,3,1}).to(at::kFloat);
      auto centered=x-x.mean(-1,true);
      x=centered/at::sqrt(centered.square().mean(-1,true)+eps);
      if(weights.count(root+p+".norm.weight"))x=x*aw(p+".norm.weight").to(at::kFloat);
      states=at::relu(x.to(dtype).permute({0,3,1,2}));
    }
    states=states.permute({0,2,3,1}).reshape({states.size(2),-1});
    states=al(states,"audio_tower.subsample_conv_projection.input_proj_linear");

    auto time=states.size(0),dim=channels/heads;
    auto relative_span=(chunk+past+future)/2+1;
    auto timing=at::arange(relative_span-1,-1,-1,states.options().dtype(at::kFloat));
    auto inv=at::exp(at::arange(channels/2,timing.options())*(-std::log(10000.)/std::max<int64_t>(channels/2-1,1)));
    auto angles=timing.unsqueeze(1)*inv;
    auto signal=at::cat({angles.sin(),angles.cos()},-1).to(dtype);
    auto ff=[&](const at::Tensor& value,const std::string& p) {
      auto x=an(value.clamp(-clip,clip),p+"pre_layer_norm");

      x=al(x,p+"ffw_layer_1");
      x=at::silu(x.to(at::kFloat)).to(dtype);
      x=al(x,p+"ffw_layer_2").clamp(-clip,clip);
      x=an(x,p+"post_layer_norm");
      x=value+x*cfg.real("audio_config.residual_weight",.5);return x;
    };
    for(int64_t layer=0;layer<layers;++layer) {
      auto p="audio_tower.layers."+std::to_string(layer)+".";
      states=ff(states,p+"feed_forward1.");
      auto x=an(states.clamp(-clip,clip),p+"norm_pre_attn");
      auto q=al(x,p+"self_attn.q_proj").to(at::kFloat).view({time,heads,dim});
      auto k=al(x,p+"self_attn.k_proj").to(at::kFloat).view({time,heads,dim});
      auto v=al(x,p+"self_attn.v_proj").to(at::kFloat).view({time,heads,dim});
      auto scale=at::softplus(aw(p+"self_attn.per_dim_scale").to(at::kFloat)).to(dtype).to(at::kFloat);
      q=q*(1./std::sqrt(static_cast<double>(dim))/std::log(2.))*scale;
      k=k*(std::log(1.+std::exp(1.))/std::log(2.));
      auto relative=al(signal,p+"self_attn.relative_k_proj").to(at::kFloat).view({relative_span,heads,dim});
      std::vector<at::Tensor> blocks;
      for(int64_t start=0;start<time;start+=chunk) {
        auto count=std::min(chunk,time-start),key_start=std::max<int64_t>(0,start-past),key_end=std::min(time,start+count+future);
        auto queries=at::arange(start,start+count,q.options().dtype(at::kLong));
        auto keys=at::arange(key_start,key_end,q.options().dtype(at::kLong));
        auto delta=keys.unsqueeze(0)-queries.unsqueeze(1);
        auto qs=q.narrow(0,start,count);
        // The reference relative branch uses FP32 GEMM rather than a separate
        // product/reduction tree. Gather only the natural key offsets afterward.
        auto relative_logits=at::matmul(qs.permute({1,0,2}),relative.permute({1,2,0}));
        // Preserve the reference's flatten/reshape relative shift without
        // materializing its artificial boundary rows or columns. A shifted
        // element may come from the preceding query row; direct position
        // lookup changes the trained attention computation.
        auto context=chunk+past+future,span=relative.size(0);
        auto shifted=(queries-start).unsqueeze(1)*context+(keys-(start-past)).unsqueeze(0);
        auto source_row=at::floor_divide(shifted,context+1);
        auto source_column=at::remainder(shifted,context+1);
        auto rel_index=(source_row*span+source_column.clamp_max(span-1)).reshape({1,-1}).expand({heads,-1});
        auto shifted_relative=relative_logits.reshape({heads,-1}).gather(-1,rel_index).reshape({heads,count,key_end-key_start});
        shifted_relative=shifted_relative.masked_fill(source_column.ge(span).unsqueeze(0),0);
        auto logits=at::matmul(qs.permute({1,0,2}),k.narrow(0,key_start,key_end-key_start).permute({1,2,0}))+
          shifted_relative;
        logits=at::tanh(logits/cap)*cap;
        // The Conformer reference excludes the exact context horizon endpoint.
        auto allowed=(((delta<=0)&(delta>-past))|((delta>0)&(delta<future)))&
          valid.narrow(0,key_start,key_end-key_start).unsqueeze(0);
        logits=logits.masked_fill(allowed.logical_not().unsqueeze(0),cfg.real("audio_config.attention_invalid_logits_value",-1e9));
        blocks.push_back(at::matmul(at::softmax(logits,-1),v.narrow(0,key_start,key_end-key_start).permute({1,0,2})).permute({1,0,2}).reshape({count,channels}));
      }
      auto attended=al(at::cat(blocks,0).to(dtype),p+"self_attn.post").clamp(-clip,clip);
      states=states+an(attended,p+"norm_post_attn");

      x=al(an(states,p+"lconv1d.pre_layer_norm"),p+"lconv1d.linear_start");
      // Conformer's fused GLU has one BF16 rounding after FP32 sigmoid/product.
      auto gated=(x.narrow(-1,0,channels).to(at::kFloat)*
        at::sigmoid(x.narrow(-1,channels,channels).to(at::kFloat))).to(dtype);
      auto filter=aw(p+"lconv1d.depthwise_conv1d.weight").to(at::kFloat);
      // ATen's convolution boundary handles the causal receptive field without
      // appending any synthetic frames to the sequence tensor.
      auto conv=at::conv1d(gated.transpose(0,1).unsqueeze(0).to(at::kFloat),filter,std::nullopt,{1},{filter.size(-1)-1},{1},channels);
      conv=conv.narrow(-1,0,time).squeeze(0).transpose(0,1).to(dtype).clamp(-clip,clip);
      states=states+al(at::silu(an(conv,p+"lconv1d.conv_norm").to(at::kFloat)).to(dtype),p+"lconv1d.linear_end");

      states=an(ff(states,p+"feed_forward2.").clamp(-clip,clip),p+"norm_out");

    }
    if(weights.count(root+"audio_tower.output_proj.weight"))states=al(states,"audio_tower.output_proj");

    states=states*valid.unsqueeze(-1);
    auto f=states.to(at::kFloat);
    auto projected=al((f/at::sqrt(f.square().mean(-1,true)+eps)).to(dtype),"embed_audio.embedding_projection");
    auto selected=media.selected_frames.defined()?media.selected_frames:at::nonzero(valid).squeeze(-1);
    return projected.index_select(0,selected);
  }
  std::vector<at::Tensor> vision(const std::vector<ImageInput>& images) const {
    if(images.empty())return {};
    auto patch=cfg.integer("vision_config.patch_size",16),pool=cfg.integer("vision_config.pooling_kernel_size",3);
    auto vh=cfg.integer("vision_config.hidden_size"),nh=cfg.integer("vision_config.num_attention_heads");
    auto nk=cfg.integer("vision_config.num_key_value_heads",nh),dim=cfg.integer("vision_config.head_dim",vh/(nh?nh:1));
    auto nlayer=cfg.integer("vision_config.num_hidden_layers");
    auto eps=cfg.real("vision_config.rms_norm_eps",1e-6);
    auto batches=vision_batches(images,device,vh,cfg.integer("vision_config.intermediate_size",vh*4));
    if(batches.size()>1) {
      std::vector<at::Tensor> features;
      for(auto [start,end]:batches) {
        auto part=vision(std::vector<ImageInput>(images.begin()+start,images.begin()+end));
        features.insert(features.end(),part.begin(),part.end());
      }
      return features;
    }
    TORCH_CHECK(vh>0 && nh>0 && nk>0 && nh%nk==0 && dim>0 && dim%4==0 && nlayer>0,"Invalid Gemma vision geometry");
    TORCH_CHECK(!cfg.boolean("vision_config.use_clipped_linears",false),"Clipped Gemma vision linears unsupported");
    auto root=weights.count("model.vision_tower.patch_embedder.input_proj.weight")?std::string("model."):std::string();
    auto vw=[&](const std::string& name)->const at::Tensor& { return weight(weights,root+name); };
    auto vl=[&](const at::Tensor& x,const std::string& name) {
      auto key=root+name+".weight";
      if(!weights.count(key)) key=root+name+".linear.weight";
      return at::linear(x,weight(weights,key));
    };
    auto vn=[&](const at::Tensor& x,const std::string& name) {
#ifdef TEI_TORCH_CUDA_KERNELS
      if(device.is_cuda()&&dtype==at::kBFloat16) {
        auto scale=name.empty()?norm_ones.at(x.size(-1)):weight(norm_scales,root+name+".weight");
        return gemma_vision_norm_cuda(x.contiguous(),scale,eps);
      }
#endif
      auto f=x.to(at::kFloat)/at::sqrt(x.to(at::kFloat).square().mean(-1,true)+eps);
      if(!name.empty()) f=f*vw(name+".weight").to(at::kFloat);
      return f.to(x.scalar_type());
    };
    std::vector<at::Tensor> patches,column_parts,row_parts;
    std::vector<int32_t> offsets{0};int64_t max_image=0;
    for(const auto& image:images) {
      auto [frames,height,width]=image.grid_thw;
      TORCH_CHECK(frames==1 && height>0 && width>0 && height%pool==0 && width%pool==0 && image.merge_size==pool &&
        image.pixels.sizes()==at::IntArrayRef({height*width,patch*patch*3}),"Invalid packed Gemma vision patches");
      TORCH_CHECK(int64_t(offsets.back())+height*width<=INT32_MAX,"Gemma image batch too large");
      offsets.push_back(offsets.back()+height*width);max_image=std::max(max_image,height*width);
      auto ix=at::arange(height*width,image.pixels.options().device(device).dtype(at::kLong));
      column_parts.push_back(ix.remainder(width));row_parts.push_back(at::floor_divide(ix,width));
      patches.push_back(image.pixels.to(device,at::kFloat));
    }
    auto columns=at::cat(column_parts,0),rows=at::cat(row_parts,0);
    auto table=vw("vision_tower.patch_embedder.position_embedding_table");
    // Normalize source pixels in FP32 before the projection's activation cast.
    auto states=vl(((at::cat(patches,0)-.5)*2.).to(dtype),"vision_tower.patch_embedder.input_proj")+
      (table.select(0,0).index_select(0,columns)+table.select(0,1).index_select(0,rows));
    patches.clear();
    auto col_angles=columns.to(at::kFloat).unsqueeze(1)*vision_frequency,row_angles=rows.to(at::kFloat).unsqueeze(1)*vision_frequency;
    auto cos=at::cat({col_angles.cos(),row_angles.cos()},-1).to(dtype).unsqueeze(1);
    auto sin=at::cat({col_angles.sin(),row_angles.sin()},-1).to(dtype).unsqueeze(1);
    auto rotate2d=[&](const at::Tensor& x) {
#ifdef TEI_TORCH_CUDA_KERNELS
      if(device.is_cuda()&&dtype==at::kBFloat16) return gemma_vision_rotary_cuda(x,cos,sin);
#endif
      std::vector<at::Tensor> parts;
      for(int axis=0;axis<2;++axis) {
        auto part=x.narrow(-1,axis*dim/2,dim/2),a=part.narrow(-1,0,dim/4),b=part.narrow(-1,dim/4,dim/4);
        auto c=cos.narrow(-1,axis*dim/4,dim/4),s=sin.narrow(-1,axis*dim/4,dim/4);
        parts.push_back(at::cat({a*c-b*s,b*c+a*s},-1));
      }
      return at::cat(parts,-1);
    };
    auto cu=at::tensor(offsets,at::TensorOptions().dtype(at::kInt)).to(device);
    PackedInput input{{},{},{},cu,offsets.data(),int64_t(images.size()),max_image};
    for(int64_t i=0;i<nlayer;++i) {
      auto p="vision_tower.encoder.layers."+std::to_string(i)+".";
      auto normalized=vn(states,p+"input_layernorm");
      auto q=rotate2d(vn(vl(normalized,p+"self_attn.q_proj").view({states.size(0),nh,dim}),p+"self_attn.q_norm"));
      auto k=rotate2d(vn(vl(normalized,p+"self_attn.k_proj").view({states.size(0),nk,dim}),p+"self_attn.k_norm"));
      auto v=vn(vl(normalized,p+"self_attn.v_proj").view({states.size(0),nk,dim}),"");
      at::Tensor attention_output;
#ifdef TEI_TORCH_CUDA_KERNELS
      if(device.is_cuda()&&dtype==at::kBFloat16&&torch_flash_fma_supported(q,k,v,false,-1,-1))
        attention_output=torch_flash_fma(q,k,v,input.cumulative,input.max_sequence,1.,false,-1,-1);
      else
#endif
      attention_output=packed_attention(q,k,v,input,1.);
      auto output=attention_output.reshape({states.size(0),nh*dim});
      states=states+vn(vl(output,p+"self_attn.o_proj"),p+"post_attention_layernorm");
      normalized=vn(states,p+"pre_feedforward_layernorm");
      auto gate=vl(normalized,p+"mlp.gate_proj");
      auto activation=cfg.string("vision_config.hidden_activation","gelu_pytorch_tanh");
#ifdef TEI_TORCH_CUDA_KERNELS
      if(device.is_cuda()&&dtype==at::kBFloat16&&activation!="silu"&&activation!="gelu")gate=gemma_activation_cuda(gate,true);
      else
#endif
      gate=activation=="silu"?at::silu(gate):at::gelu(gate,activation=="gelu"?"none":"tanh");
      auto mlp=vl(gate*vl(normalized,p+"mlp.up_proj"),p+"mlp.down_proj");
      states=states+vn(mlp,p+"post_feedforward_layernorm");
    }
    std::vector<at::Tensor> pools;
    for(size_t i=0;i<images.size();++i) {
      const auto height=images[i].grid_thw[1],width=images[i].grid_thw[2];
      auto slice=states.narrow(0,offsets[i],offsets[i+1]-offsets[i]);
      at::Tensor pooled;
#ifdef TEI_TORCH_CUDA_KERNELS
      if(device.is_cuda()&&dtype==at::kBFloat16)pooled=gemma_vision_pool_cuda(slice.contiguous(),height,width,pool);
      else
#endif
      {
        auto bins=at::floor_divide(column_parts[i],pool)+at::floor_divide(row_parts[i],pool)*(width/pool);
        pooled=at::zeros({height*width/(pool*pool),vh},states.options().dtype(at::kFloat));
        pooled.index_add_(0,bins,slice.to(at::kFloat)/(pool*pool));
        pooled=pooled.to(dtype).to(at::kFloat)*std::sqrt(static_cast<double>(vh));
      }
      pools.push_back(pooled);
    }
    auto pooled=at::cat(pools,0);
    if(cfg.boolean("vision_config.standardize",false)) pooled=(pooled-vw("vision_tower.std_bias").to(at::kFloat))*vw("vision_tower.std_scale").to(at::kFloat);
    pooled=pooled.to(dtype);
#ifdef TEI_TORCH_CUDA_KERNELS
    // Preserve the final BareRmsNorm and its activation cast before projection.
    if(device.is_cuda()&&dtype==at::kBFloat16)
      pooled=gemma_vision_norm_cuda(pooled.contiguous(),norm_ones.at(vh),eps);
    else
#endif
    {
      auto f=pooled.to(at::kFloat);pooled=(f/at::sqrt(f.square().mean(-1,true)+eps)).to(dtype);
    }
    auto projected=vl(pooled,"embed_vision.embedding_projection");
    std::vector<at::Tensor> result;int64_t offset=0;
    for(const auto& part:pools) {result.push_back(projected.narrow(0,offset,part.size(0)));offset+=part.size(0);}
    return result;
  }
  bool has(const std::string& key) const { return weights.count(prefix + key); }
  const at::Tensor& w(const std::string& key) const {
    auto it = weights.find(prefix + key);
    TORCH_CHECK(it != weights.end(), "Missing Gemma weight: ", prefix, key);
    return it->second;
  }
  std::string key(const std::string& name) const {
    return cfg.values.count("text_config." + name) ? "text_config." + name : name;
  }
  int64_t integer(const std::string& name, int64_t fallback) const { return cfg.integer(key(name), fallback); }
  double real(const std::string& name, double fallback) const { return cfg.real(key(name), fallback); }
  bool boolean(const std::string& name, bool fallback) const { return cfg.boolean(key(name), fallback); }
  std::string string(const std::string& name, const std::string& fallback) const { return cfg.string(key(name), fallback); }
  at::Tensor linear(const at::Tensor& x, const std::string& name) const {
    return at::linear(x, w(name + ".weight"), has(name + ".bias") ? std::optional<at::Tensor>(w(name + ".bias")) : std::nullopt);
  }
  at::Tensor norm(const at::Tensor& x, const std::string& name, bool unit = false) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    if(device.is_cuda() && dtype==at::kBFloat16 && x.size(-1)<=8192) {
      const auto& scale=unit?norm_ones.at(x.size(-1)):weight(norm_scales,prefix+name+".weight");
      if(embedding2_reference_cuda) return gemma2_norm_cuda(x,scale,epsilon);
      return gemma_norm_cuda(x,scale,epsilon,!legacy);
    }
#endif
    auto f = x.to(at::kFloat);
    auto variance=f.square().mean(-1,true)+epsilon;
    f = legacy ? f*at::rsqrt(variance) : f/at::sqrt(variance);
    if (!unit) {
      f = f * weight(norm_scales,prefix + name + ".weight");
    }
    return f.to(x.scalar_type());
  }
  at::Tensor residual_norm(const at::Tensor& residual,const at::Tensor& x,const std::string& name) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    if(publisher_sequence && ple_width==0 && device.is_cuda() && dtype==at::kBFloat16
       &&x.dim()==2 &&x.size(1)>0 &&x.size(1)<=8192 &&x.is_contiguous() &&residual.is_contiguous())
      return gemma4_bf16_norm_residual(residual,x,weight(norm_scales,prefix+name+".weight"),epsilon,{});
    if(embedding2_reference_cuda)
      return gemma2_residual_norm_cuda(residual,x,weight(norm_scales,prefix+name+".weight"),epsilon);
    if(fused_legacy_residual) {
      auto output=gemma3_legacy_residual_norm_cuda(residual,x,
        weight(norm_scales,prefix+name+".weight"),epsilon);
      if(output.defined())return output;
    }
#endif
    return residual+norm(x,name);
  }
  at::Tensor activate(const at::Tensor& x) const {
    const auto act = string("hidden_activation", "gelu_pytorch_tanh");
    if(act=="gelu")return at::gelu(x,"none");
#ifdef TEI_TORCH_CUDA_KERNELS
    if(x.is_cuda()&&x.scalar_type()==at::kBFloat16&&act!="relu")return gemma_activation_cuda(x,act!="silu"&&act!="swish");
#endif
    if (act == "silu" || act == "swish") return x/(at::ones({},x.options())+at::exp(-x));
    if (act == "relu") return at::relu(x);
    auto cube=(x*x)*x;
    auto alpha=x+at::scalar_tensor(.044715,x.options())*cube;
    auto value=at::tanh(at::scalar_tensor(M_2_SQRTPI*M_SQRT1_2,x.options())*alpha);
    return (at::scalar_tensor(.5,x.options())*x)*(at::ones({},x.options())+value);
  }
  at::Tensor rotate(const at::Tensor& x, const Layer& layer, const at::Tensor& positions) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    if(!legacy && (!embedding2 || embedding2_reference_cuda) && x.is_cuda() && x.scalar_type()==at::kBFloat16)
      return gemma_rotary_cuda(x,layer.cos,layer.sin,positions);
#endif
    auto cos = layer.cos.index_select(0, positions).unsqueeze(1);
    auto sin = layer.sin.index_select(0, positions).unsqueeze(1);
    auto half = x.size(-1) / 2;
    auto a = x.slice(-1, 0, half), b = x.slice(-1, half);
    // Gemma4's composed reference rounds each BF16 product before addition;
    // retaining activation-dtype ATen operations preserves those stages.
    return at::cat({a * cos - b * sin, b * cos + a * sin}, -1).to(x.scalar_type());
  }
  at::Tensor attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                       const PackedInput& input, const Layer& layer, bool use_causal,int64_t cache_layer=-1) const {
    const bool causal=use_causal;
    auto left = layer.sliding ? (causal ? layer.window - 1 : layer.window / 2) : -1;
    auto right = layer.sliding ? (causal ? 0 : layer.window / 2) : -1;
    auto scale = legacy ? 1.0 / std::sqrt(real("query_pre_attn_scalar", layer.dim)) : 1.0;
    if(input.capture_prefix&&!layer.shared)prefix_capture_attention(input,cache_layer,k,v);
    if(input.prefix) {
      TORCH_CHECK(causal,"Noncausal Gemma cannot reuse prefix state");
      auto output=prefix_attention(q,k,v,input,cache_layer,scale,left,right,true,!layer.shared);
      if(causal&&layer.sliding&&!input.images.empty()) {
        auto kv=input.prefix->attention(cache_layer,k,v,false);
        for(const auto& image:input.images) {
          auto it=std::upper_bound(input.offsets,input.offsets+input.batch+1,image.token_start);
          auto row=it-input.offsets-1,relative=image.token_start-input.offsets[row]+input.prefix->hits[row];
          auto start=std::max<int64_t>(0,relative-(layer.window-1));
          auto count=relative+image.token_count-start;
          auto iq=q.narrow(0,image.token_start,image.token_count);
          auto ik=kv.key.narrow(0,input.prefix->original.offsets[row]+start,count);
          auto iv=kv.value.narrow(0,input.prefix->original.offsets[row]+start,count);
          at::Tensor correction;
          if(q.is_cuda()) {
            auto cuq=at::tensor(std::vector<int32_t>{0,int32_t(image.token_count)},input.cumulative.options());
            auto cuk=at::tensor(std::vector<int32_t>{0,int32_t(count)},input.cumulative.options());
#ifdef TEI_TORCH_CUDA_KERNELS
            if(torch_flash_fma_supported(iq,ik,iv,false,layer.window-1,-1))
              correction=torch_flash_fma_varlen(iq,ik,iv,cuq,cuk,image.token_count,count,scale,false,layer.window-1,-1);
            else
#endif
              correction=std::get<0>(at::_flash_attention_forward(iq,ik,iv,cuq,cuk,image.token_count,count,0.,false,false,scale,layer.window-1,-1));
          } else {
            auto heads=iq.size(1)/ik.size(1);
            auto ki=at::arange(count,iq.options().dtype(at::kLong));
            auto qi=at::arange(image.token_count,iq.options().dtype(at::kLong))+(count-image.token_count);
            auto mask=(qi.unsqueeze(1)-ki.unsqueeze(0))<=layer.window-1;
            correction=at::scaled_dot_product_attention(iq.transpose(0,1).unsqueeze(0),ik.repeat_interleave(heads,1).transpose(0,1).unsqueeze(0),
              iv.repeat_interleave(heads,1).transpose(0,1).unsqueeze(0),mask,0.,false,scale).squeeze(0).transpose(0,1);
          }
          output.narrow(0,image.token_start,image.token_count).copy_(correction);
        }
      }
      return output;
    }
    if (device.is_cuda()) {
#ifdef TEI_TORCH_CUDA_KERNELS
      if(layer.dim>256 && torch_flash_fma_supported(q,k,v,causal,left,right))
        return torch_flash_fma(q,k,v,input.cumulative,input.max_sequence,scale,causal,left,right);
#endif
      if(layer.dim>256) {
        TORCH_CHECK(!layer.sliding,"Torch512-head packed sliding attention requires a specialized kernel");
        // cuDNN's long-sequence varlen path accepts head512, but mixed-length
        // parity is not established. CUTLASS handles all actual lengths here.
        auto repeat=heads/layer.kv;
        auto keys=repeat==1?k:at::repeat_interleave(k,repeat,1);
        auto values=repeat==1?v:at::repeat_interleave(v,repeat,1);
        return std::get<0>(at::_efficient_attention_forward(q.unsqueeze(0),keys.unsqueeze(0),values.unsqueeze(0),
          std::nullopt,input.cumulative,input.cumulative,input.max_sequence,input.max_sequence,0.,causal?1:0,false,scale)).squeeze(0);
      }
      at::Tensor output;
#ifdef TEI_TORCH_CUDA_KERNELS
      if(torch_flash_fma_supported(q,k,v,causal,left,right))
        output=torch_flash_fma(q,k,v,input.cumulative,input.max_sequence,scale,causal,left,right);
      else
#endif
      output=std::get<0>(at::_flash_attention_forward(q, k, v, input.cumulative,
        input.cumulative, input.max_sequence, input.max_sequence, 0., causal,
        false, scale, left, right));
      if(causal && layer.sliding) {
        for(const auto& image:input.images) {
          auto key_start=std::max(image.sequence_start,image.token_start-(layer.window-1));
          auto key_len=image.token_start+image.token_count-key_start;
          auto cuq=at::arange(2,q.options().dtype(at::kInt))*image.token_count;
          auto cuk=at::arange(2,q.options().dtype(at::kInt))*key_len;
          auto iq=q.narrow(0,image.token_start,image.token_count),ik=k.narrow(0,key_start,key_len),iv=v.narrow(0,key_start,key_len);
          at::Tensor correction;
#ifdef TEI_TORCH_CUDA_KERNELS
          if(torch_flash_fma_supported(iq,ik,iv,false,layer.window-1,-1))
            correction=torch_flash_fma_varlen(iq,ik,iv,cuq,cuk,image.token_count,key_len,scale,false,layer.window-1,-1);
          else
#endif
          correction=std::get<0>(at::_flash_attention_forward(iq,ik,iv,cuq,cuk,image.token_count,key_len,
            0.,false,false,scale,layer.window-1,-1));
          output.narrow(0,image.token_start,image.token_count).copy_(correction);
        }
      }
      return output;
    }
    std::vector<at::Tensor> rows;
    for (int64_t i = 0; i < input.batch; ++i) {
      auto start = input.offsets[i], len = input.offsets[i + 1] - start;
      auto slice = [&](const at::Tensor& t) { return t.narrow(0, start, len).transpose(0, 1).unsqueeze(0); };
      std::optional<at::Tensor> mask;
      if (layer.sliding) {
        auto ix = at::arange(len, input.ids.options());
        auto delta = ix.unsqueeze(0) - ix.unsqueeze(1);
        mask = delta >= -left;
        if(!causal || input.images.empty()) mask=*mask & (delta<=right);
        if (causal) {
          auto allowed=delta<=0;
          for(const auto& image:input.images) {
            auto image_start=image.token_start-start,image_end=image_start+image.token_count;
            if(image_start>=0 && image_end<=len)
              allowed=allowed|((ix>=image_start)&(ix<image_end)).unsqueeze(1).logical_and(((ix>=image_start)&(ix<image_end)).unsqueeze(0));
          }
          mask=*mask & allowed;
        }
      }
      rows.push_back(at::scaled_dot_product_attention(slice(q), slice(k), slice(v), mask,
        0., causal && !mask.has_value(), scale, true).squeeze(0).transpose(0, 1));
    }
    return at::cat(rows, 0);
  }
  void normalize_layer_overrides() {
    // Match Rust usize::from_str: ASCII digits, optional +, checked overflow.
    // Normalize only actual layers once; invalid/unused keys remain ignored.
    for (const std::string scope : {"per_layer_config.", "text_config.per_layer_config."}) {
      using Fields = std::map<std::string, int64_t>;
      std::map<size_t, std::map<std::string, Fields>> aliases;
      for (const auto& [name, value] : cfg.values) {
        if (!name.starts_with(scope)) continue;
        const auto end = name.find('.', scope.size());
        if (end == std::string::npos) continue;
        const auto field = name.substr(end + 1);
        if (field != "head_dim" && field != "num_key_value_heads") continue;
        const auto spelling = name.substr(scope.size(), end - scope.size());
        std::string_view digits(spelling);
        if (digits.starts_with('+')) digits.remove_prefix(1);
        if (digits.empty() || digits.front() < '0' || digits.front() > '9') continue;
        size_t index = 0;
        const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), index);
        if (parsed.ec != std::errc() || parsed.ptr != digits.data() + digits.size()
            || layers <= 0 || index >= static_cast<size_t>(layers)) continue;
        aliases[index][spelling][field] = cfg.integer(name);
      }
      for (const auto& [index, spellings] : aliases) {
        const bool k_equals_v = string("layer_types." + std::to_string(index), "full_attention")
          != "sliding_attention" && boolean("attention_k_eq_v", false);
        Fields fallback{{"head_dim", integer("head_dim", heads > 0 ? hidden / heads : 0)},
          {"num_key_value_heads", k_equals_v ? integer("num_global_key_value_heads",
             integer("num_key_value_heads", heads)) : integer("num_key_value_heads", heads)}};
        // An omitted nested field still inherits an explicit root override.
        if (scope == "text_config.per_layer_config.")
          for (auto& [field, value] : fallback)
            value = cfg.integer("per_layer_config." + std::to_string(index) + "." + field, value);
        auto effective = [&](const Fields& fields) {
          auto result = fallback;
          for (const auto& [field, value] : fields) result[field] = value;
          return result;
        };
        const auto expected = effective(spellings.begin()->second);
        Fields explicit_fields;
        for (const auto& [spelling, candidate] : spellings) {
          TORCH_CHECK(effective(candidate) == expected,
                      "Conflicting Gemma per_layer_config aliases for layer ",
                      index, ": ", spellings.begin()->first, " and ", spelling);
          explicit_fields.insert(candidate.begin(), candidate.end());
        }
        // Do not materialize defaults: preserve root/nested field precedence.
        for (const auto& [field, value] : explicit_fields)
          cfg.values[scope + std::to_string(index) + "." + field] = std::to_string(value);
      }
    }
  }
  int64_t override_int(int64_t index, const std::string& name, int64_t fallback) const {
    // Accept both "05" (released checkpoints) and "5" layer keys.
    for (auto stem : {"per_layer_config." + std::to_string(index),
                      "per_layer_config." + (index < 10 ? std::string("0") : std::string()) + std::to_string(index)}) {
      if (cfg.values.count(key(stem + "." + name))) return integer(stem + "." + name, fallback);
    }
    return fallback;
  }
 public:
  void prepare_input(PackedInput& input) const override {
    // CPU-side descriptor keys are supplied before any device transfer. The
    // tower runs outside CUDA capture; only immutable features enter forward.
    TORCH_CHECK(input.images.empty()||supports_images(),"Gemma vision tower unavailable");
    std::vector<std::optional<ImageFeatureCache::Key>> keys;
    for(const auto& image:input.images)keys.push_back(image.has_feature_key?std::optional(image.feature_key):std::nullopt);
    auto features=image_features.get_batch(keys,[&](const std::vector<size_t>& indices){
      std::vector<ImageInput> missing;for(auto i:indices)missing.push_back(input.images[i]);
      return vision(missing);
    });
    for(size_t i=0;i<input.images.size();++i) {
      auto& image=input.images[i];image.prepared_features=features[i];
      TORCH_CHECK(image.prepared_features.dim()==2 && image.prepared_features.size(0)==image.token_count
        && image.prepared_features.size(1)==hidden && image.prepared_features.device()==device
        && image.prepared_features.scalar_type()==dtype,"Gemma prepared image feature geometry mismatch");
    }

    for(auto& audio:input.audios) {
      auto valid=audio.validity.to(at::kBool);
      for(int block=0;block<2;++block) {
        auto stride=cfg.integer("audio_config.sscp_conv_stride_size."+std::to_string(block)+".0",2);
        TORCH_CHECK(stride>0&&valid.numel()>0,"Invalid Gemma audio subsampling geometry");
        auto length=(valid.numel()+stride-1)/stride;
        auto indices=at::arange(length,valid.options().dtype(at::kLong))*stride;
        valid=valid.index_select(0,indices.clamp_max(valid.numel()-1));
      }
      audio.selected_frames=at::nonzero(valid).squeeze(-1).contiguous();
      TORCH_CHECK(audio.selected_frames.numel()==audio.token_count,"Gemma selected audio frame count does not match token span");
    }
  }
  bool supports_radix() const override {
    // Both option decisions and a loaded sequence score head use causal states.
    // forward_impl still rejects folding for bidirectional embedding requests.
    return (causal || (boolean("_classifier",false) && classification_width() > 0))
      && !legacy && !embedding2;
  }
  bool supports_images() const override { return !legacy && cfg.integer("vision_config.num_hidden_layers",0)>0; }
  bool supports_audio() const override { return embedding2 && cfg.integer("audio_config.hidden_size",0)>0; }
  int64_t prefix_layers() const override {return legacy?layers:layers-integer("num_kv_shared_layers",0);}
  uint64_t prefix_payload_bytes(int64_t tokens) const override {
    auto bytes=prefix_storage_bytes(tokens,hidden,w("embed_tokens.weight").element_size());
    for(const auto& layer:stack)if(!layer.shared)
      bytes=prefix_storage_sum(bytes,prefix_storage_bytes(tokens,2*layer.kv*layer.dim,w("embed_tokens.weight").element_size()));
    return bytes;
  }
  bool supports_prefix_cache(int operation) const override {return !embedding2&&operation!=2&&(operation==1?classification_width()>0:causal);}
  int64_t output_width() const override { return embedding2 ? integer("embedding_dim", hidden) : hidden; }
  int64_t classification_width() const override {
    auto score=weights.find("score.weight");
    return legacy || embedding2 || score==weights.end()?0:score->second.size(0);
  }
  Gemma(const Options& options, Weights& tensors, c10::Device dev, at::ScalarType type)
    : cfg(options), weights(tensors), device(dev), dtype(type) {
    family = cfg.string("model_type", "");
    legacy = family == "gemma3" || family == "gemma3_text";
    embedding2 = family == "embedding_gemma2";
    publisher_sequence = !legacy && !embedding2 &&
       cfg.string("architectures").starts_with("[") &&
       cfg.string("architectures.0")=="Gemma4ForSequenceClassification" &&
       cfg.string("auto_map.AutoModelForSequenceClassification")==
         "modeling_gemma4_sequence.Gemma4ForSequenceClassification";
    hidden = integer("hidden_size", 0); heads = integer("num_attention_heads", 0);
    layers = integer("num_hidden_layers", 0); epsilon = real("rms_norm_eps", 1e-6);
    ple_width = integer("hidden_size_per_layer_input", 0);
    if (embedding2) normalize_layer_overrides();
    // The fork's Gemma4 option-token decision path always requests causal
    // hidden states; embedding requests retain their bidirectional contract.
    causal = legacy ? !boolean("use_bidirectional_attention", false)
                    : !embedding2 && cfg.string("_decision.kind")=="option_tokens";
    for (const auto& p : {"model.language_model.", "language_model.", "model.", ""}) {
      if (weights.count(std::string(p) + "embed_tokens.weight")) { prefix = p; break; }
    }
  }
  void ready() override {
    const auto fp8=cfg.boolean("_fp8_dynamic",false);
    TORCH_CHECK(cfg.string("_fp8_scope","mlp")=="mlp","Gemma FP8 supports MLP scope only");
    if(fp8) {
      TORCH_CHECK(!legacy && !embedding2,
        "Dynamic FP8 Gemma requires Gemma4");
      TORCH_CHECK(device.is_cuda() && (dtype==at::kHalf || dtype==at::kBFloat16),
        "Dynamic FP8 Gemma requires CUDA float16/bfloat16");
      for(const auto& [name,value]:cfg.values)
        TORCH_CHECK(name!="quantization_config" && name.rfind("quantization_config.",0)!=0
          && name!="text_config.quantization_config" && name.rfind("text_config.quantization_config.",0)!=0,
          "Dynamic FP8 requires an unquantized checkpoint without supplied quantization scales");
    }
    if(supports_audio()) {
      const auto chunk=cfg.integer("audio_config.attention_chunk_size",12),left=cfg.integer("audio_config.attention_context_left",13),
        future=cfg.integer("audio_config.attention_context_right",0);
      TORCH_CHECK(left>0,"Invalid audio left attention context");const auto past=left-1;
      TORCH_CHECK(chunk>0&&future>=0&&chunk<=std::numeric_limits<int64_t>::max()-past&&
        future<=std::numeric_limits<int64_t>::max()-chunk-past,"Invalid audio relative attention geometry");
    }
    if(legacy) {
      legacy_embedding_scale=at::scalar_tensor(std::sqrt(static_cast<double>(hidden)),w("embed_tokens.weight").options());
      // Complete initialization before the model moves to another inference stream.
      if(legacy_embedding_scale.is_cuda())static_cast<void>(legacy_embedding_scale.item<double>());
    }
    // The trained Gemma2 profile retains Candle's original sqrt/div norm tree
    // and BF16 intermediate rounding. Reset when preparing another model.
    embedding2_reference_cuda = embedding2 && device.is_cuda() && dtype==at::kBFloat16 &&
      hidden==512 && heads==4 && layers==24 && ple_width==512 &&
      integer("intermediate_size",0)==2048 && integer("head_dim",0)==256 &&
      integer("num_key_value_heads",0)==2 && epsilon==1e-6;
    TORCH_CHECK(hidden > 0 && heads > 0 && layers > 0, "Invalid Gemma architecture dimensions");
    if(boolean("enable_moe_block",false)) TORCH_CHECK(!legacy && !embedding2,"Gemma4 expert routing requires Gemma4 architecture");
    TORCH_CHECK(device.is_cpu() || device.is_cuda(), "Gemma packed attention supports CPU/CUDA only");
    if (device.is_cuda()) TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "Packed CUDA Gemma requires half/bfloat16");
    if(fp8 && boolean("enable_moe_block",false))
      TORCH_CHECK(dtype==at::kBFloat16,"Grouped dynamic FP8 Gemma MoE requires BF16");
    w("embed_tokens.weight"); w("norm.weight");
    norm_ones[hidden]=at::ones({hidden},at::TensorOptions().dtype(at::kFloat).device(device));
    if(cfg.integer("vision_config.num_hidden_layers",0)>0) {
      auto width=cfg.integer("vision_config.hidden_size",0),head_count=cfg.integer("vision_config.num_attention_heads",0);
      TORCH_CHECK(width>0&&head_count>0,"Invalid Gemma vision hidden/head widths");
      norm_ones[width]=at::ones({width},at::TensorOptions().dtype(at::kFloat).device(device));
      auto head_width=cfg.integer("vision_config.head_dim",width/head_count);
      TORCH_CHECK(head_width>0&&head_width%4==0,"Invalid Gemma vision rotary width");
      auto theta=cfg.real("vision_config.rope_parameters.rope_theta",100.);
      std::vector<float> inverse(head_width/4);
      for(int64_t i=0;i<head_width/4;++i)inverse[i]=1.f/static_cast<float>(std::pow(theta,4.*i/head_width));
      vision_frequency=at::from_blob(inverse.data(),{head_width/4},at::TensorOptions().dtype(at::kFloat)).clone().to(device);
      norm_ones[head_width]=at::ones({head_width},at::TensorOptions().dtype(at::kFloat).device(device));
    }
    for(const auto& [name,value]:weights) {
      // Match Gemma4's loader: convert these learned routing tensors once,
      // before inference/graph capture, retaining exactly the same F32 values.
      if(name.ends_with("router.proj.weight") || name.ends_with("router.per_expert_scale"))
        router_f32[name]=value.to(at::kFloat).contiguous();
      if(name.ends_with(".weight") && name.find("norm")!=std::string::npos) {
        auto scale=value.to(at::kFloat);
        norm_scales[name]=legacy?scale+1.:scale;
      }
      if(name.ends_with(".input_min")) {
        auto base=name.substr(0,name.size()-10);
        audio_bounds[base]={value.item<double>(),weight(weights,base+".input_max").item<double>(),
          weight(weights,base+".output_min").item<double>(),weight(weights,base+".output_max").item<double>()};
      }
    }
    fused_legacy_residual=legacy && device.is_cuda() && dtype==at::kBFloat16 &&
      hidden==768 && heads==3 && layers==24 && integer("intermediate_size",0)==1152 &&
      integer("num_key_value_heads",heads)==1 && integer("head_dim",hidden/heads)==256;
    ordinary_ffn.reset();
    // Conditional on exact original learned Gemma3 Wi/Wo tactic replay proof.
    // One model-wide bounded executor; other families/geometries remain ATen.
    if(legacy && device.is_cuda() && dtype==at::kBFloat16 && hidden==768 &&
      heads==3 && layers==24 && integer("intermediate_size",0)==1152 &&
      integer("num_key_value_heads",heads)==1 && integer("head_dim",hidden/heads)==256)
      ordinary_ffn=std::make_unique<OrdinaryLinearExecutor>();
    auto first_shared = layers - integer("num_kv_shared_layers", 0);
    TORCH_CHECK(first_shared >= 0, "Gemma shared KV layer count exceeds layer count");
    auto pattern = integer("_sliding_window_pattern", integer("sliding_window_pattern", 6));
    TORCH_CHECK(pattern > 0, "Gemma3 sliding pattern must be positive");
    for (int64_t i = 0; i < layers; ++i) {
      Layer layer;
      layer.p = "layers." + std::to_string(i) + ".";
      auto kind = string("layer_types." + std::to_string(i), "full_attention");
      layer.sliding = legacy ? (i + 1) % pattern != 0 : kind == "sliding_attention";
      TORCH_CHECK(legacy || kind == "sliding_attention" || kind == "full_attention", "Unsupported Gemma attention type ", kind);
      auto default_dim = integer("head_dim", hidden / heads);
      layer.dim = embedding2 ? override_int(i, "head_dim", default_dim) :
        (layer.sliding || legacy ? default_dim : integer("global_head_dim", default_dim));
      layer.k_equals_v = !legacy && !layer.sliding && boolean("attention_k_eq_v", false);
      layer.kv = override_int(i, "num_key_value_heads", layer.k_equals_v ?
        integer("num_global_key_value_heads", integer("num_key_value_heads", heads)) : integer("num_key_value_heads", heads));
      TORCH_CHECK(layer.dim > 0 && layer.dim % 2 == 0 && layer.kv > 0 && heads % layer.kv == 0, "Invalid Gemma attention geometry at layer ", i);
      if(embedding2_reference_cuda && ((first_shared < layers) ||
          (layer.sliding ? (layer.dim!=256 || layer.kv!=2) : (layer.dim!=512 || layer.kv!=1))))
        embedding2_reference_cuda = false;
      if(!norm_ones.count(layer.dim))norm_ones[layer.dim]=at::ones({layer.dim},at::TensorOptions().dtype(at::kFloat).device(device));
      if (device.is_cuda()) TORCH_CHECK(layer.dim % 8 == 0,
        "Torch varlen FlashAttention requires head dimension divisible by 8; this Gemma layer requires ", layer.dim);
      layer.window = integer("sliding_window", 4096) * (embedding2 ? 2 : 1);
      TORCH_CHECK(layer.window > 0, "Gemma sliding window must be positive");
      layer.shared = !legacy && first_shared < layers && i >= first_shared;
      layer.store = false;
      if (!layer.shared && first_shared < layers) {
        layer.store = true;
        for (int64_t j = i + 1; j < first_shared; ++j)
          if (string("layer_types." + std::to_string(j), "full_attention") == kind) layer.store = false;
      }
      if (legacy) {
        layer.qkv = at::cat({w(layer.p + "self_attn.q_proj.weight"), w(layer.p + "self_attn.k_proj.weight"), w(layer.p + "self_attn.v_proj.weight")}, 0);
      } else {
        w(layer.p + "self_attn.q_proj.weight");
        if (!layer.shared) { w(layer.p + "self_attn.k_proj.weight"); if (!layer.k_equals_v) w(layer.p + "self_attn.v_proj.weight"); }
      }
      layer.gate_up = at::cat({w(layer.p + "mlp.gate_proj.weight"), w(layer.p + "mlp.up_proj.weight")}, 0);
      if(fp8) {
        layer.fp8_gate_up=std::make_unique<Fp8Linear>(layer.gate_up);
        layer.fp8_down=std::make_unique<Fp8Linear>(w(layer.p+"mlp.down_proj.weight"));
        layer.gate_up=at::Tensor();
        for(const auto* projection:{"gate_proj","up_proj","down_proj"})
          weights.erase(prefix+layer.p+"mlp."+projection+".weight");
      } else w(layer.p + "mlp.down_proj.weight");
      if(fp8 && boolean("enable_moe_block",false)) {
        layer.fp8_expert_gate_up=std::make_unique<Fp8GroupedLinear>(w(layer.p+"experts.gate_up_proj"));
        layer.fp8_expert_down=std::make_unique<Fp8GroupedLinear>(w(layer.p+"experts.down_proj"));
        weights.erase(prefix+layer.p+"experts.gate_up_proj");
        weights.erase(prefix+layer.p+"experts.down_proj");
      }
      auto rope_kind = layer.sliding ? "sliding_attention" : "full_attention";
      auto theta = legacy ? real(layer.sliding ? "rope_local_base_freq" : "rope_theta", 10000.) :
        real(std::string("rope_parameters.") + rope_kind + ".rope_theta", 10000.);
      TORCH_CHECK(theta > 0, "Gemma RoPE theta must be positive");
      auto pairs = layer.dim / 2;
      // Match Candle's host FP32 powf frequency construction exactly.
      std::vector<float> inverse(pairs);
      for(int64_t j=0;j<pairs;++j)inverse[j]=1.f/std::pow(float(theta),float(2*j)/float(layer.dim));
      auto freq=at::from_blob(inverse.data(),{pairs},at::TensorOptions().dtype(at::kFloat)).clone().to(device);
      if (!legacy && !embedding2 && !layer.sliding) {
        auto factor = real("rope_parameters.full_attention.partial_rotary_factor", 1.);
        TORCH_CHECK(factor > 0. && factor <= 1., "Invalid Gemma partial rotary factor");
        freq = freq * (at::arange(pairs, freq.options()) < static_cast<int64_t>(factor * pairs));
      }
      bool reused=false;
      for(const auto& prior:stack) {
        if(prior.dim==layer.dim && prior.sliding==layer.sliding) {
          layer.cos=prior.cos;layer.sin=prior.sin;reused=true;break;
        }
      }
      if(!reused) {
        auto angles = at::arange(integer("max_position_embeddings", 8192), freq.options()).unsqueeze(1) * freq.unsqueeze(0);
        layer.cos = angles.cos().to(dtype); layer.sin = angles.sin().to(dtype);
      }
      stack.push_back(std::move(layer));
    }
  }
  at::Tensor forward(const PackedInput& input) const override { return forward_impl(input,causal); }
  at::Tensor predict(const PackedInput& input,bool tokens) const override {
    TORCH_CHECK(classification_width()>0,"Gemma classifier score head is missing");
    TORCH_CHECK(!tokens,"Gemma4 classifier supports sequence logits only");
    return classify_hidden(input,forward_impl(input,true));
  }
  at::Tensor classify_hidden(const PackedInput& input,const at::Tensor& states) const override {
    const auto& logical=prefix_logical_input(input);
    if(publisher_sequence) {
      // The declared publisher scores every logical token before readout.
      // Selecting hidden states first changes the BF16 GEMM rounding, including
      // after radix unfolding. Pointwise cap operations commute with gathering
      // the resulting score rows, preserving each intermediate BF16 rounding.
      auto logits=at::linear(states,weight(weights,"score.weight"));
      const double cap=real("final_logit_softcapping",0.);
      const bool capped=cfg.values.count(key("final_logit_softcapping"));
#ifdef TEI_TORCH_CUDA_KERNELS
      if(logits.is_cuda()&&logits.scalar_type()==at::kBFloat16&&logits.is_contiguous()
         &&logical.cumulative.scalar_type()==at::kInt&&logical.cumulative.is_contiguous())
        return gemma4_bf16_sequence_score_readout(logits,logical.cumulative,logical.batch,cap,capped);
#endif
      if(capped) {
        logits=logits/cap;
        logits=at::tanh(logits);
        logits=logits*cap;
      }
      // Read current packed ends on the producing device, including graph replay.
      auto ids=logical.cumulative.narrow(0,1,logical.batch).to(at::kLong)-1;
      return logits.index_select(0,ids);
    }
    // An LM softcap alone does not specify a sequence score contract.
    auto ids=logical.cumulative.narrow(0,1,logical.batch).to(at::kLong)-1;
    return at::linear(states.index_select(0,ids),weight(weights,"score.weight"));
  }
  at::Tensor forward_impl(const PackedInput& input,bool is_causal) const {
#ifdef TEI_TORCH_CUDA_KERNELS
    // The synchronous call owns the device guard; the session owns no guard.
    c10::cuda::OptionalCUDAGuard ordinary_guard;
    if(ordinary_ffn)ordinary_guard.set_device(device);
#endif
    // Reuse this call's stream/context/policy snapshot for all Wi/Wo pairs.
    auto ordinary_session=ordinary_ffn?ordinary_ffn->prepare(w("embed_tokens.weight")):
      OrdinaryLinearExecutor::Session{};
    TORCH_CHECK(!radix_enabled(input) || (is_causal && !legacy && !embedding2),
      "Gemma radix folding requires causal Gemma4 hidden states");
    const auto& ids=radix_ids(input);
    const auto& positions=radix_positions(input);
    const auto tokens=ids.numel();
    auto scale = legacy ? legacy_embedding_scale : at::scalar_tensor(std::sqrt(static_cast<double>(hidden)), w("embed_tokens.weight").options());
    auto states = at::embedding(w("embed_tokens.weight"), ids) * scale;
    at::Tensor ple;
    auto inject_images=[&]() {
      TORCH_CHECK(input.images.empty() || supports_images(),"Gemma vision tower unavailable");
      std::vector<ImageInput> missing;
      for(const auto& image:input.images)if(!image.prepared_features.defined())missing.push_back(image);
      auto computed=vision(missing);size_t next=0;
      for(const auto& image:input.images) {
        auto features=image.prepared_features.defined()?image.prepared_features:computed[next++];
        TORCH_CHECK(image.token_start>=0 && image.token_start+image.token_count<=states.size(0)
          && features.dim()==2 && features.size(0)==image.token_count && features.size(1)==hidden
          && features.device()==states.device() && features.scalar_type()==states.scalar_type(),"Gemma image token geometry mismatch");
        states.slice(0,image.token_start,image.token_start+image.token_count).copy_(features);
      }
    };
    if(embedding2) inject_images();
    TORCH_CHECK(input.audios.empty() || supports_audio(),"Gemma audio tower unavailable");
    for(const auto& media:input.audios) {
      auto features=audio(media);
      TORCH_CHECK(media.token_start>=0 && media.token_start+media.token_count<=states.size(0) && features.size(0)==media.token_count,"Gemma audio token geometry mismatch");
      states.narrow(0,media.token_start,media.token_count).copy_(features);
    }

    if (ple_width > 0) {
      auto stem = embedding2 ? "ple." : "";
      auto projected = linear(states, std::string(stem) + "per_layer_model_projection") *
        at::scalar_tensor(1./std::sqrt(static_cast<double>(hidden)),states.options());

      ple = norm(projected.view({tokens, layers, ple_width}), std::string(stem) + "per_layer_projection_norm");
      if (!embedding2) {
        auto token_inputs = at::embedding(w("embed_tokens_per_layer.weight"), ids) *
          at::scalar_tensor(std::sqrt(static_cast<double>(ple_width)),states.options());
        ple = (token_inputs.view_as(ple) + ple) * at::scalar_tensor(1./std::sqrt(2.),states.options());
      }
    }
    if(!embedding2 && !input.images.empty()) {
      states=radix_unfold(states,input);
      inject_images();
      states=radix_fold(states,input);
    }

    std::array<std::pair<at::Tensor,at::Tensor>,2> shared;
    std::array<int64_t,2> producers{-1,-1};
    // Request-local gathers reuse identical shared RoPE tables across layers.
    std::array<std::pair<at::Tensor,at::Tensor>,2> legacy_rotary;
    for (int64_t i = 0; i < layers; ++i) {
      const auto& layer = stack[i]; auto p = layer.p;
      auto normalized = norm(states, p + "input_layernorm");

      at::Tensor q, k, v;
      if (legacy) {
        std::optional<at::Tensor> bias;
        if (has(p + "self_attn.q_proj.bias")) bias = at::cat({w(p + "self_attn.q_proj.bias"), w(p + "self_attn.k_proj.bias"), w(p + "self_attn.v_proj.bias")}, 0);
        auto qkv = at::linear(normalized, layer.qkv, bias).view({tokens, heads + 2 * layer.kv, layer.dim});
        q = qkv.narrow(1, 0, heads); k = qkv.narrow(1, heads, layer.kv); v = qkv.narrow(1, heads + layer.kv, layer.kv);
      } else {
        q = prefix_projection(normalized, p + "self_attn.q_proj",input).view({tokens, heads, layer.dim});
        if (!layer.shared) {
          k = (radix_enabled(input) && layer.dim==512 && layer.kv==2
            ? linear_radix_projection(normalized,p+"self_attn.k_proj",input)
            : prefix_projection(normalized,p+"self_attn.k_proj",input)).view({tokens,layer.kv,layer.dim});
          v = layer.k_equals_v ? k : prefix_projection(normalized, p + "self_attn.v_proj",input).view({tokens, layer.kv, layer.dim});
        }
      }
      bool fused_qk_rotary=false;
#ifdef TEI_TORCH_CUDA_KERNELS
      if(legacy && device.is_cuda() && dtype==at::kBFloat16 && !layer.shared &&
         layer.dim>0 && layer.dim<=8192 && layer.dim%2==0) {
        auto result=gemma3_qk_norm_rotary_cuda(q,k,
          weight(norm_scales,prefix+p+"self_attn.q_norm.weight"),
          weight(norm_scales,prefix+p+"self_attn.k_norm.weight"),
          layer.cos,layer.sin,positions,epsilon);
        q=std::move(result.first);k=std::move(result.second);fused_qk_rotary=true;
      }
#endif
      if(!fused_qk_rotary) q = norm(q, p + "self_attn.q_norm");
      if (layer.shared) {
        q = radix_unfold(rotate(q, layer, positions),input);
        k = shared[layer.sliding].first; v = shared[layer.sliding].second;
        TORCH_CHECK(k.defined(), "Missing shared Gemma KV states at layer ", i);
      } else {
        if(!fused_qk_rotary) {
        k = norm(k, p + "self_attn.k_norm");
#ifdef TEI_TORCH_CUDA_KERNELS
        if (legacy && device.is_cuda()) {
          // The fork's packed Gemma3 rotary contracts the first product in
          // BF16/Half. Gemma4's composed rotary retains separate rounding.
          at::Tensor cos, sin;
          if (fused_legacy_residual && layer.dim == 256) {
            auto& tables = legacy_rotary[layer.sliding];
            if (!tables.first.defined()) {
              tables.first = layer.cos.index_select(0, positions).contiguous();
              tables.second = layer.sin.index_select(0, positions).contiguous();
            }
            cos = tables.first; sin = tables.second;
          } else {
            cos = layer.cos.index_select(0, positions).contiguous();
            sin = layer.sin.index_select(0, positions).contiguous();
          }
          // Both norms own fresh contiguous buffers, so rotation can reuse
          // them exactly as Candle does, without extra Q/K allocations.
          encoder_rotary_inplace(q,k,cos,sin);
        } else
#endif
        {
          q = rotate(q, layer, positions);
          k = rotate(k, layer, positions);
        }
        }
        if (!legacy) v = norm(v, "", true);
        auto full=radix_unfold_qkv(q,k,v,input);
        q=full[0];k=full[1];v=full[2];
        if (layer.store) {shared[layer.sliding] = {k,v};producers[layer.sliding]=i;}
      }

      auto output = radix_fold(attention(q, k, v, input, layer,is_causal,layer.shared?producers[layer.sliding]:i),input)
        .reshape({tokens, heads * layer.dim});

      output=radix_enabled(input)?linear_radix_projection(output,p+"self_attn.o_proj",input):prefix_projection(output,p+"self_attn.o_proj",input);
      bool fused_residual_prenorm=false;
#ifdef TEI_TORCH_CUDA_KERNELS
      if(fused_legacy_residual) {
        auto result=gemma3_residual_prenorm_cuda(states,output,
          weight(norm_scales,prefix+p+"post_attention_layernorm.weight"),
          weight(norm_scales,prefix+p+"pre_feedforward_layernorm.weight"),epsilon);
        if(result.first.defined()) {
          states=std::move(result.first);normalized=std::move(result.second);
          fused_residual_prenorm=true;
        }
      }
#endif
      if(!fused_residual_prenorm) {
        states = residual_norm(states,output,p+"post_attention_layernorm");
        normalized = norm(states, p + "pre_feedforward_layernorm");
      }

      auto gu=layer.fp8_gate_up ? layer.fp8_gate_up->forward(normalized) : !input.prefix && ordinary_ffn ? ordinary_ffn->forward(normalized,layer.gate_up,ordinary_session) : at::Tensor();
      if(!gu.defined())gu=prefix_linear(normalized,layer.gate_up,{},input);
      auto width = gu.size(-1) / 2;
      at::Tensor mlp;
#ifdef TEI_TORCH_CUDA_KERNELS
      auto act=string("hidden_activation","gelu_pytorch_tanh");
      if(device.is_cuda() && dtype==at::kBFloat16 && act!="relu" && act!="gelu")mlp=gemma_gated_cuda(gu,act!="silu" && act!="swish");
      else
#endif
      mlp=activate(gu.narrow(-1,0,width))*gu.narrow(-1,width,width);
      auto down=layer.fp8_down ? layer.fp8_down->forward(mlp,has(p+"mlp.down_proj.bias")?w(p+"mlp.down_proj.bias"):at::Tensor()) : !input.prefix && ordinary_ffn && !has(p+"mlp.down_proj.bias") ?
        ordinary_ffn->forward(mlp,w(p+"mlp.down_proj.weight"),ordinary_session) : at::Tensor();
      mlp=down.defined() ? down : input.prefix?prefix_linear(mlp,w(p+"mlp.down_proj.weight"),
        has(p+"mlp.down_proj.bias")?w(p+"mlp.down_proj.bias"):at::Tensor(),input):linear(mlp,p+"mlp.down_proj");

      if(boolean("enable_moe_block",false))mlp=moe(states,mlp,layer,input);
#ifdef TEI_TORCH_CUDA_KERNELS
      const auto* layer_scale=publisher_sequence && ple_width==0 ? &w(p+"layer_scalar") : nullptr;
      if(layer_scale && device.is_cuda() && dtype==at::kBFloat16 && states.is_contiguous()
         &&layer_scale->device()==device && layer_scale->scalar_type()==at::kBFloat16
         &&layer_scale->is_contiguous() && layer_scale->dim()==1 && layer_scale->numel()==1) {
        // Both original pointwise stages round to BF16. The fresh output cannot
        // overwrite residual/normalized inputs or a previous request's state.
        if(mlp.dim()==2 &&mlp.size(1)>0 &&mlp.size(1)<=8192 &&mlp.is_contiguous()) {
          states=gemma4_bf16_norm_residual(states,mlp,
            weight(norm_scales,prefix+p+"post_feedforward_layernorm.weight"),epsilon,*layer_scale);
          continue;
        }
        auto normalized_mlp=norm(mlp,p+"post_feedforward_layernorm");
        if(normalized_mlp.is_contiguous()) {
          states=gemma4_bf16_residual_scale(states,normalized_mlp,*layer_scale);
          continue;
        }
      }
#endif
      states = residual_norm(states,mlp,p+"post_feedforward_layernorm");

      if (ple.defined()) {
        auto stem = p + (embedding2 ? "ple_block." : "");
        auto contribution = activate(linear(states, stem + "per_layer_input_gate")) * ple.select(1,i);
        states = states + norm(linear(contribution, stem + "per_layer_projection"), stem + "post_per_layer_input_norm");
      }
      if (!legacy) states = states * w(p + "layer_scalar");

    }
    auto output = norm(states, "norm");
    output=embedding2 ? linear(output, "embedding_projection") : output;
    return prefix_restore(input,radix_unfold(output,input));
  }
};
}
// The actual Candle audio backend lowers convolution to row-major im2col and
// FP32 GEMM. cuDNN chooses a different accumulation order at BF16 boundaries.
// im2col represents only intrinsic filter boundaries, without appending frames.
at::Tensor gemma_audio_conv2d(const at::Tensor& input,const at::Tensor& filter,int64_t stride) {
  TORCH_CHECK(input.dim()==4&&filter.dim()==4&&input.scalar_type()==at::kFloat&&filter.scalar_type()==at::kFloat&&input.device()==filter.device()&&input.size(1)==filter.size(1)&&stride>0,"Invalid Gemma audio convolution geometry");
  const int64_t height=(input.size(2)+2*(filter.size(2)/2)-filter.size(2))/stride+1;
  const int64_t width=(input.size(3)+2-filter.size(3))/stride+1;
  TORCH_CHECK(input.size(0)>0&&height>0&&width>0,"Empty Gemma audio convolution output");
  auto columns=at::im2col(input,{filter.size(2),filter.size(3)},{1,1},{filter.size(2)/2,1},{stride,stride}).transpose(1,2).contiguous().reshape({input.size(0)*height*width,-1});
  return at::matmul(columns,filter.flatten(1).transpose(0,1)).reshape({input.size(0),height,width,filter.size(0)}).permute({0,3,1,2}).contiguous();
}
at::Tensor gemma_audio_linear(const at::Tensor& input,const at::Tensor& matrix,const at::Tensor& bias) {
  // Candle rounds the low-precision GEMM before adding its learned bias.
  auto output=at::linear(input,matrix);
  return bias.defined()?output+bias:output;
}
std::unique_ptr<Model> create_gemma(const Options& cfg, Weights& weights, c10::Device device, at::ScalarType dtype) {
  auto family = cfg.string("model_type", "");
  if (family != "gemma3" && family != "gemma3_text" && family != "gemma4" && family != "gemma4_text" && family != "embedding_gemma2") return nullptr;
  return std::make_unique<Gemma>(cfg, weights, device, dtype);
}
}

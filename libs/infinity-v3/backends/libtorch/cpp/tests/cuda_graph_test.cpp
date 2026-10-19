#include "batching/cuda_graph.h"
#include "batching/prefix_cache.h"
#include "batching/radix.h"
#include "models/decoder_models.h"
#include "kernels/fast_kernels.h"
#include "helpers/vendor_error.h"
#include "kernels/prefix_state_copy.h"
#include "kernels/prefix_input.h"
#include <ATen/ops/_fused_rms_norm.h>
#include <ATen/ops/_flash_attention_forward.h>
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/cuda/CUDAEvent.h>
#include <thread>
#include <chrono>
#include <set>
using namespace tei;
void prefix_input_test() {
 for(int batch:{1,32,65}) {
  std::vector<int32_t> full{0},suffix{0};std::vector<int64_t> hits,indices;
  for(int row=0;row<batch;++row) {
   const auto length=row%3==0?513:row%7+2;const auto hit=row%length;
   hits.push_back(hit);
   for(int token=full.back()+hit;token<full.back()+length;++token)indices.push_back(token);
   full.push_back(full.back()+length);suffix.push_back(suffix.back()+length-hit);
  }
  auto opts=at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
  auto ids=at::arange(full.back(),opts)+(int64_t(1)<<60),positions=-at::arange(full.back(),opts),types=ids.remainder(3);
  auto selected=at::tensor(indices,opts);auto packed=pack_prefix_inputs(ids,positions,types,full,suffix,hits);
  TORCH_CHECK(at::equal(packed.ids,ids.index_select(0,selected))&&at::equal(packed.positions,positions.index_select(0,selected))
    &&at::equal(packed.types,types.index_select(0,selected)),"Packed suffix gather changed fields or sequence boundaries");
  TORCH_CHECK(!prefix_inputs_supported(at::_neg_view(ids),positions,types)
    &&!prefix_inputs_supported(ids,positions.to(at::kInt),types),"Unsupported prefix input layout accepted");
  auto invalid=hits;invalid.back()=full.back()-full[batch-1];bool rejected=false;
  try{pack_prefix_inputs(ids,positions,types,full,suffix,invalid);}catch(const c10::Error&){rejected=true;}
  TORCH_CHECK(rejected,"Invalid final-row metadata reached suffix gathering");
 }
}
void prefix_state_copy_test() {
 std::vector<at::Tensor> targets,sources,expected;
 const at::ScalarType types[]={at::kByte,at::kShort,at::kFloat,at::kDouble};
 for(int i=0;i<70;++i) {
  auto dtype=types[i%4];auto bytes=int64_t(c10::elementSize(dtype));
  auto width=int64_t(i%5==0?128:129),rows=int64_t(i%3==0?513:7),gap=int64_t(i%4);
  auto offset=int64_t(i%3)*bytes;
  auto raw=at::arange(offset+(rows*(width+gap))*bytes,at::TensorOptions().device(at::kCUDA).dtype(at::kLong))
   .remainder(256).to(at::kByte);
  auto source=raw.narrow(0,offset,rows*(width+gap)*bytes).view(dtype).view({rows,width+gap}).narrow(1,0,width);
  // A size-one inner dimension must not invalidate an otherwise packed row.
  if(i%2)source=source.unsqueeze(1);
  if(i==69)source=source.narrow(0,0,1).expand(source.sizes());
  auto target_storage=at::empty({offset+int64_t(source.nbytes())},source.options().dtype(at::kByte));
  auto target=target_storage.narrow(0,offset,source.nbytes()).view(dtype).view(source.sizes());
  sources.push_back(source);targets.push_back(target);expected.push_back(source.contiguous().view(at::kByte));
 }
 TORCH_CHECK(prefix_state_copies(targets,sources),"Packed and row-strided prefix copies rejected");
 for(size_t i=0;i<targets.size();++i)
  TORCH_CHECK(at::equal(targets[i].view(at::kByte),expected[i]),"Prefix state copy changed raw bits or row boundaries");
 std::vector<at::Tensor> partial;
 for(const auto& source:sources){auto shape=source.sizes().vec();shape[0]=3;partial.push_back(at::empty(shape,source.options()));}
 TORCH_CHECK(!prefix_state_copies(partial,sources),"Exact copy silently shortened source rows");
 TORCH_CHECK(prefix_state_copies(partial,sources,true),"Leading-row copy rejected packed or strided sources");
 for(size_t i=0;i<partial.size();++i)
  TORCH_CHECK(at::equal(partial[i].view(at::kByte),sources[i].narrow(0,0,3).contiguous().view(at::kByte)),
    "Leading-row copy changed bits or crossed its source span");
 auto first=at::full({4,4},-1,at::TensorOptions().device(at::kCUDA));
 auto unsupported=at::ones({4,4},first.options()).transpose(0,1);
 TORCH_CHECK(!prefix_state_copies({first,at::empty_like(first)},{at::zeros_like(first),unsupported})
  &&at::equal(first,at::full_like(first,-1)),"Rejected copy layout submitted partial work");
 TORCH_CHECK(!prefix_state_copies({at::Tensor()},{first})&&!prefix_state_copies({}, {first}),"Invalid copy arguments accepted");
 TORCH_CHECK(!prefix_state_copies({at::empty_like(first)},{at::_neg_view(first)}),"Lazy negative view accepted by raw copy");
 std::cout<<"Batched prefix state copies preserved raw bits across 70 aligned and strided layouts\n";
}
class Dense final:public Model {
public:
 void ready() override{}
 at::Tensor forward(const PackedInput& input) const override{
  return at::sin(input.ids.to(at::kFloat).unsqueeze(1)+input.positions.to(at::kFloat).unsqueeze(1)*.1)
    *at::arange(1,9,input.ids.options().dtype(at::kFloat)).unsqueeze(0);
 }
};
class Heads final:public Model {
public:
 void ready() override{}
 at::Tensor forward(const PackedInput& input) const override {
  return input.ids.to(at::kFloat).unsqueeze(1).repeat({1,3});
 }
 at::Tensor predict(const PackedInput& input,bool tokenwise) const override {
  auto h=forward(input);
  if(!tokenwise)h=h.index_select(0,input.cumulative.narrow(0,0,input.batch).to(at::kLong));
  return h+(tokenwise?13:7);
 }
};
class FailedWarmup final:public Model {
 mutable bool injected=false;
 mutable at::cuda::CUDAEvent queued;
public:
 void ready() override{}
 at::Tensor forward(const PackedInput& input) const override {
  if(!injected) {
   injected=true;
   const auto stream=c10::cuda::getCurrentCUDAStream(input.ids.get_device());
   TORCH_CHECK(cudaLaunchHostFunc(stream.stream(),[](void*) {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
   },nullptr)==cudaSuccess,"Unable to enqueue warmup work");
   queued.record(stream);
   TORCH_CHECK(false,"Injected warmup allocation failure");
  }
  TORCH_CHECK(queued.query(),"Fallback began before failed warmup work drained");
  return input.ids.to(at::kFloat).unsqueeze(1);
 }
};
class VendorFailure final:public Model {
 const bool during_capture;
 mutable int calls=0;
 mutable bool injected=false;
 mutable at::cuda::CUDAEvent queued;
public:
 explicit VendorFailure(bool capture):during_capture(capture){}
 void ready() override{}
 at::Tensor forward(const PackedInput& input) const override {
  ++calls;
  const auto stream=c10::cuda::getCurrentCUDAStream(input.ids.get_device());
  cudaStreamCaptureStatus status;
  TORCH_CHECK(cudaStreamIsCapturing(stream.stream(),&status)==cudaSuccess,"Unable to inspect vendor failure stream");
  if(!injected && (!during_capture || status==cudaStreamCaptureStatusActive)) {
   injected=true;
   if(!during_capture) {
    TORCH_CHECK(cudaLaunchHostFunc(stream.stream(),[](void*) {
     std::this_thread::sleep_for(std::chrono::milliseconds(250));
    },nullptr)==cudaSuccess,"Unable to enqueue vendor warmup work");
    queued.record(stream);
   } else {
    TORCH_CHECK(calls==4,"Vendor capture failure must follow all three warmups");
    auto pending=input.ids.to(at::kFloat).unsqueeze(1);
   }
   throw VendorAllocationError("Injected numeric vendor allocation failure",3);
  }
  if(injected) {
   TORCH_CHECK(status==cudaStreamCaptureStatusNone,"Vendor fallback left capture active");
   if(!during_capture)TORCH_CHECK(queued.query(),"Vendor fallback began before warmup drained");
  }
  return input.ids.to(at::kFloat).unsqueeze(1);
 }
};
class Uncapturable final:public Model {
public:
 void ready() override{}
 at::Tensor forward(const PackedInput& input) const override{
  auto sum=input.ids.sum().item<int64_t>();
  return at::full({input.ids.size(0),8},sum,input.ids.options().dtype(at::kFloat));
 }
};
class Media final:public Model {
public:
 bool prepare_features=true;
 mutable bool observed_shared_images=false;
 void ready() override{}
 void prepare_input(PackedInput& input) const override {
  int active_device=-1;
  TORCH_CHECK(cudaGetDevice(&active_device)==cudaSuccess && active_device==input.ids.get_device(),"Media preparation ran without the input device guard");
  if(prepare_features)for(auto& image:input.images)image.prepared_features=image.pixels.sum().view({1,1});
  for(auto& audio:input.audios)audio.selected_frames=at::nonzero(audio.validity).flatten();
 }
 at::Tensor forward(const PackedInput& input) const override {
  observed_shared_images=input.images.size()==2&&input.images[0].pixels.unsafeGetTensorImpl()==input.images[1].pixels.unsafeGetTensorImpl();
  auto result=input.ids.to(at::kFloat);
  for(const auto& image:input.images)result=result+(image.prepared_features.defined()?image.prepared_features.sum():image.pixels.sum())+image.grid_thw[1]+image.token_start;
  for(const auto& audio:input.audios) {
   auto indices=audio.selected_frames.defined()?audio.selected_frames:at::nonzero(audio.validity).flatten();
   result=result+audio.features.index_select(0,indices).sum()+audio.token_start;
  }
  if(input.multimodal_positions.defined())result=result+input.multimodal_positions.sum();
  return result.unsqueeze(1);
 }
};
std::unique_ptr<Model> tiny_decoder(Weights* fixture=nullptr,int64_t head_dim=8,at::ScalarType dtype=at::kHalf) {
 Options cfg;cfg.values={{"model_type","qwen3"},{"hidden_size","16"},{"num_attention_heads","2"},{"num_key_value_heads","1"},{"head_dim","8"},{"intermediate_size","24"},{"num_hidden_layers","1"},{"vocab_size","32"},{"max_position_embeddings","32"}};
 cfg.values["head_dim"]=std::to_string(head_dim);
 Weights w;auto opts=at::TensorOptions().device(at::kCUDA).dtype(dtype);
 auto add=[&](std::string name,std::initializer_list<int64_t> shape,bool norm=false){w[name]=norm?at::ones(at::IntArrayRef(shape),opts):at::randn(at::IntArrayRef(shape),opts)*.1;};
 add("model.embed_tokens.weight",{32,16});add("model.norm.weight",{16},true);
 auto p=std::string("model.layers.0.");add(p+"input_layernorm.weight",{16},true);add(p+"post_attention_layernorm.weight",{16},true);
 add(p+"self_attn.q_proj.weight",{2*head_dim,16});add(p+"self_attn.k_proj.weight",{head_dim,16});add(p+"self_attn.v_proj.weight",{head_dim,16});add(p+"self_attn.o_proj.weight",{16,2*head_dim});
 add(p+"self_attn.q_norm.weight",{head_dim},true);add(p+"self_attn.k_norm.weight",{head_dim},true);
 add(p+"mlp.gate_proj.weight",{24,16});add(p+"mlp.up_proj.weight",{24,16});add(p+"mlp.down_proj.weight",{16,24});
 if(fixture)*fixture=w;
 auto result=create_decoder(cfg,w,c10::Device("cuda:0"),dtype);result->ready();return result;
}
void prefix_qkv_graph_test() {
 for(auto dtype:{at::kHalf,at::kBFloat16}) {
  at::manual_seed(409);auto model=tiny_decoder(nullptr,128,dtype);
  auto opts=at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
  std::vector<int32_t> ends{0,8,14,24};std::vector<int64_t> positions;
  for(size_t row=0;row+1<ends.size();++row)for(int token=0;token<ends[row+1]-ends[row];++token)positions.push_back(token);
  PackedInput input{at::Tensor(),at::zeros({24},opts),at::tensor(positions,opts),
   at::tensor(ends,opts.dtype(at::kInt)),ends.data(),3,10};
  PrefixCache cache(16<<20,64,32);CudaGraphCache graphs(4,64,1);
  for(int round=0;round<12;++round) {
   if(round==8)cache.clear();
   const int parent=(round/2)%2;
   auto ids=std::vector<int64_t>(24,3+parent);input.ids=at::tensor(ids,opts);
   cache.run(*model,input,0,&graphs);
   // Ragged rows retain different prefix lengths; change every suffix row to
   // force shader replay rather than a complete-output cache hit.
   for(int row=0;row<3;++row)for(int token=ends[row]+row+2;token<ends[row+1];++token)ids[token]=16+round;
   input.ids=at::tensor(ids,opts);auto expected=model->forward(input);
   auto actual=cache.run(*model,input,0,&graphs);
   TORCH_CHECK(at::allclose(actual,expected,.01,.02),"Direct prefix QKV graph changed ragged output after parent rebinding");
   actual.zero_();
  }
  auto stats=graphs.stats();
  TORCH_CHECK(stats.prefix_captures==1&&stats.prefix_hits>=10,"Direct prefix QKV graph did not capture/replay after clear");
 }
 std::cout<<"Direct prefix QKV graph preserved ragged Half/BF16 output through rebinding and clear\n";
}
void radix_graph_test(Model& model) {
 CudaGraphCache cache(3,16,1);
 auto options=at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
 std::vector<int32_t> offsets={0,3,6};
 PackedInput input{at::tensor({3,4,5,3,4,6},options),at::zeros({6},options),at::tensor({0,1,2,0,1,2},options),
   at::tensor({0,3,6},options.dtype(at::kInt)),offsets.data(),2,3};
 input.compact_ids=at::tensor({3,4,5,6},options);input.compact_positions=at::tensor({0,1,2,2},options);input.compact_types=at::zeros({4},options);
 input.scatter_unfold=at::tensor({0,1,2,0,1,3},options);input.fold_gather=at::tensor({0,1,2,5},options);
 for(int run=0;run<3;++run) {
   // Reorder compact rows and representatives without changing their shapes.
   // Replayed maps and all compact metadata must contain the new values.
   if(run==1) {
     input.compact_ids=at::tensor({6,5,4,3},options);input.compact_positions=at::tensor({2,2,1,0},options);
     input.scatter_unfold=at::tensor({3,2,1,3,2,0},options);input.fold_gather=at::tensor({5,2,1,0},options);
   }
   if(run==2) {input.ids.add_(2);input.compact_ids.add_(2);input.positions.add_(1);input.compact_positions.add_(1);}
   auto expected=model.forward(input),actual=cache.forward(model,input).clone();
   TORCH_CHECK(at::equal(actual,expected),"Radix capture/replay retained stale compact metadata or maps");
   auto full=input;full.compact_ids=at::Tensor();full.compact_positions=at::Tensor();full.compact_types=at::Tensor();full.scatter_unfold=at::Tensor();full.fold_gather=at::Tensor();
   TORCH_CHECK((actual-model.forward(full)).abs().max().item<float>()<.004,"Folded decoder changed full packed output");
 }
 TORCH_CHECK(cache.size()==1,"Radix graph failed to capture or map values changed shape key");
 input.compact_ids=input.ids.clone();input.compact_positions=input.positions.clone();input.compact_types=input.types.clone();
 input.scatter_unfold=at::arange(6,options);input.fold_gather=input.scatter_unfold.clone();
 TORCH_CHECK(at::equal(cache.forward(model,input).clone(),model.forward(input)),"Different compact token count reused stale graph");
 TORCH_CHECK(cache.size()==2,"Compact token count missing from radix graph key");
}
void full_snapshot_graph_test(Model& inner) {
 struct Observed final:Model {
  Model& inner;mutable size_t calls=0,observations=0;
  explicit Observed(Model& value):inner(value){}
  void ready() override{}
  bool supports_prefix_cache(int op)const override{return inner.supports_prefix_cache(op);}
  int64_t prefix_layers()const override{return inner.prefix_layers();}
  uint64_t prefix_payload_bytes(int64_t tokens)const override{return inner.prefix_payload_bytes(tokens);}
  at::Tensor forward(const PackedInput& input)const override{++calls;observations+=bool(input.capture_prefix);return inner.forward(input);}
 } model(inner);
 auto options=at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
 std::vector<int32_t> ends={0,3,7};
 PackedInput input{at::tensor({3,4,5,6,7,8,9},options),at::zeros({7},options),
  at::tensor({0,1,2,0,1,2,3},options),at::tensor(ends,options.dtype(at::kInt)),ends.data(),2,4};
 auto expected=inner.forward(input);
 CudaGraphCache plain(1,64,1);PrefixCache disabled(0);
 TORCH_CHECK(at::equal(disabled.run(model,input,0,&plain),expected)&&model.observations==0,
  "Disabled C++ cache retained full-forward prefix state");
 CudaGraphCache graphs(4,64,1);graphs.retain_prefix_states();
 graphs.forward(model,input);auto before=model.calls,attempts=graphs.stats().attempts;
 PrefixCache cache(1<<20,64,8);
 auto result=cache.run(model,input,0,&graphs);
 TORCH_CHECK(at::equal(result,expected)&&model.calls==before,"Warmed full graph admission reran or changed the backbone");
 TORCH_CHECK(cache.stats().admissions==2&&graphs.stats().attempts==attempts,"Full publication added another graph capture or omitted a row");
 auto owned=result.clone();result.zero_();
 input.ids.add_(2);auto changed=inner.forward(input);
 TORCH_CHECK(at::equal(cache.run(model,input,0,&graphs),changed)&&at::equal(owned,expected),"Full graph replay overwrote an owned snapshot result");
 TORCH_CHECK(model.calls==before,"Changed full parent forced eager snapshot admission");
 auto original=input;original.ids=input.ids-2;
 TORCH_CHECK(at::equal(cache.run(model,original,0,&graphs),expected)&&model.calls==before,
  "Replaying the full graph changed an earlier resident snapshot");
 cache.clear();TORCH_CHECK(cache.bytes()==0,"Full graph retained a resident snapshot after clear");
 TORCH_CHECK(graphs.stats().live_bytes>=inner.prefix_payload_bytes(input.ids.numel()),"Full graph state escaped its storage budget");
 CudaGraphCache cold(4,64,1);PrefixCache first(1<<20,64,8);
 before=model.calls;
 TORCH_CHECK(at::equal(first.run(model,input,0,&cold),changed)&&model.calls==before+1
  &&cold.stats().attempts==0,"Cold admission added graph warmup/capture instead of one ordinary forward");
 auto folded=input;folded.ids=at::tensor({3,4,5,3,4,6,7},options);
 folded.compact_ids=at::tensor({3,4,5,6,7},options);folded.compact_positions=at::tensor({0,1,2,2,3},options);
 folded.compact_types=at::zeros({5},options);folded.scatter_unfold=at::tensor({0,1,2,0,1,3,4},options);
 folded.fold_gather=at::tensor({0,1,2,5,6},options);
 CudaGraphCache radix(4,64,1);radix.retain_prefix_states();radix.forward(model,folded);
 PrefixCache snapshots(1<<20,64,8);before=model.calls;auto full=inner.forward(folded);
 TORCH_CHECK(at::equal(snapshots.run(model,folded,0,&radix),full)&&model.calls==before,
  "Folded full graph admission reran or changed logical rows");
 auto saved=folded;folded.ids=folded.ids+2;folded.compact_ids=folded.compact_ids+2;
 TORCH_CHECK(at::equal(snapshots.run(model,folded,0,&radix),inner.forward(folded))
  &&at::equal(snapshots.run(model,saved,0,&radix),full)&&model.calls==before,
  "Folded graph publication retained stale maps or shared resident output storage");
 struct External final:Model {
  at::Tensor storage=at::ones({1024,8},at::TensorOptions().device(at::kCUDA));
  void ready()override{}
  bool supports_prefix_cache(int op)const override{return op==0;}
  int64_t prefix_layers()const override{return 1;}
  uint64_t prefix_payload_bytes(int64_t n)const override{return n*8*4*3;}
  at::Tensor forward(const PackedInput& input)const override {
   auto rows=storage.narrow(0,0,input.ids.numel());
   if(input.capture_prefix)prefix_capture_attention(input,0,rows.unsqueeze(1),rows.unsqueeze(1));
   return prefix_restore(input,rows);
  }
 } external;
 CudaGraphCache budget(4,64,1,16<<10);budget.retain_prefix_states();
 TORCH_CHECK(at::equal(budget.forward(external,input),external.forward(input))&&budget.size()==0
  &&budget.stats().attempts==1,"External full-state views escaped whole-storage budget accounting");
 std::cout<<"Warmed full graph snapshot publication, independent storage and cold fallback passed\n";
}
// This copy-only model isolates raw Q/K/V bytes from attention arithmetic.
// The actual decoder graph above independently covers full/folded semantics.
class RadixQkvCopy final:public Model {
public:
 at::Tensor bits,q,k,v;
 explicit RadixQkvCopy(at::ScalarType dtype,int64_t dimension) {
  auto options=at::TensorOptions().device(at::kCUDA,0).dtype(at::kLong);
  // Includes every 16-bit pattern, including NaNs/infinities and signed zero.
  bits=at::arange(352*40*dimension,options).remainder(65536).to(at::kShort).view({352,40,dimension});
  auto packed=bits.view(dtype);q=packed.narrow(1,0,32);k=packed.narrow(1,32,4);v=packed.narrow(1,36,4);
 }
 void ready() override{}
 at::Tensor reference(const PackedInput& input) const {
  return at::cat({q.index_select(0,input.scatter_unfold).flatten(1).view(at::kShort),
   k.index_select(0,input.scatter_unfold).flatten(1).view(at::kShort),
   v.index_select(0,input.scatter_unfold).flatten(1).view(at::kShort)},1);
 }
 at::Tensor forward(const PackedInput& input) const override {
  auto output=radix_unfold_qkv(q,k,v,input);
  return at::cat({output[0].flatten(1).view(at::kShort),output[1].flatten(1).view(at::kShort),
   output[2].flatten(1).view(at::kShort)},1);
 }
};
void radix_qkv_copy_graph_test() {
 auto options=at::TensorOptions().device(at::kCUDA,0).dtype(at::kLong);
 std::vector<int32_t> offsets;std::vector<int64_t> scatter,gather,positions,compact_positions;
 for(int64_t row=0;row<8;++row){
  offsets.push_back(row*128);
  for(int64_t token=0;token<128;++token){
   auto index=token<96?token:96+row*32+token-96;
   scatter.push_back(index);positions.push_back(token);
   if(row==0||token>=96){gather.push_back(row*128+token);compact_positions.push_back(token);}
  }
 }
 offsets.push_back(1024);
 auto tensor=[&](const std::vector<int64_t>& values){return at::tensor(values,options);};
 PackedInput input{tensor(scatter),at::zeros({1024},options),tensor(positions),
  at::tensor(offsets,options.dtype(at::kInt)),offsets.data(),8,128};
 input.compact_ids=at::arange(352,options);input.compact_types=at::zeros({352},options);
 input.compact_positions=tensor(compact_positions);input.scatter_unfold=tensor(scatter);input.fold_gather=tensor(gather);
 for(auto dtype:{at::kHalf,at::kBFloat16})for(int64_t dimension:{7,64}) {
  RadixQkvCopy model(dtype,dimension);auto request=input;
  // Dimension 7 intentionally takes the unsupported vector-copy fallback.
  int devices=0;cudaGetDeviceCount(&devices);
  if(devices>1){
   cudaSetDevice(1);auto actual=model.forward(request);int current=-1;cudaGetDevice(&current);
   TORCH_CHECK(current==1,"QKV unfolding changed the caller's CUDA device");
   TORCH_CHECK(at::equal(actual,model.reference(request)),"QKV unfolding used another-device stream");
   cudaSetDevice(0);
  }
  CudaGraphCache cache(2,2048,1);
  for(int run=0;run<3;++run){
   if(run==1){
    request.compact_ids=request.compact_ids.flip({0});request.compact_positions=request.compact_positions.flip({0});
    request.fold_gather=request.fold_gather.flip({0});request.scatter_unfold=351-request.scatter_unfold;
    model.bits.copy_(model.bits.flip({0}));
   }
   if(run==2)model.bits.bitwise_xor_(255);
   auto expected=model.reference(request),actual=cache.forward(model,request).clone();
   TORCH_CHECK(at::equal(actual,expected),"QKV unfolding changed raw bits or replayed stale maps/values");
  }
  TORCH_CHECK(cache.size()==1,"Same QKV/map shapes did not reuse the graph");
 }
}
void arithmetic_policy_test() {
 struct PolicyModel final:Model {
  void ready() override {}
  at::Tensor forward(const PackedInput& input) const override {
   auto& context=at::globalContext();
   auto scale=(context.allowTF32CuBLAS()?3:2)+(context.deterministicCuDNN()?7:0);
   return input.ids.to(at::kFloat).unsqueeze(1)*scale;
  }
 } model;
 auto& global=at::globalContext();
 struct Restore {
  bool tf32,deterministic;
  ~Restore(){at::globalContext().setAllowTF32CuBLAS(tf32);at::globalContext().setDeterministicCuDNN(deterministic);}
 } restore{global.allowTF32CuBLAS(),global.deterministicCuDNN()};
 auto options=at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
 int32_t offsets[]={0,3};auto ids=at::tensor({3,4,5},options);
 PackedInput input{ids,at::zeros_like(ids),ids,at::tensor({0,3},options.dtype(at::kInt)),offsets,1,3};
 CudaGraphCache cache(4,8,1);
 global.setAllowTF32CuBLAS(false);
 TORCH_CHECK(at::equal(cache.forward(model,input),model.forward(input)),"Initial arithmetic policy capture failed");
 global.setAllowTF32CuBLAS(true);
 TORCH_CHECK(at::equal(cache.forward(model,input),model.forward(input)),"Full-request graph replay ignored changed BLAS policy");
 global.setDeterministicCuDNN(!restore.deterministic);
 TORCH_CHECK(at::equal(cache.forward(model,input),model.forward(input)),"Full-request graph replay ignored changed cuDNN policy");
 TORCH_CHECK(cache.stats().captures==3,"Changed arithmetic settings reused an incompatible shader");
}
void admission_test() {
 Dense model;
 auto options=at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
 CudaGraphCache cache(2,32,3,512ULL<<20,4);
 auto request=[&](int n) {
  std::vector<int32_t> offsets{0,n};
  auto ids=at::arange(n,options);
  PackedInput input{ids,at::zeros_like(ids),ids,at::tensor({0,n},options.dtype(at::kInt)),offsets.data(),1,n};
  TORCH_CHECK(at::equal(cache.forward(model,input).clone(),model.forward(input)),"Admission changed output");
 };
 // Shape churn never pays capture cost; a stable shape can still enter later.
 for(int repeat=0;repeat<5;++repeat)for(int n=1;n<=8;++n)request(n);
 TORCH_CHECK(cache.stats().attempts==0,"Alternating cold shapes captured");
 for(int n=1;n<=4;++n)for(int repeat=0;repeat<4;++repeat)request(n);
 auto warmed=cache.stats();
 TORCH_CHECK(warmed.captures==4 && warmed.evictions==2 && cache.size()==2,"Working-set turnover failed");
 for(int n=5;n<=8;++n)for(int repeat=0;repeat<4;++repeat)request(n);
 TORCH_CHECK(cache.stats().attempts==4,"Capture budget did not stop churn");
 request(4);TORCH_CHECK(cache.stats().hits>warmed.hits,"Budget exhaustion disabled resident replay");
 CudaGraphCache tiny_budget(2,32,1,1,4);
 std::vector<int32_t> offsets{0,3};auto ids=at::arange(3,options);
 PackedInput input{ids,at::zeros_like(ids),ids,at::tensor({0,3},options.dtype(at::kInt)),offsets.data(),1,3};
 for(int repeat=0;repeat<4;++repeat)
  TORCH_CHECK(at::equal(tiny_budget.forward(model,input),model.forward(input)),"Budget fallback changed output");
 TORCH_CHECK(tiny_budget.size()==0 && tiny_budget.stats().attempts==0,"Oversized inputs allocated a graph pool");
}
void retired_output_pool_test() {
 auto pools=[] {
  std::set<c10::MempoolId_t> result;
  for(const auto& segment:c10::cuda::CUDACachingAllocator::snapshot({0,0},false).segments)
   if(segment.owner_private_pool_id!=c10::MempoolId_t{0,0})result.insert(segment.owner_private_pool_id);
  return result;
 };
 Dense model;CudaGraphCache cache(1,64,1);
 auto options=at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
 auto request=[&](int n) {
  std::vector<int32_t> offsets{0,n};auto ids=at::arange(n,options);
  PackedInput input{ids,at::zeros_like(ids),ids,at::tensor({0,n},options.dtype(at::kInt)),offsets.data(),1,n};
  auto result=cache.forward(model,input);
  TORCH_CHECK(at::equal(result,model.forward(input)),"Retired-pool fixture changed output");
  return result;
 };
 const auto before=pools();auto held=request(4);auto added=pools();
 for(const auto& pool:before)added.erase(pool);
 TORCH_CHECK(added.size()==1,"Retired-pool fixture did not capture a private pool");
 const auto retired=*added.begin();auto expected=held.clone();
 request(8);
 TORCH_CHECK(pools().count(retired)&&at::equal(held,expected),"Eviction freed a live output");
 held=at::Tensor();request(16);
 TORCH_CHECK(!pools().count(retired),"Released output left its retired pool reserved");
 // Both a failed capture and an over-budget successful trial must release
 // their private pools before eager fallback returns.
 for(bool unsupported:{false,true}) {
  const auto initial=pools();Uncapturable uncapturable;
  const Model& trial_model=unsupported?static_cast<const Model&>(uncapturable):model;
  CudaGraphCache trial(1,64,1,unsupported?512ULL<<20:1000,4);
  std::vector<int32_t> offsets{0,4};auto ids=at::arange(4,options);
  PackedInput input{ids,at::zeros_like(ids),ids,at::tensor({0,4},options.dtype(at::kInt)),offsets.data(),1,4};
  TORCH_CHECK(at::equal(trial.forward(trial_model,input),trial_model.forward(input)),"Trial fallback changed output");
  TORCH_CHECK(trial.stats().attempts==1&&trial.size()==0,"Trial fixture did not reject a capture");
  TORCH_CHECK(pools()==initial,"Rejected capture left a private pool reserved");
 }
}
void graph_teardown_pool_test() {
 auto pools=[] {
  std::set<c10::MempoolId_t> result;
  for(const auto& segment:c10::cuda::CUDACachingAllocator::snapshot({0,0},false).segments)
   if(segment.owner_private_pool_id!=c10::MempoolId_t{0,0})result.insert(segment.owner_private_pool_id);
  return result;
 };
 Dense model;
 auto options=at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
 std::vector<int32_t> offsets{0,4};auto ids=at::arange(4,options);
 PackedInput input{ids,at::zeros_like(ids),ids,at::tensor({0,4},options.dtype(at::kInt)),offsets.data(),1,4};
 const auto before=pools();
 for(int repeat=0;repeat<8;++repeat) {
  {
   CudaGraphCache cache(1,64,1);
   TORCH_CHECK(at::equal(cache.forward(model,input),model.forward(input)),"Teardown fixture changed output");
   TORCH_CHECK(cache.size()==1,"Teardown fixture did not capture");
  }
  TORCH_CHECK(pools()==before,"Destroyed graph cache left a private pool reserved at iteration ",repeat);
 }
 // Deferred outputs must remain valid after model teardown. Once released,
 // their remaining cached reservation must be reclaimable by the allocator.
 at::Tensor held;std::set<c10::MempoolId_t> retained;
 auto expected=model.forward(input);
 {
  CudaGraphCache cache(1,64,1);
  held=cache.forward(model,input);
  retained=pools();for(const auto& pool:before)retained.erase(pool);
  TORCH_CHECK(retained.size()==1,"Deferred teardown fixture did not capture a pool");
 }
 TORCH_CHECK(at::equal(held,expected),"Graph teardown changed deferred output");
 held=at::Tensor();
 for(const auto& pool:retained)c10::cuda::CUDACachingAllocator::emptyCache(pool);
 TORCH_CHECK(pools()==before,"Released deferred graph output left an unreclaimable private pool");
}
class GemmWorkspaces final:public Model {
 at::Tensor weight=at::randn({640,640},at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16));
public:
 void ready() override{}
 at::Tensor forward(const PackedInput& input) const override {
  auto hidden=input.ids.to(at::kBFloat16).unsqueeze(1).repeat({1,640})*.001;
  // The long reduction uses cuBLAS workspace, unlike tiny forward-only GEMMs.
  auto cross=at::mm(hidden.transpose(0,1),hidden);
  return at::mm(hidden,at::mm(cross,weight));
 }
};
void gemm_workspace_lifetime_test() {
 GemmWorkspaces model;CudaGraphCache cache(2,65536,1,2ULL<<30,16);
 auto options=at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
 auto request=[&](int n) {
  std::vector<int32_t> offsets{0,n};auto ids=at::arange(n,options);
  PackedInput input{ids,at::zeros_like(ids),ids,at::tensor({0,n},options.dtype(at::kInt)),offsets.data(),1,n};
  auto expected=model.forward(input),actual=cache.forward(model,input).clone();
  TORCH_CHECK(at::equal(actual,expected),"Surviving GEMM graph changed after graph destruction");
  c10::cuda::getCurrentCUDAStream().synchronize();
 };
 for(int i=1;i<=8;++i) {
  if(i>1)request((i+126)*256);
  request((i+127)*256);
  // Simulate allocator pressure reclaiming unused storage. A surviving graph
  // must still own its workspace; production does not clear the global cache.
  c10::cuda::CUDACachingAllocator::emptyCache();
  if(i>1)request((i+126)*256);
 }
 TORCH_CHECK(cache.stats().captures==8&&cache.stats().evictions==6&&cache.stats().hits==14,
   "GEMM fixture did not exercise eviction and surviving replay");
 // The shared Torch stream pool wraps after 32 acquisitions. Another resident
 // model must not capture on a lane whose workspaces our graph still uses.
 std::vector<int32_t> offsets{0,32768};auto ids=at::arange(32768,options);
 PackedInput input{ids,at::zeros_like(ids),ids,at::tensor({0,32768},options.dtype(at::kInt)),offsets.data(),1,32768};
 auto surviving=std::make_unique<CudaGraphCache>(1,65536,1,2ULL<<30,2);
 auto expected=model.forward(input);
 TORCH_CHECK(at::equal(surviving->forward(model,input).clone(),expected),"Resident GEMM capture failed");
 for(int i=0;i<31;++i)c10::cuda::getStreamFromPool(false,0);
 {
  GemmWorkspaces other_model;CudaGraphCache other(1,65536,1,2ULL<<30,2);
  TORCH_CHECK(at::equal(other.forward(other_model,input).clone(),other_model.forward(input)),"Other model capture failed");
  TORCH_CHECK(other.size()==1,"Other resident model did not capture");
 }
 c10::cuda::CUDACachingAllocator::emptyCache();
 TORCH_CHECK(at::equal(surviving->forward(model,input).clone(),expected),"Another model destroyed a surviving workspace");
 auto caller=c10::cuda::getCurrentCUDAStream();caller.synchronize();
 TORCH_CHECK(cudaLaunchHostFunc(caller.stream(),[](void*) {
  std::this_thread::sleep_for(std::chrono::milliseconds(250));
 },nullptr)==cudaSuccess,"Unable to enqueue pending replay");
 auto held=surviving->forward(model,input);at::cuda::CUDAEvent finished;finished.record(caller);
 surviving.reset();
 TORCH_CHECK(finished.query(),"Graph cache destroyed before pending replay drained");
 TORCH_CHECK(at::equal(held,expected),"Graph teardown changed deferred output");
}
int main(int argc,char** argv){c10::InferenceMode guard;
 if(argc==2&&std::string(argv[1])=="prefix-qkv"){prefix_qkv_graph_test();return 0;}
 if(argc==2&&std::string(argv[1])=="prefix-input"){prefix_input_test();return 0;}
 prefix_state_copy_test();prefix_input_test();if(argc==2&&std::string(argv[1])=="prefix-copy")return 0;prefix_qkv_graph_test();arithmetic_policy_test();admission_test();retired_output_pool_test();graph_teardown_pool_test();gemm_workspace_lifetime_test();radix_qkv_copy_graph_test();Weights fixture;auto decoder=tiny_decoder(&fixture);auto& model=*decoder;radix_graph_test(model);full_snapshot_graph_test(model);CudaGraphCache cache(2,8,1);
 for(auto offsets:std::vector<std::vector<int32_t>>{{0,3},{0,1,3},{0,2,3},{0,3}}){
  auto options=at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
  auto ids=at::tensor({3,4,5},options),pos=at::tensor({0,1,2},options);
  auto cu=at::from_blob(offsets.data(),{int64_t(offsets.size())},at::TensorOptions().dtype(at::kInt)).to(at::kCUDA);
  PackedInput input{ids,at::zeros_like(ids),pos,cu,offsets.data(),int64_t(offsets.size()-1),3};
  for(int run=0;run<3;++run){input.ids.add_(7);input.positions.add_(1);
   auto actual=cache.forward(model,input).clone(),expected=model.forward(input);
   TORCH_CHECK((actual-expected).abs().max().item<float>()<0.0001,"Graph replay used stale inputs");
  }
  TORCH_CHECK(cache.size()<=2,"Graph cache exceeds bound");
 }
 TORCH_CHECK(cache.size()==2,"Dense graphs did not capture");
 std::vector<int32_t> offsets{0,3};auto options=at::TensorOptions().device(at::kCUDA).dtype(at::kLong);
 auto ids=at::tensor({3,4,5},options);auto cu=at::tensor({0,3},options.dtype(at::kInt));
 PackedInput input{ids,at::zeros_like(ids),at::zeros_like(ids),cu,offsets.data(),1,3};
 auto angles=input.positions.to(at::kFloat).unsqueeze(1)*at::tensor({1.f,.1f,.01f,.001f},options.dtype(at::kFloat));
 auto cosine=at::cos(angles).to(at::kHalf),sine=at::sin(angles).to(at::kHalf);
 auto direct=decoder_multimodal_forward(model,input,{}, {},cosine,sine);
 TORCH_CHECK((direct-model.forward(input)).abs().max().item<float>()<0.0001,"External RoPE changed ordinary decoder output");
 auto visual_indices=at::tensor({int64_t(1)},options),visual=at::ones({1,16},options.dtype(at::kHalf))*.25;
 auto injected=decoder_multimodal_forward(model,input,visual_indices,visual,cosine,sine);
 auto withzero=decoder_multimodal_forward(model,input,visual_indices,visual,cosine,sine,{at::zeros_like(visual)});
 // Candle's visual forward explicitly materializes h+r in model dtype
 // before adding deepstack features, even when those features are zero.
 // Compare that ordering rather than assuming a zero feature bypasses it.
 auto initial=at::embedding(fixture.at("model.embed_tokens.weight"),ids);
 initial.index_copy_(0,visual_indices,visual);
 const std::string p="model.layers.0.";
 auto n=tei::fused_add_rms_norm(initial,{},fixture.at(p+"input_layernorm.weight"),1e-6).first;
 auto projection=at::cat({fixture.at(p+"self_attn.q_proj.weight"),fixture.at(p+"self_attn.k_proj.weight"),fixture.at(p+"self_attn.v_proj.weight")},0);
 auto qkv=at::linear(n,projection).view({3,4,8});
 auto q=std::get<0>(at::_fused_rms_norm(qkv.narrow(1,0,2).contiguous(),{8},fixture.at(p+"self_attn.q_norm.weight"),1e-6));
 auto k=std::get<0>(at::_fused_rms_norm(qkv.narrow(1,2,1).contiguous(),{8},fixture.at(p+"self_attn.k_norm.weight"),1e-6));
 auto rotated=tei::fused_rotary(q,k,cosine.view({3,1,4}),sine.view({3,1,4}));
 auto attention=std::get<0>(at::_flash_attention_forward(rotated.first,rotated.second,qkv.narrow(1,3,1),cu,cu,3,3,0.,true,false)).contiguous().view({3,16});
 auto post=tei::fused_add_rms_norm(at::linear(attention,fixture.at(p+"self_attn.o_proj.weight")),initial,fixture.at(p+"post_attention_layernorm.weight"),1e-6);
 auto gu=at::linear(post.first,at::cat({fixture.at(p+"mlp.gate_proj.weight"),fixture.at(p+"mlp.up_proj.weight")},0));
 auto materialized=at::linear(tei::fused_gated_activation(gu,5),fixture.at(p+"mlp.down_proj.weight"))+post.second;
 materialized=materialized.index_add(0,visual_indices,at::zeros_like(visual));
 auto reference=tei::fused_add_rms_norm(materialized,{},fixture.at("model.norm.weight"),1e-6).first;
 TORCH_CHECK((withzero-reference).abs().max().item<float>()<0.0001,"Deepstack did not follow Candle residual materialization ordering");
 auto deepstack=at::arange(16,options.dtype(at::kHalf)).view({1,16})*.07;
 auto withdeep=decoder_multimodal_forward(model,input,visual_indices,visual,cosine,sine,{deepstack});
 TORCH_CHECK((withdeep-withzero).abs().max().item<float>()>0.001,"CUDA deepstack features were ignored");
 TORCH_CHECK((withdeep[0]-withzero[0]).abs().max().item<float>()<0.0001,"CUDA deepstack crossed causal token boundary");
 Heads heads;CudaGraphCache heads_cache(4,8,1);
 for(int repeat=0;repeat<3;++repeat) {
  input.ids.add_(1);
  for(int operation=0;operation<=2;++operation) {
   auto expected=operation==0?heads.forward(input):heads.predict(input,operation==2);
   auto actual=heads_cache.forward(heads,input,operation).clone();
   TORCH_CHECK(at::equal(actual,expected),"Graph head operation collision or stale input");
  }
 }
 TORCH_CHECK(heads_cache.size()==3,"Hidden, sequence and token head graph keys collided");
 input.ids.sub_(3);
 FailedWarmup failed_warmup;CudaGraphCache failed_cache(1,4096,1);
 for(int i=0;i<2;++i) {
  TORCH_CHECK(at::equal(failed_cache.forward(failed_warmup,input),input.ids.to(at::kFloat).unsqueeze(1)),
    "Failed warmup recovery changed output");
 }
 TORCH_CHECK(failed_cache.size()==0 && failed_cache.stats().attempts==1,
   "Failed warmup was retained or repeatedly captured");
 for(bool during_capture:{false,true}) {
  VendorFailure vendor_failure(during_capture);CudaGraphCache vendor_cache(1,4096,1);
  for(int repeat=0;repeat<2;++repeat)
   TORCH_CHECK(at::equal(vendor_cache.forward(vendor_failure,input),input.ids.to(at::kFloat).unsqueeze(1)),
     "Typed vendor allocation failure changed eager recovery output");
  TORCH_CHECK(vendor_cache.size()==0 && vendor_cache.stats().attempts==1,
    "Typed vendor allocation failure retained or repeatedly captured a graph");
 }
 Uncapturable unsupported;CudaGraphCache fallback(4,4096,1);
 for(int i=0;i<2;++i){auto actual=fallback.forward(unsupported,input);TORCH_CHECK(actual[0][0].item<float>()==12,"Graph fallback changed result");}
 TORCH_CHECK(fallback.size()==0,"Unsupported model unexpectedly captured");
 Media media;CudaGraphCache media_cache(4,8,1);
 int devices=0;TORCH_CHECK(cudaGetDeviceCount(&devices)==cudaSuccess,"Unable to query CUDA devices");
 input.images.push_back({at::ones({2,4},options.dtype(at::kFloat)),{1,2,2},1,0,1,0});
 input.audios.push_back({at::arange(12,options.dtype(at::kFloat)).view({3,4}),at::tensor({1,0,1},options).to(at::kBool),0,2});
 input.multimodal_positions=at::zeros({3,3},options);
 for(int run=0;run<3;++run) {
  input.images[0].feature_key[0]++;input.images[0].has_feature_key=true;
  input.images[0].pixels.add_(1);input.audios[0].features.add_(1);input.multimodal_positions.add_(1);
  if(run==1)input.audios[0].validity.copy_(at::tensor({0,1,1},options).to(at::kBool));
  at::Tensor actual;
  if(devices>1) {
   // The caller's current GPU can differ from the request's tensors. Tower
   // preparation must be guarded before its first operation, including misses.
   c10::cuda::CUDAGuard other_device(c10::Device(c10::kCUDA,1));
   actual=media_cache.forward(media,input).clone();
  } else actual=media_cache.forward(media,input).clone();
  TORCH_CHECK(at::equal(actual,media.forward(input)),"Media graph replay retained stale values or frame indices");
 }
 TORCH_CHECK(media_cache.size()==1,"Same exact media shapes did not reuse capture");
 input.images[0].token_start=1;
 TORCH_CHECK(at::equal(media_cache.forward(media,input).clone(),media.forward(input)),"Media metadata changed without recapture");
 TORCH_CHECK(media_cache.size()==2,"Media descriptors are missing from graph key");
 auto before=media_cache.size();
 TORCH_CHECK(at::equal(media_cache.forward(media,input).clone(),media.forward(input)),"Media descriptor graph replay failed");
 Dense other_model;
 TORCH_CHECK(at::equal(media_cache.forward(other_model,input).clone(),other_model.forward(input)),"Graph reused across different model instances");
 TORCH_CHECK(media_cache.size()==before+1,"Model instance is missing from graph key");
 // D3 skips prepared features: shared uploads must remain shared in capture.
 Media shared_media;shared_media.prepare_features=false;CudaGraphCache shared_cache(4,8,1);
 auto shared=input;shared.images.push_back(shared.images.front());
 auto shared_output=shared_cache.forward(shared_media,shared).clone();
 TORCH_CHECK(shared_media.observed_shared_images,"Graph cloned shared media pixels more than once");
 TORCH_CHECK(at::equal(shared_output,shared_media.forward(shared)),"Shared-media capture changed full outputs");
 shared.images[0].pixels.add_(2);
 TORCH_CHECK(at::equal(shared_cache.forward(shared_media,shared).clone(),shared_media.forward(shared)),"Shared-media replay retained stale pixels");
 auto distinct=shared;distinct.images[1].pixels=shared.images[0].pixels.clone()+3;
 auto distinct_output=shared_cache.forward(shared_media,distinct).clone();
 TORCH_CHECK(!shared_media.observed_shared_images&&shared_cache.size()==2,"Image alias pattern is missing from graph key");
 TORCH_CHECK(at::equal(distinct_output,shared_media.forward(distinct)),"Distinct media replay overwrote one shared clone");
 distinct.images[1].pixels.add_(4);
 TORCH_CHECK(at::equal(shared_cache.forward(shared_media,distinct).clone(),shared_media.forward(distinct)),"Distinct media replay retained stale pixels");
 shared.images[1].grid_thw[1]++;
 TORCH_CHECK(at::equal(shared_cache.forward(shared_media,shared).clone(),shared_media.forward(shared))&&shared_cache.size()==3,"Shared pixel identity hid distinct image layout");
 // A shared 40 MiB image fits a 64 MiB graph budget; two independent copies do not.
 auto budget_shared=input;budget_shared.audios.clear();budget_shared.multimodal_positions=at::Tensor();
 auto budget_pixels=at::ones({1024,10240},options.dtype(at::kFloat));
 budget_shared.images={{budget_pixels,{1,2,2},2,0,1,0},{budget_pixels,{1,2,2},2,1,1,0}};
 CudaGraphCache alias_budget(1,8,1,64ULL<<20);
 TORCH_CHECK(at::equal(alias_budget.forward(shared_media,budget_shared).clone(),shared_media.forward(budget_shared)),"Shared near-budget capture changed outputs");
 TORCH_CHECK(alias_budget.size()==1&&alias_budget.stats().attempts==1,"Shared image preflight charged duplicate clones");
 auto budget_distinct=budget_shared;budget_distinct.images[1].pixels=budget_pixels.clone();
 CudaGraphCache distinct_budget(1,8,1,64ULL<<20);
 TORCH_CHECK(at::equal(distinct_budget.forward(shared_media,budget_distinct).clone(),shared_media.forward(budget_distinct)),"Distinct near-budget fallback changed outputs");
 TORCH_CHECK(distinct_budget.size()==0&&distinct_budget.stats().attempts==0,"Distinct images bypassed preflight budget");
 std::cout<<"CUDA graph dynamic inputs, exact sequence keys, bounded eviction and eager fallback passed\n";
}

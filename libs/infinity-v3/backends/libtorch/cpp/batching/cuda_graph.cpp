// SPDX-License-Identifier: Apache-2.0
#include "batching/cuda_graph.h"
#include <algorithm>
#include <iterator>
#include <optional>
#include <string>
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <list>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#endif
namespace tei {
struct CudaGraphCache::Impl {
  size_t limit;
  int64_t max_tokens;
  size_t reuse, max_captures, consecutive=0;
  uint64_t max_bytes;
  std::string last_shape;
  bool budget_exhausted=false;
  Stats counters;
#ifdef TEI_TORCH_CUDA_KERNELS
  struct CaptureLane {
    cudaStream_t raw=nullptr;
    c10::DeviceIndex device;
    CaptureLane(const CaptureLane&)=delete;
    CaptureLane& operator=(const CaptureLane&)=delete;
    explicit CaptureLane(c10::DeviceIndex index):device(index) {
      c10::cuda::CUDAGuard guard(device);
      C10_CUDA_CHECK(cudaStreamCreateWithFlags(&raw,cudaStreamNonBlocking));
    }
    ~CaptureLane() {
      try {
        c10::cuda::CUDAGuard guard(device);
        // Warmup may fail before a CUDAGraph exists to clear its workspace.
        at::cuda::clearCublasWorkspacesForStream(raw);
        C10_CUDA_CHECK_WARN(cudaStreamDestroy(raw));
      } catch(...) { /* A failed device is reported by inference/completion. */ }
    }
    c10::cuda::CUDAStream stream() const {
      return c10::cuda::getStreamFromExternal(raw,device);
    }
  };
  // Destroy owned streams only after graphs and their workspaces are gone.
  std::unordered_map<c10::DeviceIndex,std::vector<std::unique_ptr<CaptureLane>>> capture_streams;
  struct Entry {
    std::vector<int32_t> offsets;
    PackedInput input;
    at::Tensor output;
    std::unique_ptr<at::cuda::CUDAGraph> graph;
    std::list<std::string>::iterator lru;
    uint64_t bytes=0;
    std::optional<c10::cuda::CUDAStream> capture_stream;
    std::optional<c10::cuda::CUDAStream> replay_stream;
  };
  std::unordered_map<std::string, std::unique_ptr<Entry>> entries;
  std::list<std::string> lru;
  std::unordered_set<std::string> failed;
  // Reuse retired lanes, but never share a lane between live graphs: graph
  // destruction clears cuBLAS workspaces associated with its capture stream.
  std::vector<c10::MempoolId_t> retired_pools;
  void reclaim_retired() {
    for(auto pool=retired_pools.begin();pool!=retired_pools.end();) {
      c10::cuda::CUDACachingAllocator::emptyCache(*pool);
      bool retained=false;
      for(const auto& segment:c10::cuda::CUDACachingAllocator::snapshot(*pool,false).segments)
        if(segment.owner_private_pool_id==*pool){retained=true;break;}
      if(retained)++pool;else pool=retired_pools.erase(pool);
    }
  }
  ~Impl() {
    // Deferred outputs outlive the model. Join caller work before graph
    // destruction releases stream-local workspaces used by pending replay.
    for(const auto& resident:entries) {
      try { if(resident.second->replay_stream)resident.second->replay_stream->synchronize(); }
      catch(...) { /* A failed device is reported by inference/completion. */ }
    }
    // A destroyed graph marks its private pool freeable, but does not return
    // its reserved segments to CUDA. Reclaim each pool after dropping its
    // graph/static tensors; held output tensors keep their storage alive.
    while(!entries.empty()) {
      const auto resident=entries.begin();
      const auto pool=resident->second->graph->pool();
      entries.erase(resident);
      try { c10::cuda::CUDACachingAllocator::emptyCache(pool); }
      catch(...) { /* Cleanup must not throw from model destruction. */ }
    }
    try { reclaim_retired(); }
    catch(...) { /* A failed device is reported by inference/completion. */ }
  }
#endif
  Impl(size_t count,int64_t tokens,size_t uses,uint64_t bytes,size_t captures)
    :limit(count),max_tokens(tokens),reuse(uses),max_captures(captures),max_bytes(bytes) {
    TORCH_CHECK(uses>0 && bytes>0,"CUDA graph reuse and byte budgets must be positive");
  }
};
CudaGraphCache::CudaGraphCache(size_t count,int64_t tokens,size_t uses,uint64_t bytes,size_t captures)
  :impl(std::make_unique<Impl>(count,tokens,uses,bytes,captures)){}
CudaGraphCache::~CudaGraphCache()=default;
CudaGraphCache::Stats CudaGraphCache::stats() const { return impl->counters; }
size_t CudaGraphCache::size() const {
#ifdef TEI_TORCH_CUDA_KERNELS
  return impl->entries.size();
#else
  return 0;
#endif
}
at::Tensor CudaGraphCache::forward(const Model& model,const PackedInput& original, int operation) {
  TORCH_CHECK(operation >= 0 && operation <= 2, "Invalid CUDA graph operation");
  auto run = [&](const PackedInput& input) {
    return operation == 0 ? model.forward(input) : model.predict(input, operation == 2);
  };
#ifdef TEI_TORCH_CUDA_KERNELS
  auto input=original;
  c10::cuda::OptionalCUDAGuard device;
  if(input.ids.is_cuda())device.set_device(input.ids.device());
  // Outputs can outlive eviction. Retry their retired pools before the next
  // request allocates, so dropping the last output also releases reservation.
  if(!impl->retired_pools.empty())impl->reclaim_retired();
  model.prepare_input(input);
  // Pool reservation accounting requires the native caching allocator.
  if(c10::cuda::CUDACachingAllocator::name()!="native")return run(input);
  if(!input.ids.is_cuda()||!impl->limit||input.ids.numel()>impl->max_tokens||input.ids.numel()==0)
    return run(input);
  // Include offsets, max length, device and scalar types: no sequence metadata
  // from one batch can leak into another graph with the same total token count.
  std::ostringstream key;
  key<<operation<<':'<<&model<<':'<<input.ids.device()<<':'<<input.ids.scalar_type()<<':'<<input.positions.scalar_type()<<':'<<input.types.scalar_type()<<':'<<input.cumulative.scalar_type()<<':'<<input.max_sequence<<':';
  for(int64_t i=0;i<=input.batch;++i)key<<input.offsets[i]<<',';
  auto tensor_key=[&](const at::Tensor& tensor) {
    key<<';'<<tensor.defined();
    if(tensor.defined())key<<':'<<tensor.device()<<':'<<tensor.scalar_type()<<':'<<tensor.sizes();
  };
  key<<";images="<<input.images.size();
  std::unordered_map<const at::TensorImpl*,size_t> image_aliases;
  for(const auto& image:input.images) {
    const auto& payload=image.prepared_features.defined()?image.prepared_features:image.pixels;
    auto [alias,inserted]=image_aliases.emplace(payload.unsafeGetTensorImpl(),image_aliases.size());
    key<<";image-alias="<<alias->second;
    tensor_key(image.prepared_features);
    if(!image.prepared_features.defined())tensor_key(image.pixels);
    for(auto dimension:image.grid_thw)key<<':'<<dimension;
    key<<':'<<image.merge_size<<':'<<image.token_start<<':'<<image.token_count<<':'<<image.sequence_start;
  }
  key<<";audios="<<input.audios.size();
  for(const auto& audio:input.audios) {
    tensor_key(audio.features);tensor_key(audio.validity);tensor_key(audio.selected_frames);
    key<<':'<<audio.token_start<<':'<<audio.token_count;
  }
  tensor_key(input.multimodal_positions);
  tensor_key(input.compact_ids);tensor_key(input.compact_positions);tensor_key(input.compact_types);
  tensor_key(input.scatter_unfold);tensor_key(input.fold_gather);
  auto id=key.str();
  if(id==impl->last_shape) {
    if(impl->consecutive<impl->reuse)++impl->consecutive;
  } else {impl->last_shape=id;impl->consecutive=1;}
  auto found=impl->entries.find(id);
  if(found!=impl->entries.end()) {
    ++impl->counters.hits;
    auto& entry=*found->second;
    const auto replay_stream=c10::cuda::getCurrentCUDAStream(input.ids.get_device());
    if(entry.replay_stream&&entry.replay_stream!=replay_stream)entry.replay_stream->synchronize();
    entry.replay_stream=replay_stream;
    impl->lru.erase(entry.lru);impl->lru.push_front(id);entry.lru=impl->lru.begin();
    entry.input.ids.copy_(input.ids);entry.input.types.copy_(input.types);
    entry.input.positions.copy_(input.positions);entry.input.cumulative.copy_(input.cumulative);
    std::unordered_set<const at::TensorImpl*> updated_images;
    for(size_t i=0;i<input.images.size();++i) {
      const auto& payload=input.images[i].prepared_features.defined()?input.images[i].prepared_features:input.images[i].pixels;
      if(!updated_images.insert(payload.unsafeGetTensorImpl()).second)continue;
      if(input.images[i].prepared_features.defined())
        entry.input.images[i].prepared_features.copy_(input.images[i].prepared_features);
      else entry.input.images[i].pixels.copy_(input.images[i].pixels);
    }
    for(size_t i=0;i<input.audios.size();++i) {
      entry.input.audios[i].features.copy_(input.audios[i].features);
      entry.input.audios[i].validity.copy_(input.audios[i].validity);
      if(input.audios[i].selected_frames.defined())entry.input.audios[i].selected_frames.copy_(input.audios[i].selected_frames);
    }
    if(input.multimodal_positions.defined())entry.input.multimodal_positions.copy_(input.multimodal_positions);
    if(input.compact_ids.defined()) {
      entry.input.compact_ids.copy_(input.compact_ids);entry.input.compact_positions.copy_(input.compact_positions);
      entry.input.compact_types.copy_(input.compact_types);entry.input.scatter_unfold.copy_(input.scatter_unfold);
      entry.input.fold_gather.copy_(input.fold_gather);
    }
    entry.graph->replay();
    return entry.output;
  }
  ++impl->counters.misses;
  // Bound private-pool creation, including failed captures. Alternating cold
  // shapes never pass consecutive-use admission.
  if(impl->budget_exhausted || impl->counters.attempts>=impl->max_captures
    || impl->consecutive<impl->reuse || impl->failed.count(id))return run(input);
  // Reject known oversized static inputs before cloning or creating a private
  // pool. The post-capture check still accounts for activation/workspace pools.
  uint64_t remaining=impl->max_bytes-impl->counters.admitted_bytes;
  auto fits=[&](const at::Tensor& tensor) {
    if(!tensor.defined() || !tensor.is_cuda())return true;
    const auto bytes=static_cast<uint64_t>(tensor.nbytes());
    if(bytes>remaining)return false;
    remaining-=bytes;return true;
  };
  bool inputs_fit=fits(input.ids)&&fits(input.types)&&fits(input.positions)&&fits(input.cumulative);
  std::unordered_set<const at::TensorImpl*> fitted_images;
  for(const auto& image:input.images) {
    const auto& payload=image.prepared_features.defined()?image.prepared_features:image.pixels;
    if(fitted_images.insert(payload.unsafeGetTensorImpl()).second)inputs_fit=inputs_fit&&fits(payload);
  }
  for(const auto& audio:input.audios)
    inputs_fit=inputs_fit&&fits(audio.features)&&fits(audio.validity)&&fits(audio.selected_frames);
  inputs_fit=inputs_fit&&fits(input.multimodal_positions)&&fits(input.compact_ids)
    &&fits(input.compact_positions)&&fits(input.compact_types)&&fits(input.scatter_unfold)&&fits(input.fold_gather);
  if(!inputs_fit)return run(input);
  ++impl->counters.attempts;
  auto entry=std::make_unique<Impl::Entry>();
  entry->offsets.assign(input.offsets,input.offsets+input.batch+1);
  entry->input={input.ids.clone(),input.types.clone(),input.positions.clone(),input.cumulative.clone(),entry->offsets.data(),input.batch,input.max_sequence};
  std::unordered_map<const at::TensorImpl*,at::Tensor> cloned_images;
  auto clone_image=[&](const at::Tensor& payload) {
    auto [entry,inserted]=cloned_images.emplace(payload.unsafeGetTensorImpl(),at::Tensor());
    if(inserted)entry->second=payload.clone();
    return entry->second;
  };
  for(auto image:input.images) {
    if(image.prepared_features.defined()) {
      image.prepared_features=clone_image(image.prepared_features);
      image.pixels=at::Tensor(); // Captured forward never reads raw host pixels.
    } else image.pixels=clone_image(image.pixels);
    entry->input.images.push_back(std::move(image));
  }
  for(auto audio:input.audios) {
    audio.features=audio.features.clone();audio.validity=audio.validity.clone();
    if(audio.selected_frames.defined())audio.selected_frames=audio.selected_frames.clone();
    entry->input.audios.push_back(std::move(audio));
  }
  if(input.multimodal_positions.defined())entry->input.multimodal_positions=input.multimodal_positions.clone();
  if(input.compact_ids.defined()) {
    entry->input.compact_ids=input.compact_ids.clone();entry->input.compact_positions=input.compact_positions.clone();
    entry->input.compact_types=input.compact_types.clone();entry->input.scatter_unfold=input.scatter_unfold.clone();
    entry->input.fold_gather=input.fold_gather.clone();
  }
  auto current=c10::cuda::getCurrentCUDAStream(input.ids.device().index());
  const auto capture_device=input.ids.device().index();
  auto& lanes=impl->capture_streams[capture_device];
  auto available=[&](c10::cuda::CUDAStream lane) {
    for(const auto& resident:impl->entries)
      if(resident.second->capture_stream==lane)return false;
    return true;
  };
  auto lane=std::find_if(lanes.begin(),lanes.end(),[&](const auto& owned){return available(owned->stream());});
  if(lane==lanes.end()) {
    // The shared Torch pool wraps after 32 streams and can collide with another
    // resident model's graphs. Own these bounded lanes instead.
    lanes.push_back(std::make_unique<Impl::CaptureLane>(capture_device));lane=std::prev(lanes.end());
  }
  const auto capture_stream=(*lane)->stream();
  entry->capture_stream=capture_stream;
  at::cuda::CUDAEvent input_ready;input_ready.record(current);input_ready.block(capture_stream);
  bool capturing=false;
  c10::MempoolId_t trial_pool{0,0};
  try {
    {
      c10::cuda::CUDAStreamGuard stream(capture_stream);
      // Initialize GEMM and attention handles before entering capture.
      for(int i=0;i<3;++i)entry->output=run(entry->input);
      entry->graph=std::make_unique<at::cuda::CUDAGraph>();
      // Keep the ID even if capture_end fails, when graph->pool() is invalid.
      trial_pool=at::cuda::graph_pool_handle();
      entry->graph->capture_begin(trial_pool,cudaStreamCaptureModeThreadLocal);capturing=true;
      entry->output=run(entry->input);
      entry->graph->capture_end();capturing=false;
    }
    at::cuda::CUDAEvent warmed;warmed.record(capture_stream);warmed.block(current);
    entry->graph->replay();
    entry->replay_stream=current;
  } catch(const std::exception&) {
    // Unsupported operators fall back to eager inference. Capture happens on a
    // dedicated stream so an invalidated capture never touches the caller stream.
    if(capturing) {
      try {c10::cuda::CUDAStreamGuard stream(capture_stream);entry->graph->capture_end();}catch(...){}
    }
    // Warmup can enqueue work before throwing. These clones were allocated
    // on the caller stream and used on the capture stream; join before freeing
    // them or allowing eager fallback to reuse their storage.
    capture_stream.synchronize();
    entry.reset();
    cudaGetLastError();
    if(trial_pool.first||trial_pool.second) {
      impl->retired_pools.push_back(trial_pool);impl->reclaim_retired();
    }
    // The lifetime attempt budget also bounds this negative cache.
    impl->failed.insert(id);
    return run(input);
  }
  // Charge reserved private-pool segments and static inputs separately.
  // Cumulative admission charges survive eviction: reservation can outlive graphs.
  for(const auto& segment:c10::cuda::CUDACachingAllocator::snapshot(entry->graph->pool(),false).segments)
    if(segment.device==input.ids.get_device() && segment.owner_private_pool_id==entry->graph->pool())entry->bytes+=segment.total_size;
  std::unordered_set<const c10::StorageImpl*> charged_inputs;
  auto charge=[&](const at::Tensor& t){if(t.defined() && t.is_cuda() && charged_inputs.insert(t.storage().unsafeGetStorageImpl()).second)entry->bytes+=t.storage().nbytes();};
  charge(entry->input.ids);charge(entry->input.types);charge(entry->input.positions);charge(entry->input.cumulative);
  for(const auto& image:entry->input.images){charge(image.prepared_features);charge(image.pixels);}
  for(const auto& audio:entry->input.audios){charge(audio.features);charge(audio.validity);charge(audio.selected_frames);}
  charge(entry->input.multimodal_positions);charge(entry->input.compact_ids);charge(entry->input.compact_positions);
  charge(entry->input.compact_types);charge(entry->input.scatter_unfold);charge(entry->input.fold_gather);
  if(entry->bytes>impl->max_bytes-impl->counters.admitted_bytes) {
    // Allow only one oversized trial; drain replay before pool destruction.
    current.synchronize();impl->budget_exhausted=true;entry.reset();
    impl->retired_pools.push_back(trial_pool);impl->reclaim_retired();return run(input);
  }
  if(impl->entries.size()>=impl->limit) {
    // Output copies/leases are ordered on the caller stream. Drain its work
    // before destroying an entry and its private pool.
    current.synchronize();
    impl->counters.live_bytes-=impl->entries.at(impl->lru.back())->bytes;
    const auto retired_pool=impl->entries.at(impl->lru.back())->graph->pool();
    impl->entries.erase(impl->lru.back());impl->lru.pop_back();++impl->counters.evictions;
    // Reclaim only this retired private pool, preserving eager caches and live
    // graph pools. Live output tensors continue to own their storage.
    TORCH_CHECK(retired_pool.first||retired_pool.second,"Missing retired graph pool");
    impl->retired_pools.push_back(retired_pool);
    impl->reclaim_retired();
  }
  ++impl->counters.captures;impl->counters.live_bytes+=entry->bytes;
  impl->counters.admitted_bytes+=entry->bytes;
  auto result=entry->output;
  impl->lru.push_front(id);entry->lru=impl->lru.begin();
  impl->entries.emplace(std::move(id),std::move(entry));
  return result;
#else
  auto input=original;model.prepare_input(input);return run(input);
#endif
}
}

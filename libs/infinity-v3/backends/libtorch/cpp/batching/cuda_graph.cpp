// SPDX-License-Identifier: Apache-2.0
#include "batching/cuda_graph.h"
#include <algorithm>
#include <iterator>
#include <optional>
#include "batching/prefix_cache.h"
#include "batching/tensor_copy.h"
#include <string>
#ifdef TEI_TORCH_CUDA_KERNELS
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/ops/_foreach_copy.h>
#include <list>
#include <type_traits>
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
  bool budget_exhausted=false, retain_prefix=false;
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
    std::vector<int32_t> original_offsets;
    PackedInput original;
    std::unique_ptr<PrefixContext> prefix;
    std::unique_ptr<PrefixContext> capture;
    // Resident snapshots are immutable. Remember the last copied owner without
    // extending its cache lifetime; address equality after weak locking cannot
    // confuse a retired owner with a new allocation at the same address.
    std::vector<std::weak_ptr<PrefixEntry>> parents;
    at::Tensor output;
    std::unique_ptr<at::cuda::CUDAGraph> graph;
    std::list<std::string>::iterator lru;
    uint64_t bytes=0;
    std::optional<c10::cuda::CUDAStream> capture_stream;
    std::optional<c10::cuda::CUDAStream> replay_stream;
    at::cuda::CUDAEvent consumed;
    bool recorded=false;
    ~Entry(){if((prefix||capture)&&recorded)try{consumed.synchronize();}catch(...) {}}
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
#ifdef TEI_TORCH_CUDA_KERNELS
// Keys live only in this process. Lengths delimit variable-sized fields, and
// each scalar keeps its native type so metadata needs no stream formatting.
struct GraphKey {
  std::string bytes;
  template<class T> void add(T value) {
    static_assert(std::is_trivially_copyable_v<T>);
    bytes.append(reinterpret_cast<const char*>(&value),sizeof(value));
  }
  void device(c10::Device value) {add(value.type());add(value.index());}
  void tensor(const at::Tensor& value,int64_t rows=-1) {
    add(value.defined());if(!value.defined())return;
    device(value.device());add(value.scalar_type());add(value.dim());
    TORCH_CHECK(rows<0||(value.dim()>0&&rows<=value.size(0)),"Invalid graph prefix span");
    for(int64_t dim=0;dim<value.dim();++dim)add(!dim&&rows>=0?rows:value.size(dim));
  }
};
static PackedInput clone_graph_input(const PackedInput& input,std::vector<int32_t>& offsets) {
  offsets.assign(input.offsets,input.offsets+input.batch+1);
  PackedInput owned{input.ids.clone(),input.types.clone(),input.positions.clone(),input.cumulative.clone(),
    offsets.data(),input.batch,input.max_sequence};
  std::unordered_map<const at::TensorImpl*,at::Tensor> cloned_images;
  auto clone_image=[&](const at::Tensor& payload) {
    auto [entry,inserted]=cloned_images.emplace(payload.unsafeGetTensorImpl(),at::Tensor());
    if(inserted)entry->second=payload.clone();
    return entry->second;
  };
  for(auto image:input.images) {
    if(image.prepared_features.defined()){image.prepared_features=clone_image(image.prepared_features);image.pixels=at::Tensor();}
    else image.pixels=clone_image(image.pixels);
    owned.images.push_back(std::move(image));
  }
  for(auto audio:input.audios) {
    audio.features=audio.features.clone();audio.validity=audio.validity.clone();
    if(audio.selected_frames.defined())audio.selected_frames=audio.selected_frames.clone();
    owned.audios.push_back(std::move(audio));
  }
  if(input.multimodal_positions.defined())owned.multimodal_positions=input.multimodal_positions.clone();
  if(input.compact_ids.defined()) {
    owned.compact_ids=input.compact_ids.clone();owned.compact_positions=input.compact_positions.clone();owned.compact_types=input.compact_types.clone();
    owned.scatter_unfold=input.scatter_unfold.clone();owned.fold_gather=input.fold_gather.clone();
  }
  return owned;
}
static void graph_input_copies(const PackedInput& target,const PackedInput& source,
  std::vector<at::Tensor>& targets,std::vector<at::Tensor>& sources) {
  auto copy=[&](const at::Tensor& a,const at::Tensor& b){if(a.defined()){targets.push_back(a);sources.push_back(b);}};
  copy(target.ids,source.ids);copy(target.types,source.types);copy(target.positions,source.positions);copy(target.cumulative,source.cumulative);
  std::unordered_set<const at::TensorImpl*> updated_images;
  for(size_t i=0;i<source.images.size();++i) {
    const auto& payload=source.images[i].prepared_features.defined()?source.images[i].prepared_features:source.images[i].pixels;
    if(!updated_images.insert(payload.unsafeGetTensorImpl()).second)continue;
    if(target.images[i].prepared_features.defined())copy(target.images[i].prepared_features,source.images[i].prepared_features);
    else copy(target.images[i].pixels,source.images[i].pixels);
  }
  for(size_t i=0;i<source.audios.size();++i) {
    copy(target.audios[i].features,source.audios[i].features);copy(target.audios[i].validity,source.audios[i].validity);
    copy(target.audios[i].selected_frames,source.audios[i].selected_frames);
  }
  copy(target.multimodal_positions,source.multimodal_positions);
  copy(target.compact_ids,source.compact_ids);copy(target.compact_positions,source.compact_positions);copy(target.compact_types,source.compact_types);
  copy(target.scatter_unfold,source.scatter_unfold);copy(target.fold_gather,source.fold_gather);
}
#endif
CudaGraphCache::CudaGraphCache(size_t count,int64_t tokens,size_t uses,uint64_t bytes,size_t captures)
  :impl(std::make_unique<Impl>(count,tokens,uses,bytes,captures)){}
CudaGraphCache::~CudaGraphCache()=default;
CudaGraphCache::Stats CudaGraphCache::stats() const { return impl->counters; }
void CudaGraphCache::retain_prefix_states(bool enabled) {impl->retain_prefix=enabled;}
size_t CudaGraphCache::size() const {
#ifdef TEI_TORCH_CUDA_KERNELS
  return impl->entries.size();
#else
  return 0;
#endif
}
at::Tensor CudaGraphCache::forward(const Model& model,const PackedInput& original, int operation) {
  return forward(model,original,operation,nullptr);
}
at::Tensor CudaGraphCache::forward(const Model& model,const PackedInput& original,int operation,const MathPolicy* known_policy,PrefixContext* publish) {
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
  auto fallback=[&]{auto eager=input;if(publish)eager.prefix=publish;return run(eager);};
  // Pool reservation accounting requires the native caching allocator.
  if(c10::cuda::CUDACachingAllocator::name()!="native")return fallback();
  if(!input.ids.is_cuda()||!impl->limit||input.ids.numel()>impl->max_tokens||input.ids.numel()==0)
    return fallback();
  if(input.prefix&&model.prefix_checkpoint_only())return fallback();
  const bool retain=impl->retain_prefix&&!input.prefix&&model.supports_prefix_cache(operation)
    &&!model.prefix_checkpoint_only()&&input.audios.empty();
  // Include offsets, max length, device and scalar types: no sequence metadata
  // from one batch can leak into another graph with the same total token count.
  GraphKey key;
  const auto policy=known_policy?*known_policy:input.prefix?input.prefix->math_policy:execution_math_policy(input.ids.device());
  key.bytes.append(reinterpret_cast<const char*>(policy.data()),sizeof(policy));
  key.add(operation);key.add(model.prefix_namespace());key.device(input.ids.device());
  key.add(input.ids.scalar_type());key.add(input.positions.scalar_type());
  key.add(input.types.scalar_type());key.add(input.cumulative.scalar_type());key.add(input.max_sequence);
  key.add(input.batch);for(int64_t i=0;i<=input.batch;++i)key.add(input.offsets[i]);
  auto describe_extras=[&](const PackedInput& value) {
  key.add(value.images.size());
  std::unordered_map<const at::TensorImpl*,size_t> image_aliases;
  for(const auto& image:value.images) {
    const auto& payload=image.prepared_features.defined()?image.prepared_features:image.pixels;
    auto [alias,inserted]=image_aliases.emplace(payload.unsafeGetTensorImpl(),image_aliases.size());
    key.add(alias->second);
    key.tensor(image.prepared_features);
    if(!image.prepared_features.defined())key.tensor(image.pixels);
    for(auto dimension:image.grid_thw)key.add(dimension);
    key.add(image.merge_size);key.add(image.token_start);key.add(image.token_count);key.add(image.sequence_start);
  }
  key.add(value.audios.size());
  for(const auto& audio:value.audios) {
    key.tensor(audio.features);key.tensor(audio.validity);key.tensor(audio.selected_frames);
    key.add(audio.token_start);key.add(audio.token_count);
  }
  key.tensor(value.multimodal_positions);
  key.tensor(value.compact_ids);key.tensor(value.compact_positions);key.tensor(value.compact_types);
  key.tensor(value.scatter_unfold);key.tensor(value.fold_gather);
  };
  describe_extras(input);
  key.add(bool(input.prefix));key.add(retain);
  if(input.prefix) {
    const auto& prefix=*input.prefix;
    key.add(prefix.original.max_sequence);key.add(prefix.original.batch);
    for(int64_t row=0;row<=prefix.original.batch;++row)key.add(prefix.original.offsets[row]);
    key.add(prefix.hits.size());for(auto hit:prefix.hits)key.add(hit);
    key.tensor(prefix.original.ids);key.tensor(prefix.original.multimodal_positions);
    key.tensor(prefix.original.compact_ids);key.tensor(prefix.original.scatter_unfold);
    describe_extras(prefix.original);
    key.add(prefix.readers.size());
    for(size_t row=0;row<prefix.readers.size();++row)if(prefix.hits[row]) {
      key.tensor(prefix.readers[row]->output,prefix.hits[row]);
      key.add(prefix.readers[row]->layers.size());
      for(const auto& [layer,state]:prefix.readers[row]->layers) {
        key.add(layer);key.tensor(state.key,prefix.hits[row]);key.tensor(state.value,prefix.hits[row]);
      }
    }
  }
  auto id=std::move(key.bytes);
  if(id==impl->last_shape) {
    if(impl->consecutive<impl->reuse)++impl->consecutive;
  } else {impl->last_shape=id;impl->consecutive=1;}
  auto found=impl->entries.find(id);
  if(found!=impl->entries.end()) {
    auto& entry=*found->second;
    const auto replay_stream=c10::cuda::getCurrentCUDAStream(input.ids.get_device());
    // Order shared graph buffers on the device, without making a deferred
    // caller wait on the host for the preceding stream's submission.
    if(entry.recorded&&entry.replay_stream!=replay_stream)entry.consumed.block(replay_stream);
    entry.replay_stream=replay_stream;
    if(publish&&!entry.capture)return fallback();
    ++impl->counters.hits;
    try {
    impl->lru.erase(entry.lru);impl->lru.push_front(id);entry.lru=impl->lru.begin();
    std::vector<at::Tensor> targets,sources;graph_input_copies(entry.input,input,targets,sources);
    if(entry.prefix) {
      graph_input_copies(entry.original,input.prefix->original,targets,sources);
      for(size_t row=0;row<entry.prefix->readers.size();++row)if(entry.prefix->hits[row]) {
        const auto& actual=input.prefix->readers[row];const auto& owned=entry.prefix->readers[row];
        if(entry.parents[row].lock()==actual)continue;
        entry.parents[row].reset();
        targets.push_back(owned->output);sources.push_back(actual->output);
        for(const auto& [layer,state]:owned->layers) {
          const auto& fresh=actual->layers.at(layer);
          targets.push_back(state.key);sources.push_back(fresh.key);
          targets.push_back(state.value);sources.push_back(fresh.value);
        }
      }
    }
    tensor_copies(targets,sources,bool(entry.prefix));
    if(entry.prefix)for(size_t row=0;row<entry.parents.size();++row)
      entry.parents[row]=input.prefix->readers[row];
    entry.graph->replay();
    if(entry.prefix) {
      ++impl->counters.prefix_hits;input.prefix->publish_graph(*entry.prefix);
      auto output=entry.output.clone();entry.consumed.record(c10::cuda::getCurrentCUDAStream(input.ids.get_device()));entry.recorded=true;return output;
    }
    if(entry.capture) {
      auto output=entry.output;
      if(publish){publish->publish_graph(*entry.capture);output=output.clone();}
      entry.consumed.record(c10::cuda::getCurrentCUDAStream(input.ids.get_device()));entry.recorded=true;
      return output;
    }
    entry.consumed.record(replay_stream);entry.recorded=true;
    return entry.output;
    }catch(...) {
      c10::cuda::getCurrentCUDAStream(input.ids.get_device()).synchronize();
      throw;
    }
  }
  ++impl->counters.misses;
  // Admission consumes an already warmed full shader. A missing observer uses
  // the ordinary eager capture path rather than adding graph capture latency.
  if(publish)return fallback();
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
  if(input.prefix) {
    // Reject a known oversized prefix arena before cloning inputs or starting
    // capture. Actual private-pool reservation accounting still follows below.
    auto minimum=model.prefix_payload_bytes(input.prefix->original.ids.numel());
    for(size_t row=0;row<input.prefix->readers.size();++row)if(input.prefix->hits[row]) {
      const auto& reader=input.prefix->readers[row];const auto hits=input.prefix->hits[row];
      minimum=prefix_storage_sum(minimum,reader->output.narrow(0,0,hits).nbytes());
      for(const auto& [layer,state]:reader->layers) {
        minimum=prefix_storage_sum(minimum,state.key.narrow(0,0,hits).nbytes());
        minimum=prefix_storage_sum(minimum,state.value.narrow(0,0,hits).nbytes());
      }
    }
    if(minimum>impl->max_bytes-impl->counters.admitted_bytes)return run(input);
  }
  ++impl->counters.attempts;
  auto entry=std::make_unique<Impl::Entry>();
  entry->input=clone_graph_input(input,entry->offsets);
  if(retain&&model.prefix_payload_bytes(input.ids.numel())<=impl->max_bytes-impl->counters.admitted_bytes) {
    entry->capture=std::make_unique<PrefixContext>(PrefixContext{entry->input,entry->offsets,
      std::vector<int64_t>(input.batch),std::vector<std::shared_ptr<PrefixEntry>>(input.batch),
      std::vector<std::shared_ptr<PrefixEntry>>(input.batch),0,std::make_shared<PrefixBudget>()});
    entry->capture->graph_layers.resize(input.batch);entry->capture->graph_outputs.resize(input.batch);
    entry->input.capture_prefix=entry->capture.get();
  }
  if(input.prefix) {
    const auto& original=*input.prefix;
    entry->original=clone_graph_input(original.original,entry->original_offsets);
    entry->prefix=std::make_unique<PrefixContext>(PrefixContext{entry->original,original.offsets,original.hits,{},
      std::vector<std::shared_ptr<PrefixEntry>>(original.original.batch),original.max_bytes,std::make_shared<PrefixBudget>()});
    entry->prefix->linear=original.linear;entry->prefix->math_policy=original.math_policy;entry->prefix->collect_graph=true;
    entry->prefix->graph_layers.resize(original.original.batch);entry->prefix->graph_outputs.resize(original.original.batch);
    for(size_t row=0;row<original.readers.size();++row) {
      std::shared_ptr<PrefixEntry> owned;
      if(original.hits[row]) {
        owned=std::make_shared<PrefixEntry>();const auto& source=original.readers[row];
        owned->output=source->output.narrow(0,0,original.hits[row]).clone();
        for(const auto& [layer,state]:source->layers)owned->layers[layer]=PrefixLayer{
          state.key.narrow(0,0,original.hits[row]).clone(),state.value.narrow(0,0,original.hits[row]).clone(),{}, {}};
      }
      entry->prefix->readers.push_back(std::move(owned));
      entry->parents.push_back(original.readers[row]);
    }
    entry->input.prefix=entry->prefix.get();
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
      auto reset_capture=[&]{if(entry->capture) {
        for(auto& row:entry->capture->graph_layers)row.clear();
        for(auto& output:entry->capture->graph_outputs)output=at::Tensor();
      }};
      for(int i=0;i<3;++i){reset_capture();entry->output=run(entry->input);}
      reset_capture();
      if(entry->prefix) {
        // Warmups allocate full K/V arenas. Bind verified parent views directly
        // into them so replay writes only suffix rows, without concatenation.
        for(size_t row=0;row<entry->prefix->readers.size();++row)if(entry->prefix->hits[row]) {
          auto& reader=entry->prefix->readers[row];
          if(entry->prefix->graph_hidden.defined()) {
            c10::cuda::CUDACachingAllocator::recordStream(reader->output.storage().data_ptr(),capture_stream);
            reader->output=entry->prefix->graph_hidden.narrow(0,entry->original.offsets[row],entry->prefix->hits[row]);
          }
          for(auto& [layer,state]:reader->layers) {
            c10::cuda::CUDACachingAllocator::recordStream(state.key.storage().data_ptr(),capture_stream);
            c10::cuda::CUDACachingAllocator::recordStream(state.value.storage().data_ptr(),capture_stream);
            const auto& full=entry->prefix->graph_layers[row].at(layer);
            state.key=full.key.narrow(0,0,entry->prefix->hits[row]);
            state.value=full.value.narrow(0,0,entry->prefix->hits[row]);
          }
        }
        for(auto& output:entry->prefix->graph_outputs)output=at::Tensor();
      }
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
    entry->consumed.record(current);entry->recorded=true;
  } catch(const std::exception&) {
    // Unsupported operators fall back to eager inference. Capture happens on a
    // dedicated stream so an invalidated capture never touches the caller stream.
    if(capturing) {
      try {c10::cuda::CUDAStreamGuard stream(capture_stream);entry->graph->capture_end();}catch(...){}
    }
    // Eager warmups may still use entry storage on the dedicated stream even
    // when capture fails. Complete them before releasing that storage/pool.
    capture_stream.synchronize();
    current.synchronize();
    entry.reset();
    cudaGetLastError();
    if(trial_pool.first||trial_pool.second) {
      impl->retired_pools.push_back(trial_pool);impl->reclaim_retired();
    }
    // The lifetime attempt budget also bounds this negative cache.
    impl->failed.insert(id);
    return run(input);
  }
  if(entry->capture) {
    bool complete=true;
    for(size_t row=0;row<entry->capture->graph_layers.size();++row)
      complete=complete&&int64_t(entry->capture->graph_layers[row].size())==model.prefix_layers()
        &&entry->capture->graph_outputs[row].defined();
    if(!complete){current.synchronize();entry->input.capture_prefix=nullptr;entry->capture.reset();}
  }
  // Charge reserved private-pool segments and static inputs separately.
  // Cumulative admission charges survive eviction: reservation can outlive graphs.
  const auto allocation=c10::cuda::CUDACachingAllocator::snapshot(entry->graph->pool(),false);
  for(const auto& segment:allocation.segments)
    if(segment.device==input.ids.get_device() && segment.owner_private_pool_id==entry->graph->pool())entry->bytes+=segment.total_size;
  std::unordered_set<const void*> external;
  auto charge=[&](const at::Tensor& t){if(t.defined() && t.is_cuda() && external.insert(t.storage().data_ptr().get()).second)entry->bytes+=t.storage().nbytes();};
  charge(entry->input.ids);charge(entry->input.types);charge(entry->input.positions);charge(entry->input.cumulative);
  for(const auto& image:entry->input.images){charge(image.prepared_features);charge(image.pixels);}
  for(const auto& audio:entry->input.audios){charge(audio.features);charge(audio.validity);charge(audio.selected_frames);}
  charge(entry->input.multimodal_positions);charge(entry->input.compact_ids);charge(entry->input.compact_positions);
  charge(entry->input.compact_types);charge(entry->input.scatter_unfold);charge(entry->input.fold_gather);
  if(entry->prefix) {
    charge(entry->original.ids);charge(entry->original.types);charge(entry->original.positions);charge(entry->original.cumulative);
    charge(entry->original.multimodal_positions);charge(entry->original.compact_ids);charge(entry->original.compact_positions);
    charge(entry->original.compact_types);charge(entry->original.scatter_unfold);charge(entry->original.fold_gather);
    for(const auto& image:entry->original.images){charge(image.prepared_features);charge(image.pixels);}
    for(const auto& audio:entry->original.audios){charge(audio.features);charge(audio.validity);charge(audio.selected_frames);}
    if(entry->prefix->graph_hidden.defined())charge(entry->prefix->graph_hidden);
    else for(const auto& reader:entry->prefix->readers)if(reader)charge(reader->output);
    charge(entry->prefix->graph_input_rows);
    for(const auto& row:entry->prefix->graph_layers)
      for(const auto& [layer,state]:row){charge(state.key);charge(state.value);}
  }
  if(entry->capture) {
    // Retained views normally belong to captured private-pool segments, already
    // charged above. Charge any external backing storage once, including the
    // whole allocation when a view retains more than its logical slice.
    auto retained=[&](const at::Tensor& t) {
      if(!t.defined()||!t.is_cuda())return;
      const auto pointer=t.storage().data_ptr().get();const auto address=reinterpret_cast<uintptr_t>(pointer);
      for(const auto& segment:allocation.segments)if(segment.device==input.ids.get_device()
        &&segment.owner_private_pool_id==entry->graph->pool()&&address>=segment.address
        &&address-segment.address<segment.total_size)return;
      if(external.insert(pointer).second)entry->bytes+=t.storage().nbytes();
    };
    for(const auto& row:entry->capture->graph_layers)for(const auto& [layer,state]:row){retained(state.key);retained(state.value);}
    for(const auto& output:entry->capture->graph_outputs)retained(output);
  }
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
  auto result=entry->output;
  try {
    if(entry->prefix) {
      input.prefix->publish_graph(*entry->prefix);result=result.clone();
      entry->consumed.record(current);entry->recorded=true;
    }
  }catch(...) {
    // Publication may have queued some copies before a later launch/allocation
    // fails. Drain them before releasing their graph-backed sources.
    current.synchronize();throw;
  }
  ++impl->counters.captures;impl->counters.live_bytes+=entry->bytes;
  impl->counters.admitted_bytes+=entry->bytes;
  if(entry->prefix)++impl->counters.prefix_captures;
  impl->lru.push_front(id);entry->lru=impl->lru.begin();
  impl->entries.emplace(std::move(id),std::move(entry));
  return result;
#else
  auto input=original;model.prepare_input(input);if(publish)input.prefix=publish;return run(input);
#endif
}
}

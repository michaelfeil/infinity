// SPDX-License-Identifier: Apache-2.0
#include "batching/output.h"
#include "bridge.h"
#include "batching/cuda_graph.h"
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAAllocatorConfig.h>
#include <c10/core/InferenceMode.h>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <utility>

// Exercise quota-aware request admission without reading the large payload,
// classify an actual startup CUDA OOM, and check bounded pinned staging.
void native_allocator_error_test(const at::Device& device,int pressure=0) {
  const bool bounded_staging=pressure==1;
  TeiConfig cfg{16,2,2,24,32,32,1,1e-6,0};
  std::unique_ptr<void,decltype(&tei_destroy)> engine(tei_create(&cfg,device.str().c_str(),1),tei_destroy);
  TORCH_CHECK(engine,tei_error());
  for(const auto& [key,value]:std::initializer_list<std::pair<const char*,const char*>>{
      {"model_type","qwen3"},{"hidden_size","16"},{"num_attention_heads","2"},
      {"num_key_value_heads","1"},{"head_dim","8"},{"intermediate_size","24"},
      {"num_hidden_layers","2"},{"vocab_size","32"},{"max_position_embeddings","32"},
      {"hidden_act","silu"},{"rms_norm_eps","0.000001"}})
    TORCH_CHECK(tei_option(engine.get(),key,value)==0,tei_error());
  auto weight=[&](const std::string& key,std::initializer_list<int64_t> shape,bool norm=false) {
    int64_t count=1;for(auto n:shape)count*=n;
    float offset=0;for(auto c:key)offset+=c;
    auto value=(at::sin(at::arange(count,at::TensorOptions().dtype(at::kFloat))*.13+offset*.01)*.15).view(at::IntArrayRef(shape));
    if(norm)value+=1;
    TORCH_CHECK(tei_weight(engine.get(),key.c_str(),reinterpret_cast<const uint8_t*>(value.const_data_ptr()),
      value.sizes().data(),value.dim(),0)==0,tei_error());
  };
  weight("model.embed_tokens.weight",{32,16});weight("model.norm.weight",{16},true);
  for(int layer=0;layer<2;++layer) {
    auto root="model.layers."+std::to_string(layer)+".";
    weight(root+"input_layernorm.weight",{16},true);weight(root+"post_attention_layernorm.weight",{16},true);
    weight(root+"self_attn.q_proj.weight",{16,16});weight(root+"self_attn.k_proj.weight",{8,16});
    weight(root+"self_attn.v_proj.weight",{8,16});weight(root+"self_attn.o_proj.weight",{16,16});
    weight(root+"self_attn.q_norm.weight",{8},true);weight(root+"self_attn.k_norm.weight",{8},true);
    weight(root+"mlp.gate_proj.weight",{24,16});weight(root+"mlp.up_proj.weight",{24,16});weight(root+"mlp.down_proj.weight",{16,24});
  }
  if(bounded_staging)TORCH_CHECK(tei_option(engine.get(),"_output_staging_max_bytes","256")==0,tei_error());
  TORCH_CHECK(tei_ready(engine.get())==0,tei_error());
  int64_t ids[]={3,4,5,6},types[4]{},positions[]={0,1,2,3},row=0;int32_t offsets[]={0,4};
  auto valid=[&] {
    std::unique_ptr<void,decltype(&tei_output_destroy)> output(tei_forward_submit(engine.get(),ids,types,positions,
      offsets,1,4,0,&row,1,nullptr,0,16,nullptr,nullptr,0,nullptr,0,nullptr),tei_output_destroy);
    TORCH_CHECK(output,tei_error());const auto* values=tei_output_wait(output.get());TORCH_CHECK(values,tei_error());
    for(int i=0;i<16;++i)TORCH_CHECK(std::isfinite(values[i]),"Allocator failure corrupted valid response");
    TORCH_CHECK(tei_last_error_kind()==0,"Valid result retained allocator error kind");
  };
  valid();
  if(pressure==2) {
    std::unique_ptr<void,decltype(&tei_output_destroy)> held(tei_forward_submit(engine.get(),ids,types,positions,
      offsets,1,4,0,&row,1,nullptr,0,16,nullptr,nullptr,0,nullptr,0,nullptr),tei_output_destroy);
    TORCH_CHECK(held,tei_error());const auto* before=tei_output_wait(held.get());TORCH_CHECK(before,tei_error());
    std::array<float,16> saved;std::copy_n(before,16,saved.begin());
    auto state=tei::pinned_admission_state();const auto previous_limit=state->max_bytes;
    // Retained native storage fills the cap even after reclaiming Torch cache.
    state->max_bytes=state->bytes;
    // Exact RAW capacity exceeds one platform host page, without padding.
    const auto sequences=tei::pinned_host_page_size()/(4*16*sizeof(float))+1;
    TORCH_CHECK(sequences <= static_cast<size_t>(std::numeric_limits<int32_t>::max()/4), "Host page geometry is unrepresentable");
    std::vector<int64_t> packed_ids(4*sequences),packed_types(4*sequences),packed_positions(4*sequences);
    std::vector<int32_t> packed_offsets(sequences+1);std::vector<int64_t> raw_rows(sequences);
    for(size_t sequence=0;sequence<sequences;++sequence){raw_rows[sequence]=sequence;packed_offsets[sequence]=4*sequence;
      for(int token=0;token<4;++token){packed_ids[4*sequence+token]=ids[token];packed_positions[4*sequence+token]=token;}}
    packed_offsets.back()=4*sequences;
    try {
      std::unique_ptr<void,decltype(&tei_output_destroy)> rejected(tei_forward_submit(engine.get(),packed_ids.data(),packed_types.data(),packed_positions.data(),
        packed_offsets.data(),sequences,4,0,nullptr,0,raw_rows.data(),raw_rows.size(),sequences*4*16,nullptr,nullptr,0,nullptr,0,nullptr),tei_output_destroy);
      TORCH_CHECK(!rejected&&tei_last_error_kind()==2&&std::string(tei_error()).find("Process native pinned staging")!=std::string::npos,
        "Process pinned saturation lost its recoverable admission classification");
      const auto* after=tei_output_wait(held.get());TORCH_CHECK(after,tei_error());
      for(size_t i=0;i<saved.size();++i)TORCH_CHECK(after[i]==saved[i],"Pinned rejection corrupted a pending result");
      held.reset();valid();
    } catch(...) {state->max_bytes=previous_limit;throw;}
    state->max_bytes=previous_limit;return;
  }
  if(bounded_staging) {
    std::unique_ptr<void,decltype(&tei_output_destroy)> held(tei_forward_submit(engine.get(),ids,types,positions,
      offsets,1,4,0,&row,1,nullptr,0,16,nullptr,nullptr,0,nullptr,0,nullptr),tei_output_destroy);
    TORCH_CHECK(held,tei_error());const auto* before=tei_output_wait(held.get());TORCH_CHECK(before,tei_error());
    std::array<float,16> saved;std::copy_n(before,16,saved.begin());
    // A retains128 output+112 input bytes. B's empty result needs8 bytes;
    // its first32-byte input pin cannot fit until the held result is released.
    auto rejected=tei_forward_submit(engine.get(),ids,types,positions,offsets,1,4,0,nullptr,0,
      nullptr,0,0,nullptr,nullptr,0,nullptr,0,nullptr);
    TORCH_CHECK(!rejected&&tei_last_error_kind()==2&&std::string(tei_error()).find("pending results")!=std::string::npos,
      "Nested input staging saturation lost its admission classification or diagnostic");
    const auto* after=tei_output_wait(held.get());TORCH_CHECK(after,tei_error());
    for(size_t i=0;i<saved.size();++i)TORCH_CHECK(after[i]==saved[i],"Rejected input upload corrupted a held result");
    held.reset();valid();return;
  }
  C10_CUDA_CHECK(cudaDeviceSynchronize());
  c10::cuda::CUDACachingAllocator::emptyCache();
  const double original=c10::cuda::CUDACachingAllocator::getMemoryFraction(device.index());
  size_t free=0,total=0;C10_CUDA_CHECK(cudaMemGetInfo(&free,&total));
  const auto reserved=c10::cuda::CUDACachingAllocator::getDeviceStats(device.index()).reserved_bytes[0].current;
  c10::cuda::CUDACachingAllocator::setMemoryFraction(double(reserved+(1<<20))/total,device.index());
  try {
    const auto* unread=reinterpret_cast<const int64_t*>(uintptr_t(1));int32_t huge[]={0,4000000};
    std::unique_ptr<void,decltype(&tei_output_destroy)> rejected(tei_forward_submit(engine.get(),unread,unread,
      unread,huge,1,4000000,0,nullptr,0,&row,1,64000000,nullptr,nullptr,0,nullptr,0,nullptr),tei_output_destroy);
    TORCH_CHECK(!rejected&&tei_last_error_kind()==2,"Allocator quota did not reject before output allocation/upload");
    std::vector<int64_t> real_input(4000000,0);
    std::unique_ptr<void,decltype(&tei_output_destroy)> upload_rejected(tei_forward_submit(engine.get(),
      real_input.data(),real_input.data(),real_input.data(),huge,1,4000000,0,nullptr,0,nullptr,0,0,
      nullptr,nullptr,0,nullptr,0,nullptr),tei_output_destroy);
    TORCH_CHECK(!upload_rejected&&tei_last_error_kind()==2,
      "Allocator quota did not reject before nested upload");
    // Weight upload is a startup C ABI path, outside serving admission. Use
    // real readable host data to preserve an actual allocator-OOM check.
    // Inactive graph/cache segments can be reclaimed during allocation.
    // Bound against active bytes so reclamation cannot make this probe fit.
    C10_CUDA_CHECK(cudaDeviceSynchronize());
    c10::cuda::CUDACachingAllocator::emptyCache();
    const auto stats=c10::cuda::CUDACachingAllocator::getDeviceStats(device.index());
    const auto active=stats.allocated_bytes[0].current;
    const auto reserved=stats.reserved_bytes[0].current;
    TORCH_CHECK(active>=0&&reserved>=active,"Invalid allocator geometry for startup OOM injection");
    TORCH_CHECK(uint64_t(reserved)<=uint64_t(INT64_MAX)-(2<<20)-2,
      "Startup OOM probe dimension overflow");
    const auto startup_quota=active+(1<<20);
    const int64_t elements=(reserved+(2<<20))/2+1;
    const auto target_bytes=elements*2;
    TORCH_CHECK(target_bytes>reserved+(1<<20),"Startup probe could fit a retained cached block");
    size_t current_free=0,current_total=0;C10_CUDA_CHECK(cudaMemGetInfo(&current_free,&current_total));
    TORCH_CHECK(size_t(target_bytes)<current_free/2,"Insufficient physical headroom for bounded quota probe");
    std::vector<float> startup_input(static_cast<size_t>(elements),0.f);int64_t shape[]={elements};
    std::cout<<"startup quota probe active="<<active<<" reserved="<<reserved
      <<" quota="<<startup_quota<<" target_bytes="<<target_bytes<<" driver_free="<<current_free<<"\n";
    c10::cuda::CUDACachingAllocator::setMemoryFraction(double(startup_quota)/current_total,device.index());
    const auto weight_status=tei_weight(engine.get(),"__quota_recovery_probe",
      reinterpret_cast<const uint8_t*>(startup_input.data()),shape,1,0);
    const auto oom_after=c10::cuda::CUDACachingAllocator::getDeviceStats(device.index()).num_ooms;
    TORCH_CHECK(weight_status!=0&&tei_last_error_kind()==3&&oom_after>stats.num_ooms,
      "Actual startup allocator OOM lost its recoverable classification: status=",weight_status,
      " kind=",tei_last_error_kind()," CUDA_ooms_before=",stats.num_ooms," after=",oom_after,
      " diagnostic=",tei_error());
  } catch (...) {c10::cuda::CUDACachingAllocator::setMemoryFraction(original,device.index());throw;}
  c10::cuda::CUDACachingAllocator::setMemoryFraction(original,device.index());valid();
}

class Model final : public tei::Model {
public:
  void ready() override {}
  at::Tensor forward(const tei::PackedInput& input) const override {
    return (input.ids.to(at::kFloat) + input.positions.to(at::kFloat)).unsqueeze(1).repeat({1, 768});
  }
};
int main() {
  try {
    TORCH_CHECK(tei::pinned_allocation_bytes(1,4096)==4096 && tei::pinned_allocation_bytes(4097,4096)==8192,
      "4 KiB host page accounting regressed");
    TORCH_CHECK(tei::pinned_allocation_bytes(1,65536)==65536 && tei::pinned_allocation_bytes(65537,65536)==131072,
      "64 KiB host page accounting regressed");
    TORCH_CHECK(tei::pinned_cached_bytes(8192,2,65536)==8192+2*2*65535,
      "Cached Torch block start/end page rounding is not reserved");
    c10::InferenceMode inference;
    const at::Device device("cuda:0");
    c10::cuda::CUDAGuard guard(device);
    // Aggregate admission is tied to actual storage across independent engines,
    // including idle retained slots and views that survive their pool.
    auto pinned_state = tei::pinned_admission_state();
    auto* host_allocator = at::getHostAllocator(at::kCUDA);
    host_allocator->empty_cache();
    const auto original_pinned_limit = pinned_state->max_bytes;
    TORCH_CHECK(pinned_state->bytes == 0, "Unexpected initial native pinned storage");
    const auto page=tei::pinned_host_page_size();
    std::cout << "PINNED_ALLOCATOR_SCOPE host_page_bytes=" << page
      << " host_register=" << c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::pinned_use_cuda_host_register()
      << " configured_reserve_MB=" << c10::cuda::CUDACachingAllocator::CUDAAllocatorConfig::pinned_reserve_segment_size_mb() << '\n';
    const auto foreign_stats=host_allocator->get_stats();
    const auto foreign_pins=tei::pinned_cached_bytes(foreign_stats.allocated_bytes.current,foreign_stats.allocations.current);
    pinned_state->max_bytes = foreign_pins + 2*page;
    auto engine_a = std::make_shared<tei::OutputPool>();
    auto engine_b = std::make_shared<tei::OutputPool>();
    for (auto engine : {engine_a, engine_b}) {
      tei::OutputLease lease(engine, device, 1); lease.allocate(); lease.record(); lease.wait();
    }
    TORCH_CHECK(pinned_state->bytes == 2*page && engine_a->slots[0].host.is_pinned(),
                "Pinned replicas did not share physical staging accounting");
    bool process_rejected = false;
    try { auto engine_c = std::make_shared<tei::OutputPool>();
      tei::OutputLease lease(engine_c, device, 1); lease.allocate(); }
    catch (const tei::MemoryAdmissionError&) { process_rejected = true; }
    TORCH_CHECK(process_rejected && pinned_state->bytes == 2*page,
                "Another replica bypassed process pinned admission");
    auto surviving_view = engine_a->slots[0].host.narrow(0, 0, 1);
    engine_a.reset(); engine_b.reset();
    TORCH_CHECK(pinned_state->bytes == page, "Surviving view released pinned storage credit");
    surviving_view = at::Tensor();
    TORCH_CHECK(pinned_state->bytes == 0, "Last storage reference did not release native pinned memory");
    // Torch's cached physical bytes, not only active bytes, remain charged.
    auto torch_pin = at::empty({static_cast<int64_t>(page/sizeof(float)+1)}, at::TensorOptions().dtype(at::kFloat).pinned_memory(true));
    process_rejected = false;
    try { auto blocked = tei::allocate_pinned_staging(1, at::TensorOptions().dtype(at::kFloat)); }
    catch (const tei::MemoryAdmissionError&) { process_rejected = true; }
    TORCH_CHECK(process_rejected && pinned_state->bytes == 0 && torch_pin.is_pinned(),
                "Native admission ignored active foreign Torch pins");
    torch_pin = at::Tensor();
    const auto cached_pins = host_allocator->get_stats();
    TORCH_CHECK(cached_pins.allocated_bytes.current >= cached_pins.active_bytes.current + 2*page,
                "Torch pinned size-class cache fixture was not retained");
    pinned_state->max_bytes = foreign_pins + page;
    auto reclaimed = tei::allocate_pinned_staging(1, at::TensorOptions().dtype(at::kFloat));
    TORCH_CHECK(pinned_state->bytes == page && host_allocator->get_stats().allocated_bytes.current <= foreign_pins,
                "Pinned admission did not reclaim completed Torch cache before allocation");
    // Torch cannot register externally owned pins. OutputLease must own them
    // through its stream event rather than relying on Torch's cache lifetime.
    TORCH_CHECK(!host_allocator->record_event(reclaimed.data_ptr(), reclaimed.storage().data_ptr().get_context(),
      c10::cuda::getCurrentCUDAStream().unwrap()), "Native pins unexpectedly entered Torch's cache");
    reclaimed = at::Tensor();
    TORCH_CHECK(pinned_state->bytes == 0, "Direct pinned staging was cached after release");
    pinned_state->max_bytes = original_pinned_limit;
    auto pool = std::make_shared<tei::OutputPool>();
    tei::CudaGraphCache cache(2, 4096,1);
    Model model;
    std::array<std::unique_ptr<tei::OutputLease>, 2> pending;
    std::array<int, 2> patterns{};
    std::array<int64_t, 2> rows{};
    const float* previous_host = nullptr;
    for (int step = 0; step < 12; ++step) {
      const int index = step % 2;
      if (pending[index]) {
        auto* lease = pending[index].get();
        // Complete on another CPU thread, as the production Rust worker does.
        auto actual = std::async(std::launch::async, [lease] { return lease->wait(); }).get();
        for (int64_t row = 0; row < rows[index]; ++row)
          for (int column = 0; column < 768; ++column)
            TORCH_CHECK(actual[row * 768 + column] == patterns[index] + row, "Stale graph/slot result");
        if (step == 2) previous_host = actual;
        pending[index].reset();
      }
      // Exact shapes change and evict cached graphs while another result is pending.
      const int64_t tokens = step < 4 ? 4096 : step % 3 == 0 ? 32 : step % 3 == 1 ? 128 : 4096;
      auto lease = std::make_unique<tei::OutputLease>(pool, device, tokens * 768);
      lease->allocate();
      const int pattern = step * 7;
      auto host = at::full({tokens}, pattern, at::TensorOptions().dtype(at::kLong));
      auto ids = lease->upload(host);
      host.fill_(-1); // The asynchronous upload must own its host data already.
      auto positions = lease->upload(at::arange(tokens, host.options()));
      std::array<int32_t, 2> offsets{0, static_cast<int32_t>(tokens)};
      auto cu = lease->upload(at::from_blob(offsets.data(), {2}, host.options().dtype(at::kInt)));
      tei::PackedInput input{ids, lease->upload(at::zeros_like(host)), positions, cu, offsets.data(), 1, tokens};
      lease->copy(cache.forward(model, input), 0);
      lease->record();
      patterns[index] = pattern;
      rows[index] = tokens;
      if (step == 2) TORCH_CHECK(lease->wait() == previous_host, "Pinned output was not reused");
      pending[index] = std::move(lease);
    }
    for (int index = 0; index < 2; ++index) {
      auto actual = pending[index]->wait();
      for (int64_t row = 0; row < rows[index]; ++row)
        for (int column = 0; column < 768; ++column)
          TORCH_CHECK(actual[row * 768 + column] == patterns[index] + row, "Final result mismatch");
      pending[index].reset();
    }
    // Dropping unfinished work must drain before reusing either input or output pins.
    { tei::OutputLease abandoned(pool, device, 4096 * 768); abandoned.allocate();
      abandoned.copy(at::ones({4096,768}, at::TensorOptions().device(device).dtype(at::kHalf)), 0);
      abandoned.record(); }
    at::Tensor partial_upload;
    auto partial_pool = std::make_shared<tei::OutputPool>();
    const auto before_partial = pinned_state->bytes;
    { tei::OutputLease partial(partial_pool, device, 1); partial.allocate();
      partial_upload = partial.upload(at::full({1024 * 1024}, 123, at::TensorOptions().dtype(at::kInt)));
      // Unwinding before record() must drain H2D before freeing its external pin.
    }
    TORCH_CHECK(pinned_state->bytes == before_partial && partial_pool->retained_bytes() == 0,
                "Partial upload did not drain and release native pinned storage");
    TORCH_CHECK(at::equal(partial_upload.cpu(), at::full({1024 * 1024}, 123, at::TensorOptions().dtype(at::kInt))),
                "Partial upload released its external pin before H2D completion");
    tei::OutputLease reused(pool, device, 768); reused.allocate();
    reused.copy(at::full({1,768}, 42, at::TensorOptions().device(device)), 0); reused.record();
    TORCH_CHECK(reused.wait()[767] == 42, "Abandoned lease contaminated reuse");
    // GPU completion alone must not release A while CPU packing still owns it.
    // Even if B is free, C waits for A rather than selecting any free slot.
    auto alternating = std::make_shared<tei::OutputPool>();
    auto first = std::make_unique<tei::OutputLease>(alternating, device, 1);
    first->allocate(); first->copy(at::ones({1}, at::TensorOptions().device(device)), 0); first->record();
    const auto first_host = first->wait();
    { tei::OutputLease second(alternating, device, 1); second.allocate();
      second.copy(at::full({1}, 2, at::TensorOptions().device(device)), 0); second.record();
      TORCH_CHECK(second.wait() != first_host, "A and B shared an output buffer"); }
    std::promise<void> started;
    auto third = std::async(std::launch::async, [&] {
      c10::InferenceMode worker_inference;
      c10::cuda::CUDAGuard worker_guard(device);
      started.set_value();
      tei::OutputLease lease(alternating, device, 1); lease.allocate();
      lease.copy(at::full({1}, 3, at::TensorOptions().device(device)), 0); lease.record();
      const auto host = lease.wait();
      return std::make_pair(host, host[0]);
    });
    started.get_future().wait();
    const auto waiting = third.wait_for(std::chrono::milliseconds(20));
    first.reset(); // Release only after simulated result packing finishes.
    const auto [third_host, third_value] = third.get();
    TORCH_CHECK(waiting == std::future_status::timeout, "C reused free B while A was still owned");
    TORCH_CHECK(third_host == first_host && third_value == 3, "C did not reuse A safely");
    // Admission protects pending results; only completed idle buffers may be evicted.
    auto bounded = std::make_shared<tei::OutputPool>();
    bounded->max_bytes = 256;
    auto held = std::make_unique<tei::OutputLease>(bounded, device, 8);
    held->allocate(); held->copy(at::ones({8}, at::TensorOptions().device(device)), 0); held->record();
    bool rejected = false;
    try { tei::OutputLease oversized(bounded, device, 32); oversized.allocate(); }
    catch (const tei::MemoryAdmissionError&) { rejected = true; }
    TORCH_CHECK(rejected && bounded->retained_bytes() == 64,
                "Admission allocated over budget or freed a pending result");
    TORCH_CHECK(held->wait()[7] == 1, "Budget rejection corrupted pending output");
    held.reset();
    // Next submission uses the other slot and releases the completed idle cache.
    { tei::OutputLease small(bounded, device, 8); small.allocate(); small.record(); small.wait(); }
    { tei::OutputLease larger(bounded, device, 32); larger.allocate();
      TORCH_CHECK(bounded->retained_bytes() == 256, "Idle staging eviction did not respect the budget");
      rejected = false;
      try { larger.upload(at::ones({1}, at::TensorOptions().dtype(at::kFloat))); }
      catch (const tei::MemoryAdmissionError&) { rejected = true; }
      TORCH_CHECK(rejected && bounded->retained_bytes() == 256, "Input staging exceeded the budget");
      larger.copy(at::full({32}, 9, at::TensorOptions().device(device)), 0); larger.record();
      TORCH_CHECK(larger.wait()[31] == 9, "Staging did not recover after admission rejection"); }
    rejected = false;
    try { tei::OutputLease oversized(bounded, device, std::numeric_limits<size_t>::max()); oversized.allocate(); }
    catch (const c10::Error&) { rejected = true; }
    TORCH_CHECK(rejected, "Oversized capacity overflow bypassed staging admission");
    auto varied = std::make_shared<tei::OutputPool>(); varied->max_bytes = 4096;
    { tei::OutputLease many_inputs(varied, device, 8); many_inputs.allocate();
      many_inputs.upload(at::ones({256}, at::TensorOptions().dtype(at::kFloat)));
      many_inputs.upload(at::ones({256}, at::TensorOptions().dtype(at::kFloat)));
      many_inputs.record(); many_inputs.wait(); }
    { tei::OutputLease other(varied, device, 8); other.allocate(); other.record(); other.wait(); }
    { tei::OutputLease larger(varied, device, 500); larger.allocate();
      TORCH_CHECK(varied->slots[0].inputs.empty(), "Selected slot retained unused input pins");
      larger.record(); larger.wait(); }
    { tei::OutputLease other(varied, device, 8); other.allocate(); other.record(); other.wait(); }
    { tei::OutputLease smaller(varied, device, 8); smaller.allocate();
      smaller.upload(at::ones({32}, at::TensorOptions().dtype(at::kFloat)));
      TORCH_CHECK(varied->retained_bytes() <= varied->max_bytes,
                  "Selected slot's old output capacity blocked a smaller request");
      smaller.copy(at::full({8}, 5, at::TensorOptions().device(device)), 0); smaller.record();
      TORCH_CHECK(smaller.wait()[7] == 5, "Selected-slot reclamation corrupted output"); }
    // Simulate device allocation pressure without consuming the GPU's memory.
    // An allocation failure must release the slot and its partial host allocation.
    auto pressured = std::make_shared<tei::OutputPool>();
    const auto fraction = c10::cuda::CUDACachingAllocator::getMemoryFraction(0);
    c10::cuda::CUDACachingAllocator::emptyCache();
    c10::cuda::CUDACachingAllocator::setMemoryFraction(0, 0);
    bool allocation_failed = false;
    try { tei::OutputLease failed(pressured, device, 8 * 1024 * 1024); failed.allocate(); }
    catch (const c10::OutOfMemoryError&) { allocation_failed = true; }
    catch (...) { c10::cuda::CUDACachingAllocator::setMemoryFraction(fraction, 0); throw; }
    c10::cuda::CUDACachingAllocator::setMemoryFraction(fraction, 0);
    TORCH_CHECK(allocation_failed && pressured->retained_bytes() == 0,
                "Failed allocation retained partial staging buffers");
    { tei::OutputLease recovered(pressured, device, 8); recovered.allocate();
      recovered.copy(at::full({8}, 11, at::TensorOptions().device(device)), 0); recovered.record();
      TORCH_CHECK(recovered.wait()[7] == 11, "Device allocation failure poisoned subsequent submission"); }
    native_allocator_error_test(device);
    native_allocator_error_test(device,true);
    native_allocator_error_test(device,2);
    std::cout << "output staging, overlap, graph eviction, alternation, reuse and cancellation passed\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}

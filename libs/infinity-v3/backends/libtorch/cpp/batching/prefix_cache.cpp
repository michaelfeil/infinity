/// \file
/// Resident prefix admission, snapshot publication, and suffix execution.
/// Combines cached attention/recurrent state with request-local packed metadata.

#include "batching/prefix_cache.h"
#include "kernels/infinity_fa4.h"
#include "batching/cuda_graph.h"
#include "batching/prefix_lookup.h"
#include "batching/tensor_copy.h"
#include "batching/radix.h"
#include "models/model.h"
#include "helpers/prefix_linear.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <ATen/ops/_flash_attention_forward.h>
#include <ATen/ops/_efficient_attention_forward.h>
#include <ATen/ops/_cudnn_attention_forward.h>
#include <ATen/ops/_foreach_copy.h>
#ifdef TEI_TORCH_CUDA_KERNELS
#include "kernels/torch_flash_fma.h"
#include "kernels/prefix_input.h"
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#endif

namespace tei {
PrefixEntry::~PrefixEntry() {
  if (budget) {
    budget->live_bytes -= bytes;
    budget->reserved_bytes -= reserved_bytes;
  }
}

const PackedInput& prefix_logical_input(const PackedInput& input) {
  return input.prefix ? input.prefix->original : input;
}

at::Tensor prefix_restore(const PackedInput& input, const at::Tensor& hidden) {
  if (input.capture_prefix)
    for (int64_t row = 0; row < input.batch; ++row)
      input.capture_prefix->graph_outputs[row] =
          hidden.narrow(0, input.offsets[row], input.offsets[row + 1] - input.offsets[row]);
  return input.prefix ? input.prefix->restore(hidden) : hidden;
}

void prefix_capture_attention(const PackedInput& input,
                              int64_t layer,
                              const at::Tensor& key,
                              const at::Tensor& value) {
  if (!input.capture_prefix)
    return;
  TORCH_CHECK(layer >= 0 && key.size(0) == input.ids.numel() && value.size(0) == input.ids.numel(),
              "Invalid full graph prefix state");
  for (int64_t row = 0; row < input.batch; ++row) {
    auto start = input.offsets[row], length = input.offsets[row + 1] - start;
    auto& layers = input.capture_prefix->graph_layers[row];
    TORCH_CHECK(!layers.count(layer), "Full graph prefix layer observed more than once");
    layers.emplace(
        layer, PrefixLayer{key.narrow(0, start, length), value.narrow(0, start, length), {}, {}});
  }
}

at::Tensor prefix_linear(const at::Tensor& input,
                         const at::Tensor& weight,
                         const at::Tensor& bias,
                         const PackedInput& packed) {
  if (packed.prefix && packed.prefix->linear)
    return packed.prefix->linear->forward(
        input, weight, bias, radix_ids(packed.prefix->original).numel());
  return at::linear(input, weight, bias);
}

const PrefixLayer* PrefixContext::previous(int64_t row, int64_t layer) const {
  if (!readers[row])
    return nullptr;
  auto found = readers[row]->layers.find(layer);
  return found == readers[row]->layers.end() ? nullptr : &found->second;
}

/// Allocate independent snapshot storage, copying now or appending a deferred source/target pair.
static at::Tensor snapshot_copy(const at::Tensor& source,
                                std::vector<at::Tensor>* targets,
                                std::vector<at::Tensor>* sources) {
  if (!source.defined())
    return {};
  if (!targets)
    return source.clone();
  auto target = at::empty_like(source);
  targets->push_back(target);
  sources->push_back(source);
  return target;
}

bool PrefixContext::store(int64_t row,
                          int64_t layer,
                          const PrefixLayer& source,
                          std::vector<at::Tensor>* targets,
                          std::vector<at::Tensor>* sources) {
  auto& entry = writers[row];
  if (!entry)
    return false;
  // Charge the map node as well as tensor payloads. Views are cloned below so
  // an admitted prefix cannot keep a larger batch allocation alive.
  uint64_t needed = sizeof(std::pair<const int64_t, PrefixLayer>) + 4 * sizeof(void*);
  for (const auto& t : {source.key, source.value, source.recurrent, source.convolution})
    if (t.defined())
      needed += t.numel() * t.element_size();
  if (needed > entry->reserved_bytes) {
    entry.reset();
    return false;
  }
  PrefixLayer owned;
  auto clone = [&](const at::Tensor& t) { return snapshot_copy(t, targets, sources); };
  owned.key = clone(source.key);
  owned.value = clone(source.value);
  owned.recurrent = clone(source.recurrent);
  owned.convolution = clone(source.convolution);
  TORCH_CHECK(!entry->layers.count(layer), "Prefix layer captured more than once");
  entry->layers.emplace(layer, std::move(owned));
  entry->bytes += needed;
  budget->live_bytes += needed;
  entry->reserved_bytes -= needed;
  budget->reserved_bytes -= needed;
  return true;
}

const at::Tensor& PrefixContext::prepare_graph_rows(const c10::Device& device) {
  TORCH_CHECK(collect_graph, "Prefix row mapping requires graph collection");
  if (!graph_input_rows.defined()) {
    std::vector<int32_t> rows;
    rows.reserve(offsets.back());
    for (int64_t row = 0; row < original.batch; ++row) {
      const auto full = int64_t(original.offsets[row + 1]) - original.offsets[row];
      const auto count = int64_t(offsets[row + 1]) - offsets[row];
      TORCH_CHECK(hits[row] >= 0 && count > 0 && hits[row] + count == full,
                  "Invalid direct prefix row mapping");
      for (int64_t token = 0; token < count; ++token)
        rows.push_back(original.offsets[row] + hits[row] + token);
    }
    graph_input_rows = at::tensor(rows, at::TensorOptions().dtype(at::kInt)).to(device);
  }
  return graph_input_rows;
}

PrefixLayer&
PrefixContext::prepare_graph_layer(int64_t layer, const at::Tensor& key, const at::Tensor& value) {
  TORCH_CHECK(collect_graph, "Prefix attention arena requires graph collection");
  auto& packed = graph_packed_layers[layer];
  if (!packed.key.defined()) {
    auto key_shape = key.sizes().vec(), value_shape = value.sizes().vec();
    key_shape[0] = original.ids.numel();
    value_shape[0] = original.ids.numel();
    packed.key = at::empty(key_shape, key.options());
    packed.value = at::empty(value_shape, value.options());
  }
  std::vector<at::Tensor> targets, sources;
  for (int64_t row = 0; row < original.batch; ++row) {
    auto& saved = graph_layers[row][layer];
    if (saved.key.defined())
      continue;
    const auto full = original.offsets[row + 1] - original.offsets[row];
    saved.key = packed.key.narrow(0, original.offsets[row], full);
    saved.value = packed.value.narrow(0, original.offsets[row], full);
    if (hits[row]) {
      auto old = previous(row, layer);
      TORCH_CHECK(old && old->key.defined() && old->key.size(0) >= hits[row],
                  "Missing graph prefix attention state");
      targets.push_back(saved.key.narrow(0, 0, hits[row]));
      sources.push_back(old->key.narrow(0, 0, hits[row]));
      targets.push_back(saved.value.narrow(0, 0, hits[row]));
      sources.push_back(old->value.narrow(0, 0, hits[row]));
    }
  }
  if (!targets.empty())
    tensor_copies(targets, sources);
  return packed;
}

PrefixContext::KV PrefixContext::attention(int64_t layer,
                                           const at::Tensor& key,
                                           const at::Tensor& value,
                                           bool capture) {
  if (capture && collect_graph)
    prepare_graph_layer(layer, key, value);
  const bool direct = capture && collect_graph && !key.is_neg() && !key.is_conj() &&
                      !value.is_neg() && !value.is_conj() &&
                      key.data_ptr() == graph_packed_layers.at(layer).key.data_ptr() &&
                      value.data_ptr() == graph_packed_layers.at(layer).value.data_ptr() &&
                      key.sizes() == graph_packed_layers.at(layer).key.sizes() &&
                      value.sizes() == graph_packed_layers.at(layer).value.sizes() &&
                      key.strides() == graph_packed_layers.at(layer).key.strides() &&
                      value.strides() == graph_packed_layers.at(layer).value.strides();
  std::vector<at::Tensor> keys, values;
  std::vector<at::Tensor> targets, sources;
  for (int64_t row = 0; row < original.batch; ++row) {
    int64_t length = offsets[row + 1] - offsets[row], full = length + hits[row];
    auto k = direct ? graph_layers[row].at(layer).key.narrow(0, hits[row], length)
                    : key.narrow(0, offsets[row], length);
    auto v = direct ? graph_layers[row].at(layer).value.narrow(0, hits[row], length)
                    : value.narrow(0, offsets[row], length);
    if (capture && collect_graph) {
      auto& saved = graph_layers[row][layer];
      auto key_target = saved.key.narrow(0, hits[row], length),
           value_target = saved.value.narrow(0, hits[row], length);
      if (k.is_neg() || k.is_conj() || k.data_ptr() != key_target.data_ptr() ||
          k.strides() != key_target.strides()) {
        targets.push_back(key_target);
        sources.push_back(k);
      }
      if (v.is_neg() || v.is_conj() || v.data_ptr() != value_target.data_ptr() ||
          v.strides() != value_target.strides()) {
        targets.push_back(value_target);
        sources.push_back(v);
      }
      k = saved.key;
      v = saved.value;
    } else if (hits[row]) {
      auto old = previous(row, layer);
      TORCH_CHECK(old && old->key.defined() && old->key.size(0) >= hits[row],
                  "Missing prefix attention state");
      k = at::cat({old->key.narrow(0, 0, hits[row]), k}, 0);
      v = at::cat({old->value.narrow(0, 0, hits[row]), v}, 0);
    }
    if (capture && collect_graph)
      graph_layers[row][layer] = PrefixLayer{k, v, {}, {}};
    TORCH_CHECK(full == original.offsets[row + 1] - original.offsets[row],
                "Cached keys must preserve the full logical context");
    keys.push_back(k);
    values.push_back(v);
  }
  if (!targets.empty())
    tensor_copies(targets, sources);
  if (capture)
    for (int64_t row = 0; row < original.batch; ++row)
      if (writers[row]) {
        auto saved = int64_t(writers[row]->identity.tokens.size());
        store(row,
              layer,
              PrefixLayer{keys[row].narrow(0, 0, saved), values[row].narrow(0, 0, saved), {}, {}});
      }
  if (capture && collect_graph) {
    const auto& packed = graph_packed_layers.at(layer);
    return {packed.key, packed.value, original.cumulative, original.max_sequence};
  }
  // Full K/V boundaries are unchanged by suffix execution. Reuse the existing
  // device metadata instead of synchronizing a host upload at every layer.
  // A single sequence also needs no second copy after prefix/suffix assembly.
  return {keys.size() == 1 ? keys[0].contiguous() : at::cat(keys, 0),
          values.size() == 1 ? values[0].contiguous() : at::cat(values, 0),
          original.cumulative,
          original.max_sequence};
}

at::Tensor prefix_attention(const at::Tensor& q,
                            const at::Tensor& k,
                            const at::Tensor& v,
                            const PackedInput& input,
                            int64_t layer,
                            double scale,
                            int64_t left,
                            int64_t right,
                            bool exact,
                            bool capture,
                            bool cudnn) {
  auto kv = input.prefix->attention(layer, k, v, capture);
  if (q.is_cuda()) {
    if (!cudnn) {
      auto fa4 = infinity_fa4(q,
                              kv.key,
                              kv.value,
                              input.cumulative,
                              kv.cumulative,
                              input.max_sequence,
                              scale,
                              true,
                              left,
                              right);
      if (fa4.defined())
        return fa4;
    }
    if (cudnn) {
      // Keep the original packed cuDNN geometry and causal provider. An
      // explicit dense suffix bias changes cuDNN's reduction recipe enough
      // to alter learned logits. Prefix queries are discarded, so zero rows
      // can stand in for them without retaining or projecting cached Q.
      // This replays attention rows, while QKV/MLP still process only suffixes.
      const auto& original = input.prefix->original;
      auto shape = q.sizes().vec();
      shape[0] = kv.key.size(0);
      auto queries = at::zeros(shape, q.options());
      for (int64_t row = 0; row < input.batch; ++row) {
        auto count = input.offsets[row + 1] - input.offsets[row];
        queries.narrow(0, original.offsets[row] + input.prefix->hits[row], count)
            .copy_(q.narrow(0, input.offsets[row], count));
      }
      auto full = std::get<0>(at::_cudnn_attention_forward(queries,
                                                           kv.key,
                                                           kv.value,
                                                           std::nullopt,
                                                           original.cumulative,
                                                           original.cumulative,
                                                           original.max_sequence,
                                                           original.max_sequence,
                                                           true,
                                                           0.,
                                                           true,
                                                           false,
                                                           scale));
      std::vector<at::Tensor> outputs;
      for (int64_t row = 0; row < input.batch; ++row) {
        auto count = input.offsets[row + 1] - input.offsets[row];
        outputs.push_back(full.narrow(0, original.offsets[row] + input.prefix->hits[row], count));
      }
      return outputs.size() == 1 ? outputs[0] : at::cat(outputs, 0);
    }
#ifdef TEI_TORCH_CUDA_KERNELS
    if (!exact && input.max_sequence == 1 && left < 0 && right < 0 &&
        (q.size(2) == 64 || q.size(2) == 128) &&
        torch_flash_fma_supported(q, kv.key, kv.value, true, left, right))
      return torch_flash_unfused_varlen(q,
                                        kv.key,
                                        kv.value,
                                        input.cumulative,
                                        kv.cumulative,
                                        input.max_sequence,
                                        kv.max_length,
                                        scale);
    if (exact && torch_flash_fma_supported(q, kv.key, kv.value, true, left, right))
      return torch_flash_fma_varlen(q,
                                    kv.key,
                                    kv.value,
                                    input.cumulative,
                                    kv.cumulative,
                                    input.max_sequence,
                                    kv.max_length,
                                    scale,
                                    true,
                                    left,
                                    right);
#endif
    if (q.size(2) <= 256)
      return std::get<0>(at::_flash_attention_forward(q,
                                                      kv.key,
                                                      kv.value,
                                                      input.cumulative,
                                                      kv.cumulative,
                                                      input.max_sequence,
                                                      kv.max_length,
                                                      0.,
                                                      true,
                                                      false,
                                                      scale,
                                                      left,
                                                      right));
    TORCH_CHECK(left < 0, "Cached wide-head local attention requires a qualified kernel");
    auto heads = q.size(1) / kv.key.size(1);
    auto keys = heads == 1 ? kv.key : kv.key.repeat_interleave(heads, 1);
    auto values = heads == 1 ? kv.value : kv.value.repeat_interleave(heads, 1);
    // The pinned efficient-attention enum uses 2 for bottom-right causality.
    return std::get<0>(at::_efficient_attention_forward(q.unsqueeze(0),
                                                        keys.unsqueeze(0),
                                                        values.unsqueeze(0),
                                                        std::nullopt,
                                                        input.cumulative,
                                                        kv.cumulative,
                                                        input.max_sequence,
                                                        kv.max_length,
                                                        0.,
                                                        2,
                                                        false,
                                                        scale))
        .squeeze(0);
  }
  std::vector<at::Tensor> outputs;
  for (int64_t row = 0; row < input.batch; ++row) {
    auto nq = input.offsets[row + 1] - input.offsets[row];
    auto nk = input.prefix->original.offsets[row + 1] - input.prefix->original.offsets[row];
    auto qs = q.narrow(0, input.offsets[row], nq).transpose(0, 1).unsqueeze(0);
    auto ks =
        kv.key.narrow(0, input.prefix->original.offsets[row], nk).transpose(0, 1).unsqueeze(0);
    auto vs =
        kv.value.narrow(0, input.prefix->original.offsets[row], nk).transpose(0, 1).unsqueeze(0);
    auto repeat = q.size(1) / kv.key.size(1);
    if (repeat != 1) {
      ks = ks.repeat_interleave(repeat, 1);
      vs = vs.repeat_interleave(repeat, 1);
    }
    auto qi = at::arange(nq, qs.options().dtype(at::kLong)) + input.prefix->hits[row];
    auto ki = at::arange(nk, qs.options().dtype(at::kLong));
    auto delta = qi.unsqueeze(1) - ki.unsqueeze(0);
    auto mask = delta >= 0;
    if (left >= 0)
      mask = mask & (delta <= left);
    if (right >= 0)
      mask = mask & (delta >= -right);
    outputs.push_back(at::scaled_dot_product_attention(qs, ks, vs, mask, 0., false, scale)
                          .squeeze(0)
                          .transpose(0, 1));
  }
  return at::cat(outputs, 0);
}

void PrefixContext::checkpoint(int64_t row,
                               int64_t layer,
                               const at::Tensor& state,
                               const at::Tensor& history) {
  store(row, layer, PrefixLayer{{}, {}, state, history});
}

at::Tensor PrefixContext::restore(const at::Tensor& suffix) {
  TORCH_CHECK(suffix.dim() == 2 && suffix.size(0) == offsets.back(),
              "Invalid cached suffix output");
  if (collect_graph) {
    const bool initialize = !graph_hidden.defined();
    if (initialize)
      graph_hidden = at::empty({original.ids.numel(), suffix.size(1)}, suffix.options());
    std::vector<at::Tensor> targets, sources;
    for (int64_t row = 0; row < original.batch; ++row) {
      auto value = graph_hidden.narrow(
          0, original.offsets[row], original.offsets[row + 1] - original.offsets[row]);
      if (initialize && hits[row]) {
        targets.push_back(value.narrow(0, 0, hits[row]));
        sources.push_back(readers[row]->output.narrow(0, 0, hits[row]));
      }
      auto length = offsets[row + 1] - offsets[row];
      targets.push_back(value.narrow(0, hits[row], length));
      sources.push_back(suffix.narrow(0, offsets[row], length));
      graph_outputs[row] = value;
    }
    tensor_copies(targets, sources);
    for (int64_t row = 0; row < original.batch; ++row)
      store_output(row, graph_outputs[row]);
    return graph_hidden;
  }
  std::vector<at::Tensor> rows;
  for (int64_t row = 0; row < original.batch; ++row) {
    auto value = suffix.narrow(0, offsets[row], offsets[row + 1] - offsets[row]);
    if (hits[row])
      value = at::cat({readers[row]->output.narrow(0, 0, hits[row]), value}, 0);
    if (collect_graph)
      graph_outputs[row] = value;
    store_output(row, value);
    rows.push_back(value);
  }
  return rows.size() == 1 ? rows[0] : at::cat(rows, 0);
}

void PrefixContext::store_output(int64_t row,
                                 const at::Tensor& value,
                                 std::vector<at::Tensor>* targets,
                                 std::vector<at::Tensor>* sources) {
  auto& entry = writers[row];
  if (!entry)
    return;
  auto retained = value.narrow(0, 0, entry->identity.tokens.size());
  const auto bytes = retained.numel() * retained.element_size();
  if (bytes > entry->reserved_bytes) {
    entry.reset();
    return;
  }
  entry->output = snapshot_copy(retained, targets, sources);
  entry->bytes += bytes;
  budget->live_bytes += bytes;
  entry->reserved_bytes -= bytes;
  budget->reserved_bytes -= bytes;
}

void PrefixContext::publish_graph(const PrefixContext& captured) {
  std::vector<at::Tensor> targets, sources;
  for (int64_t row = 0; row < original.batch; ++row)
    if (writers[row]) {
      auto& entry = writers[row];
      const auto retained = int64_t(entry->identity.tokens.size());
      std::vector<std::pair<int64_t, PrefixLayer>> states;
      uint64_t payload = 0;
      bool packed = true;
      auto account = [&](const at::Tensor& tensor) {
        packed = packed && payload % tensor.element_size() == 0;
        payload += tensor.nbytes();
      };
      for (const auto& [layer, value] : captured.graph_layers[row]) {
        PrefixLayer state{
            value.key.narrow(0, 0, retained), value.value.narrow(0, 0, retained), {}, {}};
        account(state.key);
        account(state.value);
        states.emplace_back(layer, std::move(state));
      }
      auto output = captured.graph_outputs[row].narrow(0, 0, retained);
      account(output);
      if (!packed) {
        for (const auto& [layer, state] : states)
          if (!store(row, layer, state, &targets, &sources))
            break;
        store_output(row, captured.graph_outputs[row], &targets, &sources);
        continue;
      }
      const auto nodes =
          states.size() * (sizeof(std::pair<const int64_t, PrefixLayer>) + 4 * sizeof(void*));
      const auto needed = prefix_storage_sum(payload, nodes);
      if (needed > entry->reserved_bytes) {
        entry.reset();
        continue;
      }
      auto slab = at::empty({int64_t(payload)}, output.options().dtype(at::kByte));
      int64_t offset = 0;
      auto view = [&](const at::Tensor& source) {
        auto target =
            slab.narrow(0, offset, source.nbytes()).view(source.scalar_type()).view(source.sizes());
        offset += source.nbytes();
        targets.push_back(target);
        sources.push_back(source);
        return target;
      };
      for (const auto& [layer, state] : states) {
        TORCH_CHECK(!entry->layers.count(layer), "Prefix layer captured more than once");
        entry->layers.emplace(layer, PrefixLayer{view(state.key), view(state.value), {}, {}});
      }
      entry->output = view(output);
      entry->slab = std::move(slab);
      entry->bytes += needed;
      budget->live_bytes += needed;
      entry->reserved_bytes -= needed;
      budget->reserved_bytes -= needed;
    }
  if (!targets.empty())
    tensor_copies(targets, sources);
}

PrefixRecurrent prefix_recurrent(
    const PackedInput& input, const at::Tensor& qkv, int64_t layer, int64_t heads, int64_t kernel) {
  PrefixRecurrent buffers;
  if (!input.prefix)
    return buffers;
  std::vector<int64_t> sources(input.batch, 0);
  std::vector<int32_t> lengths(input.batch, 0), positions(input.batch, -1),
      destinations(input.batch, -1);
  buffers.history = at::zeros({input.batch, kernel - 1, qkv.size(1)}, qkv.options());
  for (int64_t row = 0; row < input.batch; ++row) {
    auto old = input.prefix->previous(row, layer);
    if (input.prefix->hits[row]) {
      TORCH_CHECK(old && old->recurrent.defined() && old->convolution.defined(),
                  "Missing cached DeltaNet state/history");
      TORCH_CHECK(old->recurrent.scalar_type() == at::kFloat && old->recurrent.is_contiguous() &&
                      old->recurrent.sizes() == at::IntArrayRef({heads, 128, 128}),
                  "Invalid DeltaNet cached state geometry");
      sources[row] = reinterpret_cast<int64_t>(old->recurrent.const_data_ptr<float>());
      lengths[row] = old->convolution.size(0);
      TORCH_CHECK(lengths[row] <= kernel - 1, "Cached convolution history exceeds kernel width");
      if (lengths[row])
        buffers.history[row]
            .narrow(0, kernel - 1 - lengths[row], lengths[row])
            .copy_(old->convolution);
    }
    if (input.prefix->writers[row]) {
      positions[row] =
          int64_t(input.prefix->writers[row]->identity.tokens.size()) - input.prefix->hits[row];
      TORCH_CHECK(positions[row] > 0 &&
                      positions[row] <= input.offsets[row + 1] - input.offsets[row],
                  "Invalid DeltaNet checkpoint boundary");
      destinations[row] = buffers.rows.size();
      buffers.rows.push_back(row);
    }
  }
  buffers.initial_sources =
      at::tensor(sources, at::TensorOptions().dtype(at::kLong)).to(qkv.device());
  buffers.history_lengths =
      at::tensor(lengths, at::TensorOptions().dtype(at::kInt)).to(qkv.device());
  buffers.checkpoint_positions =
      at::tensor(positions, at::TensorOptions().dtype(at::kInt)).to(qkv.device());
  buffers.checkpoint_rows =
      at::tensor(destinations, at::TensorOptions().dtype(at::kInt)).to(qkv.device());
  if (!buffers.rows.empty())
    buffers.checkpoints =
        at::empty({int64_t(buffers.rows.size()), heads, 128, 128}, qkv.options().dtype(at::kFloat));
  return buffers;
}

void prefix_recurrent_commit(const PackedInput& input,
                             const at::Tensor& qkv,
                             int64_t layer,
                             int64_t kernel,
                             const PrefixRecurrent& buffers) {
  for (size_t slot = 0; slot < buffers.rows.size(); ++slot) {
    auto row = buffers.rows[slot];
    if (!input.prefix->writers[row])
      continue;
    auto count =
        int64_t(input.prefix->writers[row]->identity.tokens.size()) - input.prefix->hits[row];
    auto history = qkv.narrow(0, input.offsets[row], count);
    auto previous = input.prefix->previous(row, layer);
    if (previous && previous->convolution.defined())
      history = at::cat({previous->convolution, history}, 0);
    history = history.narrow(0,
                             std::max<int64_t>(0, history.size(0) - (kernel - 1)),
                             std::min<int64_t>(history.size(0), kernel - 1));
    input.prefix->checkpoint(row, layer, buffers.checkpoints[slot], history);
  }
}

PrefixCache::PrefixCache(
    uint64_t bytes, int64_t max_prefix, size_t count, size_t uses, StreamOrder order)
    : limit(bytes), maximum(max_prefix), capacity(count), admission_uses(uses),
      ordered_submissions(order == StreamOrder::EngineOrdered) {
  TORCH_CHECK(maximum > 0 && capacity > 0 && uses > 0, "Invalid prefix cache bounds");
  linear = std::make_unique<PrefixLinearPlans>();
  if (bytes && uses > 1) {
    short_recent = std::make_unique<ShortRecent>();
    while (short_recent->cells < std::min<size_t>(count, 4096))
      short_recent->cells *= 2;
  }
}

PrefixCache::~PrefixCache() = default;

void PrefixCache::reset_observations() {
  index.reset();
  recent = {};
  recent_next = 0;
  boundaries.clear();
  boundary_clock = 0;
  if (short_recent) {
    for (auto& depth : short_recent->depths)
      depth.clear();
    short_recent->extended.clear();
    short_recent->clock = 0;
  }
}

void PrefixCache::clear() {
  std::lock_guard guard(mutex);
  entries.clear();
  linear->clear();
  reset_observations();
}

at::Tensor PrefixCache::cumulative(const std::vector<int32_t>& offsets, const c10::Device& device) {
#ifdef TEI_TORCH_CUDA_KERNELS
  // Keep at most four immutable shapes, each no larger than the default graph
  // token bound. Larger requests retain the ordinary per-request allocation.
  if (device.is_cuda() && offsets.size() <= 4097) {
    c10::cuda::CUDAGuard guard(device);
    for (auto& buffer : boundaries)
      if (buffer.offsets == offsets && buffer.device.device() == device) {
        buffer.used = ++boundary_clock;
        buffer.consume();
        return buffer.device;
      }
    Boundaries buffer;
    buffer.offsets = offsets;
    buffer.used = ++boundary_clock;
    buffer.host = at::empty({int64_t(offsets.size())},
                            at::TensorOptions().dtype(at::kInt).pinned_memory(true));
    std::memcpy(buffer.host.mutable_data_ptr(), offsets.data(), offsets.size() * sizeof(int32_t));
    buffer.device = buffer.host.to(device, true);
    const auto producer = at::cuda::getCurrentCUDAStream(device.index());
    std::shared_ptr<at::cuda::CUDAEvent> ready;
    if (!ordered_submissions) {
      ready = std::make_shared<at::cuda::CUDAEvent>();
      ready->record(producer);
    }
    buffer.consume = [tensor = buffer.device, ready, producer] {
      const auto stream = at::cuda::getCurrentCUDAStream(producer.device_index());
      if (stream != producer) {
        if (ready)
          ready->block(stream);
        c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
      }
    };
    if (boundaries.size() == 4)
      boundaries.erase(
          std::min_element(boundaries.begin(), boundaries.end(), [](const auto& a, const auto& b) {
            return a.used < b.used;
          }));
    auto result = buffer.device;
    boundaries.push_back(std::move(buffer));
    return result;
  }
#endif
  return at::tensor(offsets, at::TensorOptions().dtype(at::kInt)).to(device);
}

/// Track bounded frequency evidence for shared text stems without treating hashes as reuse proof.
int64_t PrefixCache::observe_text(std::span<const PrefixToken> tokens,
                                  int operation,
                                  bool rotary_context,
                                  int64_t known_prefix,
                                  int64_t observed) {
  TORCH_INTERNAL_ASSERT(short_recent && tokens.size() > 0 && known_prefix >= 0 &&
                        size_t(known_prefix) <= tokens.size());
  // Position-first counters preserve interleaved questions under a shared role
  // header. Each depth has eight-way buckets, bounding an observation to eight
  // probes instead of comparing every prior document's complete token prefix.
  auto& state = *short_recent;
  uint64_t key = 0x5354454d434f554eULL + uint64_t(operation) * 4 + uint64_t(rotary_context);
  const auto tick = ++state.clock;
  int64_t confirmed = std::max<int64_t>(known_prefix, std::max<int64_t>(0, observed));
  auto count = [&](auto& table, uint64_t hash, size_t cells) {
    if (table.size() < cells) {
      std::vector<ShortRecent::Counter> grown(cells);
      for (const auto& cell : table)
        if (cell.uses) {
          const auto bucket = (cell.key & ((cells / 8) - 1)) * 8;
          auto slot = std::find_if(grown.begin() + bucket,
                                   grown.begin() + bucket + 8,
                                   [](const auto& value) { return !value.uses; });
          TORCH_INTERNAL_ASSERT(slot != grown.begin() + bucket + 8);
          *slot = cell;
        }
      table.swap(grown);
    }
    const auto bucket = (hash & ((table.size() / 8) - 1)) * 8;
    ShortRecent::Counter* selected = nullptr;
    for (size_t way = 0; way < 8; ++way) {
      auto& cell = table[bucket + way];
      if (cell.uses && cell.key == hash) {
        selected = &cell;
        break;
      }
      if (!selected || (!cell.uses && selected->uses) ||
          (cell.uses && selected->uses && cell.used < selected->used))
        selected = &cell;
    }
    if (selected->uses && selected->key == hash) {
      if (selected->uses < admission_uses)
        ++selected->uses;
    } else {
      selected->key = hash;
      selected->uses = 1;
    }
    selected->used = tick;
    return selected->uses >= admission_uses;
  };
  for (size_t depth = 0; depth < std::min<size_t>(64, tokens.size()); ++depth) {
    key = prefix_admission_step(key, tokens[depth]);
    // The miss shortcut already counted this stem, or a reader already proves
    // these rows reusable. Only unseen descendants need admission counters.
    if (observed < 0 && int64_t(depth) >= known_prefix &&
        count(state.depths[depth], key, state.cells))
      confirmed = depth + 1;
  }
  size_t factor = 8;
  while (factor < std::min<size_t>(256, (tokens.size() > 64 ? (tokens.size() - 64 + 7) / 8 : 0)))
    factor *= 2;
  if (confirmed >= 64)
    for (size_t start = 64; start < tokens.size(); start += 8) {
      const auto length = std::min<size_t>(8, tokens.size() - start);
      key = prefix_admission_block(key, tokens.subspan(start, length));
      if (int64_t(start + length) > known_prefix &&
          count(state.extended, key, std::min<size_t>(65536, state.cells * factor)))
        confirmed = start + length;
    }
  return confirmed >= int64_t(std::min<size_t>(8, tokens.size())) ? confirmed : 0;
}

bool PrefixCache::admit(const PrefixIdentity& identity, int operation) {
  if (admission_uses == 1)
    return true;
  return admit(prefix_admission_fingerprint(identity, operation));
}

bool PrefixCache::admit(PrefixFingerprint key) {
  if (admission_uses == 1)
    return true;
  for (auto& observed : recent)
    if (observed.uses && observed.key == key) {
      if (observed.uses < admission_uses)
        ++observed.uses;
      return observed.uses >= admission_uses;
    }
  recent[recent_next] = {key, 1};
  recent_next = (recent_next + 1) % recent.size();
  return false;
}

PrefixCache::Stats PrefixCache::stats() const {
  std::lock_guard guard(mutex);
  return counters;
}

uint64_t PrefixCache::bytes() const {
  std::lock_guard guard(mutex);
  return budget->live_bytes;
}

/// Borrowed host token views used to construct and compare exact prefix identities.
struct HostPrefixTokens {
  const int64_t *ids, *positions, *types, *rotary;
  int64_t total;

  explicit HostPrefixTokens(const PackedInput& input, int64_t expected) : total(input.ids.numel()) {
    TORCH_CHECK(total == expected, "Host prefix IDs must match packed token count");
    const auto& axes = input.multimodal_positions;
    for (const auto& values : {input.ids, input.positions, input.types})
      TORCH_CHECK(values.device().is_cpu() && values.scalar_type() == at::kLong &&
                      values.is_contiguous() && values.dim() == 1 && values.numel() == total,
                  "Invalid host prefix identity metadata");
    if (axes.defined())
      TORCH_CHECK(axes.device().is_cpu() && axes.scalar_type() == at::kLong &&
                      axes.is_contiguous() && axes.sizes() == at::IntArrayRef({3, total}),
                  "Invalid host prefix rotary axes");
    ids = input.ids.const_data_ptr<int64_t>();
    positions = input.positions.const_data_ptr<int64_t>();
    types = input.types.const_data_ptr<int64_t>();
    rotary = axes.defined() ? axes.const_data_ptr<int64_t>() : nullptr;
  }

  PrefixToken token(int64_t index) const {
    PrefixToken value;
    value.values = {ids[index], positions[index], types[index], 0, 0, 0};
    if (rotary)
      for (int i = 0; i < 3; ++i)
        value.values[3 + i] = rotary[i * total + index];
    return value;
  }
};

/// Build exact token/media identity from host views, with media placement relative to its sequence.
static PrefixIdentity
identity(const PackedInput& input, int64_t row, const HostPrefixTokens& host) {
  auto start = input.offsets[row], end = input.offsets[row + 1];
  PrefixIdentity key;
  // Image processing can select a residual path for every row in a batch,
  // including text-only rows. Keep that execution context in their identity.
  key.image_context = !input.images.empty();
  key.rotary_context = input.multimodal_positions.defined();
  key.tokens.reserve(end - start);
  for (int64_t t = start; t < end; ++t) {
    key.tokens.push_back(host.token(t));
  }
  for (const auto& image : input.images)
    if (image.token_start >= start && image.token_start < end) {
      PrefixMedia media;
      media.start = image.token_start - start;
      media.count = image.token_count;
      media.layout = {image.grid_thw[0], image.grid_thw[1], image.grid_thw[2], image.merge_size};
      media.layout.push_back(int64_t(image.pixels.scalar_type()));
      for (auto size : image.pixels.sizes())
        media.layout.push_back(size);
      media.hashed = image.has_feature_key;
      if (media.hashed)
        media.digest = image.feature_key;
      else {
        auto pixels = image.pixels.cpu().contiguous();
        auto bytes = pixels.numel() * pixels.element_size();
        media.content.resize(bytes);
        std::memcpy(media.content.data(), pixels.const_data_ptr(), bytes);
      }
      key.media.push_back(std::move(media));
    }
  return key;
}

at::Tensor PrefixCache::run(const Model& model,
                            const PackedInput& original,
                            int operation,
                            CudaGraphCache* graphs) {
  std::lock_guard guard(mutex);
  if (graphs && limit)
    graphs->retain_prefix_states();
  const MathPolicy* known_policy = nullptr;
  auto run = [&](const PackedInput& input) {
    return operation ? model.predict(input, operation == 2) : model.forward(input);
  };
  auto fallback = [&] {
    return graphs ? graphs->forward(model, original, operation, known_policy) : run(original);
  };
  if (!limit || !model.supports_prefix_cache(operation) || !original.audios.empty()) {
    ++counters.bypasses;
    return fallback();
  }
  try {
    auto policy = execution_math_policy(original.ids.device());
    known_policy = &policy;
    if (model_epoch != model.prefix_namespace() || math_policy != policy) {
      entries.clear();
      linear->clear();
      reset_observations();
      model_epoch = model.prefix_namespace();
      math_policy = policy;
    }
    TORCH_CHECK(original.batch > 0 && original.offsets && original.offsets[0] == 0 &&
                    original.offsets[original.batch] == original.ids.numel(),
                "Invalid full prefix-cache input boundaries");
    for (int64_t row = 0; row < original.batch; ++row)
      TORCH_CHECK(original.offsets[row + 1] > original.offsets[row] &&
                      original.offsets[row + 1] - original.offsets[row] <= original.max_sequence,
                  "Invalid full prefix-cache sequence length");
    auto host = original;
    host.ids = original.host_ids.defined() ? original.host_ids : original.ids.cpu();
    host.types = original.host_types.defined() ? original.host_types : original.types.cpu();
    host.positions =
        original.host_positions.defined() ? original.host_positions : original.positions.cpu();
    if (original.multimodal_positions.defined())
      host.multimodal_positions = original.host_multimodal_positions.defined()
                                      ? original.host_multimodal_positions
                                      : original.multimodal_positions.cpu();
    HostPrefixTokens tokens(host, original.ids.numel());
    // CUDA recipes can depend on the complete packed geometry. Old prefix rows
    // cannot be repaired by selecting the current recipe only for suffix rows.
    // Require the same geometry until cross-shape providers are qualified.
    const auto* reuse_layout = original.ids.is_cuda() ? &original : nullptr;
    // Small registries retain the cheap exact scan. Build the positional index
    // only after enough resident entries make repeated scans expensive.
    const auto* resident =
        index && index->usable() && entries.size() >= index_minimum ? index.get() : nullptr;
    const bool short_policy =
        admission_uses > 1 && original.images.empty() && !model.prefix_checkpoint_only();
    std::vector<int64_t> observed;
    if (admission_uses > 1 && original.images.empty()) {
      // A first-token mismatch proves that no resident prefix can be consumed.
      // Observe only the bounded stem before allocating complete identities or
      // suffix metadata. Keep each row's decision so a mixed batch is observed
      // exactly once, even when another row permits snapshot capture.
      bool candidate = false;
      for (int64_t row = 0; !entries.empty() && row < original.batch && !candidate; ++row) {
        auto first = tokens.token(original.offsets[row]);
        candidate =
            resident ? resident->possible(
                           first, operation, original.multimodal_positions.defined(), reuse_layout)
                     : std::any_of(entries.begin(), entries.end(), [&](const auto& entry) {
                         return entry->operation == operation && !entry->identity.image_context &&
                                !entry->identity.tokens.empty() &&
                                entry->identity.tokens.front() == first &&
                                entry->identity.rotary_context ==
                                    original.multimodal_positions.defined() &&
                                (!reuse_layout || prefix_batch_compatible(*entry, *reuse_layout));
                       });
      }
      if (!candidate) {
        observed.resize(original.batch);
        std::array<PrefixToken, prefix_block_tokens> stem;
        for (int64_t row = 0; row < original.batch; ++row) {
          auto length = original.offsets[row + 1] - original.offsets[row];
          auto retained =
              std::min<int64_t>(maximum, length - (model.prefix_checkpoint_only() ? 1 : 0));
          if (model.prefix_checkpoint_only() && retained >= 64)
            retained = retained / 64 * 64;
          const auto count = std::min<int64_t>(retained, prefix_block_tokens);
          for (int64_t t = 0; t < count; ++t)
            stem[t] = tokens.token(original.offsets[row] + t);
          if (count) {
            if (short_policy) {
              const auto common = observe_text(
                  {stem.data(), size_t(count)}, operation, original.multimodal_positions.defined());
              observed[row] = common;
            } else
              observed[row] =
                  admit(prefix_admission_fingerprint({stem.data(), size_t(count)},
                                                     operation,
                                                     false,
                                                     original.multimodal_positions.defined()))
                      ? retained
                      : 0;
          }
        }
        if (std::none_of(observed.begin(), observed.end(), [](auto value) { return value != 0; })) {
          counters.misses += original.batch;
          ++counters.bypasses;
          return fallback();
        }
      }
    }
    std::vector<PrefixIdentity> identities;
    identities.reserve(original.batch);
    for (int64_t row = 0; row < original.batch; ++row)
      identities.push_back(identity(host, row, tokens));
    if ((operation == 0 || operation == 1) && !model.prefix_checkpoint_only()) {
      std::vector<std::shared_ptr<PrefixEntry>> complete;
      complete.reserve(original.batch);
      for (const auto& key : identities) {
        std::shared_ptr<PrefixEntry> found;
        if (resident && !key.image_context && key.media.empty()) {
          found = resident->exact(key, operation, reuse_layout, true);
          if (found && !prefix_batch_compatible(*found, original))
            found.reset();
        } else {
          auto match = std::find_if(entries.begin(), entries.end(), [&](const auto& entry) {
            return entry->operation == operation && prefix_batch_compatible(*entry, original) &&
                   prefix_same(key, entry->identity) && entry->ready();
          });
          if (match != entries.end())
            found = *match;
        }
        if (!found)
          break;
        complete.push_back(std::move(found));
      }
      if (complete.size() == size_t(original.batch)) {
        std::vector<at::Tensor> outputs;
        outputs.reserve(complete.size());
        for (auto& entry : complete) {
#ifdef TEI_TORCH_CUDA_KERNELS
          if (original.ids.is_cuda())
            c10::cuda::CUDACachingAllocator::recordStream(
                entry->output.storage().data_ptr(),
                at::cuda::getCurrentCUDAStream(original.ids.get_device()));
#endif
          outputs.push_back(entry->output);
        }
        at::Tensor result;
        if (operation == 1) {
          // Run the original trained head on complete logical rows, preserving
          // padding selection and providers that score every token before readout.
          result = model.classify_hidden(original,
                                         outputs.size() == 1 ? outputs[0] : at::cat(outputs, 0));
        } else {
          // Return fresh storage: a caller may mutate its result, and another stream
          // may clear/evict the immutable source as soon as this submission returns.
          result = outputs.size() == 1 ? outputs[0].clone() : at::cat(outputs, 0);
        }
        if (result.defined()) {
          for (auto& entry : complete) {
            ++counters.hits;
            counters.reused_tokens += entry->identity.tokens.size();
            entry->used = ++clock;
          }
          return result;
        }
      }
    }
    PrefixContext context{original, {}, {}, {}, {}, limit, budget};
    context.offsets = {0};
    context.linear = linear.get();
    context.math_policy = policy;
    bool pack_inputs = false;
#ifdef TEI_TORCH_CUDA_KERNELS
    pack_inputs = original.batch > 1 &&
                  prefix_inputs_supported(original.ids, original.positions, original.types);
#endif
    std::vector<at::Tensor> ids, positions, types, axes;
    std::vector<int64_t> original_rows;
    for (int64_t row = 0; row < original.batch; ++row) {
      auto key = std::move(identities[row]);
      const auto bound = std::min<int64_t>(maximum, key.tokens.size() - 1);
      auto found =
          resident
              ? resident->lookup(
                    key, entries, operation, bound, model.prefix_checkpoint_only(), reuse_layout)
              : prefix_lookup(
                    key, entries, operation, bound, model.prefix_checkpoint_only(), reuse_layout);
      auto best = found.length;
      auto source = std::move(found.entry);
      if (best) {
        ++counters.hits;
        counters.reused_tokens += best;
        source->used = ++clock;
      } else
        ++counters.misses;
#ifdef TEI_TORCH_CUDA_KERNELS
      if (source && original.ids.is_cuda()) {
        auto stream = at::cuda::getCurrentCUDAStream(original.ids.get_device());
        auto retain = [&](const at::Tensor& tensor) {
          if (tensor.defined())
            c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
        };
        if (source->slab.defined())
          retain(source->slab);
        else {
          retain(source->output);
          for (const auto& [id, state] : source->layers)
            for (const auto& tensor : {state.key, state.value, state.recurrent, state.convolution})
              retain(tensor);
        }
      }
#endif
      context.hits.push_back(best);
      context.readers.push_back(source);
      auto start = original.offsets[row] + best, length = original.offsets[row + 1] - start;
      if (!pack_inputs) {
        ids.push_back(original.ids.narrow(0, start, length));
        positions.push_back(original.positions.narrow(0, start, length));
        types.push_back(original.types.narrow(0, start, length));
      }
      if (original.multimodal_positions.defined())
        axes.push_back(original.multimodal_positions.narrow(1, start, length));
      context.offsets.push_back(context.offsets.back() + length);
      for (int64_t t = start; t < original.offsets[row + 1]; ++t)
        original_rows.push_back(t);
      std::shared_ptr<PrefixEntry> writer;
      auto retained = std::min<int64_t>(maximum, key.tokens.size());
      if (model.prefix_checkpoint_only()) {
        retained = std::min<int64_t>(maximum, key.tokens.size() - 1);
        if (retained >= 64)
          retained = retained / 64 * 64;
      }
      if (short_policy) {
        const auto common = observe_text({key.tokens.data(), size_t(retained)},
                                         operation,
                                         key.rotary_context,
                                         best,
                                         observed.empty() ? -1 : observed[row]);
        const bool complete =
            retained > 64 &&
            admit(prefix_document_fingerprint(
                {key.tokens.data(), size_t(retained)}, operation, key.rotary_context));
        // A cold confirmed stem may seed a complete snapshot to discover longer
        // prefixes. Once a reader exists, a changing tail must not repeatedly
        // promote that stem into an unrelated complete document.
        if (!complete && (common < 64 || best))
          retained = std::min(retained, common);
      }
      for (const auto& media : key.media)
        if (media.start < retained && media.start + media.count > retained)
          retained = media.start;
      // The selected reader already owns every eligible row. No new snapshot
      // can be admitted, so trimming and duplicate lookup have no work to do.
      if (retained <= best) {
        context.writers.emplace_back();
        continue;
      }
      key.tokens.resize(retained);
      key.tokens.shrink_to_fit();
      key.media.erase(std::remove_if(key.media.begin(),
                                     key.media.end(),
                                     [&](const auto& media) { return media.start >= retained; }),
                      key.media.end());
      key.media.shrink_to_fit();
      key.fingerprints.resize(
          std::min(key.fingerprints.size(), key.tokens.size() / prefix_block_tokens));
      bool resident_exact =
          resident && !key.image_context && key.media.empty()
              ? bool(resident->exact(key, operation, reuse_layout, false))
              : std::any_of(entries.begin(), entries.end(), [&](const auto& entry) {
                  return entry->operation == operation &&
                         (!reuse_layout || prefix_batch_compatible(*entry, *reuse_layout)) &&
                         prefix_same(key, entry->identity);
                });
      if (short_policy && !resident_exact)
        resident_exact =
            std::any_of(context.writers.begin(), context.writers.end(), [&](const auto& earlier) {
              return earlier && prefix_same(key, earlier->identity);
            });
      if (retained > best && !resident_exact &&
          (short_policy || (observed.empty() ? admit(key, operation) : observed[row]))) {
        prefix_fingerprints(key, retained);
        key.fingerprints.shrink_to_fit();
        uint64_t metadata = sizeof(PrefixEntry) + key.tokens.capacity() * sizeof(PrefixToken) +
                            key.media.capacity() * sizeof(PrefixMedia) +
                            key.fingerprints.capacity() * sizeof(PrefixFingerprint);
        for (const auto& media : key.media)
          metadata += media.layout.capacity() * sizeof(int64_t) + media.content.capacity();
        writer = std::make_shared<PrefixEntry>();
        writer->identity = std::move(key);
        writer->operation = operation;
        writer->used = ++clock;
        writer->batch_offsets.assign(original.offsets, original.offsets + original.batch + 1);
        metadata += writer->batch_offsets.capacity() * sizeof(int32_t);
        if (capacity >= index_minimum)
          metadata = prefix_storage_sum(
              metadata,
              prefix_storage_sum(PrefixIndex::storage_bytes(retained),
                                 writer->batch_offsets.capacity() * sizeof(int32_t)));
        writer->batch_max_sequence = original.max_sequence;
        writer->compact_tokens = original.compact_ids.defined() ? original.compact_ids.numel() : 0;
        writer->bytes = metadata;
      }
      context.writers.push_back(std::move(writer));
    }
    // Look up every row before evicting, then reserve each complete snapshot.
    // Oversized recurrent checkpoints never allocate capture buffers. Reader
    // payloads remain charged even if their lookup slot is evicted during a run.
    size_t admitting = 0;
    for (auto& writer : context.writers)
      if (writer) {
        const auto payload = model.prefix_payload_bytes(writer->identity.tokens.size());
        const auto nodes =
            prefix_storage_bytes(model.prefix_layers(),
                                 sizeof(std::pair<const int64_t, PrefixLayer>) + 4 * sizeof(void*),
                                 1);
        const auto reservation = prefix_storage_sum(payload, nodes);
        const auto needed = prefix_storage_sum(writer->bytes, reservation);
        if (!payload || needed > limit) {
          writer.reset();
          continue;
        }
        auto fits = [&] {
          return budget->live_bytes <= limit &&
                 budget->reserved_bytes <= limit - budget->live_bytes &&
                 needed <= limit - budget->live_bytes - budget->reserved_bytes;
        };
        while (!entries.empty() && (!fits() || entries.size() + admitting >= capacity)) {
          auto victim = entries.end();
          for (auto it = entries.begin(); it != entries.end(); ++it)
            if (it->use_count() == 1 && (victim == entries.end() || (*it)->used < (*victim)->used))
              victim = it;
          if (victim == entries.end()) {
            if (!fits())
              break;
            victim =
                std::min_element(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
                  return a->used < b->used;
                });
          }
          if (index)
            index->erase(**victim, reuse_layout);
          entries.erase(victim);
          ++counters.evictions;
        }
        if (!fits() || entries.size() + admitting >= capacity) {
          writer.reset();
          continue;
        }
        writer->budget = budget;
        writer->reserved_bytes = reservation;
        budget->live_bytes += writer->bytes;
        budget->reserved_bytes += reservation;
        ++admitting;
      }
    if (!admitting &&
        std::none_of(context.hits.begin(), context.hits.end(), [](auto hit) { return hit != 0; })) {
      ++counters.bypasses;
      return fallback();
    }
    PackedInput input = original;
    // Cold capture keeps the caller's original metadata and device buffers.
    // Only a real hit needs packed suffix rows and a new Radix mapping.
    if (std::any_of(context.hits.begin(), context.hits.end(), [](auto hit) { return hit != 0; })) {
#ifdef TEI_TORCH_CUDA_KERNELS
      if (pack_inputs) {
        auto packed = pack_prefix_inputs(original.ids,
                                         original.positions,
                                         original.types,
                                         {original.offsets, size_t(original.batch) + 1},
                                         context.offsets,
                                         context.hits);
        input.ids = packed.ids;
        input.positions = packed.positions;
        input.types = packed.types;
      } else
#endif
      {
        input.ids = ids.size() == 1 ? ids[0] : at::cat(ids);
        input.positions = positions.size() == 1 ? positions[0] : at::cat(positions);
        input.types = types.size() == 1 ? types[0] : at::cat(types);
      }
      input.cumulative = cumulative(context.offsets, input.ids.device());
      input.offsets = context.offsets.data();
      input.max_sequence = 0;
      for (int64_t row = 0; row < original.batch; ++row)
        input.max_sequence =
            std::max<int64_t>(input.max_sequence, context.offsets[row + 1] - context.offsets[row]);
      if (!axes.empty())
        input.multimodal_positions = at::cat(axes, 1);
      input.images.clear();
      for (const auto& image : original.images) {
        auto found = std::upper_bound(
            original.offsets, original.offsets + original.batch + 1, image.token_start);
        auto row = found - original.offsets - 1;
        if (image.token_start + image.token_count <= original.offsets[row] + context.hits[row])
          continue;
        auto copy = image;
        copy.token_start =
            context.offsets[row] + image.token_start - original.offsets[row] - context.hits[row];
        copy.sequence_start = context.offsets[row];
        input.images.push_back(std::move(copy));
      }
      if (radix_enabled(original)) {
        auto scatter = original.host_scatter_unfold.defined() ? original.host_scatter_unfold
                                                              : original.scatter_unfold.cpu();
        TORCH_CHECK(scatter.device().is_cpu() && scatter.scalar_type() == at::kLong &&
                        scatter.is_contiguous() &&
                        scatter.sizes() == at::IntArrayRef({original.ids.numel()}),
                    "Invalid host prefix Radix mapping");
        const auto* scatter_rows = scatter.const_data_ptr<int64_t>();
        std::unordered_map<int64_t, int64_t> remap;
        std::vector<int64_t> compact, unfold, fold;
        for (int64_t row = 0; row < int64_t(original_rows.size()); ++row) {
          auto old = scatter_rows[original_rows[row]];
          auto [it, inserted] = remap.emplace(old, remap.size());
          if (inserted) {
            compact.push_back(row);
            fold.push_back(row);
          }
          unfold.push_back(it->second);
        }
        auto index =
            at::tensor(compact, at::TensorOptions().dtype(at::kLong)).to(input.ids.device());
        input.compact_ids = input.ids.index_select(0, index);
        input.compact_positions = input.positions.index_select(0, index);
        input.compact_types = input.types.index_select(0, index);
        input.scatter_unfold =
            at::tensor(unfold, at::TensorOptions().dtype(at::kLong)).to(input.ids.device());
        input.fold_gather =
            at::tensor(fold, at::TensorOptions().dtype(at::kLong)).to(input.ids.device());
      }
    }
    input.prefix = &context;
    const bool has_hit =
        std::any_of(context.hits.begin(), context.hits.end(), [](auto hit) { return hit != 0; });
    auto result = graphs && !model.prefix_checkpoint_only()
                      ? has_hit ? graphs->forward(model, input, operation)
                                : graphs->forward(model, original, operation, &policy, &context)
                      : run(input);
#ifdef TEI_TORCH_CUDA_KERNELS
    std::shared_ptr<at::cuda::CUDAEvent> event;
    // Native engine submission fences already cover snapshot writes and result
    // staging. Independent caches still need per-publication completion events.
    if (admitting && input.ids.is_cuda() && !ordered_submissions) {
      event = std::make_shared<at::cuda::CUDAEvent>();
      event->record(at::cuda::getCurrentCUDAStream(input.ids.device().index()));
    }
#endif
    for (auto& entry : context.writers)
      if (entry && entry->output.defined() &&
          int64_t(entry->layers.size()) == model.prefix_layers()) {
        budget->reserved_bytes -= entry->reserved_bytes;
        entry->reserved_bytes = 0;
#ifdef TEI_TORCH_CUDA_KERNELS
        if (event) {
          const auto producer = at::cuda::getCurrentCUDAStream(input.ids.get_device());
          // All snapshot writes and the event are queued before publication.
          // Stream ordering therefore permits immediate producer-stream reuse;
          // another stream can consume the entry only after global completion.
          entry->ready = [event, producer] {
            return at::cuda::getCurrentCUDAStream(producer.device_index()) == producer ||
                   event->query();
          };
        }
#endif
        entries.erase(std::remove_if(entries.begin(),
                                     entries.end(),
                                     [&](const auto& old) {
                                       const bool duplicate =
                                           old->operation == operation &&
                                           (!reuse_layout ||
                                            prefix_batch_compatible(*old, *reuse_layout)) &&
                                           prefix_same(old->identity, entry->identity);
                                       if (duplicate && index)
                                         index->erase(*old, reuse_layout);
                                       return duplicate;
                                     }),
                      entries.end());
        if (entries.size() < capacity) {
          entries.push_back(entry);
          ++counters.admissions;
          if (index)
            index->insert(entry, reuse_layout);
          else if (entries.size() >= index_minimum) {
            index = std::make_unique<PrefixIndex>();
            for (const auto& old : entries)
              index->insert(old, reuse_layout);
          }
        }
      }
    return result;
  } catch (...) {
    entries.clear();
    linear->clear();
    reset_observations();
    throw;
  }
}
}

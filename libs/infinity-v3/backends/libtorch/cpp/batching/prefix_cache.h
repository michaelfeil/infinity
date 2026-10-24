/// \file
/// Prefix identities, retained model snapshots, and per-submission reuse context.
/// Defines cache budgets and the interface used by native model adapters.

#pragma once
#include "batching/packed_input.h"
#include "helpers/math_policy.h"
#include <ATen/Context.h>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

namespace infinity {
class Model;
class PrefixLinearPlans;
class CudaGraphCache;
class PrefixIndex;

// Prepared identity, scoped to one loaded engine and execution operation.
struct PrefixToken {
  std::array<int64_t, 6> values{};
  bool operator==(const PrefixToken&) const = default;
};

/// Media placement, layout, and content identity participating in prefix matching.
struct PrefixMedia {
  int64_t start = 0, count = 0;
  std::vector<int64_t> layout;
  std::vector<uint8_t> content;
  std::array<uint8_t, 32> digest{};
  bool hashed = false;
  bool operator==(const PrefixMedia&) const = default;
};

/// Two-part advisory digest used to rank prefix candidates before exact comparison.
struct PrefixFingerprint {
  uint64_t low = 0, high = 0;
  bool operator==(const PrefixFingerprint&) const = default;
};

/// Token/media identity and context flags used to scope reusable prefix state.
struct PrefixIdentity {
  std::vector<PrefixToken> tokens;
  std::vector<PrefixMedia> media;
  bool image_context = false;
  bool rotary_context = false;
  std::vector<PrefixFingerprint> fingerprints;
};

/// Retained attention, recurrent, and convolution tensors for one model layer.
struct PrefixLayer {
  at::Tensor key, value, recurrent, convolution;
};

/// Shared accounting for live and reserved resident-prefix storage.
struct PrefixBudget {
  uint64_t live_bytes = 0, reserved_bytes = 0;
};

/// Retained per-layer state and output data for one exact input-prefix identity.
struct PrefixEntry {
  PrefixIdentity identity;
  // Full-result reuse retains the original projection/attention batch recipe.
  std::vector<int32_t> batch_offsets;
  int64_t batch_max_sequence = 0, compact_tokens = 0;
  std::map<int64_t, PrefixLayer> layers;
  at::Tensor output;
  at::Tensor slab;
  std::shared_ptr<PrefixBudget> budget;
  uint64_t bytes = 0, reserved_bytes = 0, used = 0;
  int operation = 0;
  // Consumable on the calling thread's stream, including ordered producer work
  // that has not completed globally yet. Independent streams require completion;
  // native engines additionally order all submissions through their own fence.
  std::function<bool()> ready = [] { return true; };
  ~PrefixEntry();
};

/// Submission-local links between logical inputs, cached readers, and new snapshots.
struct PrefixContext {
  const PackedInput& original;
  std::vector<int32_t> offsets;
  std::vector<int64_t> hits;
  std::vector<std::shared_ptr<PrefixEntry>> readers, writers;
  uint64_t max_bytes;
  std::shared_ptr<PrefixBudget> budget;

  // Attention adapters use independently packed Q and K/V boundaries.
  struct KV {
    at::Tensor key, value, cumulative;
    int64_t max_length = 0;
  };

  /// Assemble complete per-sequence K/V from retained prefixes and current suffix rows. Queries
  /// keep suffix boundaries; returned K/V use the original logical cumulative boundaries.
  KV attention(int64_t layer, const at::Tensor& key, const at::Tensor& value, bool capture = true);
  /// Allocate graph-owned full K/V arenas and seed cached prefix rows before suffix writes.
  PrefixLayer& prepare_graph_layer(int64_t layer, const at::Tensor& key, const at::Tensor& value);
  /// Retain a validated suffix-to-full row map for direct writes into graph-owned K/V arenas.
  const at::Tensor& prepare_graph_rows(const c10::Device&);
  /// Borrow a retained layer snapshot while this context keeps its reader entry alive.
  const PrefixLayer* previous(int64_t row, int64_t layer) const;
  /// Store independently owned recurrent state and convolution history at an admitted boundary.
  void checkpoint(int64_t row, int64_t layer, const at::Tensor& state, const at::Tensor& history);
  /// Restore suffix hidden rows to full logical token order, including retained prefix outputs.
  /// Graph collection may return reusable graph-owned storage; consume it before the next replay.
  at::Tensor restore(const at::Tensor& suffix);
  /// Spend reserved admission credit on an owned layer snapshot; return false if capture is
  /// abandoned. Optional copy lists defer data movement so
  /// multiple snapshots can be copied together.
  bool store(int64_t row,
             int64_t layer,
             const PrefixLayer& value,
             std::vector<at::Tensor>* targets = nullptr,
             std::vector<at::Tensor>* sources = nullptr);
  /// Retain only the admitted prefix of full hidden rows, without
  /// keeping the whole batch allocation.
  void store_output(int64_t row,
                    const at::Tensor& full,
                    std::vector<at::Tensor>* targets = nullptr,
                    std::vector<at::Tensor>* sources = nullptr);
  /// Copy replay-owned layer/output views into independent resident
  /// snapshots before buffers are reused.
  void publish_graph(const PrefixContext&);
  PrefixLinearPlans* linear = nullptr;
  MathPolicy math_policy{};
  // Graphs own execution buffers. These references keep full K/V and hidden
  // rows available for batched publication into independent resident snapshots.
  bool collect_graph = false;
  std::vector<std::map<int64_t, PrefixLayer>> graph_layers;
  std::vector<at::Tensor> graph_outputs;
  std::map<int64_t, PrefixLayer> graph_packed_layers;
  at::Tensor graph_hidden;
  at::Tensor graph_input_rows;
};

/// Borrow full request metadata during suffix execution; otherwise return the input itself.
const PackedInput& prefix_logical_input(const PackedInput&);
/// Record full-forward graph outputs when requested, then restore any cached prefix hidden rows.
at::Tensor prefix_restore(const PackedInput&, const at::Tensor&);
void prefix_capture_attention(const PackedInput&,
                              int64_t layer,
                              const at::Tensor& key,
                              const at::Tensor& value);
at::Tensor
prefix_linear(const at::Tensor&, const at::Tensor&, const at::Tensor&, const PackedInput&);
/// Attend suffix queries to full K/V with separate boundaries and the model's selected math recipe.
at::Tensor prefix_attention(const at::Tensor& q,
                            const at::Tensor& k,
                            const at::Tensor& v,
                            const PackedInput&,
                            int64_t layer,
                            double scale,
                            int64_t left = -1,
                            int64_t right = -1,
                            bool exact = false,
                            bool capture = true,
                            bool cudnn = false);

/// Recurrent-history and checkpoint metadata for uncached-suffix execution.
struct PrefixRecurrent {
  at::Tensor initial_sources, history, history_lengths;
  at::Tensor checkpoint_positions, checkpoint_rows, checkpoints;
  std::vector<int64_t> rows;
};

/// Prepare retained recurrent-state pointers, bounded convolution history, and checkpoint
/// destinations. The context's reader entries must outlive device
/// use of the uploaded state pointers.
PrefixRecurrent prefix_recurrent(
    const PackedInput&, const at::Tensor& qkv, int64_t layer, int64_t heads, int64_t kernel);
/// Publish captured FP32 recurrent states and the last kernel-1
/// convolution rows for admitted writers.
void prefix_recurrent_commit(const PackedInput&,
                             const at::Tensor& qkv,
                             int64_t layer,
                             int64_t kernel,
                             const PrefixRecurrent&);

/// Coordinates admission and reuse of bounded, engine-scoped prefix snapshots.
class PrefixCache {
public:
  // EngineOrdered requires GPU ordering between every run submission. Host
  // serialization alone is insufficient; independent callers retain events.
  enum class StreamOrder { Independent, EngineOrdered };

  /// Prefix reuse, admission, eviction, and bypass counters.
  struct Stats {
    uint64_t hits = 0, misses = 0, reused_tokens = 0, admissions = 0, evictions = 0, bypasses = 0;
  };

  explicit PrefixCache(uint64_t bytes,
                       int64_t max_prefix = 8192,
                       size_t entries = 64,
                       size_t admission_uses = 1,
                       StreamOrder order = StreamOrder::Independent);
  ~PrefixCache();
  /// Serialize lookup/admission and execute full or suffix inference for the selected operation.
  /// Model identity and math policy scope reuse; stream-order
  /// guarantees follow the constructor mode.
  at::Tensor
  run(const Model&, const PackedInput&, int operation, CudaGraphCache* fallback = nullptr);
  /// Drop cache ownership and observations; snapshots still held
  /// by readers retain their byte charges.
  void clear();
  Stats stats() const;
  uint64_t bytes() const;

private:
  mutable std::mutex mutex;
  uint64_t limit, clock = 0;
  int64_t maximum;
  size_t capacity;
  std::shared_ptr<PrefixBudget> budget = std::make_shared<PrefixBudget>();
  std::vector<std::shared_ptr<PrefixEntry>> entries;
  std::unique_ptr<PrefixIndex> index;
  static constexpr size_t index_minimum = 256;
  Stats counters;
  uint64_t model_epoch = 0;
  MathPolicy math_policy{};
  std::unique_ptr<PrefixLinearPlans> linear;

  /// Reuse count for a recently observed prefix admission candidate.
  struct Recent {
    PrefixFingerprint key;
    size_t uses = 0;
  };

  std::array<Recent, 64> recent{};

  /// Bounded frequency observations for short-prefix admission candidates.
  struct ShortRecent {
    /// One fingerprint observation with its use count and recency marker.
    struct Counter {
      uint64_t key = 0;
      size_t uses = 0;
      uint64_t used = 0;
    };

    std::array<std::vector<Counter>, 64> depths;
    std::vector<Counter> extended;
    size_t cells = 64;
    uint64_t clock = 0;
  };

  std::unique_ptr<ShortRecent> short_recent;

  /// Cached host/device sequence boundaries and their consumption callback.
  struct Boundaries {
    std::vector<int32_t> offsets;
    at::Tensor host, device;
    std::function<void()> consume;
    uint64_t used = 0;
  };

  std::vector<Boundaries> boundaries;
  uint64_t boundary_clock = 0;
  size_t recent_next = 0, admission_uses;
  bool ordered_submissions;
  bool admit(const PrefixIdentity&, int operation);
  bool admit(PrefixFingerprint);
  int64_t observe_text(std::span<const PrefixToken>,
                       int operation,
                       bool rotary_context,
                       int64_t known_prefix = 0,
                       int64_t observed = -1);
  void reset_observations();
  at::Tensor cumulative(const std::vector<int32_t>&, const c10::Device&);
};
}

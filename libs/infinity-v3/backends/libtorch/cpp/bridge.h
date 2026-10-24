/// \file
/// C ABI declarations mirrored by the Rust LibTorch adapter.
/// Input descriptors borrow caller storage; engine and output handles have explicit
/// destruction entry points.

#pragma once
#include <stdint.h>
#include <stddef.h>

// C ABI mirrored by the repr(C) descriptors in ../src/lib.rs and ../src/radix.rs.
// Keep field order and integer widths in sync when changing a descriptor.
struct TeiConfig {
  int64_t hidden, heads, layers, intermediate, vocab, positions, types;
  double epsilon;
  int32_t activation;
};

/// Borrowed image-patch matrix with packed-token placement and optional cache identity.
struct TeiImage {
  const float* pixels;
  int64_t rows, patch_dim, grid[3], merge_size, token_start, token_count, sequence_start;
  uint8_t feature_key[32]{};
  uint8_t has_feature_key = 0;
};

/// Borrowed audio features and validity mask with packed-token placement.
struct TeiAudio {
  const float* values;
  const uint8_t* mask;
  int64_t frames, feature_size, token_start, token_count;
};

/// Borrowed compact token rows and maps between compact and logical token layouts.
struct TeiRadix {
  const int64_t* ids;
  const int64_t* types;
  const int64_t* positions;
  const int64_t* scatter;
  const int64_t* fold;
  int64_t compact_tokens;
};

/// Question and option spans for one field of a structured decision request.
struct TeiDecisionField {
  int64_t kind, question_start, question_end;
  const int64_t* options;
  size_t option_count;
};

/// Counted, borrowed metadata selecting a supported decision-head request contract.
struct TeiDecision {
  // kind 4 is block option-token scoring; question_type holds its prefix length.
  int32_t kind;
  int64_t question_type;
  const int64_t* markers;
  size_t marker_count;
  const int64_t* token_ids;
  size_t token_count;
  const TeiDecisionField* fields;
  size_t field_count;
};

// A tensor range from a Rust-validated safetensors header, ordered by offset.
struct TeiWeight {
  const char* name;
  const int64_t* shape;
  size_t rank;
  int32_t dtype;
  uint64_t offset;
  uint64_t length;
};

// Every fallible entry point catches C++ exceptions; the error string is thread-local.
extern "C" {
/// Borrow the calling thread's last error text; copy it before another fallible native call.
const char* tei_error();
// Thread-local status of the last fallible call: 0 success, 1 other error,
// 2 memory admission rejection, 3 recoverable allocator OOM. Read immediately
// on the calling thread. Other device failures retain kind 1.
int32_t tei_last_error_kind();
int64_t tei_device_count(const char* device);
// Synchronous CUDA allocation/add/readback check; CPU is a no-op.
int32_t tei_device_preflight(const char* device);
/// Create an owned engine handle, or return null with thread-local error details. Load options and
/// weights, call tei_ready(), then release the handle with tei_destroy().
void* tei_create(const TeiConfig*, const char* device, int32_t dtype);
/// Copy the supplied checkpoint bytes into engine-owned storage; success is 0, failure is -1.
int32_t tei_weight(
    void*, const char* name, const uint8_t* data, const int64_t* shape, size_t rank, int32_t dtype);
#ifndef _WIN32
int32_t tei_load_safetensors(void*,
                             int fd,
                             uint64_t data_start,
                             uint64_t data_length,
                             const TeiWeight*,
                             size_t count,
                             size_t threads,
                             size_t buffer_bytes,
                             int32_t direct_io);
#endif
int32_t tei_option(void*, const char* key, const char* value);
int64_t tei_output_width(void*);
int64_t tei_pooled_width(void*);
int64_t tei_classification_width(void*);
int64_t tei_graph_count(void*);
// hits, misses, captures, attempts, evictions, live bytes, cumulative admitted bytes.
int32_t tei_graph_stats(void*, uint64_t* counters, size_t count);
// Operation 0: hidden states, 1: sequence classification, 2: token classification.
int32_t tei_prefix_cache_supported(void*, int32_t operation);
// hits, misses, reused tokens, admissions, evictions, bypasses, live bytes.
int32_t tei_prefix_cache_stats(void*, uint64_t* counters, size_t count);
int32_t tei_prefix_cache_clear(void*);
// Set probe bypass and return its previous value, or -1 on error.
int32_t tei_prefix_cache_set_probe(void*, int32_t enabled);
int32_t tei_supports_radix(void*);
/// Prepare loaded weights and model/cache state; rebuilding drains previous engine submissions.
int32_t tei_ready(void*);
/// Validate one decision descriptor per requested count and fill caller-owned logit-count storage.
int32_t tei_decision_counts(void*, const TeiDecision*, size_t count, int64_t* counts);
/// Synchronously write concatenated request logits and one action probability per sequence.
/// Capacity counts logit floats; inputs and output buffers remain valid throughout the call.
int32_t tei_decide(void*,
                   const int64_t* ids,
                   const int64_t* types,
                   const int64_t* positions,
                   const int32_t* cumulative,
                   int64_t batch,
                   int64_t max_sequence,
                   const TeiDecision*,
                   float* logits,
                   size_t capacity,
                   float* actions,
                   const int64_t* media_positions,
                   const TeiImage* images,
                   size_t image_count,
                   const TeiAudio* audios,
                   size_t audio_count);
/// Perform synchronous decisions with optional compact
/// causal-prefix maps; null maps disable sharing.
int32_t tei_decide_radix(void*,
                         const int64_t* ids,
                         const int64_t* types,
                         const int64_t* positions,
                         const int32_t* cumulative,
                         int64_t batch,
                         int64_t max_sequence,
                         const TeiDecision*,
                         float* logits,
                         size_t capacity,
                         float* actions,
                         const int64_t* media_positions,
                         const TeiImage* images,
                         size_t image_count,
                         const TeiAudio* audios,
                         size_t audio_count,
                         const TeiRadix* radix);
/// Synchronously write pooled selections first, then raw token spans, preserving selection order.
/// Capacity counts float elements, including repetitions; input
/// boundaries delimit unpadded sequences.
int32_t tei_forward(void*,
                    const int64_t* ids,
                    const int64_t* types,
                    const int64_t* positions,
                    const int32_t* cumulative,
                    int64_t batch,
                    int64_t max_sequence,
                    int32_t pool,
                    const int64_t* pooled,
                    size_t pooled_count,
                    const int64_t* raw,
                    size_t raw_count,
                    float* output,
                    size_t capacity,
                    const int64_t* media_positions,
                    const TeiImage* images,
                    size_t image_count,
                    const TeiAudio* audios,
                    size_t audio_count);
// A separate entry point preserves the existing packed C ABI for callers that
// do not provide radix maps. Maps never contain padded or unused compact rows.
int32_t tei_forward_radix(void*,
                          const int64_t* ids,
                          const int64_t* types,
                          const int64_t* positions,
                          const int32_t* cumulative,
                          int64_t batch,
                          int64_t max_sequence,
                          int32_t pool,
                          const int64_t* pooled,
                          size_t pooled_count,
                          const int64_t* raw,
                          size_t raw_count,
                          float* output,
                          size_t capacity,
                          const int64_t* media_positions,
                          const TeiImage* images,
                          size_t image_count,
                          const TeiAudio* audios,
                          size_t audio_count,
                          const TeiRadix* radix);
// Returns an owned output lease, or nullptr on failure. capacity counts floats.
// Inputs are staged during this call; caller storage can be released on return.
// Wait before reading results, then destroy the lease after copying them out.
void* tei_forward_submit(void* handle,
                         const int64_t* ids,
                         const int64_t* types,
                         const int64_t* positions,
                         const int32_t* cumulative,
                         int64_t batch,
                         int64_t max_sequence,
                         int32_t pool,
                         const int64_t* pooled,
                         size_t pooled_count,
                         const int64_t* raw,
                         size_t raw_count,
                         size_t capacity,
                         const int64_t* media_positions,
                         const TeiImage* images,
                         size_t image_count,
                         const TeiAudio* audios,
                         size_t audio_count,
                         const TeiRadix* radix);
// Owned output lease: wait and construct host results before destroy.
const float* tei_output_wait(void*);
/// Release an owned output lease after consuming its results;
/// unfinished transfers drain on destruction.
void tei_output_destroy(void*);
/// Destroy an owned engine handle after callers stop using it, draining pending producer work.
void tei_destroy(void*);
}

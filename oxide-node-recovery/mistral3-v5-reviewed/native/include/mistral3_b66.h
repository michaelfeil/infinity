/*
 * mistral3_b66.h - source-only Root packet C ABI for Mistral3 + Pixtral.
 *
 * This header is a fixed-width, C-compatible description of the packet that
 * wraps the frozen b66 libtei_torch C entry points (tei_create, tei_option,
 * tei_load_safetensors, tei_ready, tei_forward, tei_forward_submit,
 * tei_output_wait, tei_output_destroy, tei_destroy). Root compiles this header
 * and native/src/mistral3_runner.cpp later inside the GPU/Torch image.
 *
 * Nothing here was compiled, linked, loaded or executed during the CPU-only
 * control. The header contains no Torch, ATen, c10, CUDA or C++ types, and it
 * does not itself initialize CUDA. A consumer must set CUDA_VISIBLE_DEVICES
 * explicitly for a real GPU run; the bundled runner refuses an empty guard.
 *
 * Image pixels are caller-owned RGB patch rows in float32, row-major, exactly
 * rows * patch_dim values. Token buffers are caller-owned, concatenated in
 * sequence order. All descriptors are borrowed for the duration of a
 * synchronous call; deferred submission stages them before returning.
 */
#ifndef MISTRAL3_B66_H
#define MISTRAL3_B66_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MISTRAL3_B66_ABI_VERSION 1u

/* Mirrors the b66 scalar dtype selector: 0=f32, 1=f16, 2=bf16. */
enum {
    MISTRAL3_B66_DTYPE_F32 = 0,
    MISTRAL3_B66_DTYPE_F16 = 1,
    MISTRAL3_B66_DTYPE_BF16 = 2
};

/* Pool selector values accepted by the b66 forward ABI. */
enum {
    MISTRAL3_B66_POOL_FIRST = 0,
    MISTRAL3_B66_POOL_MEAN = 1,
    MISTRAL3_B66_POOL_LAST = 2,
    MISTRAL3_B66_POOL_CLASSIFICATION = 4
};

typedef void* mistral3_b66_handle;
typedef struct mistral3_b66_output_lease* mistral3_b66_output;

/*
 * One image in packed-token layout.
 *
 * grid_thw is [frames, patch_rows, patch_columns] and must have frames == 1.
 * patch_dim is 3 * patch_size * patch_size (588 for patch_size 14).
 * rows is the patch count (grid_thw[1] * grid_thw[2]).
 * pixels points at rows * patch_dim float32 values.
 * pixel_count, when nonzero, must equal rows * patch_dim; zero means derive it.
 * token_start is the first packed-token index replaced by projected features.
 * token_count is (grid_thw[1] / merge_size) * (grid_thw[2] / merge_size).
 * sequence_start is the first token index of the image's sequence.
 */
typedef struct mistral3_b66_image {
    int64_t grid_thw[3];
    int64_t patch_dim;
    int64_t rows;
    int64_t merge_size;
    const float* pixels;
    uint64_t pixel_count;
    int64_t token_start;
    int64_t token_count;
    int64_t sequence_start;
    uint8_t feature_key[32];
    uint8_t has_feature_key;
} mistral3_b66_image;

/*
 * One packed forward request. ids/types/positions contain
 * cumulative[batch] entries. cumulative has batch + 1 int32 offsets.
 * pool uses MISTRAL3_B66_POOL_*; pool == 4 emits classification scores.
 * pooled/raw are row indices into the packed batch, preserving caller order.
 * output_capacity counts float elements.
 */
typedef struct mistral3_b66_text_request {
    const int64_t* ids;
    const int64_t* types;
    const int64_t* positions;
    const int32_t* cumulative;
    int64_t batch;
    int64_t max_sequence;
    int32_t pool;
    const int64_t* pooled;
    size_t pooled_count;
    const int64_t* raw;
    size_t raw_count;
    size_t output_capacity;
} mistral3_b66_text_request;

/*
 * One safetensors range. offset is relative to the shard payload start
 * (8 + header_bytes), exactly as stored in data_offsets. length is the exact
 * payload byte length. name is a NUL-terminated UTF-8 tensor name.
 */
typedef struct mistral3_b66_weight {
    const char* name;
    const int64_t* shape;
    size_t rank;
    int32_t dtype;
    uint64_t offset;
    uint64_t length;
} mistral3_b66_weight;

/* Monotonic nanosecond counters and output metadata for one engine. */
typedef struct mistral3_b66_perf_counters {
    uint64_t create_ns;
    uint64_t option_ns;
    uint64_t load_ns;
    uint64_t ready_ns;
    uint64_t forward_sync_ns;
    uint64_t forward_submit_ns;
    uint64_t output_wait_ns;
    uint64_t output_destroy_ns;
    uint64_t destroy_ns;
    uint64_t forward_calls;
    uint64_t deferred_calls;
    uint64_t last_output_elements;
    uint64_t last_output_checksum;
    uint64_t min_forward_ns;
    uint64_t max_forward_ns;
    uint64_t total_forward_ns;
} mistral3_b66_perf_counters;

/*
 * Create an engine for the checkpoint at a directory path. device is a b66
 * device string such as "cuda:0"; dtype is MISTRAL3_B66_DTYPE_*. Returns NULL
 * on failure; use mistral3_b66_last_error().
 */
mistral3_b66_handle mistral3_b66_create(const char* checkpoint,
                                        const char* device,
                                        int32_t dtype);

/* Set one flattened config option before ready(). Returns 0 or -1. */
int32_t mistral3_b66_option(mistral3_b66_handle handle,
                            const char* key,
                            const char* value);

/*
 * Open one safetensors shard, validate its header-derived data start/length,
 * and hand every descriptor to the frozen b66 reader. The descriptor array
 * must be ordered by ascending offset. Returns 0 or -1.
 */
int32_t mistral3_b66_load_safetensors(mistral3_b66_handle handle,
                                      const char* shard_path,
                                      const mistral3_b66_weight* weights,
                                      size_t count,
                                      size_t threads,
                                      size_t buffer_bytes,
                                      int32_t direct_io);

/* Build model/caches from the loaded config and all tensors. Returns 0 or -1. */
int32_t mistral3_b66_ready(mistral3_b66_handle handle);

/*
 * Synchronous image+text forward. output must hold output_capacity floats and
 * the capacity must be at least the exact requested element count. Returns 0
 * or -1.
 */
int32_t mistral3_b66_forward(mistral3_b66_handle handle,
                             const mistral3_b66_text_request* request,
                             const mistral3_b66_image* images,
                             size_t image_count,
                             float* output);

/*
 * Deferred forward. Inputs are staged before return; output_capacity must be
 * exactly the requested element count. Returns an owned lease or NULL.
 */
mistral3_b66_output mistral3_b66_forward_submit(
    mistral3_b66_handle handle,
    const mistral3_b66_text_request* request,
    const mistral3_b66_image* images,
    size_t image_count);

/* Wait for one lease and borrow its host output until output_destroy(). */
const float* mistral3_b66_output_wait(mistral3_b66_output output);

/* Number of float elements in a completed lease; returns 0 or -1. */
int32_t mistral3_b66_output_elements(mistral3_b66_output output,
                                     size_t* elements);

/* Release one lease after its results are consumed. */
void mistral3_b66_output_destroy(mistral3_b66_output output);

/* Copy the engine's timing counters. Returns 0 or -1. */
int32_t mistral3_b66_perf(mistral3_b66_handle handle,
                          mistral3_b66_perf_counters* counters);

/* Reset engine timing counters. Returns 0 or -1. */
int32_t mistral3_b66_reset_perf(mistral3_b66_handle handle);

/* Last thread-local error text; valid until the next fallible call. */
const char* mistral3_b66_last_error(void);

/* Last thread-local error kind: 0 success, 1 error, 2/3 memory admission. */
int32_t mistral3_b66_last_error_kind(void);

/* Destroy the engine and drain any outstanding work. */
void mistral3_b66_destroy(mistral3_b66_handle handle);

#ifdef __cplusplus
}
#endif

#endif /* MISTRAL3_B66_H */

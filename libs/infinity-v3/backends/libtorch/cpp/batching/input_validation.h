// SPDX-License-Identifier: Apache-2.0
/// \file
/// Structural validation and upload-size accounting for native request descriptors.
/// Checks counted pointers, packed boundaries, media layouts, and decision metadata.

#pragma once
#include "helpers/admission_error.h"
#include "bridge.h"
#include "batching/packed_input.h"
#include "models/decision_models.h"
#include <c10/util/Exception.h>
#include <algorithm>
#include <limits>

namespace tei {
inline size_t validate_output_indices(const int32_t* cumulative,
                                      int64_t batch,
                                      const int64_t* pooled,
                                      size_t pooled_count,
                                      const int64_t* raw,
                                      size_t raw_count) {
  TORCH_CHECK(!pooled_count || pooled, "Missing counted pooled indices");
  TORCH_CHECK(!raw_count || raw, "Missing counted raw indices");
  TORCH_CHECK(pooled_count <= std::numeric_limits<size_t>::max() / sizeof(int64_t) &&
                  raw_count <= std::numeric_limits<size_t>::max() / sizeof(int64_t),
              "Overflowing output index count");
  for (size_t i = 0; i < pooled_count; ++i)
    TORCH_CHECK(pooled[i] >= 0 && pooled[i] < batch, "Invalid pooled output index");
  size_t rows = 0;
  for (size_t i = 0; i < raw_count; ++i) {
    TORCH_CHECK(raw[i] >= 0 && raw[i] < batch, "Invalid raw output index");
    const size_t length = int64_t(cumulative[raw[i] + 1]) - cumulative[raw[i]];
    TORCH_CHECK(length <= std::numeric_limits<size_t>::max() - rows,
                "Raw output row count overflow");
    rows += length;
  }
  return rows;
}

inline size_t expected_output_elements(size_t pooled_count,
                                       size_t raw_rows,
                                       int64_t pooled_width,
                                       int64_t raw_width) {
  auto product = [](size_t count, int64_t width) {
    if (!count)
      return size_t(0);
    TORCH_CHECK(width > 0 &&
                    static_cast<uint64_t>(width) <= std::numeric_limits<size_t>::max() / count,
                "Native output size overflow");
    return count * static_cast<size_t>(width);
  };
  const auto pooled = product(pooled_count, pooled_width), raw = product(raw_rows, raw_width);
  TORCH_CHECK(raw <= std::numeric_limits<size_t>::max() - pooled, "Native output size overflow");
  const auto elements = pooled + raw;
  TORCH_CHECK(elements <= std::numeric_limits<size_t>::max() / sizeof(float),
              "Native output byte size overflow");
  return elements;
}

inline int64_t input_product(int64_t left, int64_t right, const char* name) {
  TORCH_CHECK(left > 0 && right > 0 && left <= std::numeric_limits<int64_t>::max() / right,
              "Invalid or overflowing ",
              name,
              " dimensions");
  return left * right;
}

inline void input_bytes(int64_t elements, size_t element_size, const char* name) {
  TORCH_CHECK(elements > 0 &&
                  static_cast<uint64_t>(elements) <=
                      static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / element_size &&
                  static_cast<uint64_t>(elements) <=
                      std::numeric_limits<size_t>::max() / element_size,
              "Overflowing ",
              name,
              " byte size");
}

// Validate descriptors without constructing tensors or reading media payloads.
// This is structural admission; a request memory budget is a separate layer.
inline int64_t validate_input_descriptors(const int64_t* ids,
                                          const int64_t* types,
                                          const int64_t* positions,
                                          const int32_t* cumulative,
                                          int64_t batch,
                                          int64_t max_sequence,
                                          const int64_t* media_positions,
                                          const TeiImage* images,
                                          size_t image_count,
                                          const TeiAudio* audios,
                                          size_t audio_count,
                                          bool single_frame_images = false) {
  TORCH_CHECK(batch > 0 && batch < std::numeric_limits<int64_t>::max() && cumulative,
              "Invalid native packed batch");
  input_bytes(batch + 1, sizeof(int32_t), "cumulative offsets");
  TORCH_CHECK(cumulative[0] == 0 && max_sequence > 0, "Invalid native packed sequence geometry");
  for (int64_t row = 0; row < batch; ++row) {
    const int64_t start = cumulative[row], end = cumulative[row + 1];
    TORCH_CHECK(start >= 0 && end > start && end - start <= max_sequence,
                "Native packed sequences must be nonempty and fit max_sequence");
  }
  const int64_t tokens = cumulative[batch];
  TORCH_CHECK(max_sequence <= tokens, "Native maximum sequence exceeds packed token count");
  input_bytes(tokens, sizeof(int64_t), "token buffer");
  TORCH_CHECK(ids && types && positions, "Missing native token buffer");
  if (media_positions)
    input_bytes(input_product(3, tokens, "media positions"), sizeof(int64_t), "media positions");
  TORCH_CHECK(!image_count || images, "Missing counted image descriptors");
  TORCH_CHECK(!audio_count || audios, "Missing counted audio descriptors");
  TORCH_CHECK(image_count <= std::numeric_limits<size_t>::max() / sizeof(TeiImage) &&
                  audio_count <= std::numeric_limits<size_t>::max() / sizeof(TeiAudio),
              "Overflowing media descriptor count");
  auto span = [&](int64_t start, int64_t count, const char* name) {
    TORCH_CHECK(start >= 0 && start < tokens && count > 0 && count <= tokens - start,
                "Invalid ",
                name,
                " token span");
    const auto* end = std::upper_bound(cumulative, cumulative + batch + 1, start);
    TORCH_CHECK(count <= int64_t(*end) - start,
                "Media token span crosses packed sequence boundary");
    return int64_t(*(end - 1));
  };
  for (size_t i = 0; i < image_count; ++i) {
    const auto& item = images[i];
    TORCH_CHECK(item.pixels && item.merge_size > 0, "Invalid image payload or merge size");
    const auto rows = input_product(
        input_product(item.grid[0], item.grid[1], "image grid"), item.grid[2], "image grid");
    TORCH_CHECK(!single_frame_images || item.grid[0] == 1, "Gemma image grid must have one frame");
    TORCH_CHECK(item.rows == rows && item.grid[1] % item.merge_size == 0 &&
                    item.grid[2] % item.merge_size == 0,
                "Image rows or merge geometry do not match grid");
    input_bytes(
        input_product(item.rows, item.patch_dim, "image payload"), sizeof(float), "image payload");
    const auto merged_rows = input_product(
        input_product(item.grid[0], item.grid[1] / item.merge_size, "merged image grid"),
        item.grid[2] / item.merge_size,
        "merged image grid");
    TORCH_CHECK(item.token_count == merged_rows, "Image token count does not match merged grid");
    const auto sequence_start = span(item.token_start, item.token_count, "image");
    TORCH_CHECK(item.sequence_start == sequence_start,
                "Image sequence start does not match containing packed sequence");
  }
  for (size_t i = 0; i < audio_count; ++i) {
    const auto& item = audios[i];
    TORCH_CHECK(item.values && item.mask, "Missing audio features or validity mask");
    input_bytes(input_product(item.frames, item.feature_size, "audio payload"),
                sizeof(float),
                "audio payload");
    input_bytes(item.frames, sizeof(uint8_t), "audio validity mask");
    span(item.token_start, item.token_count, "audio");
  }
  return tokens;
}

// Bound real uploaded buffers and temporary F32-to-model image conversion.
// This excludes model activations/workspaces and the separate output pool.
struct InputUploadBudget {
  size_t limit, bytes = 0;

  void add(size_t count, size_t width) {
    require_memory_admission(limit > 0 && bytes <= limit && width > 0 && count <= limit / width &&
                                 count * width <= limit - bytes,
                             "Native input upload exceeds its memory budget");
    bytes += count * width;
  }
};

inline size_t validate_input_upload_budget(int64_t tokens,
                                           int64_t batch,
                                           bool media_positions,
                                           const TeiImage* images,
                                           size_t image_count,
                                           const TeiAudio* audios,
                                           size_t audio_count,
                                           const TeiRadix* radix,
                                           size_t image_element_size,
                                           size_t limit,
                                           bool normalize_images = false,
                                           bool deduplicate_images = false) {
  TORCH_CHECK(limit > 0, "Native input upload budget must be positive");
  InputUploadBudget budget{limit};
  auto add = [&](size_t count, size_t width) { budget.add(count, width); };
  add(tokens, 3 * sizeof(int64_t));
  add(batch + 1, sizeof(int32_t));
  if (media_positions)
    add(tokens, 3 * sizeof(int64_t));
  // push_back vectors can briefly retain old and replacement capacities.
  add(image_count, 3 * sizeof(ImageInput));
  add(audio_count, 3 * sizeof(AudioInput));
  std::vector<ImageUploadLayout> charged_images;
  if (deduplicate_images) {
    add(image_count,
        3 * sizeof(ImageUploadLayout)); // Admission and request-local cache bookkeeping.
    charged_images.reserve(image_count);
  }
  for (size_t i = 0; i < image_count; ++i) {
    const auto count = input_product(images[i].rows, images[i].patch_dim, "image payload");
    if (deduplicate_images) {
      const ImageUploadLayout key{images[i].pixels,
                                  at::kFloat,
                                  images[i].rows,
                                  images[i].patch_dim,
                                  images[i].patch_dim,
                                  1};
      if (std::find(charged_images.begin(), charged_images.end(), key) != charged_images.end())
        continue;
      charged_images.push_back(key);
    }
    add(count, sizeof(float));
    if (normalize_images)
      add(count, 2 * sizeof(float));
    if (image_element_size != sizeof(float))
      add(count, image_element_size);
  }
  for (size_t i = 0; i < audio_count; ++i) {
    add(input_product(audios[i].frames, audios[i].feature_size, "audio payload"), sizeof(float));
    add(audios[i].frames, sizeof(uint8_t));
  }
  if (radix) {
    TORCH_CHECK(radix->compact_tokens > 0 && radix->compact_tokens <= tokens && radix->ids &&
                    radix->types && radix->positions && radix->scatter && radix->fold,
                "Invalid native radix buffers");
    add(radix->compact_tokens, 4 * sizeof(int64_t));
    add(tokens, sizeof(int64_t));
    // validate_radix owns two compact parent/context arrays on the host.
    add(radix->compact_tokens, 2 * sizeof(int64_t));
    if (image_count || media_positions) {
      add(batch, sizeof(std::vector<const TeiImage*>));
      add(tokens, sizeof(int64_t));
      add(image_count, 3 * sizeof(const TeiImage*));
      // At most one checked context pair per full token, including tree nodes.
      add(tokens, sizeof(std::pair<int64_t, int64_t>) + 6 * sizeof(void*));
    }
  }
  return budget.bytes;
}

inline size_t validate_decision_metadata_budget(const TeiDecision* requests,
                                                size_t count,
                                                size_t limit,
                                                size_t used = 0) {
  InputUploadBudget budget{limit, used};
  budget.add(count, sizeof(DecisionRequest));
  TORCH_CHECK(requests || !count, "Missing decision metadata buffer");
  for (size_t i = 0; i < count; ++i) {
    const auto& request = requests[i];
    budget.add(request.marker_count, sizeof(int64_t));
    budget.add(request.token_count, sizeof(int64_t));
    budget.add(request.field_count, 3 * sizeof(DecisionField));
    TORCH_CHECK((!request.marker_count || request.markers) &&
                    (!request.token_count || request.token_ids) &&
                    (!request.field_count || request.fields),
                "Missing decision metadata values");
    for (size_t j = 0; j < request.field_count; ++j) {
      const auto& field = request.fields[j];
      budget.add(field.option_count, 3 * sizeof(std::pair<int64_t, int64_t>));
      TORCH_CHECK(!field.option_count || field.options, "Missing Clef option spans");
    }
  }
  return budget.bytes;
}
}

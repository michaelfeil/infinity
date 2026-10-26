/// \file
/// Packed token/media descriptors and request-local image reuse for native inference.
/// Host views, sequence offsets, and prefix contexts are borrowed for the submission;
/// callers must keep their backing data alive while native code uses it.

#pragma once
#include <ATen/ATen.h>
#include <array>
#include <vector>
#include <string_view>
#include <unordered_map>
#include <algorithm>
#include <memory>

namespace infinity {
// Native images are contiguous F32 matrices. Admission and the upload cache
// must identify their storage/layout without reading any pixel values.
struct ImageUploadLayout {
  const void* data;
  at::ScalarType dtype;
  int64_t rows, columns, row_stride, column_stride;
  bool operator==(const ImageUploadLayout&) const = default;
};

/// Build a storage/layout identity without hashing pixels; the matrix remains borrowed.
inline ImageUploadLayout image_upload_layout(const at::Tensor& host) {
  TORCH_CHECK(host.dim() == 2, "Native image upload requires a matrix");
  return {host.const_data_ptr(),
          host.scalar_type(),
          host.size(0),
          host.size(1),
          host.stride(0),
          host.stride(1)};
}

// Gemma normalizes F32 pixels first; other towers receive the model dtype directly.
inline at::Tensor upload_image_pixels(const at::Tensor& pixels,
                                      const c10::Device& device,
                                      at::ScalarType dtype,
                                      std::string_view family) {
  if (family == "gemma4" || family == "gemma4_text" || family == "embedding_gemma2")
    return pixels;
  return pixels.to(device).to(dtype);
}

// The caller's pixels are immutable for one native request. Keep this cache local
// to packed_input: no borrowed host view or cache entry survives that call.
class RequestImageUploads {
  /// Host tensor metadata paired with its request-local upload result.
  struct Entry {
    at::Tensor host, uploaded;
  };

  std::unordered_map<const void*, std::vector<Entry>> entries;

public:
  /// Reuse an upload only for the same host pointer, dtype, shape, and strides. The caller must
  /// keep host pixels immutable throughout this request.
  template <class Upload> at::Tensor get(const at::Tensor& host, Upload&& upload) {
    auto& matches = entries[host.const_data_ptr()];
    for (const auto& entry : matches)
      if (host.dim() == 2 && entry.host.dim() == 2
              ? image_upload_layout(entry.host) == image_upload_layout(host)
              : entry.host.scalar_type() == host.scalar_type() &&
                    entry.host.sizes() == host.sizes() && entry.host.strides() == host.strides())
        return entry.uploaded;
    auto result = upload();
    matches.push_back({host, result});
    return result;
  }
};

struct PrefixContext;

/// Image patches, grid metadata, token placement, and optional prepared vision features.
struct ImageInput {
  at::Tensor pixels;
  std::array<int64_t, 3> grid_thw;
  int64_t merge_size = 1, token_start = 0, token_count = 0, sequence_start = 0;
  std::array<uint8_t, 32> feature_key{};
  bool has_feature_key = false;
  at::Tensor prepared_features{};
};

// Vision depends on pixels and layout, not the question's packed token offset.
// Tensor identity comes from the request-local upload cache; no content hash,
// caller feature key, or cross-request reuse can alias distinct media.
class RequestImageFeatures {
  /// Image identity, cached features, and remaining uses within this request.
  struct Entry {
    ImageInput image;
    at::Tensor features;
    size_t remaining;
  };

  std::unordered_map<const at::TensorImpl*, std::vector<Entry>> entries;

public:
  /// Count uses of each tensor/layout pair so features can be released after their last use.
  explicit RequestImageFeatures(const std::vector<ImageInput>& images, bool enabled = true) {
    if (!enabled)
      return;
    for (const auto& image : images) {
      auto& matches = entries[image.pixels.unsafeGetTensorImpl()];
      auto match = std::find_if(matches.begin(), matches.end(), [&](const Entry& entry) {
        return entry.image.grid_thw == image.grid_thw && entry.image.merge_size == image.merge_size;
      });
      if (match == matches.end())
        matches.push_back({image, {}, 1});
      else
        ++match->remaining;
    }
  }

  /// Compute distinct cache misses together and return features in input image order. The callback
  /// returns one tensor per missing image; duplicate layouts share its result.
  template <class Compute>
  std::vector<at::Tensor> get_batch(const std::vector<ImageInput>& images, Compute compute) {
    std::vector<ImageInput> missing;
    std::vector<at::Tensor> values(images.size());
    std::vector<size_t> slots(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
      const auto& image = images[i];
      auto found = entries.find(image.pixels.unsafeGetTensorImpl());
      if (found != entries.end())
        for (const auto& entry : found->second)
          if (entry.remaining && entry.image.grid_thw == image.grid_thw &&
              entry.image.merge_size == image.merge_size) {
            values[i] = entry.features;
            break;
          }
      if (values[i].defined())
        continue;
      auto match = std::find_if(missing.begin(), missing.end(), [&](const ImageInput& other) {
        return image.pixels.is_same(other.pixels) && image.grid_thw == other.grid_thw &&
               image.merge_size == other.merge_size;
      });
      slots[i] = match - missing.begin();
      if (match == missing.end())
        missing.push_back(image);
    }
    auto computed = missing.empty() ? std::vector<at::Tensor>{} : compute(missing);
    TORCH_CHECK(computed.size() == missing.size(), "Image feature batch size mismatch");
    for (size_t i = 0; i < images.size(); ++i)
      values[i] =
          get(images[i], [&] { return values[i].defined() ? values[i] : computed[slots[i]]; });
    return values;
  }

  /// Consume one registered use, retaining an independent feature allocation only if needed again.
  template <class Compute> at::Tensor get(const ImageInput& image, Compute&& compute) {
    auto found = entries.find(image.pixels.unsafeGetTensorImpl());
    if (found != entries.end())
      for (auto& entry : found->second) {
        if (entry.image.grid_thw != image.grid_thw || entry.image.merge_size != image.merge_size ||
            !entry.remaining)
          continue;
        auto result = entry.features.defined() ? entry.features : compute();
        if (--entry.remaining) {
          // A reused image must not retain the other images' projected features.
          if (!result.is_contiguous() || result.storage_offset() != 0 ||
              result.storage().nbytes() != result.numel() * result.element_size())
            result = result.clone();
          entry.features = result;
        } else
          entry.features = at::Tensor();
        return result;
      }
    return compute();
  }
};

/// Audio features, validity metadata, and token placement for one packed audio span.
struct AudioInput {
  at::Tensor features, validity;
  int64_t token_start = 0, token_count = 0;
  at::Tensor selected_frames{};
};

// Immutable structural buffers prepared outside CUDA graph capture. A retained
// graph owns these boundaries independently of the request that supplied them.
struct BlockLayout {
  at::Tensor prefix_cumulative, suffix_cumulative, kv_cumulative, history_cumulative, cache_sources;
  std::vector<int32_t> history_offsets;
};

// Packed token rows, without padding. For sequence lengths [3, 2], cumulative
// is [0, 3, 5]. These offsets locate sequences in the packed buffer; they are
// separate from model position IDs. offsets is the host view of those boundaries.
struct PackedInput {
  at::Tensor ids, types, positions, cumulative;
  const int32_t* offsets;
  int64_t batch, max_sequence;
  std::vector<ImageInput> images{};
  std::vector<AudioInput> audios{};
  at::Tensor multimodal_positions{};
  // Logical sequences remain full-length for attention and output metadata.
  // Radix maps share only actual token rows with identical causal prefixes.
  at::Tensor compact_ids{}, compact_positions{}, compact_types{};
  at::Tensor scatter_unfold{}, fold_gather{};
  // Complete prefix boundaries and exact-match representatives, request-local only.
  std::vector<int64_t> block_prefix_lengths{}, block_prefix_sources{};
  // Packed suffix rows select immutable prefix KV/recurrent/conv snapshots.
  at::Tensor block_cache_sources{};
  // Execution-only state; original logical request metadata lives in the context.
  PrefixContext* prefix = nullptr;
  // Borrowed full-input host views, valid for this native submission only.
  // Prefix identity must not copy token metadata back from the device.
  at::Tensor host_ids{}, host_types{}, host_positions{}, host_multimodal_positions{},
      host_scatter_unfold{};
  // Read-only full-forward observation during graph capture. This does not
  // select suffix execution or change the model's projection/attention recipe.
  PrefixContext* capture_prefix = nullptr;
  std::shared_ptr<const BlockLayout> block_layout{};
};
}

#pragma once
#include <ATen/ATen.h>
#include <array>
#include <vector>
#include <string_view>
#include <unordered_map>
#include <algorithm>

namespace tei {
// Native images are contiguous F32 matrices. Admission and the upload cache
// must identify their storage/layout without reading any pixel values.
struct ImageUploadLayout {
 const void* data;at::ScalarType dtype;
 int64_t rows,columns,row_stride,column_stride;
 bool operator==(const ImageUploadLayout&) const=default;
};
inline ImageUploadLayout image_upload_layout(const at::Tensor& host) {
 TORCH_CHECK(host.dim()==2,"Native image upload requires a matrix");
 return {host.const_data_ptr(),host.scalar_type(),host.size(0),host.size(1),host.stride(0),host.stride(1)};
}
// Gemma normalizes F32 pixels first; D3 must never round through FP16.
inline at::Tensor upload_image_pixels(const at::Tensor& pixels, const c10::Device& device,
 at::ScalarType dtype, std::string_view family, bool d3) {
 if(family=="gemma4"||family=="gemma4_text"||family=="embedding_gemma2")return pixels;
 const auto image_dtype=d3?at::kBFloat16:family.starts_with("qwen3_5")?at::kHalf:dtype;
 return pixels.to(device).to(image_dtype);
}

// The caller's pixels are immutable for one native request. Keep this cache local
// to packed_input: no borrowed host view or cache entry survives that call.
class RequestImageUploads {
 struct Entry {at::Tensor host,uploaded;};
 std::unordered_map<const void*,std::vector<Entry>> entries;
 public:
 template<class Upload> at::Tensor get(const at::Tensor& host,Upload&& upload) {
  auto& matches=entries[host.const_data_ptr()];
  for(const auto& entry:matches)
   if(host.dim()==2&&entry.host.dim()==2?
      image_upload_layout(entry.host)==image_upload_layout(host):
      entry.host.scalar_type()==host.scalar_type()&&entry.host.sizes()==host.sizes()&&entry.host.strides()==host.strides())return entry.uploaded;
  auto result=upload();matches.push_back({host,result});return result;
 }
};

struct ImageInput {
  at::Tensor pixels;
  std::array<int64_t,3> grid_thw;
  int64_t merge_size=1, token_start=0, token_count=0, sequence_start=0;
  std::array<uint8_t,32> feature_key{};
  bool has_feature_key=false;
  at::Tensor prepared_features{};
};
// Vision depends on pixels and layout, not the question's packed token offset.
// Tensor identity comes from the request-local upload cache; no content hash,
// caller feature key, or cross-request reuse can alias distinct media.
class RequestImageFeatures {
 struct Entry {ImageInput image;at::Tensor features;size_t remaining;};
 std::unordered_map<const at::TensorImpl*,std::vector<Entry>> entries;
 public:
 explicit RequestImageFeatures(const std::vector<ImageInput>& images,bool enabled=true) {
  if(!enabled)return;
  for(const auto& image:images) {
   auto& matches=entries[image.pixels.unsafeGetTensorImpl()];
   auto match=std::find_if(matches.begin(),matches.end(),[&](const Entry& entry){return entry.image.grid_thw==image.grid_thw&&entry.image.merge_size==image.merge_size;});
   if(match==matches.end())matches.push_back({image,{},1});else ++match->remaining;
  }
 }
 template<class Compute> at::Tensor get(const ImageInput& image,Compute&& compute) {
  auto found=entries.find(image.pixels.unsafeGetTensorImpl());
  if(found!=entries.end())for(auto& entry:found->second) {
   if(entry.image.grid_thw!=image.grid_thw||entry.image.merge_size!=image.merge_size||!entry.remaining)continue;
   auto result=entry.features.defined()?entry.features:compute();
   if(--entry.remaining)entry.features=result;else entry.features=at::Tensor();
   return result;
  }
  return compute();
 }
};

struct AudioInput {
  at::Tensor features, validity;
  int64_t token_start=0, token_count=0;
  at::Tensor selected_frames{};
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
};
}

#pragma once
#include <ATen/ATen.h>
namespace tei {
// Newly authored packed row gather: actual last non-PAD row, or first row
// when every ID is PAD. Cumulative offsets and IDs remain on device so graph
// replay selects the current input, without padding or host index uploads.
at::Tensor qwen2_packed_classifier_rows(const at::Tensor& hidden,
  const at::Tensor& ids,const at::Tensor& cumulative,int64_t batch,int64_t pad);
}

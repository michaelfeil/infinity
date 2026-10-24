/// \file
/// Device-side selection of packed classifier rows without host index uploads.

#pragma once
#include <ATen/ATen.h>

namespace infinity {
// Newly authored packed row gather: actual last non-PAD row, or first row
// when every ID is PAD. Cumulative offsets and IDs remain on device so graph
// replay selects the current input, without padding or host index uploads.
// No PAD configuration selects the packed end; all copies preserve raw bits.
at::Tensor decoder_packed_classifier_rows(const at::Tensor& hidden,
                                          const at::Tensor& ids,
                                          const at::Tensor& cumulative,
                                          int64_t batch,
                                          int64_t pad,
                                          bool has_pad = true);
}

#pragma once
#include <ATen/ATen.h>

namespace tei {
// Gather true token/position rows and preserve Candle's FP32 residual norm.
at::Tensor distil_embedding_norm(const at::Tensor& words,
                                 const at::Tensor& positions,
                                 const at::Tensor& ids,
                                 const at::Tensor& position_ids,
                                 const at::Tensor& gamma,
                                 const at::Tensor& beta,
                                 double epsilon);
}

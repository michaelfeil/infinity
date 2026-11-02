// Upstream FlashAttention 2.8.3 file, BSD-3-Clause; see ../LICENSES.
// Local modification: do not include ATen when dropout is compiled out.
#pragma once
#ifndef FLASHATTENTION_DISABLE_DROPOUT
#include <ATen/cuda/PhiloxCudaState.h>     // For at::PhiloxCudaState
#include <ATen/cuda/detail/UnpackRaw.cuh>  // For at::cuda::philox::unpack
#endif

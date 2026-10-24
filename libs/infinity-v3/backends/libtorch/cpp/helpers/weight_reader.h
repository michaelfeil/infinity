/// \file
/// Weight-reader interface for Rust-validated safetensors ranges.
/// Produced tensors own their storage; the supplied file descriptor is borrowed.

#pragma once
#include <ATen/ATen.h>
#include "../bridge.h"
#include <functional>

namespace tei {
// Called only with Rust-validated offsets/shapes. Each result owns its storage.
// The borrowed descriptor remains open until every worker and DMA has drained.
void read_safetensors(
    int fd,
    uint64_t data_start,
    uint64_t data_length,
    const TeiWeight* weights,
    size_t count,
    const at::Device& device,
    size_t threads,
    size_t buffer_bytes,
    bool direct,
    const std::function<void(const char*, at::Tensor)>& insert,
    const std::function<at::ScalarType(const char*, at::ScalarType)>& target_dtype = {});
}

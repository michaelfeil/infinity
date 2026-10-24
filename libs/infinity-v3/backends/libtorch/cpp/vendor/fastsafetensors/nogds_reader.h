// SPDX-License-Identifier: Apache-2.0
// Copyright 2024 IBM. Adapted by Infinity Core Contributors; see provenance.json.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

namespace fastsafetensors {
using cudaError_t = int;
using cudaStream_t = void*;
inline constexpr int cudaSuccess = 0;
inline constexpr int cudaMemcpyHostToDevice = 1;

// The original runtime operation table is supplied by the embedding framework.
// CPU builds need neither CUDA headers nor a CUDA runtime.
struct ext_funcs_t {
    int (*cudaSetDevice)(int);
    int (*cudaHostAlloc)(void**, size_t, unsigned);
    int (*cudaFreeHost)(void*);
    int (*cudaMemcpy)(void*, const void*, size_t, int);
    int (*cudaDeviceSynchronize)();
    int (*cudaMemcpyAsync)(void*, const void*, size_t, int, void*);
    int (*cudaStreamCreateWithFlags)(void**, unsigned);
    int (*cudaStreamSynchronize)(void*);
    int (*cudaStreamDestroy)(void*);
};

class gds_device_buffer {
    uintptr_t pointer_;
    uint64_t length_;
public:
    gds_device_buffer(uintptr_t pointer, uint64_t length)
        : pointer_(pointer), length_(length) {}
    uintptr_t get_base_address() const { return pointer_; }
    uint64_t get_length() const { return length_; }
};

class nogds_file_reader {
    struct state;
    std::unique_ptr<state> _state;
public:
    nogds_file_reader(bool use_mmap, uint64_t buffer_kib, uint64_t threads,
                     const ext_funcs_t& operations, int device);
    int submit_read(int fd, const gds_device_buffer&, int64_t offset,
                          int64_t length, uint64_t pointer_offset,
                          bool track_progress = false);
    uint64_t wait_read_prefix(int request, uint64_t length);
    uintptr_t wait_read(int request);
    ~nogds_file_reader();
};
} // namespace fastsafetensors

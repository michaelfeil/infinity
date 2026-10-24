/// \file
/// Native safetensors tensor loading through the non-GDS worker reader.
/// Handles descriptor policy, destination allocation, conversion, and copy completion.

#include "weight_reader.h"
#include "../vendor/fastsafetensors/nogds_reader.h"
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <optional>
#include <sys/stat.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/vfs.h>
#endif
#ifdef TEI_TORCH_CUDA_KERNELS
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAGuard.h>
#endif

namespace {
int cpu_device(int) {
  return 0;
}

int cpu_allocate(void** p, size_t n, unsigned) {
  return posix_memalign(p, 4096, n);
}

int cpu_free(void* p) {
  std::free(p);
  return 0;
}

int cpu_copy(void* dst, const void* src, size_t n, int) {
  std::memcpy(dst, src, n);
  return 0;
}

int cpu_sync() {
  return 0;
}

const fastsafetensors::ext_funcs_t cpu_ops{
    cpu_device, cpu_allocate, cpu_free, cpu_copy, cpu_sync, nullptr, nullptr, nullptr, nullptr};

/// Completes worker device copies before weight-reader resources leave scope.
struct Drain {
  const fastsafetensors::ext_funcs_t* operations;
  bool complete = false;

  void finish() {
    TORCH_CHECK(operations->cudaDeviceSynchronize() == 0, "Could not finish weight copies");
    complete = true;
  }

  ~Drain() {
    if (!complete)
      operations->cudaDeviceSynchronize();
  }
};
#ifdef TEI_TORCH_CUDA_KERNELS
int gpu_device(int n) {
  return cudaSetDevice(n);
}

int gpu_allocate(void** p, size_t n, unsigned flags) {
  return cudaHostAlloc(p, n, flags);
}

int gpu_free(void* p) {
  return cudaFreeHost(p);
}

int gpu_copy(void* dst, const void* src, size_t n, int) {
  return cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice);
}

int gpu_sync() {
  return cudaDeviceSynchronize();
}

int gpu_copy_async(void* dst, const void* src, size_t n, int, void* stream) {
  return cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream));
}

int gpu_create(void** stream, unsigned flags) {
  cudaStream_t native = nullptr;
  int error = cudaStreamCreateWithFlags(&native, flags);
  *stream = native;
  return error;
}

int gpu_wait(void* stream) {
  return cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
}

int gpu_destroy(void* stream) {
  return cudaStreamDestroy(static_cast<cudaStream_t>(stream));
}

const fastsafetensors::ext_funcs_t cuda_ops{gpu_device,
                                            gpu_allocate,
                                            gpu_free,
                                            gpu_copy,
                                            gpu_sync,
                                            gpu_copy_async,
                                            gpu_create,
                                            gpu_wait,
                                            gpu_destroy};
#endif

/// Owns a duplicated file descriptor and restores any temporarily changed flags.
struct Descriptor {
  int value;
  int original_flags = -1;

  ~Descriptor() {
    if (value >= 0) {
      if (original_flags >= 0)
        fcntl(value, F_SETFL, original_flags);
      close(value);
    }
  }
};

at::ScalarType dtype(int32_t value) {
  switch (value) {
  case 0:
    return at::kFloat;
  case 1:
    return at::kHalf;
  case 2:
    return at::kBFloat16;
  default:
    TORCH_CHECK(false, "Unsupported weight dtype");
  }
}

bool direct_filesystem(int fd) {
#ifdef __linux__
  struct statfs info{};

  if (fstatfs(fd, &info) == 0) {
    // As upstream, preserve page caching on RAM and common network mounts.
    switch (static_cast<unsigned long>(info.f_type)) {
    case 0x01021994:
    case 0x858458f6:
    case 0x6969:
    case 0xff534d42:
    case 0x65735546:
      return false; // tmpfs, ramfs, NFS, CIFS, FUSE
    }
  }
  return true;
#else
  return false;
#endif
}
}

namespace infinity {
/// Validate ordered byte ranges, then read bounded groups into independently owned tensors. The
/// insert callback receives each completed tensor; file/DMA cleanup drains work on failure too.
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
    const std::function<at::ScalarType(const char*, at::ScalarType)>& target_dtype) {
  TORCH_CHECK(threads > 0 && threads <= 64, "Weight reader threads must be in 1..=64");
  TORCH_CHECK(buffer_bytes >= 65536 && buffer_bytes <= (1ULL << 30),
              "Weight reader buffer must be between 65536 and 1073741824 bytes");
  TORCH_CHECK(data_start <= INT64_MAX && data_length <= INT64_MAX - data_start,
              "Safetensors file is too large");
  TORCH_CHECK(weights || count == 0, "Missing weight descriptors");

  struct stat info{};

  TORCH_CHECK(fstat(fd, &info) == 0 && S_ISREG(info.st_mode) &&
                  static_cast<uint64_t>(info.st_size) == data_start + data_length,
              "Safetensors file size changed or is not a regular file");
  // Validate the FFI descriptors before allocating or starting asynchronous work.
  uint64_t previous_end = 0;
  for (size_t i = 0; i < count; ++i) {
    const auto& weight = weights[i];
    TORCH_CHECK(weight.name && (weight.shape || weight.rank == 0), "Invalid weight descriptor");
    uint64_t length = c10::elementSize(dtype(weight.dtype));
    for (size_t j = 0; j < weight.rank; ++j) {
      TORCH_CHECK(weight.shape[j] >= 0 &&
                      (!weight.shape[j] || length <= UINT64_MAX / weight.shape[j]),
                  "Weight shape overflow");
      length *= weight.shape[j];
    }
    TORCH_CHECK(length == weight.length && weight.offset >= previous_end &&
                    weight.offset <= data_length && length <= data_length - weight.offset,
                "Invalid weight offset or byte length");
    previous_end = weight.offset + length;
  }
  if (count == 0)
    return;
  Descriptor owned{dup(fd)};
  TORCH_CHECK(owned.value >= 0, "Could not duplicate safetensors descriptor");
  fcntl(owned.value, F_SETFD, FD_CLOEXEC);
  owned.original_flags = fcntl(owned.value, F_GETFL);
  TORCH_CHECK(owned.original_flags >= 0, "Could not read safetensors descriptor flags");
#ifdef O_DIRECT
  if (direct && direct_filesystem(owned.value)) {
    // Failure leaves the descriptor buffered; no special permissions needed.
    fcntl(owned.value, F_SETFL, owned.original_flags | O_DIRECT);
  } else if (owned.original_flags & O_DIRECT) {
    TORCH_CHECK(fcntl(owned.value, F_SETFL, owned.original_flags & ~O_DIRECT) == 0,
                "Could not select buffered safetensors reads");
  }
#endif
  bool cuda = device.is_cuda();
  const auto* ops = &cpu_ops;
  int device_id = 0;
#ifdef TEI_TORCH_CUDA_KERNELS
  std::optional<c10::cuda::CUDAGuard> guard;
  if (cuda) {
    guard.emplace(device);
    ops = &cuda_ops;
    device_id = device.index();
  }
#else
  TORCH_CHECK(!cuda, "This build has no CUDA weight reader");
#endif
  auto storage_device = cuda ? device : at::Device(at::kCPU);
  auto bytes_options = at::TensorOptions().dtype(at::kByte).device(storage_device);
  // Bound both host bounce buffers and transient GPU staging. Tiny sidecars
  // do not reserve the full default pool. Large single tensors read directly
  // into independently owned final storage, avoiding a second shard-sized copy.
  size_t budget = std::max<size_t>(65536, std::min<uint64_t>(buffer_bytes, data_length));
  size_t workers =
      std::max<size_t>(1, std::min<size_t>(threads, (data_length + (8 << 20) - 1) / (8 << 20)));
  Drain drain{ops};
  fastsafetensors::nogds_file_reader reader(
      false, (budget + 1023) / 1024, workers, *ops, device_id);
  auto read = [&](at::Tensor& dst, uint64_t offset, uint64_t length) {
    if (!length)
      return;
    fastsafetensors::gds_device_buffer buffer(reinterpret_cast<uintptr_t>(dst.data_ptr()), length);
    int request = reader.submit_read(owned.value, buffer, data_start + offset, length, 0);
    TORCH_CHECK(reader.wait_read(request) == buffer.get_base_address(),
                "Safetensors read failed or file was truncated");
  };
  for (size_t first = 0; first < count;) {
    const auto& initial = weights[first];
    size_t end = first + 1;
    uint64_t start = initial.offset, stop = initial.offset + initial.length;
    while (end < count && weights[end].offset + weights[end].length - start <= budget) {
      stop = weights[end].offset + weights[end].length;
      ++end;
    }
    if (end == first + 1) {
      auto source_type = dtype(initial.dtype);
      auto output_type = target_dtype ? target_dtype(initial.name, source_type) : source_type;
      auto tensor =
          at::empty(at::IntArrayRef(initial.shape, initial.rank), bytes_options.dtype(output_type));
      if (source_type == output_type) {
        read(tensor, start, initial.length);
      } else {
        // Convert large mismatched-dtype weights in bounded pieces,
        // rather than temporarily retaining a second large GPU tensor.
        auto flat = tensor.flatten();
        uint64_t element_bytes = c10::elementSize(source_type);
        uint64_t per_chunk = std::max<uint64_t>(1, budget / element_bytes);
        for (uint64_t position = 0; position < initial.length / element_bytes;) {
          uint64_t length =
              std::min<uint64_t>(per_chunk, initial.length / element_bytes - position);
          auto raw = at::empty({static_cast<int64_t>(length)}, bytes_options.dtype(source_type));
          read(raw, start + position * element_bytes, length * element_bytes);
          flat.narrow(0, position, length).copy_(raw);
          position += length;
        }
      }
      insert(initial.name, std::move(tensor));
    } else {
      auto staging = at::empty({static_cast<int64_t>(stop - start)}, bytes_options);
      read(staging, start, stop - start);
      for (size_t i = first; i < end; ++i) {
        const auto& weight = weights[i];
        auto source_type = dtype(weight.dtype);
        auto output_type = target_dtype ? target_dtype(weight.name, source_type) : source_type;
        auto tensor =
            at::empty(at::IntArrayRef(weight.shape, weight.rank), bytes_options.dtype(output_type));
        auto bytes = staging.narrow(0, weight.offset - start, weight.length);
        if ((weight.offset - start) % c10::elementSize(source_type) == 0) {
          tensor.flatten().copy_(bytes.view(source_type));
        } else {
          // Valid files can put an F32 tensor at offset 2 after F16.
          auto aligned =
              at::empty({static_cast<int64_t>(weight.length / c10::elementSize(source_type))},
                        bytes_options.dtype(source_type));
          aligned.view(at::kByte).copy_(bytes);
          tensor.flatten().copy_(aligned);
        }
        insert(weight.name, std::move(tensor));
      }
    }
    first = end;
  }
  // Worker DMA and ATen dtype conversions/copies must both finish before the
  // initialized engine can move to another caller's CUDA stream/thread.
  drain.finish();
}
}

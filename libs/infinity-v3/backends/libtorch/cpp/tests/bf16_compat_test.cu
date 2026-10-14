// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
#include "kernels/bf16_compat.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

static void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

__global__ void compare(unsigned* mismatches, unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  unsigned state = i * 747796405u + 2891336453u;
  auto random_bits = [&]() {
    state = state * 1664525u + 1013904223u;
    return static_cast<unsigned short>(state >> 16);
  };
  auto a = __ushort_as_bfloat16(random_bits());
  auto b = __ushort_as_bfloat16(random_bits());
  auto c = __ushort_as_bfloat16(random_bits());
  // A product exactly at a BF16 midpoint plus a term too small for FP32
  // round-to-nearest to retain. Include both signs and subnormal addends.
  if ((i & 7u) == 0) {
    a = __ushort_as_bfloat16(0x3fc0);
    b = __ushort_as_bfloat16(0x3f81);
    c = __ushort_as_bfloat16((i & 8u ? 0x8000 : 0) | (i & 16u ? 1 : 0x3080));
  }
  const auto actual = tei::bf16_fma_emulated(a, b, c);
  const auto expected = __hfma(a, b, c);
  const bool both_nan = isnan(__bfloat162float(actual)) && isnan(__bfloat162float(expected));
  if (!both_nan && __bfloat16_as_ushort(actual) != __bfloat16_as_ushort(expected))
    atomicAdd(mismatches, 1u);
}

int main() {
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, 0));
  if (properties.major < 8) {
    std::cout << "Native BF16 FMA comparison requires SM80+\n";
    return 0;
  }
  unsigned* mismatches = nullptr;
  check(cudaMallocManaged(&mismatches, sizeof(unsigned)));
  *mismatches = 0;
  constexpr unsigned count = 1u << 20;
  compare<<<count / 256, 256>>>(mismatches, count);
  check(cudaGetLastError());
  check(cudaDeviceSynchronize());
  const unsigned failures = *mismatches;
  check(cudaFree(mismatches));
  if (failures) throw std::runtime_error(std::to_string(failures) + " BF16 FMA mismatches");
  std::cout << count << " emulated BF16 FMAs match native results\n";
}

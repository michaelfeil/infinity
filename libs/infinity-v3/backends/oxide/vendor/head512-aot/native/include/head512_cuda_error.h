/* Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
 * Replaces the Torch exception shim used by the pinned Native launch template. */
#ifndef INFINITY_HEAD512_CUDA_ERROR_H
#define INFINITY_HEAD512_CUDA_ERROR_H

#include <cuda_runtime_api.h>

extern "C" void infinity_head512_cuda_note_error(int code);
extern "C" int infinity_head512_cuda_take_error(void);

#define HEAD512_CUDA_CHECK(EXPR)                         \
  do {                                                   \
    cudaError_t head512_status = (EXPR);                 \
    if (head512_status != cudaSuccess)                   \
      infinity_head512_cuda_note_error((int)head512_status); \
  } while (0)

#define HEAD512_KERNEL_LAUNCH_CHECK()                    \
  do {                                                   \
    cudaError_t head512_status = cudaGetLastError();     \
    if (head512_status != cudaSuccess)                   \
      infinity_head512_cuda_note_error((int)head512_status); \
  } while (0)

#endif

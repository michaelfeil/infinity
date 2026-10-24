// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// Run with qemu-x86_64 -cpu Nehalem: only generated CPU initialization is tested.
// CUDA APIs are stubbed; GPU execution and numerical parity are tested separately.
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

int _cudaDeviceGetAttribute(int* out, int attr, int device) {
  *out = 228;
  return 0;
}

int _cuKernelGetAttribute(int* out, int attr, void* kernel, int device) {
  *out = 0;
  return 0;
}

int _cudaLibraryGetKernel(void** out, void* library, const char* name) {
  *out = (void*)1;
  return 0;
}

int _cudaKernelSetAttributeForDevice(void* kernel, int attr, int value, int device) {
  return 0;
}

int _cudaFuncSetAttribute(void* kernel, int attr, int value) {
  return 0;
}

int _cudaGetDevice(int* out) {
  *out = 0;
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 2)
    return 2;
  printf("avx512f=%d\n", !!__builtin_cpu_supports("avx512f"));
  fflush(stdout);
  void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    fprintf(stderr, "%s\n", dlerror());
    return 3;
  }
  const char* families[] = {"bert", "modern_local", "qwen", "voyage", "deberta"};
  const char* dtypes[] = {"fp16", "bf16"};
  for (int i = 0; i < 5; i++)
    for (int j = 0; j < 2; j++) {
      char symbol[160];
      snprintf(symbol,
               sizeof(symbol),
               "__cute_internal_fa4_%s_%s_cuda_load_to_device",
               families[i],
               dtypes[j]);
      int (*load)(void**, int) = dlsym(lib, symbol);
      if (!load) {
        fprintf(stderr, "missing %s\n", symbol);
        return 4;
      }
      void* kernel = 0;
      printf("loading %s\n", symbol);
      fflush(stdout);
      if (load(&kernel, 0) != 0)
        return 5;
    }
  puts("all ten host initialization routines passed (CUDA calls stubbed)");
  return 0;
}

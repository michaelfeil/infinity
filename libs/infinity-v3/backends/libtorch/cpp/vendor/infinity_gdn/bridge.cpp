// Copyright Infinity Core Contributors. Licensed under the repository LICENSE.
// CuTe TVM-FFI host bindings; kernels remain the unmodified FlashInfer implementations.
#include "bridge.h"
#include <tvm/ffi/function.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <dlpack/dlpack.h>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>

using Entry = int (*)(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);
#define DECLARE(F) extern "C" int __tvm_ffi_gdn_##F##_fp16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*); \
                   extern "C" int __tvm_ffi_gdn_##F##_bf16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
#ifdef INFINITY_GDN_SM90
DECLARE(9)
#endif
#ifdef INFINITY_GDN_SM100
DECLARE(10)
#endif
#ifdef INFINITY_GDN_SM120
DECLARE(12)
#endif
static Entry entry(int major, int dtype) {
#ifdef INFINITY_GDN_SM90
  if (major == 9) return dtype ? __tvm_ffi_gdn_9_bf16 : __tvm_ffi_gdn_9_fp16;
#endif
#ifdef INFINITY_GDN_SM100
  if (major == 10) return dtype ? __tvm_ffi_gdn_10_bf16 : __tvm_ffi_gdn_10_fp16;
#endif
#ifdef INFINITY_GDN_SM120
  if (major == 12) return dtype ? __tvm_ffi_gdn_12_bf16 : __tvm_ffi_gdn_12_fp16;
#endif
  return nullptr;
}
thread_local std::string error;
extern "C" int infinity_gdn_supported_v1(int major) { return entry(major, 0) != nullptr; }
extern "C" const char* infinity_gdn_error_v1() { return error.c_str(); }
extern "C" int infinity_gdn_forward_v1(
    int dtype, int major, int device, void* stream, int64_t tokens, int64_t batch,
    int64_t kh, int64_t vh, int sm_count, void* q, void* k, void* v, void* alpha,
    void* beta, void* output, void* cumulative, void* state, void* workspace,
    int64_t workspace_bytes) {
  try {
    if (dtype < 0 || dtype > 1 || !entry(major, dtype) || tokens <= 0 || batch <= 0 ||
        kh <= 0 || vh < kh || vh % kh || sm_count <= 0)
      throw std::runtime_error("Unsupported FlashInfer GDN contract");
    DLDevice dev{kDLCUDA, device};
    DLDataType half{uint8_t(dtype ? kDLBfloat : kDLFloat), 16, 1};
    int64_t qs[]{tokens, kh, 128}, vs[]{tokens, vh, 128}, gs[]{tokens, vh},
        ss[]{batch, vh, 128, 128}, cs[]{batch + 1}, ws[]{workspace_bytes};
    int64_t qst[]{kh * 128, 128, 1}, vst[]{vh * 128, 128, 1}, gst[]{vh, 1},
        sst[]{vh * 128 * 128, 128 * 128, 128, 1}, one[]{1};
    auto tensor = [&](void* p, int n, int64_t* shape, int64_t* strides, DLDataType type) {
      return DLTensor{p, dev, n, type, shape, strides, 0};
    };
    auto qt=tensor(q,3,qs,qst,half), kt=tensor(k,3,qs,qst,half),
         vt=tensor(v,3,vs,vst,half), ot=tensor(output,3,vs,vst,half),
         at=tensor(alpha,2,gs,gst,{kDLFloat,32,1}), bt=tensor(beta,2,gs,gst,{kDLFloat,32,1}),
         st=tensor(state,4,ss,sst,{kDLFloat,32,1}), ct=tensor(cumulative,1,cs,one,{kDLInt,32,1}),
         wt=tensor(workspace,1,ws,one,{kDLInt,8,1});
    // Serialize host launch/initialization across CUDA contexts. GPU execution remains async.
    static std::mutex mutex;
    static std::exception_ptr failure;
    std::lock_guard<std::mutex> lock(mutex);
    if (failure) std::rethrow_exception(failure);
    void* previous = nullptr;
    if (TVMFFIEnvSetStream(kDLCUDA, device, stream, &previous))
      throw std::runtime_error("Cannot set GDN launch stream");
    try {
      auto fn = tvm::ffi::Function::FromExternC(nullptr, entry(major,dtype), nullptr);
      fn(&qt,&kt,&vt,&at,&bt,&ot,&ct,&st,&wt,int64_t(sm_count));
    } catch (...) {
      failure = std::current_exception();
      TVMFFIEnvSetStream(kDLCUDA,device,previous,nullptr);
      throw;
    }
    TVMFFIEnvSetStream(kDLCUDA,device,previous,nullptr);
    error.clear();
    return 0;
  } catch (const std::exception& e) {
    error=e.what();
    return -1;
  }
}

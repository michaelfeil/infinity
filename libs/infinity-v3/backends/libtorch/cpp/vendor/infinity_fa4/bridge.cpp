// SPDX-License-Identifier: MIT
// Adapted from candle-flash-attn-v4; see provenance.json and LICENSE-MIT.
#include <tvm/ffi/function.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <cmath>
#include <climits>
#include <cstdint>
#include <string>
#include <atomic>
#include <exception>
#include <limits>
#include <mutex>
extern "C" {
#define FA4_PROFILE(name, h, hk, d, causal, local) \
int __tvm_ffi_fa4_##name##_fp16(void*, const TVMFFIAny*, int32_t, TVMFFIAny*); \
int __tvm_ffi_fa4_##name##_bf16(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);
#include "profiles.inc"
#undef FA4_PROFILE
}
struct Profile { int width, group; bool causal, local; };
constexpr Profile profiles[] = {
#define FA4_PROFILE(name, h, hk, d, causal, local) {d, h / hk, bool(causal), bool(local)},
#include "profiles.inc"
#undef FA4_PROFILE
};
constexpr int attention_modes = sizeof(profiles) / sizeof(profiles[0]);
#ifndef FA4_COMPUTE_CAPABILITY
#define FA4_COMPUTE_CAPABILITY 90
#endif
extern "C" int infinity_fa4_compute_capability_v1(){return FA4_COMPUTE_CAPABILITY;}
#ifdef INFINITY_FA4_PORTABLE_HOST
extern "C" int infinity_fa4_portable_host_v1(){return 1;}
#endif
thread_local std::string last_error;
extern "C" const char* infinity_fa4_error_v1(){return last_error.c_str();}
struct StreamScope {
 int device;void* previous;
 StreamScope(int d,void* stream):device(d),previous(nullptr){if(TVMFFIEnvSetStream(kDLCUDA,d,stream,&previous))throw std::runtime_error("Unable to set TVM stream");}
 ~StreamScope(){TVMFFIEnvSetStream(kDLCUDA,device,previous,nullptr);}
};
// CuTe DSL 4.7.1 can retain its global loader spinlock when a second
// thread observes an export initialized after acquiring that lock. Serialize
// only the first successful call to each export; warm calls stay concurrent.
namespace {
std::mutex export_init_mutex;
// Slots: [0, 2*attention_modes) profiles, then the two DeBERTa slots, then the
// two ALiBi slots.  The DeBERTa reservations are kept even when FA4_DEBERTA is
// not defined so the additive ALiBi indices do not move between build variants.
// The native JinaCode pair follows the two reserved DeBERTa and published
// Jina8 slots. Derive its base from the profile count so ordinary additions
// cannot collide with a score-modifier family.
constexpr unsigned ALIBI_NATIVE12_BASE_SLOT = 2 * attention_modes + 4;
#ifdef FA4_ALIBI12_NATIVE_FLASH
std::atomic<bool> export_initialized[2 * attention_modes + 6]{};
#else
std::atomic<bool> export_initialized[2 * attention_modes + 4]{};
#endif
std::exception_ptr export_init_failure;

template <typename F>
void invoke_export(unsigned slot, F&& invoke) {
  if (!export_initialized[slot].load(std::memory_order_acquire)) {
    std::unique_lock<std::mutex> lock(export_init_mutex);
    if (!export_initialized[slot].load(std::memory_order_relaxed)) {
      if (export_init_failure) std::rethrow_exception(export_init_failure);
      try {
        invoke();
        export_initialized[slot].store(true, std::memory_order_release);
      } catch (...) {
        // The exported call combines loading and launch with no reliable
        // phase/status boundary. Its failed-init path may retain the lock.
        // Conservatively fail later cold calls; do not risk a process hang.
        // Already initialized exports may finish. Recovery requires restart.
        export_init_failure = std::current_exception();
        throw;
      }
      return;
    }
  }
  invoke();
}
}  // namespace

extern "C" int infinity_fa4_modes_v1(){return (1 << attention_modes) - 1;}

// Export families: 0 global d64 MHA, 1 windowed d64 MHA, 2 causal d128 GQA4, 3 global d128 GQA2, 4 GQA6, 5 GQA8, 6 global d256 GQA4.
// Pointers remain owned by the caller; no device allocation or synchronization here.
extern "C" int infinity_fa4_forward_v1(int mode,int dtype,int device,void* stream,void* q,void* k,void* v,void* o,void* offsets,void* offsets_k,int64_t total,int64_t total_k,int64_t batch,int64_t h,int64_t hk,double scale,int left,int right,const int64_t* strides,int max_q){
 try{
  if(dtype<0||dtype>1||mode<0||mode>=attention_modes||total<=0||total_k<=0||batch<=0||max_q<=0)throw std::runtime_error("unsupported attention geometry");
  static tvm::ffi::Function functions[2][attention_modes] = {
    {
#define FA4_PROFILE(name, h, hk, d, causal, local) tvm::ffi::Function::FromExternC(nullptr, __tvm_ffi_fa4_##name##_fp16, nullptr),
#include "profiles.inc"
#undef FA4_PROFILE
    }, {
#define FA4_PROFILE(name, h, hk, d, causal, local) tvm::ffi::Function::FromExternC(nullptr, __tvm_ffi_fa4_##name##_bf16, nullptr),
#include "profiles.inc"
#undef FA4_PROFILE
    }
  };
  const auto& profile = profiles[mode];
  if(h<=0||hk<=0||h!=profile.group*hk||!std::isfinite(scale)||scale<=0||
     (profile.local&&(left<0||(!profile.causal&&right<0))))
    throw std::runtime_error("unsupported attention parameters");
  auto& fn=functions[dtype][mode];
  DLDataType data_type{static_cast<uint8_t>(dtype==0?kDLFloat:kDLBfloat),16,1};
  int64_t d=profile.width;
  int64_t qs[]={total,h,d},ks[]={total_k,hk,d},cs[]={batch+1},st[12],cst[]={1};
  for(int j=0;j<12;j++)st[j]=strides[j];
  auto tensor=[&](void* p,int n,int64_t* sh,int64_t* stride,DLDataType dt){return DLTensor{p,{kDLCUDA,device},n,dt,sh,stride,0};};
  DLTensor qt=tensor(q,3,qs,st,data_type),kt=tensor(k,3,ks,st+3,data_type),vt=tensor(v,3,ks,st+6,data_type),ot=tensor(o,3,qs,st+9,data_type),ct=tensor(offsets,1,cs,cst,{kDLInt,32,1}),ckt=tensor(offsets_k,1,cs,cst,{kDLInt,32,1});
  tvm::ffi::Array<tvm::ffi::Any> aux{nullptr,nullptr};tvm::ffi::Any lb=profile.local?tvm::ffi::Any(int64_t(left)):tvm::ffi::Any(nullptr);
  tvm::ffi::Any rb=profile.local?tvm::ffi::Any(int64_t(right)):tvm::ffi::Any(nullptr);
  StreamScope scope(device,stream);
  invoke_export(dtype * attention_modes + mode, [&] {
#if FA4_COMPUTE_CAPABILITY == 100
  if (profile.width == 256)
    fn(&qt,&kt,&vt,&ot,nullptr,scale,&ct,&ckt,nullptr,nullptr,nullptr,lb,rb,nullptr,
       nullptr,nullptr,aux,int64_t(max_q));
  else
    fn(&qt,&kt,&vt,&ot,nullptr,scale,&ct,&ckt,nullptr,nullptr,nullptr,lb,rb,nullptr,
       nullptr,nullptr,aux,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,int64_t(max_q));
#else
  fn(&qt,&kt,&vt,&ot,nullptr,scale,&ct,&ckt,nullptr,nullptr,nullptr,lb,rb,nullptr,nullptr,aux,nullptr,nullptr);
#endif
  });
  last_error.clear();return 0;
 }catch(const std::exception& e){last_error=e.what();return -1;}
}

#ifdef FA4_DEBERTA
extern "C" int __tvm_ffi_fa4_deberta_fp16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
extern "C" int __tvm_ffi_fa4_deberta_bf16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
// Contiguous packed self-attention. Relative tables are [heads,total,2*span],
// and bucket IDs are a validated lookup for signed local q-k differences.
extern "C" int infinity_fa4_deberta_v1(int dtype,int device,void* stream,
 void* q,void* k,void* v,void* o,void* offsets,void* c2p,void* p2c,void* buckets,
 int64_t total,int64_t batch,int64_t heads,int64_t span,int64_t lut_len,int max_q) {
 try {
  if(dtype<0||dtype>1||total<=0||total>INT32_MAX||batch<=0||batch>=INT32_MAX||heads<=0||heads>INT32_MAX/64||span<=0||span>INT32_MAX/2||lut_len<=0||lut_len>INT32_MAX||lut_len%2!=1
     ||total>INT32_MAX/(heads*64)||total>INT32_MAX/(2*span)||heads>INT32_MAX/(total*2*span))
   throw std::runtime_error("invalid DeBERTa geometry");
  static tvm::ffi::Function fns[]={
   tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_deberta_fp16,nullptr),
   tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_deberta_bf16,nullptr)};
  DLDataType dt{static_cast<uint8_t>(dtype==0?kDLFloat:kDLBfloat),16,1}, it{kDLInt,32,1};
  int64_t shape[]={total,heads,64},stride[]={heads*64,64,1},cs[]={batch+1},one[]={1};
  int64_t rs[]={heads,total,2*span},rst[]={total*2*span,2*span,1},ls[]={lut_len};
  auto tensor=[&](void* p,int n,int64_t* sh,int64_t* st,DLDataType type){return DLTensor{p,{kDLCUDA,device},n,type,sh,st,0};};
  auto qt=tensor(q,3,shape,stride,dt),kt=tensor(k,3,shape,stride,dt),vt=tensor(v,3,shape,stride,dt),ot=tensor(o,3,shape,stride,dt);
  auto ct=tensor(offsets,1,cs,one,it),at=tensor(c2p,3,rs,rst,dt),bt=tensor(p2c,3,rs,rst,dt),lt=tensor(buckets,1,ls,one,it);
  // Borrowed storage stays owned by the caller. The stack DLPack wrappers live
  // through the synchronous launch call, and no deleter frees their data. The
  // tvm::ffi Tensor/Array wrappers may allocate host metadata on the heap.
  DLManagedTensor am{at,nullptr,nullptr},bm{bt,nullptr,nullptr},lm{lt,nullptr,nullptr};
  tvm::ffi::Array<tvm::ffi::Any> tensors{tvm::ffi::Tensor::FromDLPack(&am),tvm::ffi::Tensor::FromDLPack(&bm),tvm::ffi::Tensor::FromDLPack(&lm)};
  tvm::ffi::Array<tvm::ffi::Any> aux{tensors,nullptr};
  StreamScope scope(device,stream);
  invoke_export(2 * attention_modes + dtype, [&] {
#if FA4_COMPUTE_CAPABILITY == 100
  fns[dtype](&qt,&kt,&vt,&ot,nullptr,1.0,&ct,&ct,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
     nullptr,nullptr,aux,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,int64_t(max_q));
#else
  fns[dtype](&qt,&kt,&vt,&ot,nullptr,1.0,&ct,&ct,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,aux,nullptr,nullptr);
#endif
  });
  last_error.clear();return 0;
 } catch(const std::exception& e){last_error=e.what();return -1;}
}
#endif

#ifdef FA4_ALIBI
extern "C" int __tvm_ffi_fa4_alibi_model_bias_fp16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
extern "C" int __tvm_ffi_fa4_alibi_model_bias_bf16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
// Fixed non-causal packed self-attention: 8 heads, head_dim 64, no local
// window.  The only bias input is a borrowed F32[8] per-head slope vector;
// there is no O(T*T) bias tensor.  Q/K/V/output use the caller's strides[12]
// in the ordinary forward's Q,K,V,O order.  The caller owns the slope, model,
// and context storage until the stream drains.
extern "C" int infinity_fa4_alibi_model_bias_v1(int dtype,int device,void* stream,
 void* q,void* k,void* v,void* o,void* offsets,const float* slopes,
 int64_t total,int64_t batch,int64_t heads,const int64_t* strides,int max_q) {
 try {
  if(dtype<0||dtype>1||device<0||!q||!k||!v||!o||!offsets||!slopes||!strides
     ||total<=0||total>INT32_MAX/(8*64)||batch<=0||batch>=INT32_MAX
     ||heads!=8||max_q<=0||max_q>INT32_MAX||max_q>total)
    throw std::runtime_error("invalid ALiBi geometry");
  for(int j=0;j<12;j++) if(strides[j]<=0) throw std::runtime_error("invalid ALiBi strides");
  auto fits=[&](const int64_t* s){
   __int128 extent=(__int128)(total-1)*s[0]+(__int128)(heads-1)*s[1]+(__int128)(64-1)*s[2];
   return extent<=std::numeric_limits<int64_t>::max();
  };
  for(int base=0;base<12;base+=3) if(!fits(strides+base)) throw std::runtime_error("ALiBi stride extent overflows int64");
  static tvm::ffi::Function fns[]={
   tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_alibi_model_bias_fp16,nullptr),
   tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_alibi_model_bias_bf16,nullptr)};
  DLDataType dt{static_cast<uint8_t>(dtype==0?kDLFloat:kDLBfloat),16,1},it{kDLInt,32,1},ft{kDLFloat,32,1};
  int64_t shape[]={total,heads,64},cs[]={batch+1},one[]={1},ss[]={8},st[12];
  for(int j=0;j<12;j++)st[j]=strides[j];
  auto tensor=[&](void* p,int n,int64_t* sh,int64_t* stride,DLDataType type){return DLTensor{p,{kDLCUDA,device},n,type,sh,stride,0};};
  auto qt=tensor(q,3,shape,st,dt),kt=tensor(k,3,shape,st+3,dt),vt=tensor(v,3,shape,st+6,dt),ot=tensor(o,3,shape,st+9,dt);
  auto ct=tensor(offsets,1,cs,one,it),stl=tensor(const_cast<float*>(slopes),1,ss,one,ft);
  // Each call builds a few host-side DLTensor/DLManagedTensor wrappers on the
  // stack. Creating the tvm::ffi Tensor/Array wrappers may allocate host
  // metadata on the heap.  Borrowed device storage stays owned by the caller,
  // and no deleter frees it.  The stack wrappers live through the synchronous
  // launch call.
  DLManagedTensor sm{stl,nullptr,nullptr};
  tvm::ffi::Array<tvm::ffi::Any> tensors{tvm::ffi::Tensor::FromDLPack(&sm)};
  tvm::ffi::Array<tvm::ffi::Any> aux{tensors,nullptr};
  StreamScope scope(device,stream);
  invoke_export(2 * attention_modes + 2 + dtype, [&] {
#if FA4_COMPUTE_CAPABILITY == 100
   fns[dtype](&qt,&kt,&vt,&ot,nullptr,1.0,&ct,&ct,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
      nullptr,nullptr,aux,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,int64_t(max_q));
#else
   fns[dtype](&qt,&kt,&vt,&ot,nullptr,1.0,&ct,&ct,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,aux,nullptr,nullptr);
#endif
  });
  last_error.clear();return 0;
 } catch(const std::exception& e){last_error=e.what();return -1;}
}
#endif

#ifdef FA4_ALIBI12_NATIVE_FLASH
extern "C" int __tvm_ffi_fa4_jinacode_flash_alibi_fp16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
extern "C" int __tvm_ffi_fa4_jinacode_flash_alibi_bf16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
// Additive JinaCode ALiBi public ABI: packed non-causal self-attention with
// 12 Q/K/V/output heads, head_dim 64, and no local window.  It reuses the
// native F32 FMA score hook (no model-dtype bias rounding) and borrows one F32[12] slope vector; no
// O(T*T) bias is materialized.  Q/K/V/output strides are the caller's
// strides[12] in the ordinary Q,K,V,O order.  The caller owns the slope,
// model, and context storage until the stream drains.
extern "C" int infinity_fa4_jinacode_flash_alibi_v1(int dtype,int device,void* stream,
 void* q,void* k,void* v,void* o,void* offsets,const float* slopes,
 int64_t total,int64_t batch,int64_t heads,const int64_t* strides,int max_q) {
 try {
  if(dtype<0||dtype>1||device<0||!q||!k||!v||!o||!offsets||!slopes||!strides
     ||total<=0||total>INT32_MAX/(12*64)||batch<=0||batch>=INT32_MAX
     ||heads!=12||max_q<=0||max_q>INT32_MAX||max_q>total)
    throw std::runtime_error("invalid ALiBi12 geometry");
  for(int j=0;j<12;j++) if(strides[j]<=0) throw std::runtime_error("invalid ALiBi12 strides");
  auto fits=[&](const int64_t* s){
   __int128 extent=(__int128)(total-1)*s[0]+(__int128)(heads-1)*s[1]+(__int128)(64-1)*s[2];
   return extent<=std::numeric_limits<int64_t>::max();
  };
  for(int base=0;base<12;base+=3) if(!fits(strides+base)) throw std::runtime_error("ALiBi12 stride extent overflows int64");
  static tvm::ffi::Function fns[]={
   tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_jinacode_flash_alibi_fp16,nullptr),
   tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_jinacode_flash_alibi_bf16,nullptr)};
  DLDataType dt{static_cast<uint8_t>(dtype==0?kDLFloat:kDLBfloat),16,1},it{kDLInt,32,1},ft{kDLFloat,32,1};
  int64_t shape[]={total,heads,64},cs[]={batch+1},one[]={1},ss[]={12},st[12];
  for(int j=0;j<12;j++)st[j]=strides[j];
  auto tensor=[&](void* p,int n,int64_t* sh,int64_t* stride,DLDataType type){return DLTensor{p,{kDLCUDA,device},n,type,sh,stride,0};};
  auto qt=tensor(q,3,shape,st,dt),kt=tensor(k,3,shape,st+3,dt),vt=tensor(v,3,shape,st+6,dt),ot=tensor(o,3,shape,st+9,dt);
  auto ct=tensor(offsets,1,cs,one,it),stl=tensor(const_cast<float*>(slopes),1,ss,one,ft);
  // Borrowed storage stays owned by the caller.  The stack DLPack wrappers
  // live through the synchronous launch, and no deleter frees their data.
  DLManagedTensor sm{stl,nullptr,nullptr};
  tvm::ffi::Array<tvm::ffi::Any> tensors{tvm::ffi::Tensor::FromDLPack(&sm)};
  tvm::ffi::Array<tvm::ffi::Any> aux{tensors,nullptr};
  StreamScope scope(device,stream);
  invoke_export(ALIBI_NATIVE12_BASE_SLOT + dtype, [&] {
#if FA4_COMPUTE_CAPABILITY == 100
   fns[dtype](&qt,&kt,&vt,&ot,nullptr,1.0,&ct,&ct,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
      nullptr,nullptr,aux,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,int64_t(max_q));
#else
   fns[dtype](&qt,&kt,&vt,&ot,nullptr,1.0,&ct,&ct,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,aux,nullptr,nullptr);
#endif
  });
  last_error.clear();return 0;
 } catch(const std::exception& e){last_error=e.what();return -1;}
}
#endif

#ifdef FA4_DEBERTA_EXACT_MUL
extern "C" int __tvm_ffi_fa4_deberta_exact_mul_both_bf16(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);
extern "C" int __tvm_ffi_fa4_deberta_exact_mul_single_bf16(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);

namespace {
std::atomic<bool> exact_mul_initialized[2]{};

template <typename F>
void invoke_exact_mul(unsigned slot, F&& invoke) {
  if (!exact_mul_initialized[slot].load(std::memory_order_acquire)) {
    std::unique_lock<std::mutex> lock(export_init_mutex);
    if (!exact_mul_initialized[slot].load(std::memory_order_relaxed)) {
      if (export_init_failure) std::rethrow_exception(export_init_failure);
      try {
        invoke();
        exact_mul_initialized[slot].store(true, std::memory_order_release);
      } catch (...) {
        export_init_failure = std::current_exception();
        throw;
      }
      return;
    }
  }
  invoke();
}
}  // namespace

// Raw BF16 relative GEMM tables; the supplied existing materialized divisor
// selects one of two compiled, exactly-rounded score transforms. No tensor
// scalar or model-dtype arithmetic is reconstructed on the host.
extern "C" int infinity_fa4_deberta_exact_mul_v1(int dtype, int device, void* stream,
    void* q, void* k, void* v, void* o, void* offsets,
    void* c2p, void* p2c, void* buckets, int64_t total, int64_t batch,
    int64_t heads, int64_t span, int64_t lut_len, int max_q, float divisor) {
  try {
    if (dtype != 1 || device < 0 || !q || !k || !v || !o || !offsets || !c2p || !p2c || !buckets ||
        total <= 0 || total > INT32_MAX || batch <= 0 || batch > total || batch >= INT32_MAX ||
        heads != 12 || span <= 0 || span > 4096 || lut_len <= 0 || lut_len > INT32_MAX || lut_len % 2 != 1 ||
        max_q <= 0 || max_q > total || lut_len != 2 * static_cast<int64_t>(max_q) - 1 ||
        total > INT32_MAX / (heads * 64) || total > INT32_MAX / (2 * span) ||
        heads > INT32_MAX / (total * 2 * span))
      throw std::runtime_error("invalid exact-mul BF16 DeBERTa geometry");
    const auto aligned = [](void* pointer, uintptr_t alignment) {
      return reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
    };
    if (!aligned(q,2) || !aligned(k,2) || !aligned(v,2) || !aligned(o,2) ||
        !aligned(c2p,2) || !aligned(p2c,2) || !aligned(offsets,4) || !aligned(buckets,4))
      throw std::runtime_error("misaligned exact-mul BF16 DeBERTa device pointer");
    const unsigned variant = divisor == 13.875F ? 0 : divisor == 11.3125F ? 1 : 2;
    if (variant == 2) throw std::runtime_error("unsupported materialized BF16 DeBERTa divisor");
    static tvm::ffi::Function functions[] = {
        tvm::ffi::Function::FromExternC(nullptr, __tvm_ffi_fa4_deberta_exact_mul_both_bf16, nullptr),
        tvm::ffi::Function::FromExternC(nullptr, __tvm_ffi_fa4_deberta_exact_mul_single_bf16, nullptr)};
    DLDataType dt{kDLBfloat, 16, 1}, it{kDLInt, 32, 1};
    int64_t shape[]{total, heads, 64}, strides[]{heads * 64, 64, 1}, cs[]{batch + 1}, one[]{1};
    int64_t relative[]{heads, total, 2 * span}, rs[]{total * 2 * span, 2 * span, 1}, ls[]{lut_len};
    auto tensor = [&](void* data, int ndim, int64_t* sizes, int64_t* steps, DLDataType type) {
      return DLTensor{data, {kDLCUDA, device}, ndim, type, sizes, steps, 0};
    };
    auto qt=tensor(q,3,shape,strides,dt), kt=tensor(k,3,shape,strides,dt);
    auto vt=tensor(v,3,shape,strides,dt), ot=tensor(o,3,shape,strides,dt);
    auto ct=tensor(offsets,1,cs,one,it), at=tensor(c2p,3,relative,rs,dt);
    auto bt=tensor(p2c,3,relative,rs,dt), lt=tensor(buckets,1,ls,one,it);
    DLManagedTensor am{at,nullptr,nullptr}, bm{bt,nullptr,nullptr}, lm{lt,nullptr,nullptr};
    tvm::ffi::Array<tvm::ffi::Any> tensors{
        tvm::ffi::Tensor::FromDLPack(&am), tvm::ffi::Tensor::FromDLPack(&bm), tvm::ffi::Tensor::FromDLPack(&lm)};
    tvm::ffi::Array<tvm::ffi::Any> aux{tensors,nullptr};
    StreamScope scope(device,stream);
    invoke_exact_mul(variant, [&] {
#if FA4_COMPUTE_CAPABILITY == 100
      functions[variant](&qt,&kt,&vt,&ot,nullptr,1.0,&ct,&ct,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
          nullptr,nullptr,aux,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,int64_t(max_q));
#else
      functions[variant](&qt,&kt,&vt,&ot,nullptr,1.0,&ct,&ct,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,aux,nullptr,nullptr);
#endif
    });
    last_error.clear();
    return 0;
  } catch (const std::exception& error) {
    last_error=error.what();
    return -1;
  } catch (...) {
    last_error="unknown exact-mul BF16 DeBERTa provider exception";
    return -1;
  }
}

#endif

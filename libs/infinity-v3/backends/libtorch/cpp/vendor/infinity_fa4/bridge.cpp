// SPDX-License-Identifier: MIT
// Adapted from candle-flash-attn-v4; see provenance.json and LICENSE-MIT.
#include <tvm/ffi/function.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <cmath>
#include <string>
#include <atomic>
#include <exception>
#include <mutex>
extern "C" {
int __tvm_ffi_fa4_bert_fp16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
int __tvm_ffi_fa4_modern_local_fp16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
int __tvm_ffi_fa4_qwen_fp16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
int __tvm_ffi_fa4_voyage_fp16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
int __tvm_ffi_fa4_bert_bf16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
int __tvm_ffi_fa4_modern_local_bf16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
int __tvm_ffi_fa4_qwen_bf16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);
int __tvm_ffi_fa4_voyage_bf16(void*,const TVMFFIAny*,int32_t,TVMFFIAny*);

}
#ifndef FA4_COMPUTE_CAPABILITY
#define FA4_COMPUTE_CAPABILITY 90
#endif
extern "C" int infinity_fa4_compute_capability_v1(){return FA4_COMPUTE_CAPABILITY;}
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
std::atomic<bool> export_initialized[10]{};
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

// Export families: 0 global d64 MHA, 1 windowed d64 MHA, 2 causal d128 GQA4, 3 global d128 GQA2.
// Pointers remain owned by the caller; no device allocation or synchronization here.
extern "C" int infinity_fa4_forward_v1(int mode,int dtype,int device,void* stream,void* q,void* k,void* v,void* o,void* offsets,void* offsets_k,int64_t total,int64_t total_k,int64_t batch,int64_t h,int64_t hk,double scale,int left,int right,const int64_t* strides,int max_q){
 try{
  if(dtype<0||dtype>1||mode<0||mode>3||total<=0||total_k<=0||batch<=0||max_q<=0)throw std::runtime_error("unsupported attention geometry");
  static tvm::ffi::Function functions[2][4] = {
    {tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_bert_fp16,nullptr),
     tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_modern_local_fp16,nullptr),
     tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_qwen_fp16,nullptr),
     tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_voyage_fp16,nullptr)},
    {tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_bert_bf16,nullptr),
     tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_modern_local_bf16,nullptr),
     tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_qwen_bf16,nullptr),
     tvm::ffi::Function::FromExternC(nullptr,__tvm_ffi_fa4_voyage_bf16,nullptr)}
  };
  if(h<=0||hk<=0||(!std::isfinite(scale)||scale<=0)||(mode<2&&h!=hk)||(mode==2&&(h%hk||h/hk!=4))
     ||(mode==3&&(h%hk||h/hk!=2))
     ||(mode==1&&(left<0||right<0)))throw std::runtime_error("unsupported attention parameters");
  auto& fn=functions[dtype][mode];
  DLDataType data_type{static_cast<uint8_t>(dtype==0?kDLFloat:kDLBfloat),16,1};
  int64_t d=mode>=2?128:64;
  int64_t qs[]={total,h,d},ks[]={total_k,hk,d},cs[]={batch+1},st[12],cst[]={1};
  for(int j=0;j<12;j++)st[j]=strides[j];
  auto tensor=[&](void* p,int n,int64_t* sh,int64_t* stride,DLDataType dt){return DLTensor{p,{kDLCUDA,device},n,dt,sh,stride,0};};
  DLTensor qt=tensor(q,3,qs,st,data_type),kt=tensor(k,3,ks,st+3,data_type),vt=tensor(v,3,ks,st+6,data_type),ot=tensor(o,3,qs,st+9,data_type),ct=tensor(offsets,1,cs,cst,{kDLInt,32,1}),ckt=tensor(offsets_k,1,cs,cst,{kDLInt,32,1});
  tvm::ffi::Array<tvm::ffi::Any> aux{nullptr,nullptr};tvm::ffi::Any lb=mode==1?tvm::ffi::Any(int64_t(left)):tvm::ffi::Any(nullptr);
  tvm::ffi::Any rb=mode==1?tvm::ffi::Any(int64_t(right)):tvm::ffi::Any(nullptr);
  StreamScope scope(device,stream);
  invoke_export(dtype * 4 + mode, [&] {
#if FA4_COMPUTE_CAPABILITY == 100
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
  // through the synchronous launch call; no deleter frees their data.
  DLManagedTensor am{at,nullptr,nullptr},bm{bt,nullptr,nullptr},lm{lt,nullptr,nullptr};
  tvm::ffi::Array<tvm::ffi::Any> tensors{tvm::ffi::Tensor::FromDLPack(&am),tvm::ffi::Tensor::FromDLPack(&bm),tvm::ffi::Tensor::FromDLPack(&lm)};
  tvm::ffi::Array<tvm::ffi::Any> aux{tensors,nullptr};
  StreamScope scope(device,stream);
  invoke_export(8 + dtype, [&] {
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

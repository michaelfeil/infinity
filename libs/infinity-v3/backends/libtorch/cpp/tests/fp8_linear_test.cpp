#include "helpers/fp8_linear.h"
#include "helpers/workspace_admission.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/driver_api.h>
#include <thread>
#include <exception>
#include "tests/fp8_reference_test.h"
#include <cstdlib>
#include <array>
#include <c10/core/InferenceMode.h>
#include <iostream>

int main() {
  // Match the original independently implemented recipe's workspace budget.
  setenv("CUBLASLT_WORKSPACE_SIZE","32768",1);
  c10::InferenceMode inference;
  at::manual_seed(7249);
  at::globalContext().setAllowTF32CuBLAS(false);
  for(auto dtype:{at::kHalf,at::kBFloat16}) {
    auto options=at::TensorOptions().device(at::kCUDA,0).dtype(dtype);
    if(at::globalContext().blasPreferredBackend()==at::BlasBackend::Cublaslt) {
      tei::OrdinaryLinearExecutor ordinary;
      auto x=at::randn({32,768},options)*.3;
      auto w=at::randn({2304,768},options)*.05;
      auto expected=at::linear(x,w);
      c10::cuda::getCurrentCUDAStream().synchronize();
      {
        tei::OrdinaryLinearExecutor retry;
        auto state=tei::workspace_admission_state(0);
        size_t available;
        { std::lock_guard lock(state->mutex);
          available=(state->limit?state->limit:size_t(512)<<20)-state->retained;
        }
        auto pressure=tei::reserve_workspace_memory(0,available);
        TORCH_CHECK(pressure,"Workspace pressure fixture failed to reserve remaining allowance");
        auto session=retry.prepare(x);
        TORCH_CHECK(!retry.forward(x,w,session).defined(),"Workspace cap pressure was ignored");
        pressure.reset();
        auto recovered=retry.forward(x,w,session);
        TORCH_CHECK(recovered.defined()&&at::equal(recovered,expected),
          "Released workspace allowance did not restore the cached supported plan");
        c10::cuda::getCurrentCUDAStream().synchronize();
      }
      at::Tensor retained;
      auto streams=std::array<c10::cuda::CUDAStream,2>{
        c10::cuda::getStreamFromPool(),c10::cuda::getStreamFromPool()};
      for(auto stream:streams) {
        c10::cuda::CUDAStreamGuard stream_guard(stream);
        auto session=ordinary.prepare(x);
        auto first=ordinary.forward(x,w,session);
        TORCH_CHECK(first.defined(),"Supported ordinary request session declined");
        TORCH_CHECK(at::equal(first,expected),"Ordinary session changed original GEMM bytes");
        auto saved=first.clone();
        for(int i=0;i<4;++i) {
          auto next=ordinary.forward(x,w,session);
          TORCH_CHECK(next.defined()&&next.const_data_ptr()!=first.const_data_ptr(),
            "Request session reused a live output allocation");
          TORCH_CHECK(at::equal(next,expected)&&at::equal(first,saved),
            "Ordinary session changed or invalidated owned output");
        }
        { c10::cuda::CUDAStreamGuard other_guard(stream==streams[0]?streams[1]:streams[0]);
          TORCH_CHECK(!ordinary.forward(x,w,session).defined(),"Cross-stream session admitted"); }
        if(c10::cuda::device_count()>1) {
          c10::cuda::CUDAGuard other_device(at::Device(at::kCUDA,1));
          TORCH_CHECK(!ordinary.forward(x,w,session).defined(),"Cross-device session admitted");
          TORCH_CHECK(c10::cuda::current_device()==1,"Rejected session changed caller device");
        }
        TORCH_CHECK(at::equal(ordinary.forward(x,w,session),expected),
          "Restored stream lost session dispatch");
        retained=first;
        stream.synchronize();
      }
      TORCH_CHECK(at::equal(retained,expected),"Session destruction invalidated owned output");
      tei::OrdinaryLinearExecutor::Session retired;
      at::Tensor lifetime_output;
      { tei::OrdinaryLinearExecutor temporary;
        retired=temporary.prepare(x);lifetime_output=temporary.forward(x,w,retired);
        TORCH_CHECK(lifetime_output.defined(),"Lifetime fixture declined"); }
      tei::OrdinaryLinearExecutor different;
      TORCH_CHECK(!different.forward(x,w,retired).defined(),"Session admitted a different executor");
      std::exception_ptr worker_failure;
      std::thread worker([session=std::move(retired),&worker_failure]() mutable {
        try {
          c10::InferenceMode worker_inference;
          const int device=c10::cuda::device_count()>1?1:0;
          c10::cuda::CUDAGuard ambient(at::Device(at::kCUDA,device));
          C10_CUDA_CHECK(cudaFree(nullptr));
          CUcontext before{},after{};auto get=c10::cuda::DriverAPI::get()->cuCtxGetCurrent_;
          TORCH_CHECK(get(&before)==CUDA_SUCCESS,"Missing worker context");
          session=tei::OrdinaryLinearExecutor::Session{};
          TORCH_CHECK(c10::cuda::current_device()==device&&get(&after)==CUDA_SUCCESS&&after==before,
            "Foreign-thread last session destruction changed ambient CUDA device/context");
        } catch(...) {worker_failure=std::current_exception();}
      });worker.join();if(worker_failure)std::rethrow_exception(worker_failure);
      TORCH_CHECK(at::equal(lifetime_output,expected),"Executor/session destruction invalidated owned output");
      if(c10::cuda::device_count()>1) {
        c10::cuda::CUDAGuard ambient(at::Device(at::kCUDA,1));
        auto local=ordinary.prepare(x);
        TORCH_CHECK(c10::cuda::current_device()==1,"Prepare failed to restore caller device");
      }
      at::globalContext().setDeterministicAlgorithms(true,false);
      { auto unsupported=ordinary.prepare(x);
        TORCH_CHECK(!ordinary.forward(x,w,unsupported).defined(),"Unsupported policy admitted"); }
      at::globalContext().setDeterministicAlgorithms(false,false);
      { c10::cuda::CUDAStreamGuard guard(streams[0]);
        cudaGraph_t graph{};
        C10_CUDA_CHECK(cudaStreamBeginCapture(streams[0],cudaStreamCaptureModeThreadLocal));
        auto capturing=ordinary.prepare(x);
        TORCH_CHECK(!ordinary.forward(x,w,capturing).defined(),"Capture admitted ordinary session");
        C10_CUDA_CHECK(cudaStreamEndCapture(streams[0],&graph));
        C10_CUDA_CHECK(cudaGraphDestroy(graph)); }
      std::cout<<"Ordinary request-local stream/policy/capture and owned output: "<<dtype<<" passed\n";
    }
    auto weight=at::randn({512,256},options)*.05;
    tei::Fp8Linear linear(weight);
    for(int64_t rows:{1,3,7,17,35,128}) {
      auto input=at::randn({rows,256},options)*.3;
      // Exact zero and sub-floor rows exercise the scale clamp, including
      // fp16 underflow. Row count remains the actual number of tokens.
      input.select(0,0).zero_();
      if(rows>1)input.select(0,1).fill_(1e-14);
      auto quantized=tei::fp8_quantize_rows(input);
      auto cpu=input.to(at::kCPU).to(at::kFloat);
      auto scale=at::amax(cpu.abs(),{1},true).clamp_min(1e-12)*(1.f/448.f);
      auto reference=(cpu/scale).clamp(-448.,448.).to(at::kFloat8_e4m3fn);
      TORCH_CHECK(at::equal(quantized.second.to(at::kCPU),scale),"FP8 row-scale arithmetic differs from independent CPU oracle");
      TORCH_CHECK(at::equal(quantized.first.to(at::kCPU).view(at::kByte),reference.view(at::kByte)),
        "FP8 quantized bytes differ from independent CPU oracle");
      TORCH_CHECK(quantized.first.sizes()==input.sizes() && quantized.second.size(0)==rows,
        "FP8 quantization padded actual token rows");
      auto output=linear.forward(input);
      auto expected=fp8_original_recipe(quantized.first,linear.matrix(),quantized.second,linear.scales(),dtype);
      TORCH_CHECK(at::equal(output,expected),
        "Native Torch FP8 GEMM differs from original independent cuBLASLt recipe: rows=",rows," dtype=",dtype);
      TORCH_CHECK(output.sizes()==at::IntArrayRef({rows,512}) && output.scalar_type()==dtype,
        "FP8 GEMM changed actual row count or model dtype");
      auto bias=at::randn({512},options)*.01;
      TORCH_CHECK(at::equal(linear.forward(input,bias),output+bias),"FP8 bias lost separate model-dtype rounding");
    }
    auto empty=linear.forward(at::empty({0,256},options));
    TORCH_CHECK(empty.sizes()==at::IntArrayRef({0,512}),"FP8 empty rows not preserved");
    std::cout<<"FP8 row-scale/bytes, odd real rows, model-dtype GEMM and bias: "<<dtype<<" passed\n";
  }
  for(auto shape:{std::pair<int64_t,int64_t>{511,256},{512,255}}) {
    bool rejected=false;
    try{tei::Fp8Linear invalid(at::zeros({shape.first,shape.second},at::TensorOptions().device(at::kCUDA,0).dtype(at::kHalf)));}
    catch(const c10::Error&){rejected=true;}
    TORCH_CHECK(rejected,"Unsupported FP8 projection width silently padded");
  }
}

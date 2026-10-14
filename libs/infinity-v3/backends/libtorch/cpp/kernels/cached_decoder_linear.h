#pragma once
// Cache immutable descriptors and the original cuBLASLt heuristic result.
// Contract follows PyTorch 5c4886908584029761b579af026dcfb627c84070,
// aten/src/ATen/cuda/CUDABlas.cpp. Preserve its original notices below.
/*
From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/
#include <ATen/ATen.h>
#include "helpers/vendor_error.h"
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <array>
#include <map>
#include <memory>
#include <mutex>
namespace tei {
class CachedNoBiasLt {
 struct Plan {
  cublasLtMatmulDesc_t operation{};
  cublasLtMatrixLayout_t a{},b{},c{};
  cublasLtMatmulAlgo_t algorithm{};
  bool valid=false;
  ~Plan(){if(c)cublasLtMatrixLayoutDestroy(c);if(b)cublasLtMatrixLayoutDestroy(b);if(a)cublasLtMatrixLayoutDestroy(a);if(operation)cublasLtMatmulDescDestroy(operation);}
 };
 using Key=std::array<int64_t,17>;
 mutable std::mutex mutex;
 mutable std::map<Key,std::shared_ptr<Plan>> plans;
 static uint32_t alignment(const void* ptr){uint32_t a=256;while(reinterpret_cast<uintptr_t>(ptr)%a)a/=2;return a;}
 public:
 at::Tensor forward(const at::Tensor& x,const at::Tensor& weight) const {
  at::Tensor output;
  auto fallback=[&]{output=at::Tensor();return at::linear(x,weight);};
  if(!x.is_cuda()||x.scalar_type()!=at::kBFloat16||weight.scalar_type()!=x.scalar_type()||weight.device()!=x.device()||x.dim()!=2||weight.dim()!=2||x.size(0)<=0||x.size(1)<=0||weight.size(0)<=0||weight.size(1)!=x.size(1)||!x.is_contiguous()||!weight.is_contiguous()||at::globalContext().blasPreferredBackend()!=at::BlasBackend::Cublaslt||at::globalContext()._SMCarveout_EXPERIMENTAL().has_value())return fallback();
  c10::cuda::CUDAGuard guard(x.device());
  auto property=at::cuda::getCurrentDeviceProperties();if(property->major!=9||property->minor!=0)return fallback();
  auto stream=at::cuda::getCurrentCUDAStream(x.get_device()).stream();cudaStreamCaptureStatus capturing;TORCH_CHECK(cudaStreamIsCapturing(stream,&capturing)==cudaSuccess,"Unable to query current stream");if(capturing!=cudaStreamCaptureStatusNone)return fallback();
  output=at::empty({x.size(0),weight.size(0)},x.options());
  const auto m=x.size(0),n=weight.size(0),k=x.size(1);const auto limit=at::cuda::getCUDABlasLtWorkspaceSize();
  const auto policy=at::globalContext().allowBF16ReductionCuBLAS();
  const auto aa=alignment(weight.const_data_ptr()),ba=alignment(x.const_data_ptr()),ca=alignment(output.const_data_ptr());
  const Key key{{x.get_device(),m,n,k,x.stride(0),x.stride(1),weight.stride(0),weight.stride(1),output.stride(0),output.stride(1),static_cast<int64_t>(x.scalar_type()),static_cast<int64_t>(policy),static_cast<int64_t>(limit),aa,ba,ca,0}};
  std::shared_ptr<Plan> plan;
  {
   std::lock_guard lock(mutex);auto found=plans.find(key);
   if(found!=plans.end())plan=found->second;
   else if(plans.size()<64){
    plan=std::make_shared<Plan>();cublasLtMatmulPreference_t preference{};
    bool good=cublasLtMatmulDescCreate(&plan->operation,CUBLAS_COMPUTE_32F,CUDA_R_32F)==CUBLAS_STATUS_SUCCESS;
    const cublasOperation_t transpose=CUBLAS_OP_T,no_transpose=CUBLAS_OP_N;
    if(good)good=cublasLtMatmulDescSetAttribute(plan->operation,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose))==CUBLAS_STATUS_SUCCESS;
    if(good)good=cublasLtMatmulDescSetAttribute(plan->operation,CUBLASLT_MATMUL_DESC_TRANSB,&no_transpose,sizeof(no_transpose))==CUBLAS_STATUS_SUCCESS;
    if(good)good=cublasLtMatrixLayoutCreate(&plan->a,CUDA_R_16BF,k,n,k)==CUBLAS_STATUS_SUCCESS;
    if(good)good=cublasLtMatrixLayoutCreate(&plan->b,CUDA_R_16BF,k,m,k)==CUBLAS_STATUS_SUCCESS;
    if(good)good=cublasLtMatrixLayoutCreate(&plan->c,CUDA_R_16BF,n,m,n)==CUBLAS_STATUS_SUCCESS;
    if(good)good=cublasLtMatmulPreferenceCreate(&preference)==CUBLAS_STATUS_SUCCESS;
    if(good)good=cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&limit,sizeof(limit))==CUBLAS_STATUS_SUCCESS;
    if(good)good=cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,&aa,sizeof(aa))==CUBLAS_STATUS_SUCCESS;
    if(good)good=cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,&ba,sizeof(ba))==CUBLAS_STATUS_SUCCESS;
    if(good)good=cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,&ca,sizeof(ca))==CUBLAS_STATUS_SUCCESS;
    if(good&&policy!=at::CuBLASReductionOption::AllowReducedPrecisionWithSplitK){
     const uint32_t mask=policy==at::CuBLASReductionOption::DisallowReducedPrecisionAllowSplitK?CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE|CUBLASLT_REDUCTION_SCHEME_NONE:CUBLASLT_REDUCTION_SCHEME_NONE;
     good=cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,&mask,sizeof(mask))==CUBLAS_STATUS_SUCCESS;
    }
    cublasLtMatmulHeuristicResult_t heuristic{};int count=0;
    if(good)good=cublasLtMatmulAlgoGetHeuristic(at::cuda::getCurrentCUDABlasLtHandle(),plan->operation,plan->a,plan->b,plan->c,plan->c,preference,1,&heuristic,&count)==CUBLAS_STATUS_SUCCESS&&count==1&&heuristic.state==CUBLAS_STATUS_SUCCESS&&heuristic.workspaceSize<=limit;
    if(preference)cublasLtMatmulPreferenceDestroy(preference);
    if(good){plan->algorithm=heuristic.algo;plan->valid=true;}plans.emplace(key,plan);
   }
  }
  if(!plan||!plan->valid)return fallback();
  // Retain ATen's current-handle/current-stream workspace ownership and policy.
  // No retained raw workspace pointer, and no reuse across different streams.
  auto workspace=at::cuda::getCUDABlasLtWorkspace();if(!workspace)return fallback();
  const float alpha=1.f,beta=0.f;
  auto result=cublasLtMatmul(at::cuda::getCurrentCUDABlasLtHandle(),plan->operation,&alpha,weight.const_data_ptr(),plan->a,x.const_data_ptr(),plan->b,&beta,output.const_data_ptr(),plan->c,output.mutable_data_ptr(),plan->c,&plan->algorithm,workspace,limit,stream);
  if(result==CUBLAS_STATUS_ALLOC_FAILED)
   throw VendorAllocationError{"Cached decoder cublasLtMatmul",static_cast<int32_t>(result)};
  if(result!=CUBLAS_STATUS_SUCCESS) {
   // Retire only this generation. Other callers may still own the immutable
   // plan; a negative entry prevents repeatedly retrying a rejected algorithm.
   {
    std::lock_guard lock(mutex);auto found=plans.find(key);
    if(found!=plans.end()&&found->second==plan)found->second=std::make_shared<Plan>();
   }
   return fallback();
  }
  return output;
 }
};
}

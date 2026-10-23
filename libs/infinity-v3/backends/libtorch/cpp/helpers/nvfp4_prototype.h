// Copyright 2026 Infinity Core Contributors.
// Covered by Infinity Restricted MIT License 1.0; see LICENSE.
#pragma once
#include <ATen/ATen.h>
#include <ATen/ops/_scaled_mm_v2.h>
#include <ATen/cuda/CUDAContext.h>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <memory>
#include "helpers/fp8_linear.h"
#include "kernels/nvfp4_prototype.h"

namespace tei {
// Research only.
// Emulation measures quality, never FP4 throughput. Native mode needs Blackwell.
class Nvfp4PrototypeLinear {
  std::string mode,name,path;
  at::Tensor matrix,scales,global,input_global;
  std::unique_ptr<Fp8Linear> calibration_baseline;
 public:
  Nvfp4PrototypeLinear(const at::Tensor& weight,std::string mode_,std::string name_,std::string path_)
      :mode(std::move(mode_)),name(std::move(name_)),path(std::move(path_)) {
    TORCH_CHECK(mode=="emulate" || mode=="native" || mode=="calibrate","Unknown NVFP4 prototype mode");
    TORCH_CHECK(!path.empty(),"NVFP4 prototype requires activation calibration file");
    if(mode=="calibrate") {calibration_baseline=std::make_unique<Fp8Linear>(weight);return;}
    std::ifstream in(path);std::string key;float value,maximum=0;
    while(in>>key>>value)if(key==name)maximum=std::max(maximum,value);
    TORCH_CHECK(std::isfinite(maximum) && maximum>0,"Missing NVFP4 activation calibration for ",name);
    input_global=at::scalar_tensor(maximum,weight.options().dtype(at::kFloat))/(448.f*6.f);
    if(mode=="native")TORCH_CHECK(weight.is_cuda() && at::cuda::getDeviceProperties(weight.device().index())->major==10,
                                  "Native NVFP4 prototype requires B200/B300 (SM100/103)");
    auto weight_global=weight.abs().amax().to(at::kFloat).clamp_min(1.e-20)/(448.f*6.f);
    auto q=nvfp4_prototype_quantize_cuda(weight.contiguous(),weight_global,mode=="emulate");
    if(mode=="emulate")matrix=q.dequantized;
    else {matrix=q.packed;scales=q.scales;global=weight_global;}
  }
  at::Tensor forward(const at::Tensor& x) const {
    if(calibration_baseline) {
      std::ofstream out(path,std::ios::app);
      TORCH_CHECK(out.good(),"Cannot write NVFP4 calibration ",path);
      out<<name<<' '<<std::setprecision(9)<<x.abs().amax().item<float>()<<'\n';
      return calibration_baseline->forward(x);
    }
    auto q=nvfp4_prototype_quantize_cuda(x.contiguous(),input_global,mode=="emulate");
    if(mode=="emulate")return at::matmul(q.dequantized,matrix.t());
    // BlockWise1x16 + TensorWise global, canonical cuBLAS 32/4/4 scales.
    return at::_scaled_mm_v2(q.packed,matrix.t(),{q.scales,input_global},{2,0},{1,0},
                             {scales,global},{2,0},{1,0},std::nullopt,x.scalar_type());
  }
};
} // namespace tei

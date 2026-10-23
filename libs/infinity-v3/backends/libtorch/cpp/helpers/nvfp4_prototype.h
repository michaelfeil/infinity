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

namespace tei {
// Research only: deliberately uses unfused ATen quantization. Emulation measures
// quantization quality, never FP4 throughput. Native mode requires Blackwell.
struct Nvfp4PrototypeOperand {
  at::Tensor codes, scales, global, dequantized;
};
inline Nvfp4PrototypeOperand nvfp4_prototype_quantize(const at::Tensor& x,
                                                    const at::Tensor& amax={},bool dequantize=true) {
  TORCH_CHECK(x.dim()==2 && x.size(1)%16==0 && x.scalar_type()==at::kBFloat16,
              "NVFP4 prototype requires BF16 matrices with K divisible by 16");
  if(x.size(0)>2048) {
    // Bound reference-quantizer scratch space without changing tensor scales.
    auto maximum=amax.defined()?amax:x.abs().amax().to(at::kFloat);
    Nvfp4PrototypeOperand out{at::empty(x.sizes(),x.options().dtype(at::kByte)),
      at::empty({x.size(0),x.size(1)/16},x.options().dtype(at::kFloat8_e4m3fn)),
      maximum.clamp_min(1.e-20)/(448.f*6.f),dequantize?at::empty_like(x):at::Tensor()};
    for(int64_t start=0;start<x.size(0);start+=2048) {
      auto n=std::min<int64_t>(2048,x.size(0)-start);
      auto q=nvfp4_prototype_quantize(x.narrow(0,start,n),maximum,dequantize);
      out.codes.narrow(0,start,n).copy_(q.codes);out.scales.narrow(0,start,n).copy_(q.scales);
      if(dequantize)out.dequantized.narrow(0,start,n).copy_(q.dequantized);
    }
    return out;
  }
  auto f=x.to(at::kFloat).reshape({x.size(0),x.size(1)/16,16});
  auto global=(amax.defined()?amax:f.abs().amax()).clamp_min(1.e-20)/(448.f*6.f);
  auto scale=f.abs().amax(-1)/(6.f*global);
  scale=at::where(scale==0,at::ones_like(scale),scale).clamp_max(448).to(at::kFloat8_e4m3fn);
  auto combined=scale.to(at::kFloat)*global;
  auto normalized=at::where(combined.unsqueeze(-1)>0,f/combined.unsqueeze(-1),at::zeros_like(f));
  auto magnitude=normalized.abs().contiguous();
  auto bounds=at::tensor({.25f,.75f,1.25f,1.75f,2.5f,3.5f,5.f},f.options());
  auto ordinal=at::bucketize(magnitude,bounds,true,false);
  // E2M1 round-to-nearest, ties-to-even (including negative values).
  ordinal+=(magnitude==.75f).to(at::kInt)+(magnitude==1.75f).to(at::kInt)+(magnitude==3.5f).to(at::kInt);
  auto levels=at::tensor({0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f},f.options());
  auto dequantized=dequantize?(levels.index_select(0,ordinal.flatten()).reshape(f.sizes())*
                    at::where(normalized<0,-1.f,1.f)*combined.unsqueeze(-1)).reshape(x.sizes()).to(x.scalar_type()):at::Tensor();
  auto codes=(ordinal+((normalized<0).to(at::kInt)*8)).reshape(x.sizes()).to(at::kByte);
  return {codes,scale,global,dequantized};
}
inline at::Tensor nvfp4_prototype_pack(const at::Tensor& codes) {
  auto pair=codes.reshape({codes.size(0),-1,2});
  return (pair.select(-1,0)+pair.select(-1,1)*16).to(at::kByte).contiguous().view(at::kFloat4_e2m1fn_x2);
}
inline at::Tensor nvfp4_prototype_swizzle(const at::Tensor& scales) {
  const auto m=(scales.size(0)+127)/128*128,k=(scales.size(1)+3)/4*4;
  auto padded=at::zeros({m,k},scales.options());
  padded.narrow(0,0,scales.size(0)).narrow(1,0,scales.size(1)).copy_(scales);
  return padded.reshape({m/128,4,32,k/4,4}).permute({0,3,2,1,4}).contiguous().flatten();
}
class Nvfp4PrototypeLinear {
  std::string mode,name,path;
  at::Tensor matrix,scales,global,input_amax;
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
    input_amax=at::scalar_tensor(maximum,weight.options().dtype(at::kFloat));
    if(mode=="native")TORCH_CHECK(weight.is_cuda() && at::cuda::getDeviceProperties(weight.device().index())->major==10,
                                  "Native NVFP4 prototype requires B200/B300 (SM100/103)");
    auto q=nvfp4_prototype_quantize(weight,{},mode=="emulate");
    if(mode=="emulate")matrix=q.dequantized;
    else {matrix=nvfp4_prototype_pack(q.codes);scales=nvfp4_prototype_swizzle(q.scales);global=q.global;}
  }
  at::Tensor forward(const at::Tensor& x) const {
    if(calibration_baseline) {
      std::ofstream out(path,std::ios::app);
      TORCH_CHECK(out.good(),"Cannot write NVFP4 calibration ",path);
      out<<name<<' '<<std::setprecision(9)<<x.abs().amax().item<float>()<<'\n';
      return calibration_baseline->forward(x);
    }
    auto q=nvfp4_prototype_quantize(x,input_amax,mode=="emulate");
    if(mode=="emulate")return at::matmul(q.dequantized,matrix.t());
    auto packed=nvfp4_prototype_pack(q.codes);
    auto input_scales=nvfp4_prototype_swizzle(q.scales);
    // BlockWise1x16 + TensorWise global, canonical cuBLAS 32/4/4 scales.
    return at::_scaled_mm_v2(packed,matrix.t(),{input_scales,q.global},{2,0},{1,0},
                             {scales,global},{2,0},{1,0},std::nullopt,x.scalar_type());
  }
};
} // namespace tei

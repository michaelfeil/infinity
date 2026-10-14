// SPDX-License-Identifier: MIT
#include "kernels/moe_kernels.h"
#include "kernels/fast_kernels.h"
#include "kernels/qwen35_kernels.h"
#include <iostream>

int main() {
 try {
  at::manual_seed(773);
  auto opts=at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  for(int experts:{128,256})for(bool renormalize:{false,true}) {
   const int tokens=13,hidden=128,intermediate=128;
   auto x=at::randn({tokens,hidden},opts)*.25;
   auto gu=at::randn({experts,2*intermediate,hidden},opts)*.04;
   auto down=at::randn({experts,hidden,intermediate},opts)*.04;
   auto logits=at::randn({tokens,experts},opts.dtype(at::kFloat));
   // Stable ties and experts with no routed rows are both valid.
   logits[0].fill_(0);
   auto actual=tei::routed_moe_cuda(x,logits,gu,down,renormalize);
   TORCH_CHECK(actual.defined(),"Native BF16 routed kernel unavailable");
   auto p=at::softmax(logits,-1);
   auto ids=at::argsort(p,true,-1,true).narrow(-1,0,8);
   auto routes=p.gather(-1,ids);
   if(renormalize)routes=routes/routes.sum(-1,true);
   auto both=at::bmm(x.to(at::kFloat).unsqueeze(0).expand({experts,tokens,hidden}),gu.to(at::kFloat).transpose(-2,-1)).to(at::kBFloat16);
   auto activation=tei::fused_gated_activation(both.reshape({experts*tokens,2*intermediate}),0).reshape({experts,tokens,intermediate});
   // FP32 down accumulators are weighted before each route's BF16 store.
   auto acc=at::bmm(activation.to(at::kFloat),down.to(at::kFloat).transpose(-2,-1)).transpose(0,1);
   auto selected=acc.gather(1,ids.unsqueeze(-1).expand({tokens,8,hidden}));
   auto reference=(selected*routes.unsqueeze(-1)).to(at::kBFloat16).to(at::kFloat).sum(1).to(at::kBFloat16);
   auto delta=(actual.to(at::kFloat)-reference.to(at::kFloat)).abs().max().item<float>();
   TORCH_CHECK(delta<.002,"FP32 routed epilogue oracle mismatch: ",delta);
   auto cos=at::cosine_similarity(actual.to(at::kFloat),reference.to(at::kFloat),-1).min().item<float>();
   TORCH_CHECK(cos>.9999,"Routed oracle cosine mismatch: ",cos);
   std::cout<<"experts="<<experts<<" renormalize="<<renormalize<<" max_abs="<<delta<<" cosine="<<cos<<"\n";
  }
  {
   const int experts=128,tokens=3,hidden=2816,intermediate=704;
   auto x=at::randn({tokens,hidden},opts)*.25;
   auto gu=at::randn({experts,2*intermediate,hidden},opts)*.02;
   auto down=at::randn({experts,hidden,intermediate},opts)*.02;
   auto logits=at::randn({tokens,experts},opts.dtype(at::kFloat));
   auto scales=at::linspace(.25,1.5,experts,logits.options());
   auto actual=tei::routed_moe_cuda(x,logits,gu,down,true,scales);
   TORCH_CHECK(actual.defined(),"Native Gemma weighted epilogue unavailable");
   auto p=at::softmax(logits,-1);
   auto ids=at::argsort(p,true,-1,true).narrow(-1,0,8);
   auto routes=p.gather(-1,ids);routes=routes/routes.sum(-1,true);
   routes=routes*scales.index_select(0,ids.flatten()).view_as(routes);
   auto both=at::bmm(x.to(at::kFloat).unsqueeze(0).expand({experts,tokens,hidden}),gu.to(at::kFloat).transpose(-2,-1)).to(at::kBFloat16);
   auto activation=tei::gemma_gated_cuda(both.reshape({experts*tokens,2*intermediate}),true).reshape({experts,tokens,intermediate});
   auto acc=at::bmm(activation.to(at::kFloat),down.to(at::kFloat).transpose(-2,-1)).transpose(0,1);
   auto selected=acc.gather(1,ids.unsqueeze(-1).expand({tokens,8,hidden}));
   auto reference=(selected*routes.unsqueeze(-1)).to(at::kBFloat16).to(at::kFloat).sum(1).to(at::kBFloat16);
   auto delta=(actual.to(at::kFloat)-reference.to(at::kFloat)).abs().max().item<float>();
   auto cos=at::cosine_similarity(actual.to(at::kFloat),reference.to(at::kFloat),-1).min().item<float>();
   TORCH_CHECK(delta<.002&&cos>.9999,"Gemma FP32 weighted epilogue mismatch: ",delta," / ",cos);
   std::cout<<"Gemma learned expert scales max_abs="<<delta<<" cosine="<<cos<<"\n";
  }
  TORCH_CHECK(!tei::routed_moe_cuda(at::zeros({2,16}),{}, {},{},true).defined(),"CPU must retain reference path");
  std::cout<<"Native routed MoE FP32 epilogue tests passed\n";
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}

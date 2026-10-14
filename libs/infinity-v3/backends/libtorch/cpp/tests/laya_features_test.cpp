// Fork contribution; see LICENSE and LICENSING.md at the repository root.
#include "kernels/laya_features.h"
#include <c10/core/InferenceMode.h>
#include <cmath>
#include <iostream>
int main(){
 c10::InferenceMode inference;
 const auto device=c10::Device("cuda:0");
 std::vector<at::Tensor> rows,expected;
 std::vector<int64_t> offsets{0};
 for(auto count:{1,2,5,16,32,127,255,513}){
  for(auto scale:{0.f,.1f,10.f,100.f}){
   auto logits=at::sin(at::arange(count).to(at::kFloat)*.17)*scale;
   if(count>1&&scale==0)logits.fill_(100.f); // Tied maxima and uniform entropy.
   rows.push_back(logits);offsets.push_back(offsets.back()+count);
   auto probabilities=at::softmax(logits,0),sorted=std::get<0>(probabilities.sort(0,true));
   auto top1=sorted.select(0,0),top2=count>1?sorted.select(0,1):at::zeros_like(top1);
   auto entropy=-(probabilities*probabilities.clamp_min(1e-9).log()).sum()/std::log(double(std::max(2,count)));
   expected.push_back(at::stack({top1,top1-top2,entropy,at::full({},double(std::max(2,count))/255.)}));
  }
 }
 auto logits=at::cat(rows).to(device),indices=at::tensor(offsets,at::TensorOptions().dtype(at::kLong)).to(device);
 auto actual=tei::laya_action_features(logits,indices).cpu();
 auto reference=at::stack(expected);
 TORCH_CHECK((actual-reference).abs().max().item<float>()<2e-6,"Fused Laya features changed published formulas");
 TORCH_CHECK(at::isfinite(actual).all().item<bool>(),"Fused features produced nonfinite output");
 auto changed=logits.clone();changed.narrow(0,offsets[1],offsets[2]-offsets[1]).fill_(-37.);
 auto updated=tei::laya_action_features(changed,indices).cpu();
 TORCH_CHECK(at::equal(actual[0],updated[0])&&at::equal(actual.narrow(0,2,actual.size(0)-2),updated.narrow(0,2,actual.size(0)-2)),"Feature reduction leaked across option segments");
 std::cout<<"Laya fused features: singleton, tied maxima, count 1-513, extreme logits, entropy clamp and segment isolation passed\n";
}

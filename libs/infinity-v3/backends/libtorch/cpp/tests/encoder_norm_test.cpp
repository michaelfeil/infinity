// Native CUDA test for the exact Candle normalization adapter.
#include "kernels/encoder_norm.h"
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <c10/cuda/CUDAGuard.h>

int main(int argc,char** argv) {
  c10::InferenceMode inference;
  for(auto dtype:{at::kHalf,at::kBFloat16}) {
    auto opts=at::TensorOptions().device(c10::Device(argc>1?argv[1]:"cuda:0")).dtype(dtype);
    {
      auto x=at::randn({4097,1024},opts),residual=at::randn_like(x);
      auto weight=at::randn({1024},opts),bias=at::randn_like(weight);
      auto actual=tei::encoder_layer_norm(x,weight,1e-5,residual,bias,true);
      auto a=tei::encoder_layer_norm(x.narrow(0,0,4095),weight,1e-5,residual.narrow(0,0,4095),bias,true);
      auto b=tei::encoder_layer_norm(x.narrow(0,4095,2),weight,1e-5,residual.narrow(0,4095,2),bias,true);
      TORCH_CHECK(at::equal(actual.first.view(at::kByte),at::cat({a.first,b.first},0).view(at::kByte)),
        "Large norm row grouping changes the original per-row reduction bits");
      TORCH_CHECK(at::equal(actual.second,residual+x),"Large norm residual mismatch");
      auto scratch=at::empty_like(x),owned=residual.clone();
      auto reused=tei::encoder_layer_norm_update_residual_scratch(x,weight,1e-5,owned,scratch);
      auto expected=tei::encoder_layer_norm(x,weight,1e-5,residual);
      TORCH_CHECK(at::equal(reused.first.view(at::kByte),expected.first.view(at::kByte))&&
        at::equal(reused.second,expected.second),"Large norm scratch update changes output");
    }
    auto base=at::full({2,768},32,opts);
    auto delta=(at::arange(768,opts.dtype(at::kFloat)).remainder(2)*.008-.004).to(dtype).repeat({2,1});
    auto gamma=at::ones({768},opts),beta=at::full({768},.25,opts);
    auto exact=tei::encoder_layer_norm(base,gamma,1e-5,delta,beta,false);
    auto unrounded=base.to(at::kFloat)+delta.to(at::kFloat);
    auto oracle=at::layer_norm(unrounded,{768},gamma.to(at::kFloat),beta.to(at::kFloat),1e-5).to(dtype);
    TORCH_CHECK((exact.first-oracle).abs().max().item<float>()<=(dtype==at::kHalf?.003:.03125),"Unrounded residual statistics or beta differs");
    TORCH_CHECK(at::equal(exact.second,base+delta),"Returned residual sum must retain model dtype");
    auto only=tei::encoder_layer_norm_only(base,gamma,1e-5,delta,beta,false);
    TORCH_CHECK((only-oracle).abs().max().item<float>()<=(dtype==at::kHalf?.003:.03125),"Normalized-only unrounded statistics or beta differs from independent FP32 oracle");
    TORCH_CHECK(at::equal(only,exact.first),"Suppressing residual output changed exact arithmetic");
    auto rounded=tei::encoder_layer_norm(base,gamma,1e-5,delta,beta,true);
    TORCH_CHECK((exact.first-rounded.first).abs().max().item<float>()>.5,"Rounding discriminator did not exercise distinct normalization contracts");
    {
      auto x=at::randn({17,384},opts),residual=at::randn_like(x)*.2;
      auto scale=at::randn({384},opts)*.1+1,bias=at::randn({384},opts)*.1;
      auto stats=x.to(at::kFloat)+residual.to(at::kFloat);
      auto reference=at::layer_norm(stats,{384},scale.to(at::kFloat),bias.to(at::kFloat),1e-12).to(dtype);
      auto actual=tei::encoder_layer_norm_only(x,scale,1e-12,residual,bias,false);
      TORCH_CHECK((actual-reference).abs().max().item<float>()<=(dtype==at::kHalf?.003:.03125),"Width384 biased unrounded norm differs from independent FP32 reference");
    }
    if(dtype==at::kHalf) {
      // Both rounding boundaries matter: round word+type, retain FP32 position.
      auto word=at::full({17,768},32,opts);
      auto type=(at::arange(768,opts.dtype(at::kFloat)).remainder(3)*.016-.008).to(dtype).repeat({17,1});
      auto position=(at::arange(768,opts.dtype(at::kFloat)).remainder(2)*.008-.004).to(dtype).repeat({17,1});
      auto fused=tei::encoder_embedding_layer_norm(word,type,position,gamma,1e-5,beta);
      auto separate=tei::encoder_layer_norm_only(word+type,gamma,1e-5,position,beta,false);
      TORCH_CHECK(at::equal(fused,separate),"Three-input embedding norm changed sequential Half addition or norm arithmetic");
      auto all_float=word.to(at::kFloat)+type.to(at::kFloat)+position.to(at::kFloat);
      auto wrong_first=at::layer_norm(all_float,{768},gamma.to(at::kFloat),beta.to(at::kFloat),1e-5).to(dtype);
      auto wrong_second=tei::encoder_layer_norm_only(word+type,gamma,1e-5,position,beta,true);
      TORCH_CHECK((fused-wrong_first).abs().max().item<float>()>.1,"Embedding fixture does not discriminate missing first Half rounding");
      TORCH_CHECK((fused-wrong_second).abs().max().item<float>()>.1,"Embedding fixture does not discriminate erroneous position rounding");
      auto random_word=at::randn({17,768},opts),random_type=at::randn_like(random_word),random_position=at::randn_like(random_word);
      auto random_gamma=at::randn({768},opts)*.1+1,random_beta=at::randn({768},opts)*.1;
      auto actual=tei::encoder_embedding_layer_norm(random_word,random_type,random_position,random_gamma,1e-12,random_beta);
      auto expected=tei::encoder_layer_norm_only(random_word+random_type,random_gamma,1e-12,random_position,random_beta,false);
      TORCH_CHECK(at::equal(actual,expected),"Three-input embedding norm differs from original exact adapter");
    }
    for(int64_t width:{8,32,40,256,384,768,1024,1152,1536,2048}) {
      auto options=at::TensorOptions().device(c10::Device(argc>1?argv[1]:"cuda:0")).dtype(dtype);
      auto x=at::randn({7,width},options),residual=at::randn_like(x);
      auto weight=at::randn({width},options)*0.1+1;
      for(bool add:{false,true}) {
        auto result=tei::encoder_layer_norm(x,weight,1e-5,add?residual:at::Tensor());
        auto sum=add?x+residual:x;
        auto expected=at::layer_norm(sum.to(at::kFloat),{width},weight.to(at::kFloat),at::Tensor(),1e-5).to(dtype);
        auto normalized_only=tei::encoder_layer_norm_only(x,weight,1e-5,add?residual:at::Tensor());
        TORCH_CHECK((normalized_only-expected).abs().max().item<float>()<=(dtype==at::kHalf?0.003:0.03125),"Normalized-only differs from independent FP32 oracle at width ",width);
        TORCH_CHECK(at::equal(normalized_only,result.first),"Normalized-only changed rounded normalization at width ",width);
        auto error=(result.first-expected).abs().max().item<float>();
        TORCH_CHECK(error<=(dtype==at::kHalf?0.003:0.03125),"Encoder norm error ",error," at width ",width);
        if(add) {
          TORCH_CHECK(at::equal(result.second,sum),"Residual sum must round exactly to model dtype");
          auto owned=residual.clone(),saved_x=x.clone();
          auto updated=tei::encoder_layer_norm_update_residual(x,weight,1e-5,owned);
          TORCH_CHECK(at::equal(updated.first,result.first)&&at::equal(updated.second,result.second),
            "Private residual update changed original bytes at width ",width);
          TORCH_CHECK(at::equal(x,saved_x),"Private residual update mutated projection input");
          if(width<=1536)TORCH_CHECK(updated.second.const_data_ptr()==owned.const_data_ptr(),
            "Supported private residual norm allocated replacement storage");
          auto next_expected=tei::encoder_layer_norm(x,weight,1e-5,updated.second);
          auto next=tei::encoder_layer_norm_update_residual(x,weight,1e-5,updated.second);
          TORCH_CHECK(at::equal(next.first,next_expected.first)&&at::equal(next.second,next_expected.second),
            "Repeated residual updates changed rounded normalization bytes");
          if(width==768) {
            auto model_stream=c10::cuda::getStreamFromPool(false,options.device().index());
            c10::cuda::CUDAStreamGuard model_guard(model_stream);
            auto scratch=at::empty_like(x),private_sum=residual.clone();
            auto reused=tei::encoder_layer_norm_update_residual_scratch(x,weight,1e-5,private_sum,scratch);
            TORCH_CHECK(reused.first.is_alias_of(scratch)&&at::equal(reused.first,result.first),"Private norm scratch changed original bytes");
            auto matrix=at::randn({width,32},options),read=at::matmul(reused.first,matrix);
            auto expected_read=at::matmul(result.first,matrix);
            auto final=tei::encoder_layer_norm_only(x,weight,1e-5,private_sum),saved_final=final.clone();
            auto next_expected=tei::encoder_layer_norm(x,weight,1e-5,private_sum);
            auto overwritten=tei::encoder_layer_norm_update_residual_scratch(x,weight,1e-5,private_sum,scratch);
            TORCH_CHECK(at::equal(read,expected_read)&&at::equal(overwritten.first,next_expected.first),"Scratch overwrite raced an earlier consuming GEMM");
            TORCH_CHECK(!final.is_alias_of(scratch)&&at::equal(final,saved_final),"Returned final norm aliases overwritten request scratch");
            for(const auto& invalid:{x,residual,weight.unsqueeze(0).expand_as(x),at::empty({width,7},options).t(),at::empty(x.sizes(),options.device(at::kCPU))}) {
              auto before=residual.clone(),saved=invalid.clone();
              auto fallback=tei::encoder_layer_norm_update_residual_scratch(x,weight,1e-5,residual,invalid);
              TORCH_CHECK(!fallback.first.is_alias_of(invalid)&&at::equal(fallback.first,result.first)&&at::equal(fallback.second,result.second),"Invalid scratch changed allocating fallback");
              TORCH_CHECK(at::equal(residual,before)&&at::equal(invalid.contiguous().view(at::kByte),saved.contiguous().view(at::kByte)),"Invalid scratch fallback mutated supplied storage bytes");
              if(!at::equal(invalid,saved))std::cout<<"Invalid scratch preserved bytes including NaNs on "<<invalid.device()<<"\n";
            }
          }
        } else {
          TORCH_CHECK(!result.second.defined(),"Plain normalization must not allocate residual output");
        }
      }
    }
  }
  std::cout<<"Exact encoder CUDA normalization tests passed\n";
}

#include "kernels/fast_kernels.h"
#include "kernels/transformers_norm.h"
#include "kernels/qwen35_vision.h"
#include "kernels/nomic_moe.h"
#include "kernels/encoder_activation.h"
#include <ATen/ops/_fused_rms_norm.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <c10/core/InferenceMode.h>
#include <iostream>
#include <cmath>
#include <bit>
void close(const at::Tensor& x,const at::Tensor& expected,double tolerance,const char* label){
 auto error=(x.to(at::kFloat)-expected.to(at::kFloat)).abs().max().item<float>();
 TORCH_CHECK(error<=tolerance,label," max abs=",error," tolerance=",tolerance);
}
uint64_t tensor_byte_hash(const at::Tensor& value) {
 auto host=value.to(at::kCPU).contiguous();
 auto bytes=static_cast<const unsigned char*>(host.const_data_ptr());
 uint64_t hash=1469598103934665603ull;
 for(size_t i=0;i<host.nbytes();++i)hash=(hash^bytes[i])*1099511628211ull;
 return hash;
}
// Round the double oracle directly to the model format. ATen's CPU double
// conversion passes through float, which double-rounds rare HFMA midpoint cases.
at::Tensor round_model_double(const at::Tensor& values,at::ScalarType dtype) {
 auto input=values.contiguous();auto output=at::empty(input.sizes(),input.options().dtype(dtype));
 const int mantissa=dtype==at::kHalf?10:7,bias=dtype==at::kHalf?15:127;
 const int maximum=dtype==at::kHalf?0x7bff:0x7f7f;
 auto decode=[&](int bits){int exponent=bits>>mantissa,fraction=bits&((1<<mantissa)-1);
  return std::ldexp(double(exponent?(1<<mantissa)+fraction:fraction),(exponent?exponent:1)-bias-mantissa);};
 const auto* source=input.const_data_ptr<double>();auto* dest=reinterpret_cast<uint16_t*>(output.mutable_data_ptr());
 for(int64_t i=0;i<input.numel();++i){double v=source[i],a=std::abs(v);TORCH_CHECK(std::isfinite(v)&&a<=decode(maximum),"Oracle requires finite representable range");
  int lo=0,hi=maximum;while(lo<hi){int mid=(lo+hi)/2;if(decode(mid)<a)lo=mid+1;else hi=mid;}
  int bits=lo;if(lo>0){double below=a-decode(lo-1),above=decode(lo)-a;if(below<above||(below==above&&((lo-1)&1)==0))bits=lo-1;}
  dest[i]=uint16_t(bits|(std::signbit(v)?0x8000:0));
 }
 return output;
}
int main(int argc,char** argv){c10::InferenceMode guard;
 {
  // Captured strict failure: generic Half4096 RMS returned -8.28125 at
  // row14,col3320; original Candle4096, FP32 and direct FP64 return -8.2734375.
  // Restore these exact operands rather than selecting a passing random seed.
  at::manual_seed(248);auto opts=at::TensorOptions().device(at::kCUDA).dtype(at::kHalf);
  at::Tensor x,residual,weight;
  for(int width:{7,16,63,128,257,513,768,769,1024,1152,2048,4096}) {
   x=at::randn({33,width},opts);residual=at::randn_like(x);weight=at::randn({width},opts);
  }
  TORCH_CHECK(tensor_byte_hash(x)==9953450669979115434ull
   &&tensor_byte_hash(residual)==4182254403794363395ull
   &&tensor_byte_hash(weight)==14218794327604558931ull,"Captured RMS operands changed");
  auto actual=tei::fused_add_rms_norm(x,residual,weight,1e-5);
  TORCH_CHECK(tensor_byte_hash(actual.first)==13182671632785942803ull,
   "Half4096 RMS differs from independently executed original registered Candle kernel");
  auto unrounded=x.to(at::kFloat)+residual.to(at::kFloat);
  auto reference=(unrounded*at::rsqrt(unrounded.square().mean(-1,true)+1e-5)*weight.to(at::kFloat)).to(at::kHalf);
  close(actual.first,reference,.006,"Captured Half4096 original RMS gate");
  close(actual.second,x+residual,0,"Captured RMS residual model-dtype bytes");
 }

 {
  auto opts=at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  auto unaligned=[](const at::Tensor& value) {
   auto storage=at::empty({value.numel()+1},value.options());
   auto view=storage.narrow(0,1,value.numel()).view_as(value);
   return view.copy_(value);
  };
  for(int rows:{127,128,257})for(bool residual:{false,true}) {
   auto x=at::randn({rows,4096},opts),r=at::randn_like(x),w=at::randn({4096},opts);
   for(double eps:{1e-6,1e-5}) {
    auto expected=tei::fused_add_rms_norm(unaligned(x),residual?r:at::Tensor(),w,eps);
    auto actual=tei::fused_add_rms_norm(x,residual?r:at::Tensor(),w,eps);
    close(actual.first,expected.first,0,"BF16 vector RMS versus scalar fallback");
    close(actual.second,expected.second,0,"BF16 vector RMS residual");
    if(!residual)TORCH_CHECK(actual.second.is_alias_of(x),"BF16 RMS input alias");
   }
  }
  for(int width:{12288,14336}) {
   auto both=at::randn({17,2*width},opts)*8;
   close(tei::fused_gated_activation(both,5),
     tei::fused_gated_activation(unaligned(both),5),0,
     "BF16 paired SiLU at Qwen3/Llama MLP widths");
  }
 }

 {
  // Actual learned layer7/token678 router values: the previous softmax lands
  // exactly on a Half midpoint, which changes later MoE embedding outputs.
  auto logits=at::tensor({.10430908203125f,.5166015625f,-.0386962890625f,
    -1.689453125f,.90869140625f,.60693359375f,-2.291015625f,.122314453125f},
    at::TensorOptions().device(at::kCUDA).dtype(at::kFloat)).view({1,8});
  std::vector<float> reference;
  for(uint32_t bits:{0x3defc892u,0x3e3511d2u,0x3dcfd50au,0x3c9f892bu,
      0x3e85ff77u,0x3e462fffu,0x3c2ed63au,0x3df423d3u})
    reference.push_back(std::bit_cast<float>(bits));
  auto expected=at::tensor(reference,logits.options()).view({1,8});
  auto probabilities=tei::qwen35_vision_softmax(logits);
  close(probabilities,expected,0,"Candle router FP32 midpoint");
  auto selected=tei::nomic_router_topk8(probabilities,2);
  close(selected.first.to(at::kHalf),at::tensor({.26171875f,.1934814453125f},
    logits.options()).to(at::kHalf).view({1,2}),0,"Original selected Half weights");
  TORCH_CHECK(selected.second.equal(at::tensor({4,5},
    logits.options().dtype(at::kLong)).view({1,2})),"Original router indices");
 }

 {
  // Original router1 row250 and router11 row148 tied probabilities, plus
  // all-equal and strict-order cases: index policy and unchanged score dtype.
  auto p=at::tensor({
    .10291401296854019f,.03378089517354965f,.04312211647629738f,.07706139981746674f,.20502214133739471f,.038391076028347015f,.2946861982345581f,.20502214133739471f,
    .13516762852668762f,.11556561291217804f,.15189653635025024f,.14100748300552368f,.08601745963096619f,.09910222142934799f,.13023561239242554f,.14100748300552368f,
    .125f,.125f,.125f,.125f,.125f,.125f,.125f,.125f,
    0.f,1.f,2.f,3.f,4.f,5.f,6.f,7.f},at::TensorOptions().dtype(at::kFloat)).view({4,8}).cuda();
  auto expected=at::tensor({6,7,4,0,3,2,5,1,2,3,7,0,6,1,5,4,0,1,2,3,4,5,6,7,7,6,5,4,3,2,1,0},at::TensorOptions().dtype(at::kLong)).view({4,8}).cuda();
  for(int k:{1,2,8}) {
    auto selected=tei::nomic_router_topk8(p,k);auto ids=expected.narrow(-1,0,k);
    TORCH_CHECK(at::equal(selected.second,ids),"Nomic router must match pinned Candle bitonic tie order");
    TORCH_CHECK(selected.first.scalar_type()==at::kFloat&&selected.second.scalar_type()==at::kLong,"Nomic router score/index dtype changed");
    TORCH_CHECK(at::equal(selected.first,p.gather(-1,ids)),"Nomic router probabilities changed during selection");
    TORCH_CHECK(at::equal(selected.first.to(at::kHalf),p.gather(-1,ids).to(at::kHalf)),"Nomic original Half score conversion changed");
  }
 }

 {
  auto opts=at::TensorOptions().device(at::kCUDA).dtype(at::kHalf);
  auto words=at::randn({257,768},opts),types=at::randn({3,768},opts),positions=at::randn({514,768},opts);
  auto weight=at::randn({768},opts),bias=at::randn({768},opts);
  // Exact packed ragged rows: lengths32/64/128/512, repeated/nonzero IDs and
  // router offset2 including final legal position513. No padded rows exist.
  std::vector<int64_t> token_ids,type_ids,position_ids;
  for(int length:{32,64,128,512})for(int i=0;i<length;++i){token_ids.push_back((i*17+31)%257);type_ids.push_back(i%3);position_ids.push_back(i+2);}
  auto longs=opts.dtype(at::kLong);
  auto ids=at::tensor(token_ids,longs),tt=at::tensor(type_ids,longs),pp=at::tensor(position_ids,longs);
  auto word=at::embedding(words,ids),type=at::embedding(types,tt),position=at::embedding(positions,pp);
  auto actual=tei::fused_embedding_layer_norm(ids,tt,pp,words,types,positions,weight,bias,1e-5);
  close(actual,tei::fused_add_layer_norm(word,type,weight,bias,1e-5,position),0,"Embedding fusion preserves staged model-dtype kernel bits");
  // Independent CPU formula: dtype-rounded additions, FP32 mean/variance,
  // affine then final model-dtype cast. This guards the operation contract.
  auto staged=(word.to(at::kCPU)+type.to(at::kCPU)+position.to(at::kCPU)).to(at::kFloat);
  auto centered=staged-staged.mean(-1,true);
  auto reference=(centered*at::rsqrt(centered.square().mean(-1,true)+1e-5)*weight.to(at::kCPU,at::kFloat)+bias.to(at::kCPU,at::kFloat)).to(at::kHalf).to(actual.device());
  close(actual,reference,.006,"Embedding fusion independent rounded CPU oracle");
  auto empty=ids.narrow(0,0,0);
  TORCH_CHECK(tei::fused_embedding_layer_norm(empty,empty,empty,words,types,positions,weight,bias,1e-5).sizes()==at::IntArrayRef({0,768}),"Empty embedding shape");
 }

 for(auto dtype:{at::kHalf,at::kBFloat16,at::kFloat}) {
  auto opts=at::TensorOptions().device(at::kCUDA).dtype(dtype);
  for(int width:{7,16,63,128,257,513,768,769,1024,1152,2048,2624,4096,8192}) {
   auto x=at::randn({33,width},opts),r=at::randn_like(x),r2=at::randn_like(x);
   auto w=at::randn({width},opts),b=at::randn({width},opts);
   auto tol=dtype==at::kBFloat16?0.04:dtype==at::kHalf?0.006:0.00001;
   close(tei::fused_add_layer_norm(x,r,w,b,1e-5),at::layer_norm(x+r,{width},w,b,1e-5),tol,"residual LN");
   close(tei::fused_add_layer_norm(x,r,w,b,1e-5,r2),at::layer_norm(x+r+r2,{width},w,b,1e-5),tol,"two residual LN");
   auto rms=tei::fused_add_rms_norm(x,r,w,1e-5);
   close(rms.second,x+r,0,"RMS residual output");
   auto unrounded=x.to(at::kFloat)+r.to(at::kFloat);
   auto rms_reference=(unrounded*at::rsqrt(unrounded.square().mean(-1,true)+1e-5))*w.to(at::kFloat);
   close(rms.first,rms_reference.to(dtype),tol,"unrounded residual RMSNorm");
   auto initial=tei::fused_add_rms_norm(x,{},w,1e-5);
   close(initial.second,x,0,"Initial RMS residual aliases original input");
   TORCH_CHECK(initial.second.data_ptr()==x.data_ptr(),"Initial RMS residual must reuse input storage");
   auto both=at::randn({33,2*width},opts);
   for(int code=0;code<4;++code)for(bool gate_first:{false,true}) {
     auto gate=both.narrow(-1,gate_first?0:width,width),up=both.narrow(-1,gate_first?width:0,width);
     auto activated=code==0?at::silu(gate):code==1?at::gelu(gate,"none"):code==2?at::gelu(gate,"tanh"):at::relu(gate);
     close(tei::fused_gated_activation(both,code,gate_first),activated*up,tol,"gated activation");
   }
   for(bool gate_first:{false,true}) {
     auto gate=both.narrow(-1,gate_first?0:width,width),up=both.narrow(-1,gate_first?width:0,width);
     auto scalar=[&](double value){return at::full({},value,opts);};
     auto square=gate*gate,cube=square*gate;
     // Candle/NVCC contracts this expression into a modeldtype FMA.
     // FP64 exactly represents these bounded half/BF16 operands' product
     // and sum; round once instead of inserting an extra product rounding.
     auto alpha=(gate.to(at::kDouble)+scalar(.044715).to(at::kDouble)*cube.to(at::kDouble)).to(dtype);
     auto tangent=at::tanh(scalar(.7978845608028654)*alpha);
     auto precise=(scalar(.5)*gate)*(scalar(1)+tangent);
     close(tei::fused_gated_activation(both,4,gate_first),precise*up,tol,"Candle precise gated GELU");
   }
   for(bool gate_first:{false,true}) {
    auto gate=both.narrow(-1,gate_first?0:width,width),up=both.narrow(-1,gate_first?width:0,width);
    auto exponential=at::exp(-gate),denominator=at::ones_like(gate)+exponential;
    close(tei::fused_gated_activation(both,5,gate_first),(gate/denominator)*up,tol,"Candle precise gated SiLU");
   }
   if(width%4==0) {
    for(int offset:{1,4}) {
      auto storage=at::empty({both.numel()+offset},opts);
      auto unaligned=storage.narrow(0,offset,both.numel()).view_as(both);
      unaligned.copy_(both);
      for(int code=0;code<=5;++code)for(bool gate_first:{false,true})
        close(tei::fused_gated_activation(both,code,gate_first),tei::fused_gated_activation(unaligned,code,gate_first),0,"aligned/misaligned gated activation parity");
    }
   }
   for(bool approx:{false,true}) {
    close(tei::fused_gelu(x,approx),at::gelu(x,approx?"tanh":"none"),tol,"GELU");
    close(tei::fused_bias_gelu(x,b,approx),at::gelu(x+b,approx?"tanh":"none"),tol,"bias GELU");
   }
  }
  if(dtype!=at::kFloat){
   auto qkv=at::randn({7,8,128},opts);
   auto q=qkv.narrow(1,0,4),k=qkv.narrow(1,4,2);
   auto qw=at::randn({128},opts),kw=at::randn({128},opts);
   auto angle=at::randn({7,1,64},opts);auto cosine=at::cos(angle),sine=at::sin(angle);
   auto rotary=tei::fused_rotary(q,k,cosine,sine);
   auto rope=[&](const at::Tensor& x){auto a=x.narrow(-1,0,64),b=x.narrow(-1,64,64);return at::cat({a*cosine-b*sine,b*cosine+a*sine},-1);};
   close(rotary.first,rope(q),0,"Q rotary");close(rotary.second,rope(k),0,"K rotary");
   // A double reference preserves the exact half/BF16 HFMA result and
   // catches FP32-to-model-dtype double rounding at midpoint values.
   auto contracted_ref=[&](const at::Tensor& x){
    auto cpu=x.to(at::kCPU,at::kDouble),c=cosine.to(at::kCPU,at::kDouble),s=sine.to(at::kCPU,at::kDouble);
    auto a=cpu.narrow(-1,0,64),b=cpu.narrow(-1,64,64);
    auto bs=(b*s).to(dtype).to(at::kDouble),as=(a*s).to(dtype).to(at::kDouble);
    return round_model_double(at::cat({a*c-bs,b*c+as},-1),dtype).to(x.device());
   };
   auto contracted=tei::fused_rotary(q,k,cosine,sine,true);
   close(contracted.first,contracted_ref(q),0,"Q contracted rotary");
   close(contracted.second,contracted_ref(k),0,"K contracted rotary");
   auto inplace_qkv=qkv.clone(),inplace_q=inplace_qkv.narrow(1,0,4),inplace_k=inplace_qkv.narrow(1,4,2);
   tei::encoder_rotary_inplace(inplace_q,inplace_k,cosine,sine);
   close(inplace_q,contracted.first,0,"Q inplace encoder rotary");
   close(inplace_k,contracted.second,0,"K inplace encoder rotary");
   close(inplace_qkv.narrow(1,6,2),qkv.narrow(1,6,2),0,"Inplace encoder rotary preserves V");
   auto positions=at::tensor(std::vector<int64_t>{2,1,4,0,3,2,1},opts.dtype(at::kLong));
   auto expected_indexed=qkv.clone(),actual_indexed=qkv.clone();
   tei::encoder_rotary_inplace(expected_indexed.narrow(1,0,4),expected_indexed.narrow(1,4,2),
     cosine.squeeze(1).index_select(0,positions).unsqueeze(1),sine.squeeze(1).index_select(0,positions).unsqueeze(1));
   tei::encoder_rotary_indexed_inplace(actual_indexed.narrow(1,0,4),actual_indexed.narrow(1,4,2),
     cosine.squeeze(1),sine.squeeze(1),positions);
   TORCH_CHECK(at::equal(actual_indexed.view(at::kByte),expected_indexed.view(at::kByte)),
     "Indexed rotary differs IEEE bits on repeated/irregular positions or overwrites V");
   // Pitched GQA heads, incomplete launch blocks and partial-rotary fallback.
   // An even token pitch does not imply an even head pitch. Odd-width heads
   // must preserve their unrotated tail and every subsequent head's origin.
   for(int dim:{65,67})for(int offset:{0,1})for(bool indexed:{false,true}) {
    auto storage=at::randn({17*8*dim+offset},opts);
    auto packed=storage.narrow(0,offset,17*8*dim).view({17,8,dim});
    auto original=packed.clone(),qs=packed.narrow(1,0,4),ks=packed.narrow(1,4,2);
    auto angle=at::randn({17,1,32},opts),c=at::cos(angle),s=at::sin(angle);
    auto positions=at::arange(17,opts.dtype(at::kLong)).flip({0});
    auto expected=tei::fused_rotary(qs,ks,indexed?c.index_select(0,positions):c,
      indexed?s.index_select(0,positions):s,true);
    if(indexed)tei::encoder_rotary_indexed_inplace(qs,ks,c.squeeze(1),s.squeeze(1),positions);
    else tei::encoder_rotary_inplace(qs,ks,c,s);
    TORCH_CHECK(at::equal(qs,expected.first)&&at::equal(ks,expected.second),"Odd-width rotary changed head offsets or tail lanes");
    TORCH_CHECK(at::equal(packed.narrow(1,6,2),original.narrow(1,6,2)),"Odd-width rotary overwrote V");
   }
   for(int dim:{64,128,256})for(bool partial:{false,true})for(bool contract:{false,true}) {
    auto packed=at::randn({17,16,dim},opts),original=packed.clone();
    auto qs=packed.narrow(1,0,12),ks=packed.narrow(1,12,2);
    const int half=partial?dim/4:dim/2;
    auto c=at::randn({17,1,half},opts),s=at::randn_like(c);
    auto result=tei::fused_rotary(qs,ks,c,s,contract);
    auto reference=[&](const at::Tensor& input) {
     auto f=input.to(at::kCPU,at::kDouble),cd=c.to(at::kCPU,at::kDouble),sd=s.to(at::kCPU,at::kDouble);
     auto a=f.narrow(-1,0,half),b=f.narrow(-1,half,half);
     auto bs=(b*sd).to(dtype).to(at::kDouble),as=(a*sd).to(dtype).to(at::kDouble);
     auto ac=a*cd,bc=b*cd;
     if(!contract){ac=ac.to(dtype).to(at::kDouble);bc=bc.to(dtype).to(at::kDouble);}
     auto rotated=at::cat({ac-bs,bc+as},-1);
     if(partial)rotated=at::cat({rotated,f.narrow(-1,2*half,dim-2*half)},-1);
     return round_model_double(rotated,dtype).to(input.device());
    };
    close(result.first,reference(qs),0,"Q paired/partial rotary independent rounded oracle");
    close(result.second,reference(ks),0,"K paired/partial rotary independent rounded oracle");
    close(packed,original,0,"Out-of-place rotary preserves all QKV storage");
   }
   auto actual=tei::fused_qk_norm_rope(q,k,qw,kw,cosine,sine,1e-6);
   auto vstore=at::randn({k.size(0),2*k.size(1),128},k.options()),v=vstore.narrow(1,k.size(1),k.size(1));
   auto key_owner=at::full({k.size(0)+3,k.size(1),128},17,k.options());
   auto value_owner=at::full_like(key_owner,-17);
   auto key_target=key_owner.narrow(0,3,k.size(0)),value_target=value_owner.narrow(0,3,k.size(0));
   auto direct=tei::fused_qkv_norm_rope_into(q,k,v,qw,kw,cosine,sine,1e-6,key_target,value_target);
   close(direct,actual.first,0,"Direct Q normalization retains original arithmetic");
   close(key_target,actual.second,0,"Direct K normalization retains original arithmetic");
   TORCH_CHECK(at::equal(value_target.view(at::kByte),v.contiguous().view(at::kByte)),"Direct V copy changed raw bits");
   TORCH_CHECK(at::equal(key_owner.narrow(0,0,3),at::full_like(key_owner.narrow(0,0,3),17))
     &&at::equal(value_owner.narrow(0,0,3),at::full_like(value_owner.narrow(0,0,3),-17)),"Direct writes corrupted prefix rows");
   std::vector<int32_t> mapping;for(int32_t row=0;row<k.size(0);++row)mapping.push_back(row+3+(row>=7?3:0));
   auto rows=at::tensor(mapping,k.options().dtype(at::kInt)),indices=rows.to(at::kLong);
   auto mapped_key=at::full({k.size(0)+6,k.size(1),128},17,k.options()),mapped_value=at::full_like(mapped_key,-17);
   auto mapped_q=tei::fused_qkv_norm_rope_into(q,k,v,qw,kw,cosine,sine,1e-6,mapped_key,mapped_value,rows);
   auto expected_key=at::full_like(mapped_key,17),expected_value=at::full_like(mapped_value,-17);
   expected_key.index_copy_(0,indices,actual.second);expected_value.index_copy_(0,indices,v);
   close(mapped_q,actual.first,0,"Mapped Q normalization retains original arithmetic");
   close(mapped_key,expected_key,0,"Mapped K writes retain prefix and skipped rows");
   TORCH_CHECK(at::equal(mapped_value.view(at::kByte),expected_value.view(at::kByte)),"Mapped V writes changed bits or skipped rows");
   auto contiguous_k=k.contiguous();bool alias_rejected=false;
   try{tei::fused_qkv_norm_rope_into(q,contiguous_k,v,qw,kw,cosine,sine,1e-6,contiguous_k,value_target);}
   catch(const c10::Error&){alias_rejected=true;}
   TORCH_CHECK(alias_rejected&&at::equal(contiguous_k,k),"Aliasing prefix targets were written before rejection");
   auto ref=[&](const at::Tensor& x,const at::Tensor& weight){auto n=std::get<0>(at::_fused_rms_norm(x.contiguous(),{128},weight,1e-6));auto a=n.narrow(-1,0,64),b=n.narrow(-1,64,64);return at::cat({a*cosine-b*sine,b*cosine+a*sine},-1);};
   close(actual.first,ref(q,qw),dtype==at::kBFloat16?0.0625:0.008,"Q norm+RoPE");
   close(actual.second,ref(k,kw),dtype==at::kBFloat16?0.0625:0.008,"K norm+RoPE");
  }
  std::cout<<dtype<<" fused CUDA operator parity passed\n";
 }
 if(argc>1&&std::string(argv[1])=="prefix-qkv")return 0;
 {
  auto opts=at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  auto x=at::ones({2,256},opts);x.narrow(-1,128,128).fill_(2);
  auto residual=at::full_like(x,.005),weight=at::full({256},64.,opts);
  auto exact=x.to(at::kFloat)+residual.to(at::kFloat);
  auto reference=(exact*at::rsqrt(exact.square().mean(-1,true)+1e-5)*weight.to(at::kFloat)).to(at::kBFloat16);
  auto rounded=x+residual;
  auto wrong=std::get<0>(at::_fused_rms_norm(rounded,{256},weight,1e-5));
  TORCH_CHECK((wrong-reference).abs().max().item<float>()>=.125,"Residual rounding fixture must distinguish semantics");
  close(tei::fused_add_rms_norm(x,residual,weight,1e-5).first,reference,0,"RMS statistics retain FP32 residual sum");
 }
 {
  auto opts=at::TensorOptions().device(at::kCUDA).dtype(at::kBFloat16);
  auto& context=at::globalContext();
  const auto preferred=context.blasPreferredBackend();
  for(auto fallback:{at::BlasBackend::Default,at::BlasBackend::Cublas}) {
   context.setBlasPreferredBackend(fallback);
   TORCH_CHECK(!tei::nomic_moe_split_policy(c10::Device(at::kCUDA,0),at::kBFloat16,768,3072,8,2).defined(),
               "Default/cuBLAS must retain reference Nomic dispatch");
  }
  context.setBlasPreferredBackend(preferred);
  auto policy=tei::nomic_moe_split_policy(c10::Device(at::kCUDA,0),at::kBFloat16,768,3072,8,2);
  auto* properties=at::cuda::getCurrentDeviceProperties();
  if(properties->major==9 && properties->minor==0 && cublasLtGetVersion()==130200)
   TORCH_CHECK(policy.defined(),"Qualified Nomic grouped assertions require cuBLASLt preference");
  TORCH_CHECK(!tei::nomic_moe_split_policy(c10::Device(at::kCPU),at::kBFloat16,768,3072,8,2).defined(),
              "CPU must retain the Nomic reference dispatch");
  if(policy.defined()) {
   // A single regression for packed uneven/empty experts, reversed slots and
   // the small-M split-K rounding boundary that unsplit grouped GEMM loses.
   at::manual_seed(7);
   auto first=at::randn({8,3072,768},opts)*.02,second=at::randn_like(first)*.02;
   for(int rows:{2,32,137,1024}) {
    auto x=at::randn({rows,768},opts),scores=at::rand({rows,2},opts)*.5;
    std::vector<int64_t> routes;
    for(int row=0;row<rows;++row){routes.push_back(row%3==0?7:0);routes.push_back(row%3==0?0:2);}
    auto ids=at::tensor(routes,opts.dtype(at::kLong)).view({rows,2});
    auto reference=at::zeros_like(x);
    for(int expert=0;expert<8;++expert) {
     auto matches=at::nonzero(ids==expert);
     if(!matches.size(0))continue;
     auto tokens=matches.select(1,0),slots=matches.select(1,1);
     auto contribution=at::matmul(tei::encoder_gelu(at::linear(x.index_select(0,tokens),first[expert])),second[expert]);
     reference.index_add_(0,tokens,contribution*scores.index_select(0,tokens).gather(1,slots.unsqueeze(1)));
    }
    auto actual=tei::nomic_grouped_experts(x,first,second,ids,scores,policy);
    close(actual,reference,0,"Grouped Nomic preserves original GEMM split-K and BF16 weighting");
    auto saved=actual.clone();
    auto changed=tei::nomic_grouped_experts(x*2,first,second,ids,scores,policy);
    TORCH_CHECK(!at::equal(actual,changed),"Grouped experts ignored changed tokens");
    close(actual,saved,0,"Grouped requests retain earlier owned outputs");
   }
  }
 }
 for(auto dtype:{at::kHalf,at::kBFloat16}) {
  auto opts=at::TensorOptions().device(at::kCUDA).dtype(dtype);
  auto x=at::randn({31,2048},opts),residual=at::randn_like(x),weight=at::randn({2048},opts);
  for(bool add_residual:{false,true}) {
   auto sum=add_residual?x+residual:x,fp32=sum.to(at::kFloat);
   auto expected=(fp32*at::rsqrt(fp32.square().mean(-1,true)+1e-5)).to(dtype)*weight;
   auto actual=tei::transformers_rms_norm(x,add_residual?residual:at::Tensor(),weight,1e-5);
   TORCH_CHECK(at::equal(actual.second,sum),"Transformers residual must round before normalization");
   auto difference=actual.first.to(at::kFloat)-expected.to(at::kFloat);
   TORCH_CHECK(difference.norm().item<float>()/expected.to(at::kFloat).norm().item<float>()<1e-4,
     "Transformers RMSNorm changed normalized-activation rounding");
  }
 }

}

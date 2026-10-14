// Independent composed BF16 product-rounding oracle for native packed Gemma rotary.
#include <ATen/ATen.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "kernels/qwen35_kernels.h"
#include "kernels/fast_kernels.h"
at::Tensor reference(const at::Tensor& x,const at::Tensor& cos,const at::Tensor& sin,const at::Tensor& pos) {
 auto half=x.size(2)/2;auto a=x.slice(2,0,half),b=x.slice(2,half);
 auto c=cos.index_select(0,pos).unsqueeze(1),s=sin.index_select(0,pos).unsqueeze(1);
 return at::cat({a*c-b*s,b*c+a*s},2);
}
int main(int argc,char** argv) {
 c10::Device device(argc>1?argv[1]:"cuda:0");at::manual_seed(7312);
 auto bf=at::TensorOptions().device(device).dtype(at::kBFloat16);
 for(int64_t tokens:{1,3,33,129})for(int64_t heads:{1,2,16})for(int64_t width:{8,72,256,512}) {
  auto storage=at::randn({tokens,heads+3,width},bf);auto x=storage.narrow(1,1,heads);auto angles=at::randn({257,width/2},bf.dtype(at::kFloat));
  auto cos=angles.cos().to(bf),sin=angles.sin().to(bf);
  auto pos=(at::arange(tokens,at::TensorOptions().dtype(at::kLong).device(device))*17).remainder(257);
  auto actual=tei::gemma_rotary_cuda(x,cos,sin,pos),expected=reference(x,cos,sin,pos);
  TORCH_CHECK(at::equal(actual,expected),"Staged rotary mismatch T/H/D ",tokens,"/",heads,"/",width);
 }
 // Compare against the actual unfused native Gemma3 operations, including
 // noncontiguous slices from one QKV allocation and the untouched V view.
 for(int64_t tokens:{1,3,33})for(int64_t heads:{1,3,16})for(int64_t width:{8,72,256,512,1024,8192}) {
  const int64_t kv=heads==16?2:1;
  auto storage=at::randn({tokens,heads+2*kv,width},bf);
  auto edges=at::tensor({0.f,-0.f,0.0078125f,-0.0078125f,1.00390625f,
    -1.00390625f,257.f,-257.f,65536.f,-65536.f},bf.dtype(at::kFloat)).to(bf);
  storage.flatten().slice(0,0,std::min<int64_t>(edges.numel(),storage.numel()))
    .copy_(edges.slice(0,0,std::min<int64_t>(edges.numel(),storage.numel())));
  auto before=storage.clone();
  auto q=storage.narrow(1,0,heads),k=storage.narrow(1,heads,kv);
  auto qs=at::randn({width},bf.dtype(at::kFloat))+1.f;
  auto ks=at::randn({width},bf.dtype(at::kFloat))+1.f;
  auto angles=at::randn({257,width/2},bf.dtype(at::kFloat));
  auto cos=angles.cos().to(bf),sin=angles.sin().to(bf);
  auto pos=(at::arange(tokens,at::TensorOptions().dtype(at::kLong).device(device))*37).remainder(257);
  auto expected_q=tei::gemma_norm_cuda(q,qs,1e-6,false);
  auto expected_k=tei::gemma_norm_cuda(k,ks,1e-6,false);
  tei::encoder_rotary_inplace(expected_q,expected_k,cos.index_select(0,pos),sin.index_select(0,pos));
  auto actual=tei::gemma3_qk_norm_rotary_cuda(q,k,qs,ks,cos,sin,pos,1e-6);
  TORCH_CHECK(at::equal(actual.first.view(at::kShort),expected_q.view(at::kShort))&&
    at::equal(actual.second.view(at::kShort),expected_k.view(at::kShort)),
    "Gemma3 norm/rotary IEEE mismatch T/H/D ",tokens,"/",heads,"/",width);
  TORCH_CHECK(at::equal(storage.view(at::kShort),before.view(at::kShort)),"Gemma3 fused norm/rotary changed original QKV/V");
 }
 // Compare the two actual original legacy operations with the fused pair.
 // Stride holes belong to physical backing storage, not synthetic token rows.
 for(int64_t rows:{1,3,32,257,4096,16384})for(bool strided:{false,true})for(double eps:{1e-6,1e-4}) {
  auto xs=at::randn({rows,768+(strided?13:0)},bf);
  auto rs=at::randn({rows,768+(strided?17:0)},bf);
  auto x=xs.narrow(1,0,768),residual=rs.narrow(1,0,768);
  auto edges=at::tensor({0.f,-0.f,1e-20f,-1e-20f,1.00390625f,-1.00390625f,
    257.f,-257.f,65536.f,-65536.f},bf.dtype(at::kFloat)).to(bf);
  x.select(0,0).slice(0,0,edges.numel()).copy_(edges);
  auto post=at::randn({768},bf.dtype(at::kFloat))+1.f;
  auto pre=at::randn({768},bf.dtype(at::kFloat))+1.f;
  // Exercise exact BF16 cancellation in a complete real input row.
  residual.select(0,0).copy_(-tei::gemma_norm_cuda(x.narrow(0,0,1),post,eps,false).select(0,0));
  auto xb=xs.clone(),rb=rs.clone();
  auto expected_states=tei::gemma3_legacy_residual_norm_cuda(residual.contiguous(),x.contiguous(),post,eps);
  TORCH_CHECK(expected_states.defined(),"Existing legacy residual reference must admit width768");
  auto expected_normalized=tei::gemma_norm_cuda(expected_states,pre,eps,false);
  auto actual=tei::gemma3_residual_prenorm_cuda(residual,x,post,pre,eps);
  TORCH_CHECK(actual.first.defined()&&actual.second.defined(),"Fused pair must admit original or strided BF16 width768");
  TORCH_CHECK(at::equal(actual.first.view(at::kShort),expected_states.view(at::kShort))&&
    at::equal(actual.second.view(at::kShort),expected_normalized.view(at::kShort)),
    "Gemma3 residual/prenorm IEEE mismatch rows/stride/epsilon ",rows,"/",strided,"/",eps);
  TORCH_CHECK(at::equal(xs.view(at::kShort),xb.view(at::kShort))&&
    at::equal(rs.view(at::kShort),rb.view(at::kShort)),"Fused residual/prenorm mutated caller backing storage");
 }
 auto unsupported=at::zeros({3,768},bf.dtype(at::kFloat));
 auto scales=at::ones({768},bf.dtype(at::kFloat));
 auto declined=tei::gemma3_residual_prenorm_cuda(unsupported,unsupported,scales,scales,1e-6);
 TORCH_CHECK(!declined.first.defined()&&!declined.second.defined(),"F32 fallback eligibility changed");
 auto short_input=at::zeros({3,512},bf);
 declined=tei::gemma3_residual_prenorm_cuda(short_input,short_input,scales,scales,1e-6);
 TORCH_CHECK(!declined.first.defined()&&!declined.second.defined(),"Other hidden-width fallback eligibility changed");
 std::cout<<"Gemma staged rotary, 54 QK-fusion and 24 residual/prenorm packed cases passed\n";
}

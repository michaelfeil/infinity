#include "helpers/attention.h"
// Independent packed Torch ALiBi bias storage and sequence isolation regression.
#include "models/model.h"
#include "helpers/options.h"
#include <c10/core/InferenceMode.h>
#include <ATen/ops/_efficient_attention_forward.h>
#include <iostream>
#include <vector>
#include <cmath>
void overflowing_bias_storage_test() {
  const auto huge=std::numeric_limits<int64_t>::max()/128;
  // A zero-stride CPU view describes huge geometry using one actual scalar.
  auto q=at::zeros({1}).as_strided({huge,2,1},{0,0,0});
  tei::PackedInput input;input.batch=1;input.max_sequence=huge;
  bool sequence_rejected=false,alibi_rejected=false;
  try { tei::packed_sequence_bias(q,input); } catch(const c10::Error&) { sequence_rejected=true; }
  try { tei::packed_alibi_bias(q,input,at::ones({2})); } catch(const c10::Error&) { alibi_rejected=true; }
  TORCH_CHECK(sequence_rejected&&alibi_rejected,"Overflowing attention geometry reached allocation");
}
int main(int argc, char** argv) {
  c10::InferenceMode guard;
  overflowing_bias_storage_test();
  {
    auto q=at::zeros({1}).as_strided({4096,12,1},{0,0,0});
    tei::PackedInput input;input.batch=1;input.max_sequence=4096;
    for(bool alibi:{false,true}) {
      bool rejected=false;
      try {
        if(alibi)tei::packed_alibi_bias(q,input,at::ones({12}),-1,-1,1024);
        else tei::packed_sequence_bias(q,input,1024);
      } catch(const tei::MemoryAdmissionError& ex) {
        rejected=std::string(ex.what()).find("exceeds its memory budget")!=std::string::npos;
      }
      TORCH_CHECK(rejected,"Finite attention geometry bypassed byte admission");
    }
    input.max_sequence=1;
    auto small=q.narrow(0,0,1);
    TORCH_CHECK(tei::packed_sequence_bias(small,input,1024).defined()
      &&tei::packed_alibi_bias(small,input,at::ones({12}),-1,-1,1024).defined(),
      "Valid bias failed after admission rejection");
  }
  const c10::Device device(argc>1 ? argv[1] : "cuda:0");
  at::manual_seed(2718);
  for(auto dtype: {at::kHalf,at::kBFloat16}) for(auto lengths: {std::vector<int>{3,13,7},std::vector<int>{13,3,7},std::vector<int>{33},std::vector<int>{128,128,128,128},std::vector<int>{32,64,96,128,160,192,224,256}}) {
    const int H=12,D=64; int total=0,m=0; std::vector<int> offsets{0};
    for(int n:lengths){ total+=n;m=std::max(m,n);offsets.push_back(total); }
    auto opt=at::TensorOptions().device(device).dtype(dtype);
    auto q=at::randn({total,H,D},opt), k=at::randn_like(q),v=at::randn_like(q);
    auto cu=at::from_blob(offsets.data(),{(int)offsets.size()},at::kInt).to(opt.device());
    auto slopes=at::pow(2.,-at::arange(1,H+1,opt.dtype(at::kFloat))/3.);
    const int stride=((m+7)/8)*8, tile=m*stride;
    const long backing=(H-1L)*tile+(total-1L)*stride+total;
    tei::PackedInput input{{},{},{},cu,offsets.data(),(int64_t)lengths.size(),m};
    auto logical=tei::packed_alibi_bias(q,input,slopes);
    auto storage=logical.as_strided({backing},{1});
    auto local=storage.as_strided({H,m,m},{tile,stride,1});
    // Poison all unused storage, including alignment gaps and the logical
    // view's extra backing. Only actual local tiles may be read by the kernel.
    auto saved=local.clone();storage.fill_(NAN);local.copy_(saved);
    auto got=tei::packed_attention(q,k,v,input,1./std::sqrt(D),false,-1,-1,slopes,logical);
    TORCH_CHECK(at::isfinite(got).all().item<bool>(),"Unused bias storage was read");
    double max_perseq=0,max_reference=0,max_isolation=0;
    for(size_t row=0;row<lengths.size();++row){
      int n=lengths[row],start=offsets[row];
      auto slice=[&](const at::Tensor& t){return t.narrow(0,start,n).unsqueeze(0);};
      auto c=cu.narrow(0,row,2)-start;
      auto b=local.narrow(1,0,n).narrow(2,0,n).unsqueeze(0);
      auto expected=std::get<0>(at::_efficient_attention_forward(slice(q),slice(k),slice(v),b,c,c,n,n,0.,0,false,1./std::sqrt(D))).squeeze(0);
      max_perseq=std::max(max_perseq,(got.narrow(0,start,n)-expected).abs().max().item<double>());
      auto qr=q.narrow(0,start,n).to(at::kFloat).transpose(0,1),kr=k.narrow(0,start,n).to(at::kFloat).transpose(0,1),vr=v.narrow(0,start,n).to(at::kFloat).transpose(0,1);
      auto ref=at::matmul(at::softmax(at::matmul(qr,kr.transpose(1,2))/std::sqrt(D)+b.squeeze(0).to(at::kFloat),-1),vr).transpose(0,1);
      max_reference=std::max(max_reference,(got.narrow(0,start,n).to(at::kFloat)-ref).abs().max().item<double>());
    }
    auto changed=v.clone();changed.narrow(0,0,lengths[0]).add_(100);
    auto isolate=tei::packed_attention(q,k,changed,input,1./std::sqrt(D),false,-1,-1,slopes,logical);
    if(total>lengths[0]) max_isolation=(got.narrow(0,lengths[0],total-lengths[0])-isolate.narrow(0,lengths[0],total-lengths[0])).abs().max().item<double>();
    std::cout<<"dtype="<<dtype<<" total="<<total<<" max="<<m<<" backing="<<backing<<" logical="<<(long)H*total*total<<" perseq="<<max_perseq<<" independent="<<max_reference<<" isolation="<<max_isolation<<std::endl;
    TORCH_CHECK(max_perseq==0 && max_isolation==0 && max_reference < (dtype==at::kHalf ? .003 : .025),"Packed ALiBi failed");
  }
}

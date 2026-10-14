#include "helpers/attention.h"
#include <c10/core/InferenceMode.h>
#include <ATen/ops/_efficient_attention_forward.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <iostream>
#include <vector>
#include <cmath>
#include "kernels/encoder_bias.h"
int main(int argc,char** argv) {
  c10::InferenceMode guard;
  const c10::Device device(argc>1 ? argv[1] : "cuda:0");
  at::manual_seed(2718);
  for(auto dtype: {at::kHalf,at::kBFloat16}) for(auto lengths: {std::vector<int>(16,3),std::vector<int>{3,13,7},std::vector<int>{13,3,7},std::vector<int>{33},std::vector<int>{128,128,128,128},std::vector<int>{32,64,96,128,160,192,224,256}}) {
    const int H=12,D=64; int total=0,m=0; std::vector<int> offsets{0};
    for(int n:lengths){ total+=n;m=std::max(m,n);offsets.push_back(total); }
    auto opt=at::TensorOptions().device(device).dtype(dtype);
    auto q=at::randn({total,H,D},opt), k=at::randn_like(q),v=at::randn_like(q);
    auto cu=at::from_blob(offsets.data(),{(int)offsets.size()},at::kInt).to(opt.device());
    const int stride=((m+7)/8)*8, tile=m*stride;
    const long backing=std::max((long)lengths.size()*H*tile,(H-1L)*tile+(total-1L)*stride+total);
    tei::PackedInput input{{},{},{},cu,offsets.data(),(int64_t)lengths.size(),m};
    auto logical=tei::packed_sequence_bias(q,input);
    auto storage=logical.as_strided({backing},{1});storage.fill_(NAN);
    auto local=storage.as_strided({(long)lengths.size(),H,m,m},{H*tile,tile,stride,1});
    const int bins=256;
    auto ck=at::randn({H,total,bins},opt),pc=at::randn({H,total,bins},opt);
    std::vector<at::Tensor> index_rows;std::vector<int64_t> index_offsets{0};
    for(size_t r=0;r<lengths.size();++r) {
      int n=lengths[r];auto idx=at::randint(bins,{n,n},opt.dtype(at::kLong));
      index_rows.push_back(idx.reshape({-1}));index_offsets.push_back(index_offsets.back()+n*n);
    }
    auto packed_indices=at::cat(index_rows,0);
    auto ix=at::from_blob(index_offsets.data(),{int64_t(index_offsets.size())},at::kLong).to(device);
    auto ck_cpu=ck.cpu(),pc_cpu=pc.cpu();
    for(int mode:{3,0,1,2}) {
      auto use_ck=(mode==0||mode==2)?ck:at::Tensor(),use_pc=(mode==1||mode==2)?pc:at::Tensor();
      storage.fill_(NAN);
      tei::packed_bias_fill(logical,use_ck,use_pc,packed_indices,ix,cu,m);
      for(size_t r=0;r<lengths.size();++r) {
        int n=lengths[r],start=offsets[r];auto idx=index_rows[r].view({n,n}).cpu();
        // Independent CPU gather and separate model-dtype additions, unchanged rounding.
        auto expected=at::zeros({H,n,n},opt.device(at::kCPU));
        if(use_ck.defined())expected=expected+ck_cpu.narrow(1,start,n).gather(2,idx.unsqueeze(0).expand({H,n,n}));
        if(use_pc.defined())expected=expected+pc_cpu.narrow(1,start,n).gather(2,idx.transpose(0,1).unsqueeze(0).expand({H,n,n})).transpose(1,2);
        TORCH_CHECK(at::equal(local[r].narrow(1,0,n).narrow(2,0,n),expected.to(device)),"Packed gather/add bias differs from original operation oracle");
      }
    }

    // Shared metadata is indexed using the true longest real sequence, including
    // ragged order and cases where m*m is smaller than total tokens.
    auto shared_indices=at::randint(bins,{m,m},opt.dtype(at::kLong));
    auto shared_cpu=shared_indices.cpu();
    for(int mode:{3,0,1,2}) {
      auto use_ck=(mode==0||mode==2)?ck:at::Tensor(),use_pc=(mode==1||mode==2)?pc:at::Tensor();
      storage.fill_(NAN);
      tei::packed_bias_fill(logical,use_ck,use_pc,shared_indices,at::Tensor(),cu,m);
      for(size_t r=0;r<lengths.size();++r) {
        int n=lengths[r],start=offsets[r];auto idx=shared_cpu.narrow(0,0,n).narrow(1,0,n);
        auto expected=at::zeros({H,n,n},opt.device(at::kCPU));
        if(use_ck.defined())expected=expected+ck_cpu.narrow(1,start,n).gather(2,idx.unsqueeze(0).expand({H,n,n}));
        if(use_pc.defined())expected=expected+pc_cpu.narrow(1,start,n).gather(2,idx.transpose(0,1).unsqueeze(0).expand({H,n,n})).transpose(1,2);
        TORCH_CHECK(at::equal(local[r].narrow(1,0,n).narrow(2,0,n),expected.to(device)),"Shared relative bias differs from independent CPU gather/add oracle");
      }
    }

    // Capture and replay the same packed layout with changed real bucket values.
    // No host metadata upload occurs during capture; allocation addresses stay fixed.
    if(lengths==std::vector<int>{3,13,7}) {
      c10::cuda::CUDAGuard device_guard(device);
      C10_CUDA_CHECK(cudaDeviceSynchronize());
      auto stream=c10::cuda::getStreamFromPool(false,device.index());
      c10::cuda::CUDAStreamGuard stream_guard(stream);
      at::cuda::CUDAGraph graph;
      graph.capture_begin();
      tei::packed_bias_fill(logical,ck,pc,packed_indices,ix,cu,m);
      tei::packed_bias_fill(logical,ck,pc,shared_indices,at::Tensor(),cu,m);
      graph.capture_end();
      packed_indices.zero_();shared_indices.zero_();graph.replay();
      for(size_t r=0;r<lengths.size();++r) {
        int n=lengths[r],start=offsets[r];
        auto idx=at::zeros({n,n},at::TensorOptions().dtype(at::kLong));
        auto expected=at::zeros({H,n,n},opt.device(at::kCPU));
        expected=expected+ck_cpu.narrow(1,start,n).gather(2,idx.unsqueeze(0).expand({H,n,n}));
        expected=expected+pc_cpu.narrow(1,start,n).gather(2,idx.transpose(0,1).unsqueeze(0).expand({H,n,n})).transpose(1,2);
        TORCH_CHECK(at::equal(local[r].narrow(1,0,n).narrow(2,0,n),expected.to(device)),"Packed relative bias changed-bucket graph replay mismatch");
      }
      stream.synchronize();
    }
    auto got=std::get<0>(at::_efficient_attention_forward(q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0),logical,cu,cu,m,m,0.,0,false,1./std::sqrt(D))).squeeze(0);
    TORCH_CHECK(at::isfinite(got).all().item<bool>(),"Unused relative bias storage was read");
    double max_perseq=0,max_reference=0,max_isolation=0;
    for(size_t row=0;row<lengths.size();++row){
      int n=lengths[row],start=offsets[row];
      auto slice=[&](const at::Tensor& t){return t.narrow(0,start,n).unsqueeze(0);};
      auto c=cu.narrow(0,row,2)-start;
      auto b=local[row].narrow(1,0,n).narrow(2,0,n).unsqueeze(0);
      auto expected=std::get<0>(at::_efficient_attention_forward(slice(q),slice(k),slice(v),b,c,c,n,n,0.,0,false,1./std::sqrt(D))).squeeze(0);
      max_perseq=std::max(max_perseq,(got.narrow(0,start,n)-expected).abs().max().item<double>());
      auto qr=q.narrow(0,start,n).to(at::kFloat).transpose(0,1),kr=k.narrow(0,start,n).to(at::kFloat).transpose(0,1),vr=v.narrow(0,start,n).to(at::kFloat).transpose(0,1);
      auto ref=at::matmul(at::softmax(at::matmul(qr,kr.transpose(1,2))/std::sqrt(D)+b.squeeze(0).to(at::kFloat),-1),vr).transpose(0,1);
      max_reference=std::max(max_reference,(got.narrow(0,start,n).to(at::kFloat)-ref).abs().max().item<double>());
    }
    auto changed=v.clone();changed.narrow(0,0,lengths[0]).add_(100);
    auto isolate=std::get<0>(at::_efficient_attention_forward(q.unsqueeze(0),k.unsqueeze(0),changed.unsqueeze(0),logical,cu,cu,m,m,0.,0,false,1./std::sqrt(D))).squeeze(0);
    TORCH_CHECK(at::isfinite(isolate).all().item<bool>(),"Isolation output is not finite");
    if(total>lengths[0]) max_isolation=(got.narrow(0,lengths[0],total-lengths[0])-isolate.narrow(0,lengths[0],total-lengths[0])).abs().max().item<double>();
    std::cout<<"dtype="<<dtype<<" total="<<total<<" max="<<m<<" backing="<<backing<<" logical="<<(long)H*total*total<<" perseq="<<max_perseq<<" independent="<<max_reference<<" isolation="<<max_isolation<<std::endl;
    TORCH_CHECK(max_perseq==0 && max_isolation==0 && max_reference < (dtype==at::kHalf ? .003 : .025),"Packed per-sequence relative bias failed");
  }
}

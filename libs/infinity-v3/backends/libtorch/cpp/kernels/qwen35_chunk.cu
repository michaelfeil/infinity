// SPDX-License-Identifier: Apache-2.0
// Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
// Native adaptation of Transformers 5.19.0 torch_chunk_gated_delta_rule.
// Retained license and adaptation provenance: vendor/transformers_gdn/.
#include "kernels/qwen35_chunk.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <limits>
#include <mutex>
#include <set>

namespace tei {
namespace {
constexpr int CHUNK=32,DIM=128;

// Condense each chunk independently into U=(I+L)^-1 beta V and
// W=(I+L)^-1 beta diag(prefix_decay) K. Only this small triangular
// substitution is sequential; chunks, heads and feature columns are parallel.
// Decay products avoid log(0) and inf-inf for gates that reset the state.
__global__ void prepare_chunks(const __nv_bfloat16* values,const float* qk,
 const float* gates,const int32_t* starts,const int32_t* lengths,float* u,
 float* w,float* attention,float* prefix_decay,float* tail_decay,int kh,int vh) {
 const int chunk=blockIdx.x,head=blockIdx.y,column=threadIdx.x;
 const int start=starts[chunk],length=lengths[chunk],qhead=head/(vh/kh);
 extern __shared__ float shared[];
 float* keys=shared;
 float* queries=keys+CHUNK*(DIM+1);
 float* lower=queries+CHUNK*(DIM+1);
 float* su=lower+CHUNK*CHUNK;
 float* sw=su+CHUNK*DIM;
 __shared__ float prefix[CHUNK],tail[CHUNK],decay[CHUNK],beta[CHUNK];
 if(column<length) {
   decay[column]=gates[int64_t(start+column)*2*vh+head];
   beta[column]=gates[int64_t(start+column)*2*vh+vh+head];
 }
 for(int row=0;row<length;++row) {
   keys[row*(DIM+1)+column]=qk[(int64_t(start+row)*kh+qhead)*256+128+column];
   queries[row*(DIM+1)+column]=qk[(int64_t(start+row)*kh+qhead)*256+column];
 }
 __syncthreads();
 if(column<length) {
   float before=1.f,after=1.f;
   for(int i=0;i<=column;++i)before*=decay[i];
   for(int i=column+1;i<length;++i)after*=decay[i];
   prefix[column]=before;tail[column]=after;
   prefix_decay[int64_t(start+column)*vh+head]=before;
   tail_decay[int64_t(start+column)*vh+head]=after;
 }
 // Each thread owns one entry at a time; all valid (i,j) share the key tile.
 for(int pair=column;pair<length*length;pair+=DIM) {
   const int i=pair/length,j=pair%length;
   float kk=0.f,qk_dot=0.f,relative=1.f;
   if(j<=i) {
     for(int k=j+1;k<=i;++k)relative*=decay[k];
     for(int k=0;k<DIM;++k) {
       kk=fmaf(keys[i*(DIM+1)+k],keys[j*(DIM+1)+k],kk);
       qk_dot=fmaf(queries[i*(DIM+1)+k],keys[j*(DIM+1)+k],qk_dot);
     }
   }
   lower[i*CHUNK+j]=j<i?beta[i]*relative*kk:0.f;
   attention[((int64_t(chunk)*vh+head)*CHUNK+i)*CHUNK+j]=j<=i?relative*qk_dot:0.f;
 }
 __syncthreads();
 const int channels=(2*kh+vh)*DIM;
 for(int row=0;row<length;++row) {
   float uv=beta[row]*__bfloat162float(values[int64_t(start+row)*channels+2*kh*DIM+head*DIM+column]);
   float wk=beta[row]*prefix[row]*keys[row*(DIM+1)+column];
   for(int j=0;j<row;++j) {
     uv=fmaf(-lower[row*CHUNK+j],su[j*DIM+column],uv);
     wk=fmaf(-lower[row*CHUNK+j],sw[j*DIM+column],wk);
   }
   su[row*DIM+column]=uv;sw[row*DIM+column]=wk;
   // Each thread only reads its own solved feature column; no row barrier.
   u[(int64_t(start+row)*vh+head)*DIM+column]=uv;
   w[(int64_t(start+row)*vh+head)*DIM+column]=wk;
 }
}

// Two BF16 components preserve substantially more than BF16 input precision.
// Three tensor-core products include both cross terms while accumulating in
// FP32. Recurrent state itself stays in FP32 registers, never quantized storage.
__device__ __forceinline__ void split(float value,__nv_bfloat16& high,__nv_bfloat16& low) {
 high=__float2bfloat16_rn(value);
 low=__float2bfloat16_rn(value-__bfloat162float(high));
}
template<int K,int COLUMNS>
__device__ __forceinline__ void chunk_mma(const __nv_bfloat16* ah,const __nv_bfloat16* al,int lda,
 const __nv_bfloat16* bh,const __nv_bfloat16* bl,float* out,int tile_row,int tile_column,bool add=false) {
 using namespace nvcuda;
 wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a,low_a;
 wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::row_major> b,low_b;
 wmma::fragment<wmma::accumulator,16,16,16,float> c;
 auto* result=out+tile_row*16*COLUMNS+tile_column*16;
 if(add)wmma::load_matrix_sync(c,result,COLUMNS,wmma::mem_row_major);
 else wmma::fill_fragment(c,0.f);
 for(int k=0;k<K;k+=16) {
   const auto offset_a=tile_row*16*lda+k,offset_b=k*COLUMNS+tile_column*16;
   wmma::load_matrix_sync(a,ah+offset_a,lda);wmma::load_matrix_sync(low_a,al+offset_a,lda);
   wmma::load_matrix_sync(b,bh+offset_b,COLUMNS);wmma::load_matrix_sync(low_b,bl+offset_b,COLUMNS);
   wmma::mma_sync(c,a,b,c);wmma::mma_sync(c,low_a,b,c);wmma::mma_sync(c,a,low_b,c);
 }
 wmma::store_matrix_sync(result,c,COLUMNS,wmma::mem_row_major);
}

// Scan chunks rather than tokens. Tensor-core tiles process all token reads
// together; only the FP32 chunk boundary state is carried sequentially. Scratch
// tiles may mask a tail, but input/output activations remain strictly packed.
template<int COLUMNS,int SCAN_THREADS>
__global__ __launch_bounds__(SCAN_THREADS) void scan_chunks(const float* qk,const float* u,const float* w,
 const float* attention,const float* prefix_decay,const float* tail_decay,
 const int32_t* chunk_offsets,const int32_t* starts,const int32_t* lengths,
 __nv_bfloat16* out,int kh,int vh,const float* initial,const int64_t* sources,float* final_state) {
 const int seq=blockIdx.x,head=blockIdx.y,column_base=blockIdx.z*COLUMNS;
 const int tid=threadIdx.x,warp=tid/32,qhead=head/(vh/kh);
 extern __shared__ __align__(32) char scratch[];
 auto* ah=reinterpret_cast<__nv_bfloat16*>(scratch);
 auto* al=ah+CHUNK*DIM;
 auto* bh=al+CHUNK*DIM;
 auto* bl=bh+DIM*COLUMNS;
 auto* temporary=reinterpret_cast<float*>(bl+DIM*COLUMNS);
 auto* updates=temporary+(DIM*COLUMNS>CHUNK*(DIM+1)?DIM*COLUMNS:CHUNK*(DIM+1));
 float state[DIM*COLUMNS/SCAN_THREADS];
 const auto source=initial?sources[seq]:seq;
 for(int i=0;i<DIM*COLUMNS/SCAN_THREADS;++i) {
   const int index=tid+i*SCAN_THREADS;
   state[i]=initial?initial[((source*vh+head)*DIM+index/COLUMNS)*DIM+column_base+index%COLUMNS]:0.f;
 }
 for(int chunk=chunk_offsets[seq];chunk<chunk_offsets[seq+1];++chunk) {
   const int start=starts[chunk],length=lengths[chunk];
   for(int i=0;i<DIM*COLUMNS/SCAN_THREADS;++i) {
     const int index=tid+i*SCAN_THREADS;
     split(state[i],bh[index],bl[index]);
   }
   for(int index=tid;index<CHUNK*DIM;index+=SCAN_THREADS) {
     const int row=index/DIM,k=index%DIM;
     split(row<length?w[(int64_t(start+row)*vh+head)*DIM+k]:0.f,ah[index],al[index]);
   }
   __syncthreads();
   for(int tile=warp;tile<(CHUNK/16)*(COLUMNS/16);tile+=SCAN_THREADS/32)
     chunk_mma<DIM,COLUMNS>(ah,al,DIM,bh,bl,temporary,tile/(COLUMNS/16),tile%(COLUMNS/16));
   __syncthreads();
   for(int index=tid;index<CHUNK*COLUMNS;index+=SCAN_THREADS) {
     const int row=index/COLUMNS,column=index%COLUMNS;
     updates[index]=row<length?u[(int64_t(start+row)*vh+head)*DIM+column_base+column]-temporary[index]:0.f;
   }
   for(int index=tid;index<CHUNK*DIM;index+=SCAN_THREADS) {
     const int row=index/DIM,k=index%DIM;
     split(row<length?qk[(int64_t(start+row)*kh+qhead)*256+k]*prefix_decay[int64_t(start+row)*vh+head]:0.f,ah[index],al[index]);
   }
   __syncthreads();
   for(int tile=warp;tile<(CHUNK/16)*(COLUMNS/16);tile+=SCAN_THREADS/32)
     chunk_mma<DIM,COLUMNS>(ah,al,DIM,bh,bl,temporary,tile/(COLUMNS/16),tile%(COLUMNS/16));
   __syncthreads();
   for(int index=tid;index<CHUNK*COLUMNS;index+=SCAN_THREADS) {
     split(updates[index],bh[index],bl[index]);
   }
   for(int index=tid;index<CHUNK*CHUNK;index+=SCAN_THREADS) {
     const int row=index/CHUNK,column=index%CHUNK;
     split(row<length&&column<length?attention[((int64_t(chunk)*vh+head)*CHUNK+row)*CHUNK+column]:0.f,ah[index],al[index]);
   }
   __syncthreads();
   for(int tile=warp;tile<(CHUNK/16)*(COLUMNS/16);tile+=SCAN_THREADS/32)
     chunk_mma<CHUNK,COLUMNS>(ah,al,CHUNK,bh,bl,temporary,tile/(COLUMNS/16),tile%(COLUMNS/16),true);
   __syncthreads();
   for(int index=tid;index<length*COLUMNS;index+=SCAN_THREADS) {
     const int row=index/COLUMNS,column=index%COLUMNS;
     out[(int64_t(start+row)*vh+head)*DIM+column_base+column]=__float2bfloat16_rn(temporary[index]);
   }
   // Coalesced global reads followed by a shared-memory transpose. The
   // extra shared column avoids bank conflicts when reading feature columns.
   for(int index=tid;index<CHUNK*DIM;index+=SCAN_THREADS) {
     const int row=index/DIM,k=index%DIM;
     temporary[row*(DIM+1)+k]=row<length?qk[(int64_t(start+row)*kh+qhead)*256+128+k]*tail_decay[int64_t(start+row)*vh+head]:0.f;
   }
   __syncthreads();
   for(int index=tid;index<DIM*CHUNK;index+=SCAN_THREADS) {
     const int k=index/CHUNK,row=index%CHUNK;
     split(temporary[row*(DIM+1)+k],ah[index],al[index]);
   }
   __syncthreads();
   for(int tile=warp;tile<(DIM/16)*(COLUMNS/16);tile+=SCAN_THREADS/32)
     chunk_mma<CHUNK,COLUMNS>(ah,al,CHUNK,bh,bl,temporary,tile/(COLUMNS/16),tile%(COLUMNS/16));
   __syncthreads();
   const float decay=prefix_decay[int64_t(start+length-1)*vh+head];
   for(int i=0;i<DIM*COLUMNS/SCAN_THREADS;++i)state[i]=fmaf(decay,state[i],temporary[tid+i*SCAN_THREADS]);
   __syncthreads();
 }
 if(final_state)for(int i=0;i<DIM*COLUMNS/SCAN_THREADS;++i) {
   const int index=tid+i*SCAN_THREADS;
   final_state[((int64_t(seq)*vh+head)*DIM+index/COLUMNS)*DIM+column_base+index%COLUMNS]=state[i];
 }
}
}

void qwen35_chunk_delta_cuda(const at::Tensor& values,const at::Tensor& qk,
 const at::Tensor& gates,const PackedInput& input,at::Tensor& output,
 int64_t key_heads,int64_t value_heads,const at::Tensor& initial,at::Tensor* final_state) {
 c10::cuda::CUDAGuard guard(values.device());
 TORCH_CHECK(values.is_cuda()&&values.scalar_type()==at::kBFloat16&&values.is_contiguous()&&values.dim()==2&&
   key_heads>0&&value_heads>0&&value_heads<=128&&value_heads%key_heads==0&&
   values.size(1)==(2*key_heads+value_heads)*DIM,"Invalid packed chunk DeltaNet values");
 const auto tokens=values.size(0);
 TORCH_CHECK(tokens>0&&tokens<=INT32_MAX&&input.batch>0&&input.batch<=INT32_MAX&&input.offsets&&input.offsets[0]==0&&input.offsets[input.batch]==tokens,
   "Invalid packed chunk DeltaNet boundaries");
 TORCH_CHECK(qk.device()==values.device()&&qk.scalar_type()==at::kFloat&&qk.is_contiguous()&&qk.sizes()==at::IntArrayRef({tokens*key_heads,256})&&
   gates.device()==values.device()&&gates.scalar_type()==at::kFloat&&gates.is_contiguous()&&gates.sizes()==at::IntArrayRef({tokens,2*value_heads})&&
   output.device()==values.device()&&output.scalar_type()==at::kBFloat16&&output.is_contiguous()&&output.sizes()==at::IntArrayRef({tokens,value_heads,DIM}),
   "Invalid packed chunk DeltaNet buffers");
 std::vector<int32_t> starts,lengths,offsets{0};
 for(int64_t seq=0;seq<input.batch;++seq) {
   TORCH_CHECK(input.offsets[seq+1]>input.offsets[seq],"Chunk DeltaNet sequences must be nonempty");
   for(int64_t start=input.offsets[seq];start<input.offsets[seq+1];start+=CHUNK) {
     starts.push_back(start);lengths.push_back(std::min(int64_t(CHUNK),int64_t(input.offsets[seq+1])-start));
   }
   offsets.push_back(starts.size());
 }
 if(initial.defined()) {
   TORCH_CHECK(initial.device()==values.device()&&initial.scalar_type()==at::kFloat&&initial.is_contiguous()&&initial.dim()==4&&
     initial.size(1)==value_heads&&initial.size(2)==DIM&&initial.size(3)==DIM&&
     input.block_cache_sources.device()==values.device()&&input.block_cache_sources.scalar_type()==at::kLong&&input.block_cache_sources.is_contiguous()&&
     input.block_cache_sources.numel()==input.batch&&input.block_prefix_sources.size()==size_t(input.batch),"Invalid chunk DeltaNet initial state");
   for(auto source:input.block_prefix_sources)TORCH_CHECK(source>=0&&source<initial.size(0),"Invalid chunk DeltaNet state source");
 }
 auto options=values.options().dtype(at::kFloat);
 auto u=at::empty({tokens,value_heads,DIM},options),w=at::empty_like(u);
 auto attention=at::empty({int64_t(starts.size()),value_heads,CHUNK,CHUNK},options);
 auto prefix_decay=at::empty({tokens,value_heads},options),tail_decay=at::empty_like(prefix_decay);
 auto ints=values.options().dtype(at::kInt);
 auto chunk_starts=at::tensor(starts,ints),chunk_lengths=at::tensor(lengths,ints),chunk_offsets=at::tensor(offsets,ints);
 if(final_state)*final_state=at::empty({input.batch,value_heads,DIM,DIM},options);
 const auto stream=at::cuda::getCurrentCUDAStream();
 constexpr int shared_bytes=(4*CHUNK*DIM+2*CHUNK+CHUNK*CHUNK)*sizeof(float);
 // Reserve headroom for compiler-generated static shared memory in the
 // 160-KiB wide tile. Smaller devices retain the narrow scan.
 const bool wide=input.batch>=4&&at::cuda::getDeviceProperties(values.get_device())->sharedMemPerBlockOptin>=192*1024;
 constexpr int narrow_bytes=2*(CHUNK*DIM+DIM*32)*sizeof(__nv_bfloat16)+(CHUNK*(DIM+1)+CHUNK*32)*sizeof(float);
 constexpr int wide_bytes=2*(CHUNK*DIM+DIM*128)*sizeof(__nv_bfloat16)+(DIM*128+CHUNK*128)*sizeof(float);
 {
   static std::mutex mutex;
   static std::set<int> initialized;
   const std::lock_guard<std::mutex> lock(mutex);
   if(!initialized.count(values.get_device())) {
     C10_CUDA_CHECK(cudaFuncSetAttribute(prepare_chunks,cudaFuncAttributeMaxDynamicSharedMemorySize,shared_bytes));
     C10_CUDA_CHECK(cudaFuncSetAttribute(scan_chunks<32,512>,cudaFuncAttributeMaxDynamicSharedMemorySize,narrow_bytes));
     if(at::cuda::getDeviceProperties(values.get_device())->sharedMemPerBlockOptin>=192*1024)
       C10_CUDA_CHECK(cudaFuncSetAttribute(scan_chunks<128,512>,cudaFuncAttributeMaxDynamicSharedMemorySize,wide_bytes));
     initialized.insert(values.get_device());
   }
 }
 prepare_chunks<<<dim3(starts.size(),value_heads),DIM,shared_bytes,stream>>>(
   reinterpret_cast<const __nv_bfloat16*>(values.const_data_ptr()),qk.const_data_ptr<float>(),gates.const_data_ptr<float>(),
   chunk_starts.const_data_ptr<int32_t>(),chunk_lengths.const_data_ptr<int32_t>(),u.mutable_data_ptr<float>(),w.mutable_data_ptr<float>(),
   attention.mutable_data_ptr<float>(),prefix_decay.mutable_data_ptr<float>(),tail_decay.mutable_data_ptr<float>(),key_heads,value_heads);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 if(wide)scan_chunks<128,512><<<dim3(input.batch,value_heads,1),512,wide_bytes,stream>>>(
   qk.const_data_ptr<float>(),u.const_data_ptr<float>(),w.const_data_ptr<float>(),attention.const_data_ptr<float>(),prefix_decay.const_data_ptr<float>(),tail_decay.const_data_ptr<float>(),
   chunk_offsets.const_data_ptr<int32_t>(),chunk_starts.const_data_ptr<int32_t>(),chunk_lengths.const_data_ptr<int32_t>(),
   reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),key_heads,value_heads,initial.defined()?initial.const_data_ptr<float>():nullptr,
   initial.defined()?input.block_cache_sources.const_data_ptr<int64_t>():nullptr,final_state?final_state->mutable_data_ptr<float>():nullptr);
 else scan_chunks<32,512><<<dim3(input.batch,value_heads,DIM/32),512,narrow_bytes,stream>>>(
   qk.const_data_ptr<float>(),u.const_data_ptr<float>(),w.const_data_ptr<float>(),attention.const_data_ptr<float>(),prefix_decay.const_data_ptr<float>(),tail_decay.const_data_ptr<float>(),
   chunk_offsets.const_data_ptr<int32_t>(),chunk_starts.const_data_ptr<int32_t>(),chunk_lengths.const_data_ptr<int32_t>(),
   reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()),key_heads,value_heads,initial.defined()?initial.const_data_ptr<float>():nullptr,
   initial.defined()?input.block_cache_sources.const_data_ptr<int64_t>():nullptr,final_state?final_state->mutable_data_ptr<float>():nullptr);
 C10_CUDA_KERNEL_LAUNCH_CHECK();

}
}

// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "helpers/options.h"
#include "helpers/admission_error.h"
#include "bridge.h"
#include <algorithm>
#include <limits>
#include <optional>

namespace tei {
// Tensor storage only. Vendor workspaces, allocator rounding/reservation and
// other in-flight requests require separate admission and accounting.
struct TowerMemoryProfile {
  int64_t hidden=0,heads=0,linear=0,saved_layers=0,convolution=0,converted_parameters=0;
  bool per_frame_attention=false,tiled_scores=false;
  bool efficient_scores=false;
  int64_t relative_span=0;
  int64_t attention_chunk=0,attention_window=0;
};
struct TransientMemoryProfile {
  int64_t hidden=0,heads=0,linear=0,element=0,vocab=0,top_k=1,experts=0;
  int64_t retained_kv_width=0,retained_kv_layers=0;
  int64_t recurrent_heads=0,recurrent_key=0,recurrent_value=0,retained_recurrent_layers=0,retained_conv_width=0,retained_conv_length=0;
  int64_t relative_bins=0;
  size_t retained_vision_bytes=0;
  bool dense_attention=false,packed_bias=false,relative_buckets=false;
  std::optional<TowerMemoryProfile> vision,audio;
};
inline int64_t transient_linear_width(const Weights& weights) {
  int64_t width=0;
  for(const auto& [name,value]:weights) {
    if(!value.defined()||value.dim()<2||name.ends_with("word_embeddings.weight")||name.ends_with("tok_embeddings.weight")||name.ends_with("position_embeddings.weight")||
       name.ends_with("token_type_embeddings.weight")||name.ends_with("rel_embeddings.weight")||
       name.ends_with("embed_tokens.weight")||name.ends_with("lm_head.weight")||
       name.ends_with("decoder.weight")||name.ends_with("vocab_projector.weight"))continue;
    // Nomic stores expert matrices as [experts * intermediate, hidden].
    // Their actual per-expert width is supplied by the ready encoder profile.
    if(name.ends_with("experts.mlp.w1")||name.ends_with("experts.mlp.w2"))
      width=std::max(width,value.size(-1));
    else width=std::max({width,value.size(-1),value.size(-2)});
  }
  return width;
}
inline TransientMemoryProfile transient_profile(const Weights& weights,int64_t hidden,
    int64_t heads,at::ScalarType dtype,bool dense_attention) {
  TransientMemoryProfile p;
  p.hidden=hidden;p.heads=heads;p.element=c10::elementSize(dtype);
  p.linear=std::max(hidden,transient_linear_width(weights));p.dense_attention=dense_attention;
  for(const auto& [name,value]:weights) {
    if(value.defined()&&value.dim()==2&&(name.ends_with("word_embeddings.weight")||
       name.ends_with("tok_embeddings.weight")||
       name.ends_with("embed_tokens.weight")||name.ends_with("decoder.weight")||
       name.ends_with("vocab_projector.weight")))p.vocab=std::max(p.vocab,value.size(0));
  }
  return p;
}
class TransientMemoryBudget {
  size_t limit,used=0;
public:
  explicit TransientMemoryBudget(size_t bytes):limit(bytes) {
    TORCH_CHECK(bytes>0,"Native transient tensor memory budget must be positive");
  }
  void add(std::initializer_list<size_t> factors) {
    size_t bytes=1;
    for(auto factor:factors) {
      require_memory_admission(!factor||bytes<=limit/factor,"Native transient tensor estimate exceeds its memory budget");
      bytes*=factor;
    }
    require_memory_admission(bytes<=limit-used,"Native transient tensor estimate exceeds its memory budget");
    used+=bytes;
  }
  size_t bytes()const{return used;}
};
inline size_t positive_memory_dimension(int64_t value) {
  TORCH_CHECK(value>0&&uint64_t(value)<=std::numeric_limits<size_t>::max(),
    "Unknown or invalid native transient tensor dimensions");
  return static_cast<size_t>(value);
}
inline size_t estimate_transient_memory(const TransientMemoryProfile& p,int64_t tokens,
    int64_t batch,int64_t maximum,int32_t pool,size_t output_count,const TeiImage* images,
    size_t image_count,const TeiAudio* audios,size_t audio_count,size_t limit,
    bool shared_image_uploads=false,bool shared_image_features=false) {
  const auto t=positive_memory_dimension(tokens),b=positive_memory_dimension(batch),
    m=positive_memory_dimension(maximum),h=positive_memory_dimension(p.hidden),
    heads=positive_memory_dimension(p.heads),linear=positive_memory_dimension(p.linear),
    element=positive_memory_dimension(p.element),top=positive_memory_dimension(p.top_k);
  TransientMemoryBudget cost(limit);
  // Embedding/residual/QKV/rotary/normalization/readout tensors and F32
  // activation/FFN conversions. Sequential layers reuse this peak envelope.
  cost.add({t,h,32,element});cost.add({t,linear,8,std::max(element,size_t(4)),top});
  cost.add({t,3,sizeof(int64_t)});cost.add({b+1,sizeof(int32_t)});
  cost.add({output_count,3,sizeof(float)});
  if(p.dense_attention)cost.add({t,m,heads,4,sizeof(float)});
  if(p.packed_bias) {
    TORCH_CHECK(m<=std::numeric_limits<size_t>::max()-7,"Native transient bias stride overflow");
    const size_t stride=((m+7)/8)*8;
    // Charge both legal backing extents rather than just the logical view.
    cost.add({b,heads,m,stride,4,std::max(element,size_t(4))});
    cost.add({heads,m,stride,4,std::max(element,size_t(4))});
    cost.add({t,stride,4,std::max(element,size_t(4))});
  }
  if(p.relative_buckets)cost.add({m,m,128});
  if(p.relative_bins>0) {
    const auto bins=positive_memory_dimension(p.relative_bins);
    cost.add({t,heads,bins,4,sizeof(float)});cost.add({bins,h,4,sizeof(float)});
  }
  if(p.experts>0) {
    cost.add({t,positive_memory_dimension(p.experts),8,sizeof(float)});
    cost.add({positive_memory_dimension(p.experts),32,linear,8,std::max(element,size_t(4))});
    cost.add({t,top,8,sizeof(int64_t)});cost.add({t,top,h,8,element});
  }
  if(p.retained_kv_layers>0)cost.add({t,positive_memory_dimension(p.retained_kv_width),
    positive_memory_dimension(p.retained_kv_layers),2,element});
  if(p.recurrent_heads>0) {
    // DeltaNet chunk-prefix states plus the per-sequence recurrent state.
    TORCH_CHECK(t<=std::numeric_limits<size_t>::max()-63,"Native recurrent chunk count overflow");
    cost.add({(t+63)/64+b,positive_memory_dimension(p.recurrent_heads),
      positive_memory_dimension(p.recurrent_key),positive_memory_dimension(p.recurrent_value),8,sizeof(float)});
  }
  if(p.retained_recurrent_layers>0) {
    // Packed block execution retains every distinct prefix concurrently. Charge
    // one snapshot per logical sequence (an upper bound when prefixes share).
    cost.add({b,positive_memory_dimension(p.retained_recurrent_layers),positive_memory_dimension(p.recurrent_heads),positive_memory_dimension(p.recurrent_key),positive_memory_dimension(p.recurrent_value),sizeof(float)});
    cost.add({b,positive_memory_dimension(p.retained_recurrent_layers),positive_memory_dimension(p.retained_conv_width),positive_memory_dimension(p.retained_conv_length),element});
  }
  if(pool==3)cost.add({t,positive_memory_dimension(p.vocab),4,std::max(element,size_t(4))});
  auto tower=[&](const TowerMemoryProfile& v,size_t rows,size_t emitted,bool audio,size_t features,size_t frame_rows) {
    const auto vh=positive_memory_dimension(v.hidden),vl=positive_memory_dimension(v.linear),
      nh=positive_memory_dimension(v.heads);
    cost.add({rows,vh,32,sizeof(float)});cost.add({rows,vl,8,sizeof(float)});
    const auto keys=v.per_frame_attention?frame_rows:rows;
    if(v.attention_chunk>0) {
      // Conformer materializes scores for one query chunk and its context.
      // Input rows bound the subsampled length without underestimating it.
      cost.add({std::min(rows,positive_memory_dimension(v.attention_chunk)),
        std::min(rows,positive_memory_dimension(v.attention_window)),nh,8,sizeof(float)});
    } else if(v.efficient_scores)cost.add({rows,nh,4,sizeof(float)});
    else if(v.tiled_scores) {
      require_memory_admission(keys<=limit/sizeof(float)/nh,"Native transient vision score estimate exceeds its memory budget");
      const auto row_bytes=keys*nh*sizeof(float);
      cost.add({std::max(size_t(16)<<20,row_bytes),8});
    } else cost.add({keys,keys,nh,8,sizeof(float)});
    if(v.converted_parameters>0)cost.add({positive_memory_dimension(v.converted_parameters),sizeof(float)});
    if(v.relative_span>0) {
      const auto span=positive_memory_dimension(v.relative_span);
      cost.add({span,vh,8,sizeof(float)});cost.add({rows,span,nh,8,sizeof(float)});
    }
    cost.add({emitted,h,4,element});
    if(v.saved_layers>0) {
      const auto saved=positive_memory_dimension(v.saved_layers);
      cost.add({rows,vh,saved,element});cost.add({emitted,h,saved,element});
    }
    if(audio&&v.convolution>0)cost.add({rows,features,positive_memory_dimension(v.convolution),4,sizeof(float)});
  };
  TORCH_CHECK(!image_count||p.vision,"Unknown native vision transient tensor profile");
  TORCH_CHECK(!audio_count||p.audio,"Unknown native audio transient tensor profile");
  for(size_t i=0;i<image_count;++i) {
    const auto& image=images[i];size_t grid=1;
    for(auto axis:image.grid) {
      const auto n=positive_memory_dimension(axis);
      TORCH_CHECK(grid<=std::numeric_limits<size_t>::max()/n,"Native vision token geometry overflow");grid*=n;
    }
    const auto rows=std::max(grid,positive_memory_dimension(image.rows));
    bool uploaded=false,featured=false;
    for(size_t j=0;(shared_image_uploads||shared_image_features)&&j<i;++j) {
      const auto& previous=images[j];
      // from_blob creates contiguous F32 matrices; mirror RequestImageUploads
      // identity/layout and RequestImageFeatures grid/merge identity exactly.
      if(previous.pixels!=image.pixels||previous.rows!=image.rows||previous.patch_dim!=image.patch_dim)continue;
      uploaded=shared_image_uploads;
      if(shared_image_uploads&&shared_image_features&&previous.merge_size==image.merge_size&&previous.token_count==image.token_count&&
         std::equal(std::begin(previous.grid),std::end(previous.grid),std::begin(image.grid)))featured=true;
    }
    if(!uploaded)cost.add({positive_memory_dimension(image.rows),positive_memory_dimension(image.patch_dim),4,sizeof(float)});
    if(!featured)tower(*p.vision,rows,positive_memory_dimension(image.token_count),false,0,grid/positive_memory_dimension(image.grid[0]));
  }
  if(image_count&&p.retained_vision_bytes)cost.add({p.retained_vision_bytes});
  for(size_t i=0;i<audio_count;++i) {
    const auto& audio=audios[i];tower(*p.audio,positive_memory_dimension(audio.frames),
      positive_memory_dimension(audio.token_count),true,positive_memory_dimension(audio.feature_size),0);
  }
  return cost.bytes();
}
}

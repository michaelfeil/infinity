#include "batching/prefix_lookup.h"
#include "models/model.h"
#include <algorithm>
#include <cstring>
#include <limits>
#include <type_traits>
#include <unordered_map>
#define XXH_INLINE_ALL
#define XXH_NAMESPACE tei_prefix_
#include "vendor/xxhash/xxhash.h"

namespace tei {
PrefixFingerprint prefix_admission_fingerprint(const PrefixIdentity& key,int operation) {
  return prefix_admission_fingerprint(key.tokens,operation,key.image_context,key.rotary_context);
}
PrefixFingerprint prefix_admission_fingerprint(std::span<const PrefixToken> tokens,int operation,bool image_context,bool rotary_context) {
  // Collisions only permit a capture; retained identities still compare exactly.
  // Observe common causal stems rather than requiring an identical changing tail.
  auto count=std::min<size_t>(prefix_block_tokens,tokens.size());
  auto hash=XXH3_128bits_withSeed(tokens.data(),count*sizeof(PrefixToken),
    uint64_t(operation)*4+uint64_t(image_context)*2+uint64_t(rotary_context));
  return {hash.low64,hash.high64};
}
uint64_t prefix_admission_step(uint64_t parent,const PrefixToken& token) {
  // A collision only permits a capture; resident lookup verifies exact tokens.
  // The previous digest seeds the next token, retaining the whole prefix.
  return XXH3_64bits_withSeed(token.values.data(),sizeof(token.values),parent);
}
uint64_t prefix_admission_block(uint64_t parent,std::span<const PrefixToken> tokens) {
  return XXH3_64bits_withSeed(tokens.data(),tokens.size_bytes(),parent);
}
PrefixFingerprint prefix_document_fingerprint(std::span<const PrefixToken> tokens,int operation,bool rotary_context) {
  // Separate complete-document observations from the sampled admission stem.
  // A collision can only permit a capture, never authorize resident reuse.
  auto hash=XXH3_128bits_withSeed(tokens.data(),tokens.size_bytes(),
    0x444f43554d454e54ULL+uint64_t(operation)*4+uint64_t(rotary_context));
  return {hash.low64,hash.high64};
}
void prefix_fingerprints(PrefixIdentity& key,int64_t tokens) {
  auto count=std::min<int64_t>(tokens,key.tokens.size())/prefix_block_tokens;
  if(int64_t(key.fingerprints.size())>count)key.fingerprints.resize(count);
  key.fingerprints.reserve(count);
  static_assert(std::is_trivially_copyable_v<PrefixToken> && sizeof(PrefixToken)==6*sizeof(int64_t));
  static_assert(sizeof(PrefixFingerprint)==2*sizeof(uint64_t));
  std::array<uint8_t,sizeof(PrefixFingerprint)+prefix_block_tokens*sizeof(PrefixToken)> bytes;
  auto parent=key.fingerprints.empty()?PrefixFingerprint{0x1122334455667788ULL,0x8877665544332211ULL}:key.fingerprints.back();
  for(int64_t block=key.fingerprints.size();block<count;++block) {
    std::memcpy(bytes.data(),&parent,sizeof(parent));
    std::memcpy(bytes.data()+sizeof(parent),key.tokens.data()+block*prefix_block_tokens,
      prefix_block_tokens*sizeof(PrefixToken));
    auto hash=XXH3_128bits(bytes.data(),bytes.size());parent={hash.low64,hash.high64};
    key.fingerprints.push_back(parent);
  }
}
static int64_t media_common(const PrefixIdentity& a,const PrefixIdentity& b,int64_t n) {
  for(const auto* key:{&a,&b})for(const auto& media:key->media)if(media.start<n) {
    const auto& other=key==&a?b:a;
    auto match=std::find(other.media.begin(),other.media.end(),media);
    if(match==other.media.end()||media.start+media.count>n)n=std::min(n,media.start);
  }
  return n;
}
static bool ordered_media(const PrefixIdentity& key) {
  int64_t end=0;
  for(const auto& media:key.media) {
    if(media.start<end||media.count<=0||media.count>std::numeric_limits<int64_t>::max()-media.start)return false;
    end=media.start+media.count;
  }
  return true;
}
static int64_t media_bound(const PrefixIdentity& a,const PrefixIdentity& b,int64_t end) {
  // Non-overlapping ordered images make the atomic media prefix bound
  // idempotent. Check it before comparing token rows that cannot be reused.
  // Preserve the original scalar ordering for other native caller layouts.
  return ordered_media(a)&&ordered_media(b)?media_common(a,b,end):end;
}
static int64_t token_common(const PrefixIdentity& a,const PrefixIdentity& b,int64_t n,int64_t end) {
  while(n<end) {
    auto count=std::min<int64_t>(prefix_block_tokens,end-n);
    if(std::memcmp(a.tokens.data()+n,b.tokens.data()+n,count*sizeof(PrefixToken))==0) {n+=count;continue;}
    while(n<end&&a.tokens[n]==b.tokens[n])++n;
    break;
  }
  return n;
}
int64_t prefix_common(const PrefixIdentity& a,const PrefixIdentity& b,int64_t limit) {
  if(a.image_context!=b.image_context||a.rotary_context!=b.rotary_context)return 0;
  auto end=std::min<int64_t>({limit,int64_t(a.tokens.size()),int64_t(b.tokens.size())});
  end=media_bound(a,b,end);
  return media_common(a,b,token_common(a,b,0,end));
}
bool prefix_same(const PrefixIdentity& a,const PrefixIdentity& b) {
  if(a.image_context!=b.image_context||a.rotary_context!=b.rotary_context||a.tokens.size()!=b.tokens.size())return false;
  auto blocks=a.tokens.size()/prefix_block_tokens;
  if(blocks&&a.fingerprints.size()>=blocks&&b.fingerprints.size()>=blocks
    &&a.fingerprints[blocks-1]!=b.fingerprints[blocks-1])return false;
  return prefix_common(a,b,a.tokens.size())==int64_t(a.tokens.size());
}
static int64_t estimate(const PrefixIdentity& a,const PrefixIdentity& b,int64_t end) {
  auto blocks=end/prefix_block_tokens;
  if(int64_t(a.fingerprints.size())<blocks||int64_t(b.fingerprints.size())<blocks)
    return prefix_common(a,b,end);
  // A differing parent-chained digest bounds the true prefix. Even a hash
  // collision that makes matches non-monotonic only overestimates this bound:
  // every genuinely equal prefix has equal digests at every earlier block.
  int64_t low=0,high=blocks;
  while(low<high) {
    auto mid=(low+high+1)/2;
    if(a.fingerprints[mid-1]==b.fingerprints[mid-1])low=mid;else high=mid-1;
  }
  auto start=low*prefix_block_tokens;
  return media_common(a,b,token_common(a,b,start,std::min(end,start+prefix_block_tokens)));
}
bool prefix_batch_compatible(const PrefixEntry& entry,const PackedInput& input) {
  return entry.batch_max_sequence==input.max_sequence
    &&entry.compact_tokens==(input.compact_ids.defined()?input.compact_ids.numel():0)
    &&entry.batch_offsets.size()==size_t(input.batch+1)
    &&std::equal(entry.batch_offsets.begin(),entry.batch_offsets.end(),input.offsets);
}
PrefixLookup prefix_lookup(PrefixIdentity& key,const std::vector<std::shared_ptr<PrefixEntry>>& entries,
  int operation,int64_t limit,bool checkpoint_only,const PackedInput* layout) {
  limit=std::min<int64_t>(limit,key.tokens.size());PrefixLookup best;
  bool hashed=false,advisory=false;
  for(const auto& entry:entries) {
    auto end=std::min<int64_t>(limit,entry->identity.tokens.size());
    if((layout&&!prefix_batch_compatible(*entry,*layout))
      ||entry->operation!=operation||entry->identity.image_context!=key.image_context||entry->identity.rotary_context!=key.rotary_context
      ||end<=best.length||key.tokens[best.length]!=entry->identity.tokens[best.length])continue;
    end=media_bound(key,entry->identity,end);
    if(end<=best.length)continue;
    // Before fingerprint ranking is active, avoid a long exact comparison of
    // snapshots still pending on another stream. Metadata misses above still
    // avoid readiness queries entirely.
    const bool ready_checked=!hashed;
    if(ready_checked&&!entry->ready())continue;
    if(!hashed&&best.length>=prefix_block_tokens) {
      prefix_fingerprints(key,std::min<int64_t>(key.tokens.size(),limit+prefix_block_tokens-1));hashed=true;
      // An exact full-bound hit often exists among many late-diverging entries.
      // Try its complete-block digest before refining every competing tail.
      auto probe=(limit+prefix_block_tokens-1)/prefix_block_tokens;
      if(probe&&int64_t(key.fingerprints.size())>=probe)for(const auto& candidate:entries) {
        const auto& identity=candidate->identity;
        if((layout&&!prefix_batch_compatible(*candidate,*layout))
          ||candidate->operation!=operation||identity.image_context!=key.image_context||identity.rotary_context!=key.rotary_context
          ||int64_t(identity.tokens.size())<limit||int64_t(identity.fingerprints.size())<probe
          ||(checkpoint_only&&int64_t(identity.tokens.size())!=limit)
          ||identity.fingerprints[probe-1]!=key.fingerprints[probe-1]
          ||media_bound(key,identity,limit)<limit||!candidate->ready())continue;
        if(prefix_common(key,identity,limit)==limit) {
          // A ceil-block digest includes tokens beyond the reuse bound. An
          // earlier entry may match the entire bound while differing only in
          // those recomputed tokens. Preserve the original first-match tie.
          for(const auto& earlier:entries) {
            if(earlier==candidate)break;
            const auto& prior=earlier->identity;
            if((layout&&!prefix_batch_compatible(*earlier,*layout))
              ||earlier->operation!=operation||prior.image_context!=key.image_context||prior.rotary_context!=key.rotary_context
              ||int64_t(prior.tokens.size())<limit||(checkpoint_only&&int64_t(prior.tokens.size())!=limit))continue;
            if(estimate(key,prior,limit)==limit&&earlier->ready()
              &&prefix_common(key,prior,limit)==limit)return {earlier,limit,false};
          }
          return {candidate,limit,false};
        }
      }
    }
    auto n=hashed?estimate(key,entry->identity,end):prefix_common(key,entry->identity,end);
    if(n<=best.length||(checkpoint_only&&n!=int64_t(entry->identity.tokens.size()))
      ||(!ready_checked&&!entry->ready()))continue;
    best={entry,n,false};advisory=hashed;
    if(n==limit)break;
  }
  if(!advisory||prefix_common(key,best.entry->identity,best.length)==best.length)return best;
  // Do not let a colliding overestimate hide another valid candidate, including
  // a shorter recurrent checkpoint or a prefix with matching media.
  best={nullptr,0,true};
  for(const auto& entry:entries)if((!layout||prefix_batch_compatible(*entry,*layout))
    &&entry->operation==operation&&entry->ready()) {
    auto n=prefix_common(key,entry->identity,limit);
    if(n>best.length&&(!checkpoint_only||n==int64_t(entry->identity.tokens.size()))) {
      best.entry=entry;best.length=n;
    }
  }
  return best;
}

struct PrefixIndex::Impl {
  static constexpr int64_t stride=8;
  static constexpr int64_t short_positions=16;
  static int64_t endpoint(int64_t tokens) {
    return tokens<=short_positions?tokens:short_positions+(tokens-short_positions)/stride*stride;
  }
  struct Scope {
    int operation=0;
    bool rotary=false,layout=false;
    int64_t maximum=0,compact=0;
    std::vector<int32_t> offsets;
    bool operator==(const Scope&) const=default;
  };
  struct ScopeHash {
    size_t operator()(const Scope& scope) const {
      std::array<int64_t,5> values={scope.operation,scope.rotary,scope.layout,scope.maximum,scope.compact};
      auto seed=XXH3_64bits(values.data(),sizeof(values));
      return XXH3_64bits_withSeed(scope.offsets.data(),scope.offsets.size()*sizeof(int32_t),seed);
    }
  };
  struct Key {
    int64_t position=0;
    PrefixFingerprint hash;
    bool operator==(const Key&) const=default;
  };
  struct KeyHash {
    size_t operator()(const Key& key) const {
      return key.hash.low^(key.hash.high<<1)^uint64_t(key.position)*0x9e3779b97f4a7c15ULL;
    }
  };
  using Bucket=std::vector<std::weak_ptr<PrefixEntry>>;
  struct Table {
    std::unordered_map<Key,Bucket,KeyHash> prefixes,exact;
    std::map<int64_t,size_t> depths;
  };
  std::unordered_map<Scope,Table,ScopeHash> tables;
  size_t unsupported=0;
  uint64_t mask;
  explicit Impl(uint64_t mask):mask(mask) {}
  static bool supported(const PrefixEntry& entry) {
    return !entry.identity.image_context&&entry.identity.media.empty()&&!entry.identity.tokens.empty();
  }
  static Scope scope(int operation,bool rotary,const PackedInput* layout) {
    Scope result;result.operation=operation;result.rotary=rotary;result.layout=layout;
    if(layout) {
      result.maximum=layout->max_sequence;
      result.compact=layout->compact_ids.defined()?layout->compact_ids.numel():0;
      result.offsets.assign(layout->offsets,layout->offsets+layout->batch+1);
    }
    return result;
  }
  static Scope scope(const PrefixEntry& entry,bool layout) {
    Scope result;result.operation=entry.operation;result.rotary=entry.identity.rotary_context;result.layout=layout;
    if(layout) {
      result.maximum=entry.batch_max_sequence;result.compact=entry.compact_tokens;result.offsets=entry.batch_offsets;
    }
    return result;
  }
  PrefixFingerprint fingerprint(const void* data,size_t size) const {
    auto hash=XXH3_128bits(data,size);return {hash.low64&mask,hash.high64&mask};
  }
  std::vector<Key> prefixes(const PrefixIdentity& identity,int64_t limit) const {
    std::vector<Key> result;
    const auto small=std::min(limit,short_positions);
    const auto blocks=(limit-small)/stride;result.reserve(small+blocks);
    // The first short question often diverges immediately after an eight-token
    // role header. Index those positions individually, avoiding a large scan
    // of twelve-token questions that otherwise all share the same block hash.
    for(int64_t position=1;position<=small;++position)
      result.push_back({position,fingerprint(identity.tokens.data(),position*sizeof(PrefixToken))});
    auto parent=result.back().hash;
    std::array<uint8_t,sizeof(PrefixFingerprint)+stride*sizeof(PrefixToken)> bytes;
    for(int64_t block=0;block<blocks;++block) {
      std::memcpy(bytes.data(),&parent,sizeof(parent));
      std::memcpy(bytes.data()+sizeof(parent),identity.tokens.data()+small+block*stride,stride*sizeof(PrefixToken));
      parent=fingerprint(bytes.data(),bytes.size());result.push_back({small+(block+1)*stride,parent});
    }
    return result;
  }
  Key exact_key(const PrefixIdentity& identity) const {
    return {int64_t(identity.tokens.size()),fingerprint(identity.tokens.data(),identity.tokens.size()*sizeof(PrefixToken))};
  }
};
PrefixIndex::PrefixIndex(uint64_t mask):impl(std::make_unique<Impl>(mask)) {}
PrefixIndex::~PrefixIndex()=default;
bool PrefixIndex::usable() const {return !impl->unsupported;}
uint64_t PrefixIndex::storage_bytes(int64_t tokens) {
  // Bound bucket capacity, weak references, hash nodes/buckets and the geometry
  // namespace against the resident budget, including the complete-key index.
  auto positions=std::min(tokens,Impl::short_positions)+std::max<int64_t>(0,tokens-Impl::short_positions)/Impl::stride;
  return prefix_storage_sum(256,prefix_storage_bytes(positions+1,128,1));
}
void PrefixIndex::insert(const std::shared_ptr<PrefixEntry>& entry,bool layout) {
  if(!Impl::supported(*entry)){++impl->unsupported;return;}
  auto& table=impl->tables[Impl::scope(*entry,layout)];
  for(const auto& key:impl->prefixes(entry->identity,entry->identity.tokens.size()))table.prefixes[key].push_back(entry);
  table.exact[impl->exact_key(entry->identity)].push_back(entry);
  ++table.depths[Impl::endpoint(entry->identity.tokens.size())];
}
void PrefixIndex::erase(const PrefixEntry& entry,bool layout) {
  if(!Impl::supported(entry)){--impl->unsupported;return;}
  auto found=impl->tables.find(Impl::scope(entry,layout));if(found==impl->tables.end())return;
  auto& table=found->second;
  auto remove=[&](auto& map,const auto& key) {
    auto bucket=map.find(key);if(bucket==map.end())return;
    auto& values=bucket->second;
    values.erase(std::remove_if(values.begin(),values.end(),[&](const auto& value){
      auto reader=value.lock();return !reader||reader.get()==&entry;
    }),values.end());
    if(values.empty())map.erase(bucket);
  };
  for(const auto& key:impl->prefixes(entry.identity,entry.identity.tokens.size()))remove(table.prefixes,key);
  remove(table.exact,impl->exact_key(entry.identity));
  auto depth=table.depths.find(Impl::endpoint(entry.identity.tokens.size()));
  if(--depth->second==0)table.depths.erase(depth);
  if(table.depths.empty())impl->tables.erase(found);
}
bool PrefixIndex::possible(const PrefixToken& token,int operation,bool rotary,const PackedInput* layout) const {
  if(!usable())return true;
  auto table=impl->tables.find(Impl::scope(operation,rotary,layout));
  return table!=impl->tables.end()&&table->second.prefixes.contains({1,impl->fingerprint(&token,sizeof(token))});
}
std::shared_ptr<PrefixEntry> PrefixIndex::exact(const PrefixIdentity& key,int operation,const PackedInput* layout,bool ready) const {
  if(!usable()||key.image_context||!key.media.empty()||key.tokens.empty())return {};
  auto table=impl->tables.find(Impl::scope(operation,key.rotary_context,layout));if(table==impl->tables.end())return {};
  auto found=table->second.exact.find(impl->exact_key(key));if(found==table->second.exact.end())return {};
  for(const auto& weak:found->second)if(auto entry=weak.lock();entry&&prefix_same(key,entry->identity)
    &&(!layout||prefix_batch_compatible(*entry,*layout))&&(!ready||entry->ready()))return entry;
  return {};
}
PrefixLookup PrefixIndex::lookup(PrefixIdentity& key,const std::vector<std::shared_ptr<PrefixEntry>>& entries,
  int operation,int64_t limit,bool checkpoint,const PackedInput* layout) const {
  if(!usable()||checkpoint||key.image_context||!key.media.empty())return prefix_lookup(key,entries,operation,limit,checkpoint,layout);
  limit=std::min<int64_t>(limit,key.tokens.size());if(limit<=0)return {};
  auto table=impl->tables.find(Impl::scope(operation,key.rotary_context,layout));if(table==impl->tables.end())return {};
  const auto& values=table->second;
  auto hashes=impl->prefixes(key,std::min(limit,values.depths.rbegin()->first));
  // All ancestors of a genuine match are present. Hash collisions can only
  // overestimate the deepest matching position; exact checking below detects
  // that case and restores the full scalar-equivalent search.
  int64_t low=-1,high=hashes.size()-1;
  while(low<high) {
    const auto middle=(low+high+1)/2;
    if(values.prefixes.contains(hashes[middle]))low=middle;else high=middle-1;
  }
  if(low<0)return {};
  const auto& bucket=values.prefixes.at(hashes[low]);PrefixLookup best;
  for(const auto& weak:bucket)if(auto entry=weak.lock();entry&&(!layout||prefix_batch_compatible(*entry,*layout))) {
    auto common=prefix_common(key,entry->identity,limit);
    if(common>best.length&&entry->ready())best={std::move(entry),common,false};
    if(best.length==limit)break;
  }
  if(best.length>=hashes[low].position)return best;
  // Pending candidates, expired borrowed entries or a colliding digest may
  // hide a shorter ready reader. Never turn advisory indexing into a miss.
  auto result=prefix_lookup(key,entries,operation,limit,false,layout);result.collision_fallback=true;return result;
}
}

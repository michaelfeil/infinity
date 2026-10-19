#pragma once
#include "batching/prefix_cache.h"
#include <span>

namespace tei {
constexpr int64_t prefix_block_tokens=64;
void prefix_fingerprints(PrefixIdentity&,int64_t tokens);
PrefixFingerprint prefix_admission_fingerprint(const PrefixIdentity&,int operation);
PrefixFingerprint prefix_admission_fingerprint(std::span<const PrefixToken>,int operation,bool image_context,bool rotary_context=false);
uint64_t prefix_admission_step(uint64_t,const PrefixToken&);
uint64_t prefix_admission_block(uint64_t,std::span<const PrefixToken>);
PrefixFingerprint prefix_document_fingerprint(std::span<const PrefixToken>,int operation,bool rotary_context);
int64_t prefix_common(const PrefixIdentity&,const PrefixIdentity&,int64_t limit);
bool prefix_same(const PrefixIdentity&,const PrefixIdentity&);
struct PrefixLookup {
  std::shared_ptr<PrefixEntry> entry;
  int64_t length=0;
  bool collision_fallback=false;
};
bool prefix_batch_compatible(const PrefixEntry&,const PackedInput&);
// Fingerprints only rank candidates. A selected prefix is always verified
// exactly; a collision falls back to the complete exact scan.
PrefixLookup prefix_lookup(PrefixIdentity&,const std::vector<std::shared_ptr<PrefixEntry>>&,
  int operation,int64_t limit,bool checkpoint_only,const PackedInput* layout=nullptr);
// Position-first text index. Buckets borrow snapshots, so indexing never keeps
// an evicted GPU allocation alive. Owners serialize updates with cache runs.
class PrefixIndex {
public:
  explicit PrefixIndex(uint64_t fingerprint_mask=~uint64_t{0});
  ~PrefixIndex();
  void insert(const std::shared_ptr<PrefixEntry>&,bool require_layout);
  void erase(const PrefixEntry&,bool require_layout);
  bool usable() const;
  bool possible(const PrefixToken&,int operation,bool rotary,const PackedInput*) const;
  std::shared_ptr<PrefixEntry> exact(const PrefixIdentity&,int operation,const PackedInput*,bool ready) const;
  PrefixLookup lookup(PrefixIdentity&,const std::vector<std::shared_ptr<PrefixEntry>>&,
    int operation,int64_t limit,bool checkpoint_only,const PackedInput*) const;
  static uint64_t storage_bytes(int64_t tokens);
private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
}

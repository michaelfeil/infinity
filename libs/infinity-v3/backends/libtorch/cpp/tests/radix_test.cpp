#include "batching/radix.h"
#include <iostream>
using namespace tei;
static at::Tensor vector(std::initializer_list<int64_t> x) {return at::tensor(x,at::TensorOptions().dtype(at::kLong));}
static void rejects(const PackedInput& p,bool causal=true) {
 bool threw=false;try {validate_radix(p,causal);}catch(const c10::Error&){threw=true;}
 TORCH_CHECK(threw,"Malformed or context-changing radix metadata accepted");
}
int main() {
 const int32_t offsets[]={0,3,7,10};
 PackedInput p{vector({3,4,5,3,4,6,7,3,4,5}),at::zeros({10},at::kLong),vector({0,1,2,0,1,2,3,0,1,2}),{},offsets,3,4};
 p.compact_ids=vector({3,4,5,6,7});p.compact_positions=vector({0,1,2,2,3});p.compact_types=at::zeros({5},at::kLong);
 p.scatter_unfold=vector({0,1,2,0,1,3,4,0,1,2});p.fold_gather=vector({0,1,2,5,6});
 validate_radix(p,true);rejects(p,false);
 TORCH_CHECK(at::equal(radix_unfold(radix_ids(p),p),p.ids),"Token reconstruction mismatch");
 auto rows=at::arange(15,at::kFloat).view({5,3});
 TORCH_CHECK(at::equal(radix_fold(radix_unfold(rows,p),p),rows),"Representative gather mismatch");
 auto qkv=radix_unfold_qkv(rows.view({5,1,3}),rows.view({5,1,3}),rows.view({5,1,3}),p);
 for(const auto& x:qkv)TORCH_CHECK(at::equal(x,rows.view({5,1,3}).index_select(0,p.scatter_unfold)),"CPU QKV unfold fallback mismatch");
 auto bad=p;bad.compact_positions=at::Tensor();rejects(bad);
 bad=p;bad.fold_gather=vector({0,0,2,5,6});rejects(bad);
 bad=p;bad.scatter_unfold=vector({5,1,2,0,1,3,4,0,1,2});rejects(bad);
 bad=p;bad.types=vector({0,0,0,1,0,0,0,0,0,0});rejects(bad);
 bad=p;bad.compact_ids=vector({3,4,5,6,7,0});bad.compact_positions=vector({0,1,2,2,3,0});bad.compact_types=at::zeros({6},at::kLong);bad.fold_gather=vector({0,1,2,5,6,0});rejects(bad);
 // Equal token/position/type does not justify merging a different causal past.
 const int32_t other[]={0,3,6};bad=p;bad.ids=vector({3,4,5,8,4,5});bad.positions=vector({0,1,2,0,1,2});bad.types=at::zeros({6},at::kLong);bad.offsets=other;bad.batch=2;bad.max_sequence=3;
 bad.compact_ids=vector({3,4,5,8});bad.compact_positions=vector({0,1,2,0});bad.compact_types=at::zeros({4},at::kLong);bad.scatter_unfold=vector({0,1,2,3,1,2});bad.fold_gather=vector({0,1,2,3});rejects(bad);
 // Entire duplicated sequences can share bidirectional context, unlike prefixes.
 bad.ids=vector({3,4,5,3,4,5});bad.compact_ids=vector({3,4,5});bad.compact_positions=vector({0,1,2});bad.compact_types=at::zeros({3},at::kLong);bad.scatter_unfold=vector({0,1,2,0,1,2});bad.fold_gather=vector({0,1,2});validate_radix(bad,false);
 PackedInput plain=p;plain.compact_ids=at::Tensor();plain.compact_positions=at::Tensor();plain.compact_types=at::Tensor();plain.scatter_unfold=at::Tensor();plain.fold_gather=at::Tensor();
 TORCH_CHECK(!radix_enabled(plain)&&radix_ids(plain).is_same(plain.ids)&&radix_unfold(rows,plain).is_same(rows),"Identity radix path copies data");
 auto identity=radix_unfold_qkv(rows,rows,rows,plain);
 for(const auto& x:identity)TORCH_CHECK(x.is_same(rows),"Identity QKV path copies data");
 std::cout<<"radix metadata/context invariants passed\n";
}

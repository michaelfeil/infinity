#include "batching/prefix_lookup.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace infinity;

static void check(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}

static PrefixIdentity identity(int64_t n) {
  PrefixIdentity key;
  for (int64_t i = 0; i < n; ++i)
    key.tokens.push_back({{i + 10, i, i % 2, i / 3, i / 4, i / 5}});
  return key;
}

static auto entry(PrefixIdentity key, int operation = 0, bool ready = true) {
  auto value = std::make_shared<PrefixEntry>();
  prefix_fingerprints(key, key.tokens.size());
  value->identity = std::move(key);
  value->operation = operation;
  value->ready = [ready] { return ready; };
  return value;
}

// Original scalar token/media search, independent of the fingerprint ranking.
static int64_t reference(const PrefixIdentity& a, const PrefixIdentity& b, int64_t limit) {
  if (a.image_context != b.image_context || a.rotary_context != b.rotary_context)
    return 0;
  int64_t n = 0;
  while (n < limit && n < int64_t(a.tokens.size()) && n < int64_t(b.tokens.size()) &&
         a.tokens[n] == b.tokens[n])
    ++n;
  for (const auto* key : {&a, &b})
    for (const auto& media : key->media)
      if (media.start < n) {
        const auto& other = key == &a ? b : a;
        if (std::find(other.media.begin(), other.media.end(), media) == other.media.end() ||
            media.start + media.count > n)
          n = std::min(n, media.start);
      }
  return n;
}

static void compare(PrefixIdentity key,
                    const std::vector<std::shared_ptr<PrefixEntry>>& entries,
                    int operation,
                    int64_t limit,
                    bool checkpoint = false) {
  int64_t expected = 0;
  for (const auto& e : entries)
    if (e->operation == operation && e->ready()) {
      auto n = reference(key, e->identity, limit);
      if (!checkpoint || n == int64_t(e->identity.tokens.size()))
        expected = std::max(expected, n);
    }
  auto result = prefix_lookup(key, entries, operation, limit, checkpoint);
  check(result.length == expected, "Fingerprint lookup changed longest available prefix");
  check(bool(result.entry) == bool(expected), "Fingerprint lookup returned an invalid reader");
  if (result.entry)
    check(reference(key, result.entry->identity, limit) == expected,
          "Fingerprint lookup selected a non-matching identity");
  for (auto mask : {~uint64_t{0}, uint64_t{0}}) {
    PrefixIndex index(mask);
    for (const auto& e : entries)
      index.insert(e, false);
    auto indexed = index.lookup(key, entries, operation, limit, checkpoint, nullptr);
    check(indexed.length == expected && bool(indexed.entry) == bool(expected),
          "Position index changed longest ready prefix, including forced collisions");
    if (indexed.entry)
      check(reference(key, indexed.entry->identity, limit) == expected,
            "Position index selected a non-matching reader");
  }
}

int main() {
  auto key = identity(257);
  prefix_fingerprints(key, 128);
  auto first = key.fingerprints;
  prefix_fingerprints(key, 257);
  check(key.fingerprints.size() == 4, "Incomplete block was hashed");
  check(std::equal(first.begin(), first.end(), key.fingerprints.begin()),
        "Extending fingerprints changed an earlier parent");
  prefix_fingerprints(key, 65);
  check(key.fingerprints.size() == 1, "Truncated identity retained later block digests");
  prefix_fingerprints(key, 257);
  auto fresh = identity(257);
  prefix_fingerprints(fresh, 257);
  check(key.fingerprints == fresh.fingerprints,
        "Rebuilt fingerprints differ from fresh prepared identity");

  std::mt19937_64 random(8943);
  for (int trial = 0; trial < 160; ++trial) {
    auto query = identity(128 + random() % 130);
    query.image_context = trial % 3 == 0;
    query.rotary_context = trial % 2 == 0;
    if (query.image_context) {
      PrefixMedia media;
      media.start = 61;
      media.count = 16;
      media.layout = {1, 4, 4, 2};
      media.hashed = trial % 2;
      media.digest[0] = 93;
      media.content = {12, 44, 19};
      query.media.push_back(media);
    }
    std::vector<std::shared_ptr<PrefixEntry>> entries;
    for (int row = 0; row < 64; ++row) {
      auto candidate = query;
      candidate.tokens.resize(1 + random() % query.tokens.size());
      if (row % 5)
        candidate.tokens[random() % candidate.tokens.size()].values[random() % 6] ^= 31 + row;
      if (!candidate.media.empty() && row % 4 == 0)
        candidate.media[0].digest[0] ^= 1;
      if (!candidate.media.empty() && row % 7 == 0)
        candidate.media[0].content.push_back(83);
      if (row % 11 == 0)
        candidate.image_context = !candidate.image_context;
      if (row % 13 == 0)
        candidate.rotary_context = !candidate.rotary_context;
      entries.push_back(entry(std::move(candidate), row % 6 == 0 ? 1 : 0, row % 9 != 0));
    }
    entries.push_back(entry(query, 0, true));
    for (auto limit : {int64_t(0),
                       int64_t(63),
                       int64_t(64),
                       int64_t(65),
                       int64_t(127),
                       int64_t(query.tokens.size() - 1)})
      for (bool checkpoint : {false, true})
        compare(query, entries, 0, limit, checkpoint);
    compare(query, entries, 1, query.tokens.size() - 1);
  }
  // Deterministic late divergences exercise both complete-block selection and
  // partial-final-block refinement, including an unready maximum candidate.
  for (int64_t n : {128, 191, 192, 257, 2048}) {
    auto query = identity(n + 1);
    std::vector<std::shared_ptr<PrefixEntry>> entries;
    for (int row = 0; row < 64; ++row) {
      auto candidate = identity(n);
      candidate.tokens[n - 64 + row].values[0] += 900;
      entries.push_back(entry(std::move(candidate)));
    }
    entries.push_back(entry(identity(n), 0, false));
    compare(query, entries, 0, n);
    compare(query, entries, 0, n - 1);
    compare(query, entries, 0, n, true);
  }
  // Simulate a collision across every digest. An overestimated long candidate
  // must not hide a valid shorter recurrent checkpoint or trigger false reuse.
  auto query = identity(192), shorter = identity(128), wrong = identity(192);
  wrong.tokens[3].values[4] ^= 1;
  prefix_fingerprints(query, 192);
  auto good = entry(shorter), bad = entry(wrong);
  for (auto* hashes :
       {&query.fingerprints, &good->identity.fingerprints, &bad->identity.fingerprints})
    for (auto& hash : *hashes)
      hash = {};
  auto result = prefix_lookup(query, {good, bad}, 0, 191, false);
  check(result.collision_fallback && result.entry == good && result.length == 128,
        "Collision did not fall back to the exact longest prefix");
  result = prefix_lookup(query, {good, bad}, 0, 191, true);
  check(result.entry == good && result.length == 128,
        "Collision changed recurrent checkpoint eligibility");
  // A colliding reader or complete-block shortcut must never bypass the
  // geometry that produced the saved rows. Invalid readers need no GPU query.
  std::vector<int32_t> ends = {0, 192};
  PackedInput layout;
  layout.offsets = ends.data();
  layout.batch = 1;
  layout.max_sequence = 192;
  good->batch_offsets = ends;
  good->batch_max_sequence = 192;
  bad->batch_offsets = ends;
  bad->batch_max_sequence = 193;
  int incompatible_probes = 0;
  bad->ready = [&] {
    ++incompatible_probes;
    return true;
  };
  result = prefix_lookup(query, {good, bad}, 0, 191, false, &layout);
  check(result.entry == good && result.length == 128 && incompatible_probes == 0,
        "Colliding incompatible geometry remained eligible");
  auto incompatible = entry(identity(192));
  incompatible->batch_offsets = ends;
  incompatible->batch_max_sequence = 193;
  incompatible->ready = bad->ready;
  auto partial = identity(192);
  partial.tokens[190].values[0] ^= 1;
  auto compatible = entry(partial);
  compatible->batch_offsets = ends;
  compatible->batch_max_sequence = 192;
  auto layout_key = identity(192);
  result = prefix_lookup(layout_key, {good, incompatible, compatible}, 0, 191, false, &layout);
  check(result.entry == compatible && result.length == 190 && incompatible_probes == 0,
        "Complete-block shortcut ignored original geometry");
  bad->batch_max_sequence = 192;
  result = prefix_lookup(query, {good, bad}, 0, 191, false, &layout);
  check(result.collision_fallback && result.entry == good && result.length == 128,
        "Geometry-aware collision fallback changed exact selection");
  check(!prefix_same(query, bad->identity), "Collision treated distinct snapshots as identical");
  auto tied = identity(192);
  tied.tokens.back().values[0] ^= 1;
  auto first_tied = entry(tied), later = entry(identity(192));
  auto tie_key = identity(192);
  result = prefix_lookup(tie_key, {entry(identity(128)), first_tied, later}, 0, 191, false);
  check(result.entry == first_tied && result.length == 191,
        "Complete-block shortcut changed the first valid tie");
  // Stale/missing advisory metadata cannot break ordinary exact lookup.
  good->identity.fingerprints.clear();
  compare(query, {good, bad}, 0, 191);
  // Out-of-order/overlapping native media layouts retain the original oracle
  // semantics rather than relying on the ordered-image early bound.
  auto images = identity(192);
  images.image_context = true;
  PrefixMedia wide;
  wide.start = 0;
  wide.count = 100;
  wide.hashed = true;
  wide.digest[0] = 29;
  images.media.push_back(wide);
  auto overlap = images;
  auto extra = wide;
  extra.start = 50;
  extra.count = 8;
  extra.digest[0] ^= 1;
  overlap.media.push_back(extra);
  compare(images, {entry(overlap)}, 0, 191);
  std::reverse(overlap.media.begin(), overlap.media.end());
  compare(images, {entry(overlap)}, 0, 191);
  auto unrelated = identity(192);
  unrelated.tokens[0].values[0] = -1;
  int probes = 0;
  bad->ready = [&] {
    ++probes;
    return true;
  };
  prefix_lookup(unrelated, {bad}, 0, 191, false);
  check(probes == 0, "Unrelated identity unnecessarily queried GPU readiness");
  // Related questions share the same role header; deepest positional buckets
  // must distinguish them, refine partial blocks and preserve insertion ties.
  std::vector<std::shared_ptr<PrefixEntry>> questions;
  PrefixIndex position;
  for (int row = 0; row < 2048; ++row) {
    auto question = identity(12);
    question.tokens[8].values[0] = 10000 + row;
    auto saved = entry(question);
    questions.push_back(saved);
    position.insert(saved, false);
    check(saved.use_count() == 2, "Position index retained ownership of a resident snapshot");
  }
  auto request = questions[1777]->identity;
  request.tokens[11].values[3] ^= 9;
  auto selected = position.lookup(request, questions, 0, 12, false, nullptr);
  check(selected.entry == questions[1777] && selected.length == 11,
        "Shared-role position index did not refine the matching partial block");
  check(position.possible(request.tokens[0], 0, false, nullptr),
        "First-token index lost a resident candidate");
  check(position.exact(questions[1777]->identity, 0, nullptr, true) == questions[1777],
        "Exact index lost the trained readout snapshot");
  auto held = questions[1777];
  position.erase(*held, false);
  questions.erase(questions.begin() + 1777);
  selected = position.lookup(request, questions, 0, 12, false, nullptr);
  check(selected.length == 8 && selected.entry == questions.front(),
        "Eviction left a stale indexed reader or changed the first tie");
  check(!position.exact(held->identity, 0, nullptr, false) && held.use_count() == 1,
        "Evicted index retained the held reader or returned its complete output");
  held.reset();
  auto other_operation = request;
  other_operation.rotary_context = true;
  check(!position.lookup(other_operation, questions, 0, 30, false, nullptr).entry,
        "Position index crossed rotary presence");
  check(!position.lookup(request, questions, 1, 30, false, nullptr).entry,
        "Position index crossed operations");
  // The highest matching bucket may contain only pending work. A shorter ready
  // prefix must remain eligible, even with every digest deliberately colliding.
  for (auto mask : {~uint64_t{0}, uint64_t{0}}) {
    PrefixIndex pending(mask);
    auto ready_short = entry(identity(8)), pending_long = entry(identity(31), 0, false);
    std::vector<std::shared_ptr<PrefixEntry>> values = {ready_short, pending_long};
    for (const auto& e : values)
      pending.insert(e, false);
    auto query = identity(32);
    auto read = pending.lookup(query, values, 0, 31, false, nullptr);
    check(read.entry == ready_short && read.length == 8,
          "Pending deepest bucket hid a ready shorter prefix");
    pending_long->ready = [] { return true; };
    read = pending.lookup(query, values, 0, 31, false, nullptr);
    check(read.entry == pending_long && read.length == 31, "Position index cached stale readiness");
  }
  // Geometry namespaces compare offsets exactly; an incompatible long reader
  // cannot hide a compatible short one or issue a readiness query.
  PrefixIndex geometry;
  auto valid = entry(identity(8)), invalid = entry(identity(31));
  valid->batch_offsets = ends;
  invalid->batch_offsets = ends;
  valid->batch_max_sequence = 192;
  invalid->batch_max_sequence = 193;
  int invalid_queries = 0;
  invalid->ready = [&] {
    ++invalid_queries;
    return true;
  };
  geometry.insert(valid, true);
  geometry.insert(invalid, true);
  auto geometry_key = identity(32);
  selected = geometry.lookup(geometry_key, {valid, invalid}, 0, 31, false, &layout);
  check(selected.entry == valid && selected.length == 8 && invalid_queries == 0,
        "Position index crossed saved CUDA geometry");
  for (const auto& e : questions)
    position.erase(*e, false);
  check(!position.possible(request.tokens[0], 0, false, nullptr),
        "Evicting the registry left stale position buckets");
  std::cout
      << "Prefix lookup: scalar equivalence, media boundaries, checkpoint/readiness, collisions and eviction-independent readers passed\n";
}

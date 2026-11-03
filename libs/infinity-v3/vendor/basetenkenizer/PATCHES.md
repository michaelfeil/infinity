# Basetenkenizer 0.2.9 vendor patch

This directory is the published `basetenkenizer` 0.2.9 source from crates.io,
vendored locally so the startup-only patch below can be built without an
upstream release. The package name, version, license, and public API are
unchanged.

- Crate: `basetenkenizer`
- Version: `0.2.9`
- Upstream source checksum (from the baseline `Cargo.lock`):
  `370b74a776779a158eb432006cd4935fbe25627bf830656c6260d4455b6cd6e5`
- Upstream source path used for the copy:
  `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/basetenkenizer-0.2.9`

## Patch

`src/pre_tokenizers/split.rs`: `Split` construction still keeps the upstream
`max_parallel()` number of regex handles for every pattern and behavior, so
runtime chunking and parallelism are unchanged. It no longer recompiles the
compiled regex for each handle.

- `Pcre2Regex::clone` now calls `self.0.clone()`.
  `pcre2` 0.2.11's `Regex::clone` shares the immutable `Arc<Code>` and creates
  an independent `MatchDataPool`; see `pcre2` 0.2.11's
  `src/bytes.rs`, `impl Clone for Regex`.
  The previous wrapper unnecessarily rebuilt and re-JITed the pattern for every
  copy.
- `try_compile_pcre2_regexes` compiles one PCRE2 regex, then fills the
  `max_parallel()` vector by cloning that handle.
- `compile_regexes` compiles one `fancy_regex::Regex`, then fills the vector by
  cloning it. `fancy_regex` derives `Clone`; its Fancy variant shares
  `Arc<Prog>` compiled code, and its regex-automata wrapper shares the compiled
  implementation while creating a fresh cache pool.
- `TryFrom<SplitRaw>` and `from_config` retain the upstream `max_parallel()`
  handle count and scanner recognition. The construction improvement applies
  to unrecognized patterns such as baselabs/joey-1 as well as scanner patterns.

An additional guard in `pre_tokenize_pcre2_isolated` only enters
`find_matches_pcre2_parallel` when at least two handles exist. Upstream's
parallel splitter assumes multiple handles and panics on a single-CPU process
where `max_parallel() == 1`; with one handle the existing sequential PCRE2
fallback is used. This is a general single-handle guard, not a scanner-specific
precompile shortcut, and the many-core path is unchanged.

### BPE constructor

`src/models/bpe.rs`: `Bpe::new` now avoids startup work the runtime does not
consume and replaces repeated reverse-vocabulary lookups with one-pass dense
tables. BPE IDs, merge ranks/order, sparse-vocabulary rejection, duplicate-ID
handling, and the encoding algorithms are unchanged.

- The reverse vocabulary first records last-write-wins `id -> &str` entries in
  an Fx hash map bounded by the vocabulary size. If `max_id + 1` differs from
  the number of unique IDs, it sorts those IDs and returns the same first
  missing-ID error without allocating the ID range. An input ID of `u32::MAX`
  therefore still costs O(vocabulary) memory instead of attempting a huge
  reverse table.
- Only after contiguity is proven is `id_to_token` built from a dense
  `Vec<Option<&str>>`; filling it in vocabulary iteration order keeps the
  previous last-write-wins behavior for duplicate IDs.
- `id_to_token` stores `Arc<str>`. `token_to_id` reuses those allocations as
  keys; only duplicate-ID strings that lost the reverse lookup need their own
  allocation. This removes a second copy of essentially every vocabulary
  string without changing either map's contents.
- A winning vocabulary token that is the empty string is rejected with the
  upstream DAAC error
  `error building DAAC: InvalidArgumentError: pattern.len() must be >= 1`.
  This preserves the previous invalid-input behavior now that the automaton is
  gone.
- The BPE `DoubleArrayAhoCorasick` and the unused `next_prefix_map` are gone.
  `next_match` used leftmost-longest matching only to test whether the whole
  input is one non-orphan vocabulary token before cache/merge work. The
  replacement marks the single canonical `token_to_id` entry for each non-orphan
  `id_to_token` value and accepts only that entry. This is equivalent because
  `tokenize_into` uses a `next_match` result only when
  `token_lens[token] == input.len()`: a DAAC match starts at byte 0, so equal
  byte length forces it to be the entire input, and a vocabulary token equal to
  the entire input is returned by leftmost-longest matching. Partial-prefix
  DAAC matches failed that length check and had no effect, so `None` produces
  the same branch outcome. Whole-token IDs, cache/merge behavior, and merge
  order are preserved.
- `encoding_decomposition` uses a `char -> id` map built from one-character
  vocabulary entries, a reusable token scratch buffer, and a private Fx-hashed
  copy of the parsed merge pairs for its lookups. The public
  `ParsedMergeMap` and `Bpe::new` signature are unchanged, and the runtime
  `MergeMap`, `RankedMergeMap`, and adjacency construction are unchanged.
- `PartialEq` compares the canonical `id_to_token` text along with the merge
  and unmerge maps, token lengths, orphan flags, `ignore_merges`, and byte
  fallback. Same-layout vocabularies with different token text therefore remain
  unequal, as they were under DAAC pattern equality. A regression test covers
  both this distinction and equality after cloning.

Stage timings on the local `baselabs/joey-1` tokenizer, fresh release process
pinned to one CPU, with temporary instrumentation (stripped afterwards):

| Stage | Before | After |
| --- | ---: | ---: |
| `parse_merges` (unchanged, outside `Bpe::new`) | 77-83 ms | 79-83 ms |
| Reverse vocabulary / `id_to_token` | 18.5 ms | 10.8-11.8 ms |
| Character map / decomposition merge copy | - | 7.3-8.2 ms |
| Decomposition loop | 157-166 ms | 63-65 ms |
| `token_lens` + `next_prefix_map` | 49-52 ms | 0.13 ms |
| BPE automaton | 249-275 ms | removed |
| Flat/ranked merge maps | 34-35 ms | 33 ms |
| Initial byte-pair table | 6.9-7.7 ms | 6.9 ms |
| Merge adjacency | 3.8-5.9 ms | 3.9-4.1 ms |
| `token_to_id` | 34-38 ms | 30.5-30.9 ms |
| `Bpe::new` total | 562-589 ms | 156-162 ms |

Full Basetenkenizer construction fell from 790-827 ms to a 387.7 ms median
(377.7-396.1 ms over ten fresh single-CPU processes), about 52% below the
preceding patched-regex single-CPU median (803.9 ms) and 63% below the
published 256-CPU 0.2.9 baseline (1,055.8 ms).

Runtime checks used the same temporary benchmark. The 50,000 repeated cached
short encodes took ~23.9-24.5 ms versus ~23.0-23.5 ms before (about 3-5% slower
for the exact-token lookup), while 100 long corrected inputs took ~18.6-18.9 ms
versus ~19.2-19.3 ms (faster) and 5,000 unique inputs were ~2.05-2.21 ms versus
~2.04-2.16 ms (within noise). BPE merge paths, merge order, and cache contents
are unchanged.

The vendored manifest also omits example targets whose files are not bundled,
uses the workspace's locked `tokenizers` 0.21.1 for tests, and aligns the
optional `hf-hub` lower bound with the
workspace lock (0.4.2). These packaging changes do not affect the compiled
runtime dependency, which is used with `default-features = false`.

The local `.gitattributes` disables whitespace checks for the unmodified
`THIRD_PARTY_LICENSES.md` so its upstream legal notice formatting stays verbatim.

## Tests

The crate's existing scanner parity tests are retained. Added regression
coverage uses the exact `baselabs/joey-1` Split regex (unrecognized by the
scanner) and a large mixed UTF-8 input above the parallel chunk threshold:

- constructor deserialization and `from_config` produce the full
  `max_parallel()` PCRE2 and fancy-regex vectors and agree on all segments;
- generic PCRE2 parallel matching agrees with one sequential handle;
- generic fancy-regex parallel matching agrees with the sequential path;
- clones from eight threads produce identical segments, exercising shared
  compiled code with independent match-data scratch.


The BPE constructor regressions are pure local synthetic tests:

```sh
cd libs/infinity-v3/vendor/basetenkenizer
cargo test --locked --no-default-features --lib \
  models::bpe::tests::orphan_token_does_not_shortcut_character_merge -- --exact
cargo test --locked --no-default-features --lib \
  models::bpe::tests::non_contiguous_vocab_ids_are_rejected -- --exact
cargo test --locked --no-default-features --lib \
  models::bpe::tests::duplicate_vocab_ids_keep_all_token_lookups -- --exact
cargo test --locked --no-default-features --lib \
  models::bpe::tests::sparse_vocab_ids_are_rejected_without_allocating_the_id_range -- --exact
cargo test --locked --no-default-features --lib \
  models::bpe::tests::empty_vocab_token_is_rejected_like_daac -- --exact
cargo test --locked --no-default-features --lib \
  models::bpe::tests::equality_includes_canonical_token_text_and_survives_clone -- --exact
```

They cover the orphan-token fast-path exclusion that replaces the automaton,
sparse-ID rejection including `u32::MAX` without a huge allocation, empty-token
rejection, duplicate-ID map entries, and canonical-text equality across clones.
The existing BPE tests that read excluded `vendored_tokenizers` fixtures are
not part of these runs.

Run the Split suite with:

```sh
cargo test --locked -p basetenkenizer --no-default-features --lib pre_tokenizers::split::tests
taskset -c 0 cargo test --locked -p basetenkenizer --no-default-features --lib pre_tokenizers::split::tests
```

## Licensing

The original `LICENSE`, `LICENSES/`, `THIRD_PARTY_LICENSES.md`, `README.md`,
and source headers are retained. This patch is an original contribution under
the repository's scoped licensing terms; the vendored component remains under
Apache-2.0 and its bundled third-party notices.

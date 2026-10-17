# Native cache design references

Source review: 2026-10-12. These are implementation references, not comparative performance measurements. The LibTorch cache remains experimental; its learned-model qualification is tracked in [the cache notes](libtorch-prefix-cache.md).

## Ownership and language

Keep the loaded engine's cache manager in C++. It owns ATen tensors, CUDA stream ordering, completion events, and native model adapters. Rust continues to prepare inputs, configure the engine, and dispatch requests through the existing bridge. A useful separation inside C++ is prefix identity and reservation, typed resident storage, then model execution and publication. This preserves one ownership boundary while allowing the lookup implementation to evolve.

## What to adopt

| Reference | Verified mechanism | Application here |
| --- | --- | --- |
| [mistral.rs block pool](https://github.com/EricLBuehler/mistral.rs/blob/28ed6daca455b4ae063896014d1ea698b912c539/mistralrs-core/src/paged_attention/block_pool.rs) | Flat physical block registry, reference counts and an intrusive free queue. Freed blocks keep their cache identity until storage is reassigned. | A future paged arena can share immutable prefix pages and recycle unpinned storage without repeatedly cloning whole prefixes. |
| [mistral.rs cache manager](https://github.com/EricLBuehler/mistral.rs/blob/28ed6daca455b4ae063896014d1ea698b912c539/mistralrs-core/src/paged_attention/kv_cache_manager.rs) | Pins computed blocks and reserves remaining prompt capacity before allocating the tail. Lookups include required cache groups. | Preserve complete admission reservations and count active, resident, reserved and evictable storage separately. |
| [mistral.rs window pool](https://github.com/EricLBuehler/mistral.rs/blob/28ed6daca455b4ae063896014d1ea698b912c539/mistralrs-core/src/paged_attention/windowed_pool.rs) | Separate live and checkpoint capacity; bounded window storage accounts for query length and page alignment. | Replace whole-prefix storage for window layers with a bounded tail, retaining logical positions and checkpoint coverage. |
| [SGLang component tree](https://github.com/sgl-project/sglang/blob/6a7abcfc9b603d494cdc58fa54bc793a915b547c/python/sglang/srt/mem_cache/unified_cache/components/README.md) | One prefix tree coordinates Full, SWA and Mamba components. Every component must validate a reuse boundary; recurrent continuation copies state into private storage. | Coordinate full KV, window KV and DeltaNet/convolution checkpoints through one identity. A tree split must not manufacture a recurrent checkpoint. |
| [llama.cpp hybrid memory](https://github.com/ggml-org/llama.cpp/blob/cbe28bb3cb0758f0c566a9a3561766821aeb0748/src/llama-memory-hybrid.cpp) | Native attention and recurrent memory have coordinated sequence operations. Recurrent removal is attempted before attention removal because it can reject the requested rewind. | Validate all required state components before mutating an entry or exposing a resume point. |
| [FlashInfer attention API](https://docs.flashinfer.ai/api/attention.html) | Paged attention has separate planning and execution interfaces with explicit query offsets, page indices and visible lengths. | Evaluate a native kernel adapter after defining page ownership. Planning and workspace sizing should happen outside repeated layer execution where geometry is shared. |

The mistral.rs paged manager explicitly builds on vLLM v1, so it is a useful native translation of that design rather than an independent cache algorithm. Its [block identity](https://github.com/EricLBuehler/mistral.rs/blob/28ed6daca455b4ae063896014d1ea698b912c539/mistralrs-core/src/paged_attention/block_hash.rs) chains parent context, block tokens and extra keys such as media content and adapter generation. Our prepared identities additionally preserve positions, token types and every rotary axis. Keep exact identity validation or a qualified collision-resistant representation when replacing the current scan.

SGLang's [Rust tree core](https://github.com/sgl-project/sglang/blob/6a7abcfc9b603d494cdc58fa54bc793a915b547c/rust/sglang-radix-tree/README.md) moves tree operations into native code while its controller handles transfers and waits. The transferable boundary is logical planning versus physical I/O. It does not require moving our ATen cache ownership across another language boundary.

## Attention and state constraints

mistral.rs uses an explicit [prefill plan](https://github.com/EricLBuehler/mistral.rs/blob/28ed6daca455b4ae063896014d1ea698b912c539/mistralrs-core/src/paged_attention/plan.rs) selected from dtype, head geometry, windows, masks, causality, cache format and backend support, with a gather fallback. Adopt that capability-driven dispatch principle. A shape being accepted does not establish numerical equivalence with our model's original attention provider.

Its [hybrid state storage](https://github.com/EricLBuehler/mistral.rs/blob/28ed6daca455b4ae063896014d1ea698b912c539/mistralrs-core/src/kv_cache/hybrid_cache.rs) describes recurrent layouts and convolution/checkpoint pools explicitly. Our Qwen3.5 contract must retain its existing FP32 state layout and arithmetic; state quantization and speculative transition logs require separate work and qualification.

For embeddings and classifiers, KV alone does not satisfy all output contracts. Retained final hidden rows support mean/CLS pooling, raw outputs and arbitrary trained selectors. Prefix stability follows the actual causal operation. Bidirectional encoders remain ineligible for arbitrary prefix reuse.

## Integration order

1. Maintain provider qualification. Complete logical projection plans and bounded original-provider scratch resolve the tested Gemma2048 and Mistral/MoE differences. Optional cuDNN preserves original packed geometry through attention replay. New shapes, task/media paths and providers require their own controls.
2. Preserve complete reservations, immutable snapshots and original output semantics. Producer-stream reuse can use queued snapshots immediately; other streams require completion and allocation retention.
3. Introduce an indexed prefix representation while preserving complete parent identity in suffix RadixMLP maps.
4. Add typed page/window/recurrent pools and coordinate their jointly valid boundaries. Keep the current contiguous path as a reference.
5. Qualify native paged attention and stable graph/workspace plans across the supported geometry and dtype matrix.

No runtime dependency on these engines has been added. Reviewed source snapshots retain upstream license files in the external evidence bundle. Any future source port must preserve the applicable notices.

## Prefix lookup cost and index design

The manager retains a bounded registry with64 entries by default. Its adaptive lookup now uses parent-chained64-token XXH3-128 fingerprints stored with each immutable identity. Initial exact comparisons use contiguous token blocks. Once a competing long prefix can improve the result, query hashing ranks candidates and refines their partial blocks. Only an exactly verified available prefix is returned; collisions fall back to the complete scalar-equivalent scan. The original first-entry tie, media boundaries and jointly valid recurrent endpoint rules remain intact.

A Xeon8480+ CPU microbenchmark of the final production selector with64 entries and2048 cached tokens measured a varied late-divergence hit at248.9us before versus15.0us after, a partial-prefix hit at248.9us versus15.7us, and an image-identity hit at252.6us versus2.47us. Unrelated hits improved4.75us to1.96us; immediate misses improved0.263us to0.102us. An all-unready CPU-only fixture increased0.110us to0.386us; pending snapshots avoid long token comparisons. Fresh fingerprint allocation/hashing and exact verification are included. These are warm CPU-cache batch means, excluding prepared identity construction, locks, actual CUDA readiness and inference. The original8.09us flat-table prototype remains separate from production and did not preserve the full media/collision/checkpoint/tie contract.

The current fingerprint-assisted ranking is O(entries × log(blocks)) plus hashing, partial-block refinement and exact verification; the bounded registry keeps it small. It is not a constant-time lookup for an entire fresh sequence. A future flat block table can reduce dependence on entry count if larger registries justify its index maintenance and memory cost.

Use a flat table with parent-chained block hashes, following [vLLM's identity design](https://docs.vllm.ai/en/stable/design/prefix_caching/). XXH3-128 is a reasonable candidate when hash matches undergo exact identity verification and collision buckets preserve alternate candidates. Include model/operation/math policy, token IDs/types/positions/rotary axes and media context in the identity. Hashing a new input requires reading its tokens; expected constant-time lookup applies to one already-computed block key, not an entire fresh sequence or arbitrary longest-prefix search.

The current64-token fingerprints refine an incomplete final block by exact comparison. Choose a reuse boundary only when every required full/window/recurrent component is present and ready. A future index must remove lookup references on eviction without releasing storage still pinned by readers. A reserved contiguous hash table and compact entry handles avoid per-token node allocation. If contention appears, shard by well-mixed hash bits; depth can be part of a key, but all requests visit shallow depths and make them poor lock shards. The current engine serializes inference, so sharding has no demonstrated benefit yet.

For RadixMLP, the same parent-chain identity can supply compact node handles during input preparation. Cached state identity and token-local MLP sharing remain distinct: equal suffix tokens with different parents cannot share a cached causal state.

## Dynamo positional lookup

[Dynamo's positional index](https://github.com/ai-dynamo/dynamo/blob/6c661ddfcef2a75c737b996185e52a400f3bcdac/lib/kv-router/src/indexer/positional.rs) probes depths directly and scans skipped windows when candidate workers drop out. The implementation verifies chained sequence hashes even for a single candidate; hashing still costs O(depth). Its count shortcuts require monotonically shrinking worker sets, which sparse eviction can violate.

For this engine, evaluate a reserved flat table keyed by depth and parent-chained fingerprint. Preserve exact verification, alternate collision candidates, original ties and all component coverage. Acquire an owning snapshot before releasing lookup protection. Benchmark against the existing64-entry selector, including insertion, eviction, first-block misses and varied divergence depths. Depth alone is a poor lock shard because every request visits shallow positions.

## Short prefixes and small blocks

The fingerprint stride is distinct from reusable token count and physical GPU page size. Ordinary causal models refine matches token by token; the current cache stores whole snapshots rather than64-token GPU pages. Recurrent boundaries remain subject to available checkpoints.

Eight-token fingerprints deserve consideration for short, branching prefixes. Three pinned CPU rounds of the existing selector, including fresh query hashing and exact verification, measured32-token late-divergence lookup with64 entries at2.04us for stride8 versus3.40us for stride64;128-token lookup measured3.54us versus8.55us. At4096 entries,32-token cases still took about212us with stride8 versus317us with stride64. Smaller hashing strides help, but the scan still scales poorly. These are warm CPU-cache batch means, excluding inference, locking and identity construction. Longer-prefix cases have different tradeoffs.

The larger opportunity is admitting and cheaply reusing many short prefixes. Current admission hashes up to64 tokens, so changing documents after a shared8–32-token query can produce no admissions. A prepared causal-scoring control demonstrates zero natural hits, while explicitly primed snapshots reuse the expected small prefixes with bitwise-equal scores. Future work should separate admission stem length from lookup stride, retain only useful shared blocks, index thousands of prefixes under a byte budget, and avoid spending more on restoration than the computation saved. Physical8-token pages must be qualified against attention kernels and all required state components.

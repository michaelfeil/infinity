# Independent Read-Only CPU Review — Qwen3-VL RoPE Construction Budget

**Verdict: PASS**

- Candidate: `/tmp/oxide-qwen3vl-rope-budget-fix-v1/candidate`
- Review inputs: `SOURCE-READY.json`, `SOURCE-FREEZE.json`, `patches/`, `proof/`, `build/`, frozen snapshot `027b359`, public HEAD `252789`
- Review mode: read-only source/patch/used-helper/harness review. No GPU, CUDA, query, Torch, Docker, network, compile, optimized build, or writes to any input tree.
- Verified issues: none.

## 1. Actual failure

The preserved cold failure is exactly:

- `/tmp/oxide-qwen3vl-root-oxide-quality-v6/report.json`
- sha256 `cf9548e94235df635ffa6c7464b63d626a41a36b01593097b05b44acab129959`
- `RoPE construction requires 134217984 bytes, exceeding its 134217728-byte budget`
- pins `907a5b0d…f3194`, identical to the successful V6 Native report.

The supplied arithmetic is correct: `262144 × (128/2) × 2 × 4 = 134217728` table bytes, plus `(128/2) × 4 = 256` inverse-frequency bytes, equals `134217984`. The old caller compared the exact required bytes against the bare 128 MiB table cap.

## 2. Production fix is minimal and mathematically correct

Only `libs/infinity-v3/backends/oxide/src/decoder.rs` changes in production:

- `rope_construction_budget(head_dim)` at `decoder.rs:52`
- the actual constructor call site now passes it at `decoder.rs:767`

The helper is exactly:

```
inverse_bytes = checked((head_dim / 2) * size_of::<f32>())
budget        = checked(ROPE_TABLE_BYTE_LIMIT + inverse_bytes)
```

`BaseRopeTables::new` computes `required = checked(inverse_bytes + table_bytes)` (`rotary.rs:106-110`, unchanged). Therefore:

```
required <= budget  <=>  table_bytes + inverse_bytes <= cap + inverse_bytes
                    <=>  table_bytes <= cap
```

- The 128 MiB **table payload cap is retained**.
- The only allowance is the exact checked F32 inverse-frequency scratch, not unbounded slack.
- Overflow paths remain typed `SizeOverflow` / `Start`, never wrapping.
- `ROPE_TABLE_BYTE_LIMIT`, `max_position_embeddings`, RoPE math, dtype, initialization, and position validation are unchanged.
- `rotary.rs` is byte-identical in candidate, frozen snapshot, and public HEAD: sha256 `5ac97a7e119ece22f84a34b692162b1306e1850338c026d555489103817d910a`.

Independent arithmetic check of the boundary cases matches the tests:

| case | table | required | budget | result |
|---|---:|---:|---:|---|
| 262144 × d128 | 134217728 | 134217984 | 134217984 | accept |
| 262145 × d128 | 134218240 | 134218496 | 134217984 | reject |
| budget − 1 | 134217728 | 134217984 | 134217983 | reject |
| multiply overflow | — | overflow | — | typed size error |
| add overflow | fits (2^64−64) | overflow | fits | typed size error |
| tiny d8 × 10 | 320 | 336 | cap+16 | accept, payload byte-identical to legacy |

## 3. Tests exercise the production helper, not a mirror

The three new tests live in `decoder.rs` and call the production `rope_construction_budget` plus the real `BaseRopeTables::new`:

- `decoder.rs:2194` exact 262144/d128 table + 256-byte inverse accepted; bare cap still rejected; next position rejected; budget−1 rejected.
- `decoder.rs:2232` multiplication overflow, add overflow, and helper metadata overflow return typed errors.
- `decoder.rs:2259` ordinary tiny geometry remains byte-identical to the previous constructor.

No test was added to an unused/obsolete helper module. Existing `rope_table_budget_has_explicit_room_for_gte_capacity` is unchanged and still uses the bare cap for GTE geometry.

## 4. Patch scope

Applying each patch to its pinned base via `patch -o -` reproduces the claimed hashes exactly, without touching any input file:

| patch | base | result | expected |
|---|---|---|---|
| `decoder-rope-budget-private.patch` (`8f0fbd78…`) | frozen `027b359` decoder `6850cbb9…` | `28d022df…` | candidate decoder `28d022df…` |
| `qwen3_vl_host_consumer-owning-life.patch` (`8fa61be4…`) | frozen consumer `bfcd6a0c…` | `921a0e56…` | candidate consumer `921a0e56…` |
| `decoder-rope-budget-public.patch` (`63b6c34e…`) | public HEAD `252789` decoder `13dcc849…` | `62a2fcf3…` | `build/public-decoder.rs` `62a2fcf3…` |

The public patch is function-only: one new helper, one call-site change inside `new_with_classifier`, and three tests. It is not an obsolete whole-file copy of the private decoder. Public `new()` delegates to `new_with_classifier`, so the public patch covers the Root-composed Qwen path as well.

`FUNCTION-DELTA.json` sha256 is `ab4ec5f4…`; it reports exactly the new helper, the constructor, and the three new tests, with the generic rotary constructor unchanged.

## 5. Source integrity

- `source-freeze.sha256` (sha256 `0adca4be…`, 1079 files) verifies completely in the candidate.
- Recursive diff candidate vs frozen snapshot shows only the two intended files differ: `src/decoder.rs` and `examples/qwen3_vl_host_consumer.rs`.
- Public HEAD working tree is clean; public decoder and rotary retain their pinned old hashes.
- Frozen tracked decoder/consumer retain old hashes `6850cbb9…` / `bfcd6a0c…`; rotary `5ac97a7e…`.
- The failed capsule report and the failed quality binary are preserved and untouched.

## 6. Host consumer `gpu-quality-life` / `gpu-owning-life`

`gpu_quality_life` (`qwen3_vl_host_consumer.rs:728`) constructs **one** `Qwen3VlBackend` and calls the quality and owning-life paths with `&backend`. No second model load, no per-call weight upload, no fake or Tower-only backend. Routing is via the real `Backend::submit_embed` → `Qwen3VlEngine::submit` path.

Quality contract is unchanged: the original `gpu_quality` body was only factored into `gpu_quality_with_backend` (`:445`), preserving Cls/Mean/LastToken pools plus raw F32 rows for all fixtures and the same output files/schema.

Owning-life contract (`:547-726`):

- All 8 real fixtures are embedded through the ordinary route and their baselines written (`:568-590`). Root fixtures contain the same 8 names and files.
- A=`single-cats`, B=`varying-shapes` are held on the actual Backend route; their geometries differ (1 image vs 4 images). C=`three-frame-cats` and D=`ragged-three-seq` differ in shape.
- A is dropped without calling `complete()` (`:612`); health is checked; B then completes and is compared bit-for-bit with its ordinary baseline using `to_bits()` (`:309-315`, `:615-625`).
- The next different-shape request is compared exactly with its own earlier baseline (`:627-641`).
- A zero-selector malformed-media request is submitted after clearing both selectors; it is rejected by the config-aware preflight before any lease/ticket, accepted only as a typed `Start`/`Inference` error, then the backend recovers and a different fixture matches its baseline exactly (`:643-696`). `prepare_batch` validates media via `pack_batch` before the empty-selector shortcut, and the malformed payload (3 pixels vs expected 16) is a media error, not merely an empty-selector rejection.
- No third submit is attempted while two are held, so there is no `Overloaded` claim; `SlotPool::acquire` is the normal blocking path.
- Resources are retained through drop/errors by the existing `FencedOwner`/workspace-cache design; the same backend and model are reused for recovery.

All new fallible branches and loops propagate with `?`; the unexpected `Ok` malformed case is completed and then still fails the typed check.

## 7. CPU proof and build state

Existing proof logs (hashes match `SOURCE-READY.json`) show:

- focused decoder budget tests: 3 passed
- oxide lib: 489 passed, 0 failed, 56 ignored
- host consumer contract: 3 passed (parser, real-fixture packing/invalid media, report shape)
- factory: 22 passed
- real fixtures: 1 passed
- fmt: pass; clippy `--all-targets -D warnings`: pass
- debug example build: pass
- public HEAD patched focused tests: 3 passed; full lib: 667 passed

The 3 host-consumer CPU tests invoke actual `parse_args`, `fixture_batch`/`pack_batch`, and `OwningLifeEvidence::report`; there is no source-string mirroring.

Only debug/dev builds are evidenced. The two optimized target directories do not exist. `build_once_after_pass.sh` (sha256 `282653e4…`) refuses unless `OXIDE_INDEPENDENT_PASS=1`; the guard-refusal proof is present. The reviewer performed no build.

## 8. Reuse of Native a26 / V6 gold

`/tmp/oxide-qwen3vl-root-quality-v6/report.json` (sha256 `baf89570…`) is a successful Native V6 run with `returncode 0`, exactly 8 observations, and `library_sha256 b66d70b6…`, matching the pinned Native b66 hash. The gold consumer binary `/tmp/opencode/qwen-vl-native-release-target/release/qwen3-vl-b66-consumer` is present with sha256 `89bd6274…` (the pinned a26). It is dynamically linked only to libc/libgcc, so the absent `native214-fa4-mode20` `.so` path does not affect reuse; the V6 report already binds the library hash.

## 9. Non-claims

This review does not claim GPU execution, CUDA context, device compile, quality, lifetime, or performance. It validates the source/patch/harness and the existing CPU evidence only. Root retains the already-specified single conditional opt3/no-LTO build and GPU execution after this PASS; the reviewer did not build or run them.

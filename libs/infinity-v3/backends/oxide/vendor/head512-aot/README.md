# infinity-head512-aot

This is an isolated, Torch-free AOT bridge for packed attention with
`head_dim = 512`. It exists because the pinned FA4 CuTe revision cannot express
a d512 Q/K/V geometry (see [Oxide backend](../../../../docs/oxide-backend.md)). It is a bounded
**Native512 bridge derived from isolated FlashAttention 2.8.3 sources** and the
already-used Native head512 launch. It is *not* a general FlashAttention-2
backend and it leaves the normal FA4 path unchanged. It makes no performance
claim.

The provider supports FP16 and BF16, global causal/noncausal masks, local
windows (kernel launch-template support; Native's adapter currently gates d512
local masks off, so provider-local windows remain experimental and are validated
only against an FP64 reference), independent ragged Q/KV boundaries, runtime GQA
and caller-owned row/head strides with a unit d stride.

## ABI v2 and mandatory cold preparation

ABI v2 is intentionally incompatible with the old ABI. Every public C export
uses a `_v2` suffix, no `_v1` export is emitted, and the manifest schema is
`infinity-head512-aot/v2` with `abi_version: 2`,
`cold_prepare_required: true` and `warm_launch_only: true`. An old host loading
a new bundle fails on the missing `_v1` symbols; a new host loading an old
bundle fails on the mandatory `infinity_head512_prepare_v2` load symbol or the
old schema, before any capability inference or enqueue.

The caller binds exactly one owned CUDA context, then calls the exclusive
`Provider::prepare_current_context(&mut self)`. That cold call applies the fixed
opt-in shared-memory attribute to the six compiled kernel specializations
(FP16/BF16 x noncausal-global/causal-global/noncausal-local). It performs no
allocation, device/context query, GPU launch, synchronization, graph capture or
JIT work. On success the provider stores a `ready` state; repeated preparation
is idempotent. On failure the provider is poisoned and every later
`supports`/`enqueue` call returns a typed error.

`Provider::enqueue` and `Provider::supports` require the prepared state.
`Capabilities::supports` remains a pure geometry/dtype policy check for CPU
validation. Enqueue validates checked extents, physical output capacity and
vector-storage alignment, then launches once directly on the caller's stream;
the warm path performs no context query, `cudaFuncSetAttribute`, allocation,
synchronization, lock or graph capture.

The caller engine owns the `Arc<Provider>` and must keep the library, the single
bound context, the stream and every Work/input/output region alive through GPU
completion **and** provider error drain. The context must not be reset or
migrated while the provider exists.

## Kernel variants

Kernel variants are explicit and cold-selected from caller-supplied hardware
facts:

- `native`: exact SM90/SM100 M64/N32/W4 tile, 128 KiB opt-in shared memory.
- `compact`: SM120 minimum M32/N32/W2 tile, 96 KiB opt-in shared memory.

`Provider::load_for_device(root, architecture, max_optin_smem)` selects the
standard artifact for the architecture and rejects when the device limit is too
small. It performs no device or context query. Native SM90/SM100 arithmetic and
the 128 KiB kernel body remain unchanged; only the compact variant changes the
warp-M tile. SM120 support must not be inferred from a successful compile.

The upstream `flash_fwd_kernel.h`, softmax, mask and trait bodies are copied
byte-for-byte from the isolated FlashAttention 2.8.3 tree. Only the provider's
host selection is new: it instantiates the upstream kernel template directly
with `IsEvenMN=false`, `IsEvenK=true`, dropout/alibi/softcap/return-softmax
false, and the three valid mask pairs. It never calls upstream
`run_flash_fwd`, split-KV or `run_mha_*` dispatch helpers.

## Build (Root, with CUDA compiler and CUTLASS/CuTe headers)

```sh
CUTLASS_INCLUDE=/path/to/cutlass/include HEAD512_OUT=/path/to/out \
  scripts/build_aot.sh
```

The script always writes the fixed four artifact directories: `sm_90` and
`sm_100` native, `sm_120` compact, and `sm_90_compact` compact qualification.
Each manifest records schema v2, the variant, tile, shared-memory contract,
cold-prepare requirement and the source hashes. The script checks the six v2
exports, rejects any `_v1` export, and fails if the host `DT_NEEDED` set names
Torch, ATen, c10 or Python.

`sm_90_compact` is H100 math-qualification evidence for the compact tile only;
it is not SM120 runtime evidence. Device compilation, sanitizer runs, ELF
inspection and the Native oracle are intentionally outside this CPU-only handoff
and remain runtime-pending.

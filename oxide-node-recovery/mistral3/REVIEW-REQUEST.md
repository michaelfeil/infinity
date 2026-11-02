# Independent review request (Mistral3 v5 composer)

Freeze: `evidence/freeze.json`, `BASE-PINS.json`, `SOURCE-PINS.txt`.
Base: current Root working tree at public `55380b7` plus the V5/Qwen35/MeanV6
six-file overlay. Patch: `selective.patch` (source-only composer).

Review points:
1. V5 positive Llama window rejection is preserved; the composer does not apply
   obsolete upstream V4.
2. V6 mean route is preserved for shared/non-Mistral3 decoders; Mistral3 uses
   the Native model-dtype masked mean (BF16 sum round once, BF16 divide, F32).
3. Mistral window semantics: positive sliding_window uniformly/local even at the
   position limit; flat 2410 W32768 is local left=32767, not global.
4. params.json service helper: downloader fallback + router source wiring; raw
   Mistral-common weight prefixes return a typed HF-conversion gap.
5. Cold `OxideBackend::new` Mistral3 hook, CLS accepted, dynamic
   `score.weight.shape[0]` labels, text/raw/image paths.
6. Quality positions 16384/32768/262143 and shifted image case.
7. Production `submit_embed`/`submit_predict` stage and enqueue all device work
   before returning `Inference::Deferred`; only `Deferred` waits/packs. Two
   works may be in flight; failed fences quarantine.
8. Debug/default tests only; no release all-target build, no GPU/Docker/Torch/
   network/device compile.

Router `oxide,http` check/test passes; no LibTorch build was attempted.

## Independent review v1 closure

Consumed `/tmp/oxide-mistral3-serving-independent-review-v1/REVIEW.md` and
`SOURCE-HASHES.sha256`. Findings H1-H10, M1-M5, L1 and P1 are addressed in this
v5 successor: serving-thread CUDA context bind, params.json-only factory
admission, vocab/selector/image-grid/marker validation, pooled+raw capacity,
real Deferred enqueue, Native model-dtype mean, CLS, dynamic labels, result
memory reservation, gated GELU, fatal-fence health poisoning, empty-selector
short-circuit, and synchronized source pins. Optional vision bias is a typed
pre-load rejection because the prepared vision linear path is bias-free.
Source/CPU tests only; no new optimized host, device/AOT, GPU, Torch, Docker or
network execution was performed.

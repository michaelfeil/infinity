# Independent source review (read-only) and resolutions

Reviewer scope: factory ownership, HostIo/lease/synchronization, dynamic-head FA4,
F32 correction, checkpoint preservation, MoE separation.

## Blockers/HIGH found and resolved

1. **Tower feature D2H/H2D + per-call sync** — resolved. `forward_image_device`
   now returns device-resident merged rows (`VisionDeviceOutput`); the serving
   path has no `to_host_vec`, no feature re-upload, and no successful-path
   synchronize. `VisionWorkspace::mark_completed` runs only after `lease.wait`;
   error/drop paths still fence/quarantine.
2. **Manual `Qwen35VlEngine::new` wrong block route** — resolved. Block
   checkpoints use the dynamic-block Decision constructor with `causal=false`;
   `attach` requires a ready run plan and reuses an existing runtime instead of
   double-attaching.
3. **Media-free/audio early-return gaps** — resolved. One
   `validate_media_batch` runs before early returns on both embed and Decision
   routes; media without a VL runtime and audio are rejected.
4. **Source pins stale** — re-baselined after the fixes (`source-pins.json`,
   `freeze.json`, patch, consumer pin).

## Medium items addressed

- Normal factory no longer swallows a declared malformed vision/media config.
- `whole-factory` uses the normal public backend and `Batch.multimodal`; the
  controller uses it for paired whole-model capture.
- Consumer `--pool` is honored.

## Residual, explicitly not claimed

- 35B-A3B remains a typed gap: ratio8 d256 causal mode 26 + MoE text owner.
- FP16/F32 device recipes remain unsupported until the Native exact recipe.
- No GPU/device execution was performed by the author.


## P1 follow-up (Qwen3VL shared audit)

- Added `Qwen35VlInference` RAII owner constructed before returning
  `Inference::Deferred`. Uncalled closure Drop fences the inner lease first.
- On `lease.wait`/fence failure the fusion workspace is forgotten/quarantined,
  not freed as proven.
- Added CPU drop-witness test:
  `dropping_uncalled_vl_deferred_owner_runs_the_fence`.

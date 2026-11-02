# Oxide node recovery

Emergency snapshot before loss of the GPU node, based on public commit `252789b05ea2e09c14971c04dfcce458f863fead`.

This branch preserves unfinished architecture implementations, build sources, controllers, tests, reviews and validation receipts. It is a recovery branch, not a production-ready merge. Existing public behavior remains on `feat/v3-oxide-bert` / PR #876.

`MANIFEST.json` maps every saved file to its original location, SHA-256 and size. Repository files identical to the public base are omitted. Snapshots from different architecture workers must be composed selectively; do not overlay their shared decoder/lib files blindly. Check each source packet's base pins.

Qwen3 MoE is already public at the base commit. Qwen3.5 VL has an actual unresolved numeric mismatch. Mistral mean reduction and lifecycle work remain under validation. H5120 kernels passed 20 Native controls in plain/mem/init; whole trained H5120 model qualification is pending. Device130 compiles for SM90/SM100/SM120; runtime qualification is H100 only.

Model weights, build caches, compiled GPU libraries, and large raw tensor outputs are intentionally excluded. Rebuild them from the preserved build scripts and checkpoint/runtime manifests. The root work state and model/runtime paths are retained in `root-results/WORKSTATE.json`.

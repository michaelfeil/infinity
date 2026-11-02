# Qwen3-VL real SDK controller v2 (e7baa453)

`controller_v2.py` replaces the old plan-only controller. The old
`controller.py` remains immutable history.

- Models are read only from the existing container `/bench/...` mapping.
- No checkpoint weight is copied; only controller files, fixture pixels and the
  two host consumers are staged.
- CPU phase uses `CUDA_VISIBLE_DEVICES=` and runs b66/vendor GOT proof, the
  whole-model Native CPU probe for the 8B packet, and Oxide
  manifest/factory/lifecycle.
- Quality phase uses explicit `CUDA_VISIBLE_DEVICES=0` and explicit Native /
  Oxide / Oxide-lifecycle jobs under the inner `remote_guard.py`.
- Pins include e7baa453 source/readiness, host consumer SHA, Native consumer SHA,
  b66, vendor 13.2.0.0, central FA4 (`/tmp/fa4-root`), and union124.
- `--pins`, `--plan`, and the mocked CPU test are executable without Docker/GPU.

Commands (Root):

```sh
python3 native-b66-cpu-stage/controller_v2.py --pins
python3 native-b66-cpu-stage/controller_v2.py --phase cpu --plan
python3 native-b66-cpu-stage/controller_v2.py --phase cpu --run --output <fresh-outside-source>
python3 native-b66-cpu-stage/controller_v2.py --phase quality --run --output <fresh-outside-source>
```

No SDK/GPU execution is claimed by the author. Root runs the container phases.

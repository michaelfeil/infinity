# Performance

See the [README highlights](../README.md#performance-highlights) for measured
latency, throughput and startup times.

See the [H100 benchmark](h100-alpha19.md) for
per-model measurements, methodology and numerical caveats.

## Radix MLP

[RadixMLP](https://arxiv.org/abs/2601.15013) shares MLP computation across common
prefixes within a batch—for example, a query shared by reranking candidates or
context shared by decision questions. It applies to supported causal models;
reuse depends on matching prefix context and requires no persistent cache.
Use `--radix-mlp-threshold` to control when it applies; `0` disables it.

## Ragged attention

Variable-length inputs are packed without padding to the longest sequence,
reducing memory waste and padding-related OOM risk. Large workloads can still
exhaust memory; bound them with `--max-batch-tokens` and `--max-sequence-length`.

For further tuning, see [CLI options](../README.md#cli-overview),
[efficient client formats](embedding-tensors.md) and
[startup measurements](modernbert-startup.md).

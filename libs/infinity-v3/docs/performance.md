# Performance

Infinity v3 combines native Rust/C++ serving, token-based dynamic batching and
optimized inference. See the [README highlights](../README.md#performance-highlights)
for measured latency, throughput and startup times.

## Radix MLP

[RadixMLP — Intra-batch Deduplication for Causal Transformers](https://arxiv.org/abs/2601.15013)
describes the technique used for supported decision models and rerankers.
It shares MLP computation for common token prefixes instead of repeating that
work for every sequence in the batch.

Typical opportunities include a shared query across reranking candidates or
shared context across decision questions. The benefit depends on how much
eligible prefix content overlaps within the batch; identical tokens with
different preceding context cannot be treated as the same prefix.

The server enables Radix MLP only for compatible models and backends. The
`--radix-mlp-threshold` option controls when compression is worthwhile;
`0` disables it. Bidirectional models do not use this causal prefix optimization.
Reuse is within the batch and does not require a persistent cross-request cache.

## Other performance options

- Tune `--max-batch-tokens` and request concurrency for your latency target;
  see [CLI options](../README.md#cli-overview).
- Use [packed embedding tensors and NumPy views](embedding-tensors.md) to reduce
  client-side conversion overhead.
- Keep weights local for faster startup; see [startup measurements](modernbert-startup.md).

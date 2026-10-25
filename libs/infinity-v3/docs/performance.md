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

## Ragged attention

Infinity packs variable-length sequences instead of padding every input to the
longest sequence in its batch. Mixing short and long inputs therefore avoids
padding-related memory spikes and reduces OOM risk.

For example, sequences of 100, 100 and 8,000 tokens occupy 8,200 token positions
instead of 24,000 padded positions. This illustrates token storage savings, not
total GPU memory usage: long sequences and large batches can still exhaust
memory. Use `--max-batch-tokens` and `--max-sequence-length` to bound the workload.

## Other performance options

- Tune `--max-batch-tokens` and request concurrency for your latency target;
  see [CLI options](../README.md#cli-overview).
- Use [packed embedding tensors and NumPy views](embedding-tensors.md) to reduce
  client-side conversion overhead.
- Keep weights local for faster startup; see [startup measurements](modernbert-startup.md).

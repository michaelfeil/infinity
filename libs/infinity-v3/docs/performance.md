# Performance

The [November 8, 2026 B200 comparison](benchmarks/b200-20261108/results.md)
measures Infinity, vLLM and SGLang on 30 checkpoint candidates. It includes
larger Qwen3, Mistral, SFR, KaLM and Harrier models alongside BGE and Snowflake
encoders, plus three rerankers. Revisions, parameter counts, workload lengths
and per-case numerical outcomes are recorded in the dataset.

The Infinity build uses clean source at commit
`53dd6f2cb3083e72b65fbe2406913bc09ab75952`; comparator packages are vLLM 0.33.0
and SGLang 0.5.23. Results apply to the recorded configuration. Failed or
numerically mismatched cases remain diagnostic.

## Current B200 examples

End-to-end server-counted **input tokens/s**, BF16, one input per HTTP request.
The examples below require qualified results from all three runtimes, including
the independent FP32 probe check. Cases with more than twofold variation
between timing windows are excluded from these examples.

<!-- benchmark-examples -->
| Checkpoint (512 tokens, concurrency 32) | Infinity | vLLM | SGLang |
|---|---:|---:|---:|
| `tencent/KaLM-Embedding-Gemma3-12B-2511` | 43,965 | 54,739 | 43,418 |
| `Qwen/Qwen3-Embedding-8B` | 86,223 | 87,108 | 88,611 |
| `intfloat/e5-mistral-7b-instruct` | 86,429 | 88,775 | 89,982 |
| `Qwen/Qwen3-Embedding-4B` | 148,325 | 144,927 | 150,472 |
| `Snowflake/snowflake-arctic-embed-l-v2.0` | 1,176,442 | 800,265 | 446,064 |
| `BAAI/bge-large-en-v1.5` | 1,155,862 | 724,468 | 360,990 |
<!-- /benchmark-examples -->

Harrier 27B and both SFR 7B checkpoints completed measurements in all three
runtimes, but SGLang's saved probes failed the numerical check in this
configuration. They remain in the full report as diagnostic results. KaLM 12B
qualifies at 128 and 512 tokens; its 2,048-token comparator probes failed the
check. These outcomes do not establish task quality or general runtime support.

The [full comparison](benchmarks/b200-20261108/results.md) includes 128- and
512-token cases and 2,048-token cases for larger checkpoints. Short cases use
concurrency 32; long cases use 8. Each case has three ten-second windows after
warmup. Each GPU lane has separate server and client CPU cores; host activity
and cache state are uncontrolled. Minimum serial latency, latency under
concurrency and startup to health are separate measurements.

See the [method and reproduction steps](benchmark-method.md),
[CSV](benchmarks/b200-20261108/results.csv),
[JSON](benchmarks/b200-20261108/results.json) and
[compressed probe evidence](benchmarks/b200-20261108/evidence.jsonl.gz).
Numerical agreement is a limited smoke check, not a task-quality evaluation or
a guarantee of maximum serving capacity.

## Historical measurements

The [H100 benchmark](h100-alpha19.md) records the October 2026 Infinity V3.0.1
experiment, including vLLM and TEI comparisons. The
[startup study](modernbert-startup.md) measures ModernBERT and base Qwen3-8B.
Their builds, precision, workloads and sampling differ from the B200 dataset;
keep the experiments separate when interpreting the numbers.

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

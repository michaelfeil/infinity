# Telemetry

## Principles:
- **No surprises** — you will be notified before we begin collecting data. You will be notified of any changes to the data being collected or how it is used.
- **Easy opt-out:** You will be able to easily opt-out of data collection
- **Transparency** — you will be able to review all data that is sent to us
- We will **not** sell data or buy data about you.

## Why telemetry is useful for the infinity project
All CLI arguments are currently logged, and system info: OS, GPU, Processor, arm/x86

Examples how this improves infinity:
- GPU model effects:
    - if e.g. flash-attn build would be useful / worth the effort, or not worth it because most card are still
    - default parameters, e.g. batch size
- Model Architecture:
    - Helps create custom flash attention builds
    - If enough people deploy a DebertaV2 based model, work on faster implementation for that.
- Devices: 
    - prioritizes: apple mps / neuron / AMD depelopment 
- CLI args, to deprecate old ones or less popular ones such as onnx/tensorrt gpu + optimum or ctranslate2. Or to see if the v1 of the cli is still used.

## Disable Telemetry
You can disable tracking like the following:

```bash
# set 
export DO_NOT_TRACK="1"
# infinity specific setting
export INFINITY_ANONYMOUS_USAGE_STATS="0"
```

This is in line with various FOSS projects:
- https://docs.vllm.ai/en/latest/serving/usage_stats.html#usage-stats-collection
- https://docs.ray.io/en/latest/cluster/usage-stats.html
- https://github.com/langflow-ai/langflow/blob/eedfe43e6983bfed3c9a15e197c0206ea21b14eb/docs/docs/Contributing/contributing-telemetry.md?plain=1#L25
- https://github.com/LineaLabs/lineapy/blob/eebe5d5862f9888eaf619ba9bbeaa21b3ea5d2e5/README.md?plain=1#L266
- https://github.com/dbt-labs/dbt-core/blob/78c05718c589a56bc49f7520b47474690ae1cbe0/core/dbt/contracts/project.py#L19

## Review which data is sent:
https://github.com/michaelfeil/infinity/blob/main/libs/infinity_emb/infinity_emb/telemetry.py 
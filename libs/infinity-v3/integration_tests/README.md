# Reference fixtures

This directory contains Python scripts that generate independent Transformers
reference fixtures for the Candle backend's integration tests:

- `embeddinggemma_reference.py`
- `deberta_reference.py`

See the corresponding Rust tests under `backends/candle/tests` for the required
checkpoints, fixture paths, and feature flags. Python is used to generate
reference data; inference serving uses Candle.

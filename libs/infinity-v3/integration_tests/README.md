# Reference fixtures

This directory contains Python scripts that generate independent Transformers
reference fixtures for the native backend's reference checks:

- `embeddinggemma_reference.py`
- `deberta_reference.py`

Python is used only to generate reference data; inference serving uses LibTorch.
Native parity runners live under `backends/libtorch/tests`.

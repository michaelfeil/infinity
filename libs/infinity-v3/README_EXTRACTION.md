# Voyage AI Linear Layer Extraction

This directory contains scripts to extract linear output projection layers from Voyage AI models and convert them to the sentence-transformers dense module format.

## Why Use Dense Modules Instead of Built-in Linear Projection?

1. **Modularity**: Dense modules can be used with any model architecture
2. **Standardization**: Follows sentence-transformers ecosystem conventions
3. **Flexibility**: Easy to swap projection layers without modifying model code
4. **Compatibility**: Works with existing infinity_v3 dense_paths system

## Files

- `extract_voyage_linear.py` - Main extraction script
- `test_extraction.py` - Test script to verify extraction
- `requirements_extract.txt` - Python dependencies
- `README_EXTRACTION.md` - This documentation

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements_extract.txt
```

### 2. Extract Linear Layer

```bash
# Extract from voyage-4-nemo (default)
python extract_voyage_linear.py

# Extract from a different Voyage model
python extract_voyage_linear.py --model-id voyageai/voyage-4-nano

# Custom output directory
python extract_voyage_linear.py --output-dir ./my_voyage_dense
```

### 3. Verify Extraction

```bash
python test_extraction.py
```

## Output Structure

The extraction creates a sentence-transformers compatible structure:

```
voyage_dense/
├── config.json          # Main model config (without linear projection)
├── modules.json         # Points to dense module
├── 1_Dense/
│   ├── config.json      # Dense layer configuration
│   └── model.safetensors # Dense layer weights (linear.weight, linear.bias)
├── README.md           # Usage instructions
└── README_EXTRACTION.md # This documentation
```

## Dense Module Configuration

The extracted dense module will have a config like:

```json
{
  "in_features": 1024,
  "out_features": 2048,
  "bias": true,
  "activation_function": "torch.nn.modules.linear.Identity"
}
```

## Integration with Text-Embeddings-Inference

Once extracted, the dense module integrates seamlessly:

```bash
infinity_v3 --model-id ./voyage_dense
```

The system will automatically:
1. Load the main model
2. Detect `modules.json`
3. Load the dense module from `1_Dense/`
4. Apply the linear projection during inference

## Manual Loading Example

```python
from infinity_v3_backend_candle import CandleBackend
from infinity_v3_backend_core import ModelType, Pool

# The dense_paths will be auto-detected from modules.json
backend = CandleBackend::new(
    &model_path,
    "float32".to_string(),
    ModelType::Embedding(Pool::LastToken),
    None,  # Auto-detect dense modules
    0,
)
```

## Troubleshooting

### Linear Layer Not Found

If the script can't find the linear layer, it will print all available tensor names. Look for patterns like:
- `linear.weight`
- `output_projection.weight`
- `dense.weight`
- `projection.weight`

### Memory Issues

For large models, use GPU acceleration:
```bash
python extract_voyage_linear.py --device cuda
```

### Permission Errors

Ensure you have write permissions to the output directory:
```bash
chmod 755 ./voyage_dense
```

## Supported Models

The script is designed to work with Voyage AI models that have linear output projection layers:

- `voyageai/voyage-4-nemo`
- `voyageai/voyage-4-nano`
- And other Voyage models with similar architecture

## How It Works

1. **Download**: Downloads model files from HuggingFace Hub
2. **Load**: Loads safetensors and identifies linear projection layer
3. **Extract**: Extracts linear weights and bias (if present)
4. **Structure**: Creates sentence-transformers compatible directory structure
5. **Configure**: Generates appropriate config files for dense module loading

## Contributing

To add support for other model families:

1. Update the linear layer detection patterns in `extract_linear_layer()`
2. Test with the target model
3. Update documentation

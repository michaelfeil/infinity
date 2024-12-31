# CLI v2 Documentation

The current version of Infinity uses the following arguments in its CLI:
```bash
$ infinity_emb v2 --help
```

```
                                                                                                                        
 Usage: infinity_emb v2 [OPTIONS]                                                                                       
                                                                                                                        
 Infinity API ♾️  cli v2. MIT License. Copyright (c) 2023-now Michael Feil                                               
 Multiple Model CLI Playbook:                                                                                           
 - 1. cli options can be overloaded i.e. `v2 --model-id model/id1 --model-id model/id2 --batch-size 8 --batch-size 4`   
 - 2. or adapt the defaults by setting ENV Variables separated by `;`: INFINITY_MODEL_ID="model/id1;model/id2;" &&      
 INFINITY_BATCH_SIZE="8;4;"                                                                                             
 - 3. single items are broadcasted to `--model-id` length, making `v2 --model-id model/id1 --model-id/id2 --batch-size  
 8` both models have batch-size 8.                                                                                      
                                                                                                                        
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --model-id                                             TEXT                           Huggingface model repo id.     │
│                                                                                       Subset of possible models:     │
│                                                                                       https://huggingface.co/models… │
│                                                                                       [env var: `INFINITY_MODEL_ID`] │
│                                                                                       [default:                      │
│                                                                                       michaelfeil/bge-small-en-v1.5] │
│ --served-model-name                                    TEXT                           the nickname for the API,      │
│                                                                                       under which the model_id can   │
│                                                                                       be selected                    │
│                                                                                       [env var:                      │
│                                                                                       `INFINITY_SERVED_MODEL_NAME`]  │
│ --batch-size                                           INTEGER                        maximum batch size for         │
│                                                                                       inference                      │
│                                                                                       [env var:                      │
│                                                                                       `INFINITY_BATCH_SIZE`]         │
│                                                                                       [default: 32]                  │
│ --revision                                             TEXT                           huggingface  model repo        │
│                                                                                       revision.                      │
│                                                                                       [env var: `INFINITY_REVISION`] │
│ --trust-remote-code       --no-trust-remote-code                                      if potential remote modeling   │
│                                                                                       code from huggingface repo is  │
│                                                                                       trusted.                       │
│                                                                                       [env var:                      │
│                                                                                       `INFINITY_TRUST_REMOTE_CODE`]  │
│                                                                                       [default: trust-remote-code]   │
│ --engine                                               [torch|ctranslate2|optimum|ne  Which backend to use. `torch`  │
│                                                        uron|debugengine]              uses Pytorch GPU/CPU, optimum  │
│                                                                                       uses ONNX on                   │
│                                                                                       GPU/CPU/NVIDIA-TensorRT,       │
│                                                                                       `CTranslate2` uses             │
│                                                                                       torch+ctranslate2 on CPU/GPU.  │
│                                                                                       [env var: `INFINITY_ENGINE`]   │
│                                                                                       [default: torch]               │
│ --model-warmup            --no-model-warmup                                           if model should be warmed up   │
│                                                                                       after startup, and before      │
│                                                                                       ready.                         │
│                                                                                       [env var:                      │
│                                                                                       `INFINITY_MODEL_WARMUP`]       │
│                                                                                       [default: model-warmup]        │
│ --vector-disk-cache       --no-vector-disk-cache                                      If hash(request)/results       │
│                                                                                       should be cached to SQLite for │
│                                                                                       latency improvement.           │
│                                                                                       [env var:                      │
│                                                                                       `INFINITY_VECTOR_DISK_CACHE`]  │
│                                                                                       [default: vector-disk-cache]   │
│ --device                                               [cpu|cuda|mps|tensorrt|auto]   device to use for computing    │
│                                                                                       the model forward pass.        │
│                                                                                       [env var: `INFINITY_DEVICE`]   │
│                                                                                       [default: auto]                │
│ --device-id                                            TEXT                           device id defines the model    │
│                                                                                       placement. e.g. `0,1` will     │
│                                                                                       place the model on             │
│                                                                                       MPS/CUDA/GPU 0 and 1 each      │
│                                                                                       [env var:                      │
│                                                                                       `INFINITY_DEVICE_ID`]          │
│ --lengths-via-tokenize    --no-lengths-via-tokenize                                   if True, returned tokens is    │
│                                                                                       based on actual tokenizer      │
│                                                                                       count. If false, uses          │
│                                                                                       len(input) as proxy.           │
│                                                                                       [env var:                      │
│                                                                                       `INFINITY_LENGTHS_VIA_TOKENIZ… │
│                                                                                       [default:                      │
│                                                                                       lengths-via-tokenize]          │
│ --dtype                                                [float32|float16|bfloat16|int  dtype for the model weights.   │
│                                                        8|fp8|auto]                    [env var: `INFINITY_DTYPE`]    │
│                                                                                       [default: auto]                │
│ --embedding-dtype                                      [float32|int8|uint8|binary|ub  dtype post-forward pass. If != │
│                                                        inary]                         `float32`, using Post-Forward  │
│                                                                                       Static quantization.           │
│                                                                                       [env var:                      │
│                                                                                       `INFINITY_EMBEDDING_DTYPE`]    │
│                                                                                       [default: float32]             │
│ --pooling-method                                       [mean|cls|auto]                overwrite the pooling method   │
│                                                                                       if inferred incorrectly.       │
│                                                                                       [env var:                      │
│                                                                                       `INFINITY_POOLING_METHOD`]     │
│                                                                                       [default: auto]                │
│ --compile                 --no-compile                                                Enable usage of                │
│                                                                                       `torch.compile(dynamic=True)`  │
│                                                                                       if engine relies on it.        │
│                                                                                       [env var: `INFINITY_COMPILE`]  │
│                                                                                       [default: compile]             │
│ --bettertransformer       --no-bettertransformer                                      Enables varlen                 │
│                                                                                       flash-attention-2 via the      │
│                                                                                       `BetterTransformer`            │
│                                                                                       implementation. If available   │
│                                                                                       for this model.                │
│                                                                                       [env var:                      │
│                                                                                       `INFINITY_BETTERTRANSFORMER`]  │
│                                                                                       [default: bettertransformer]   │
│ --preload-only            --no-preload-only                                           If true, only downloads models │
│                                                                                       and verifies setup, then exit. │
│                                                                                       Recommended for pre-caching    │
│                                                                                       the download in a Dockerfile.  │
│                                                                                       [env var:                      │
│                                                                                       `INFINITY_PRELOAD_ONLY`]       │
│                                                                                       [default: no-preload-only]     │
│ --host                                                 TEXT                           host for the FastAPI uvicorn   │
│                                                                                       server                         │
│                                                                                       [env var: `INFINITY_HOST`]     │
│                                                                                       [default: 0.0.0.0]             │
│ --port                                                 INTEGER                        port for the FastAPI uvicorn   │
│                                                                                       server                         │
│                                                                                       [env var: `INFINITY_PORT`]     │
│                                                                                       [default: 7997]                │
│ --url-prefix                                           TEXT                           prefix for all routes of the   │
│                                                                                       FastAPI uvicorn server. Useful │
│                                                                                       if you run behind a proxy /    │
│                                                                                       cascaded API.                  │
│                                                                                       [env var:                      │
│                                                                                       `INFINITY_URL_PREFIX`]         │
│ --redirect-slash                                       TEXT                           where to redirect `/` requests │
│                                                                                       to.                            │
│                                                                                       [env var:                      │
│                                                                                       `INFINITY_REDIRECT_SLASH`]     │
│                                                                                       [default: /docs]               │
│ --log-level                                            [critical|error|warning|info|  console log level.             │
│                                                        debug|trace]                   [env var:                      │
│                                                                                       `INFINITY_LOG_LEVEL`]          │
│                                                                                       [default: info]                │
│ --permissive-cors         --no-permissive-cors                                        whether to allow permissive    │
│                                                                                       cors.                          │
│                                                                                       [env var:                      │
│                                                                                       `INFINITY_PERMISSIVE_CORS`]    │
│                                                                                       [default: no-permissive-cors]  │
│ --api-key                                              TEXT                           api_key used for               │
│                                                                                       authentication headers.        │
│                                                                                       [env var: `INFINITY_API_KEY`]  │
│ --proxy-root-path                                      TEXT                           Proxy prefix for the           │
│                                                                                       application. See:              │
│                                                                                       https://fastapi.tiangolo.com/… │
│                                                                                       [env var:                      │
│                                                                                       `INFINITY_PROXY_ROOT_PATH`]    │
│ --help                                                                                Show this message and exit.    │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```
Note: This doc is auto-generated. Do not edit this file directly.

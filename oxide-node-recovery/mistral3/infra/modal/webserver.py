"""Michael Feil, MIT License, 2024-06-17

This is a reference implementation for infinity server via CLI.
For deploying private endpoints/models, add the secrets (`INFINITY_API_KEY` and `HF_TOKEN`) to modal.
"""
import subprocess
import os
from modal import Image, App, gpu, web_server

# Configuration
PORT = 7997

VERSION_INF = os.environ.get("VERSION_INF", "0.0.63")
ENV = {
    # Per model args, padded by `;`
    "INFINITY_MODEL_ID": "jinaai/jina-clip-v1;michaelfeil/bge-small-en-v1.5;mixedbread-ai/mxbai-rerank-xsmall-v1;philschmid/tiny-bert-sst2-distilled;",
    "INFINITY_REVISION": "1cbe5e8b11ea3728df0b610d5453dfe739804aa9;ab7b31bd10f9bfbb915a28662ec4726b06c6552a;1d1a9dfbd0fde63df646402cf33e157e5852ead3;874eb28543ea7a7df80b6158bbf772d203efcab6;",
    "INFINITY_MODEL_WARMUP": "false;false;false;false;",
    "INFINITY_BATCH_SIZE": "32;32;32;32;",
    # One-off args
    "INFINITY_QUEUE_SIZE": "4096",
    "INFINITY_PORT": str(PORT),
    "DO_NOT_TRACK": os.environ.get("CI_DEPLOY_INF", ""),
}

CMD = "infinity_emb v2"

MINUTES = 60  # in seconds


def download_models():
    """downloads the models into the docker container at build time.
    Ensures no downtime when huggingface is down.
    """
    print(f"downloading models {os.environ.get('INFINITY_MODEL_ID')}")
    exit_code = subprocess.Popen(CMD + " " + "--preload-only", shell=True).wait()
    print("downloading models done.")
    assert exit_code == 0, f"Failed to download models. Exit code: {exit_code}"


# ### Image definition
# We'll start from a slim Linux image and install `infinity_emb` and a few dependencies.
image = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        f"infinity_emb[all]=={VERSION_INF}",
    )
    .env(ENV)
    .run_function(
        download_models,
        timeout=20 * MINUTES,
    )
)

app = App("infinity", image=image)


# Run a web server on port 7997 and expose the Infinity embedding server
@app.function(
    # allow up to 16 requests pending on a single container
    allow_concurrent_inputs=8,
    # boots take around 30 seconds, so keep containers alive for a few times longer
    container_idle_timeout=3 * MINUTES,
    # max 5 container instances
    concurrency_limit=9,
    # scale to zero
    keep_warm=0,
    # use an inexpensive GPU
    gpu=gpu.L4(),
)
@web_server(
    PORT,
    startup_timeout=5 * MINUTES,
    custom_domains=(
        ["infinity.modal.michaelfeil.eu"] if os.environ.get("CI_DEPLOY_INF", "") else []
    ),
)
def serve():
    subprocess.Popen(CMD, shell=True)

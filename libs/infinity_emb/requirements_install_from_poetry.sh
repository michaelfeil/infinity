#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Use PYTHON variable if set, else default to 'python3'
PYTHON_COMMAND=${PYTHON:-python3}

# Initialize variables
EXTRA_URL=""
NO_ROOT=0
WITHOUT=""
WITH=""

# Parse command-line arguments
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --no-root)
    NO_ROOT=1
    shift
    ;;
    --without)
    WITHOUT="$2"
    shift # past argument
    shift # past value
    ;;
    --with)
    WITH="$2"
    shift # past argument
    shift # past value
    ;;
    *)
    if [[ -z "$EXTRA_URL" ]]; then
        EXTRA_URL="$1"
        shift
    else
        echo "Unknown argument: $1"
        echo "Usage: $0 [--no-root] [--without <value>] [--with <value>] <extra_torch_url_py>"
        exit 1
    fi
    ;;
esac
done

# Check if EXTRA_URL is set
if [ -z "$EXTRA_URL" ]; then
    echo "Usage: $0 [--no-root] [--without <value>] [--with <value>] <extra_torch_url_py>"
    exit 1
fi

# Step 1: Build the poetry export command with optional arguments
POETRY_EXPORT_CMD=(poetry export --without-hashes --format=requirements.txt --extras "${EXTRAS}")

if [[ -n "$WITHOUT" ]]; then
    POETRY_EXPORT_CMD+=("--without" "$WITHOUT")
fi

if [[ -n "$WITH" ]]; then
    POETRY_EXPORT_CMD+=("--with" "$WITH")
fi

# Export the dependencies to requirements.txt without hashes
"${POETRY_EXPORT_CMD[@]}" > requirements.txt

# Step 2: Delete the existing virtual environment
if [ -d "./.venv" ]; then
    rm -rf ./.venv
fi

# Step 3: Create a new virtual environment using the specified Python command
$PYTHON_COMMAND -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install matching Torch and TorchVision wheels from the requested CPU/ROCm index
# before the remaining requirements, so dependent packages cannot pull CUDA wheels.
read -r TORCH_VERSION TORCHVISION_VERSION < <($PYTHON_COMMAND - <<'PYTHON'
from pathlib import Path
import tomllib
packages = {p["name"]: p["version"] for p in tomllib.loads(Path("poetry.lock").read_text())["package"]}
print(packages["torch"], packages["torchvision"])
PYTHON
)
python -m pip install --no-cache-dir "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" --index-url "$EXTRA_URL"
sed -i '/^torch==/d; /^torchvision==/d; /^nvidia-/d; /^triton==/d' requirements.txt
python -m pip install --no-cache-dir -r requirements.txt

# Step 6: Optionally install the current package
if [[ $NO_ROOT -eq 0 ]]; then
    python -m pip install -e . --no-deps --no-cache-dir --extra-index-url "$EXTRA_URL"
fi

# rm requirements.txt

echo "Script executed successfully!"
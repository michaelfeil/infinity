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

# Step 4: Extract the version of torch from requirements.txt
TORCH_VERSION=$(grep '^torch==' requirements.txt | awk -F'==' '{print $2}')

# Check if torch version was found
if [ -z "$TORCH_VERSION" ]; then
    echo "Torch version not found in requirements.txt"
    exit 1
fi

# Remove lines containing 'torch', 'nvidia', and 'triton' from requirements.txt
sed -i '/^torch==/d' requirements.txt
sed -i '/torchvision/d' requirements.txt
sed -i '/nvidia/d' requirements.txt
sed -i '/triton/d' requirements.txt

# Step 5: Install torch with the extracted version and other dependencies

python -m pip install -r requirements.txt --no-cache-dir --extra-index-url "$EXTRA_URL"
python -m pip list --format=freeze | grep nvidia | xargs python -m  pip uninstall -y triton torch torchvision
python -m pip install torch torchvision --index-url "$EXTRA_URL" --no-cache-dir

# Step 6: Optionally install the current package
if [[ $NO_ROOT -eq 0 ]]; then
    python -m pip install -e . --no-deps --no-cache-dir --extra-index-url "$EXTRA_URL"
fi

# rm requirements.txt

echo "Script executed successfully!"
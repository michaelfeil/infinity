#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Function to handle cleanup
cleanup() {
  echo "Cleaning up..."
  if [[ -n "${INFINITY_PID:-}" ]]; then
    kill "$INFINITY_PID"
  fi
}

# Set up the trap to run the cleanup function on EXIT or any error
trap cleanup EXIT

# Start infinity_emb in the background
DO_NOT_TRACK=1 infinity_emb v2 --log-level error --engine debugengine --port 7996 &
INFINITY_PID=$!
echo "infinity_emb started with PID $INFINITY_PID"

# Wait for infinity_emb to be ready
for i in {1..20}; do
  if wget -q --spider http://0.0.0.0:7996/openapi.json; then
    echo "infinity_emb is ready."
    break
  else
    echo "Waiting for infinity_emb to be ready..."
    sleep 1
  fi
done

# Download the openapi.json
wget http://0.0.0.0:7996/openapi.json -O "$SCRIPT_DIR/openapi.json"
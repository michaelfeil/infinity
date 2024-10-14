#!/bin/bash

set -euo pipefail

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
infinity_emb v2 --log-level error --engine debugengine --port 7993 &
INFINITY_PID=$!
echo "infinity_emb started with PID $INFINITY_PID"

# Wait for infinity_emb to be ready
for i in {1..10}; do
  if wget -q --spider http://0.0.0.0:7993/openapi.json; then
    echo "infinity_emb is ready."
    break
  else
    echo "Waiting for infinity_emb to be ready..."
    sleep 1
  fi
done

# Run the tests
pip install openapi-python-client==0.21.1
	 openapi-python-client generate  \
	  --url http://0.0.0.0:7993/openapi.json \
	  --config client_config.yaml \
	   --overwrite \
	   --custom-template-path=./template

# Cleanup will be called due to the trap
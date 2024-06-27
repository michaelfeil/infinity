#!/bin/bash

# Function to handle cleanup
cleanup() {
  echo "Cleaning up..."
  pkill -f infinity_emb
}

# Set up the trap to run the cleanup function on EXIT or any error
trap cleanup EXIT

# Start infinity_emb in the background
infinity_emb v2 --log-level error &
echo "infinity_emb started with PID $!"

# Run the tests
cd infinity_client && \
poetry install && \
poetry run pip install pytest requests && \
poetry run python -m pytest ../tests
# Cleanup will be called due to the trap
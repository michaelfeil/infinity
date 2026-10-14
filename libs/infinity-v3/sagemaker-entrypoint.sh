#!/bin/bash

if [[ -z "${HF_MODEL_ID}" ]]; then
    echo "HF_MODEL_ID must be set"
    exit 1
fi
export MODEL_ID="${HF_MODEL_ID}"

if [[ -n "${HF_MODEL_REVISION}" ]]; then
    export REVISION="${HF_MODEL_REVISION}"
fi

exec infinity_v3 --port 8080 --json-output

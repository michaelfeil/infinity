#!/bin/bash

if ! command -v nvidia-smi &>/dev/null; then
    echo "Error: 'nvidia-smi' command not found."
    exit 1
fi

compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv | sed -n '2p' | sed 's/\.//g')

if [ ${compute_cap} -ge 80 -a ${compute_cap} -lt 90 ]; then
    exec infinity_v3-80 "$@"
elif [ ${compute_cap} -eq 90 ]; then
    exec infinity_v3-90 "$@"
else
    echo "cuda compute cap ${compute_cap} is not supported; CUDA requires SM80 or newer"
    exit 1
fi

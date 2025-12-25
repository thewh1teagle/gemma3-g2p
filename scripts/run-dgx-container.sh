#!/bin/bash
# Launch DGX Spark training container with proper volume mounts

# Get absolute paths
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_CACHE_DIR="${HOME}/.cache/huggingface"

# Create cache directory if it doesn't exist
mkdir -p "${HF_CACHE_DIR}"

# Launch container
docker run -it --rm \
    --gpus=all \
    --net=host \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${PROJECT_DIR}:/workspace/gemma3-g2p" \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    -v "${HOME}/.wandb:/root/.wandb" \
    -e WANDB_API_KEY="${WANDB_API_KEY}" \
    -e WANDB_PROJECT="gemma3" \
    -w /workspace/gemma3-g2p \
    unsloth-dgx-spark \
    "$@"

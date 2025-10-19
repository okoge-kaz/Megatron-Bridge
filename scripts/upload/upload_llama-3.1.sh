#!/bin/bash

set -e

ITERATIONS=(1000 2000 2500 3000 4000 5000 6000 7000 7500 8000 9000 10000 11000 12000 12500)

upload_base_dir=/groups/gch51639/fujii/checkpoints/megatron-to-hf/Llama-3.1-8B/swallow-code/exp20

upload_checkpoint() {
  local upload_dir=$1
  local repo_name=$2
  local max_retries=5
  local retry_count=0

  while [ $retry_count -lt $max_retries ]; do
    if python scripts/upload/upload.py \
        --ckpt-path "$upload_dir" \
        --repo-name "$repo_name"; then
        echo "Successfully uploaded $repo_name"
        return 0
    else
        echo "Upload failed for $repo_name. Retrying..."
        ((retry_count++))
        sleep 5
    fi
  done

  echo "Failed to upload $repo_name after $max_retries attempts"
  return 1
}

for ITERATION in "${ITERATIONS[@]}"; do
  FORMATTED_ITERATION="$(printf "%07d" "${ITERATION}")"
  upload_dir=$upload_base_dir/iteration_${FORMATTED_ITERATION}
  repo_name="tokyotech-llm/Llama-3.1-8B-swallow-code-v2-exp20-iter${FORMATTED_ITERATION}"

  if ! upload_checkpoint "$upload_dir" "$repo_name"; then
    echo "Skipping to next checkpoint after repeated failures for $repo_name"
  fi
done

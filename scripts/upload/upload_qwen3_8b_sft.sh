#!/bin/bash

set -e

ITERATIONS=(7800)

upload_base_dir=/groups/gch51639/fujii/checkpoints/megatron-to-hf/Qwen3-8B-reasoning-ablation/sft/exp19/LR1.50E-5-MINLR1.50E-6-WD0.1

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
  repo_name="tokyotech-llm/Qwen3-8B-SFT-Reasoning-exp19-LR1.5E-5-iter${FORMATTED_ITERATION}"

  if ! upload_checkpoint "$upload_dir" "$repo_name"; then
    echo "Skipping to next checkpoint after repeated failures for $repo_name"
  fi
done

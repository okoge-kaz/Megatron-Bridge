#!/bin/bash
#SBATCH --job-name=mcore-to-hf
#SBATCH --time=1:00:00
#SBATCH --partition=part-group_9d80ef
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=outputs/mcore-to-hf/%x-%j.out
#SBATCH --error=outputs/mcore-to-hf/%x-%j.out

set -eu -o pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "Node name: $SLURMD_NODENAME"
echo "----------------------------------------"

module load singularitypro/4.1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_CUDA_ARCH_LIST="9.0"
nvidia-smi --query-gpu=index,name,memory.free --format=csv

ITERATIONS=(100 1600 3200 6400 9000 12800 18000 25000)

for ITERATION in "${ITERATIONS[@]}"; do
  FORMATTED_ITERATION="$(printf "%07d" "${ITERATION}")"
  echo -e "Converting iteration ${ITERATION}\n"

  # model config
  HF_CHECKPOINT_DIR=/home/group_9d80ef/kazuki_fujii/hf_checkpoints/Qwen3-14B-Base
  MEGATRON_CHECKPOINT_DIR=/home/group_9d80ef/kazuki_fujii/checkpoints/Qwen3-Swallow-14B-v0.2/tp4-pp1-ct1/LR1.50E-5-MINLR1.50E-6-WD0.1/iter_${FORMATTED_ITERATION}
  HF_CHECKPOINT_SAVE_DIR=/home/group_9d80ef/kazuki_fujii/checkpoints/megatron-to-hf/Qwen3-Swallow-14B-v0.2/tp4-pp1-ct1/LR1.50E-5-MINLR1.50E-6-WD0.1/iteration_${FORMATTED_ITERATION}
  mkdir -p "${HF_CHECKPOINT_SAVE_DIR}"

  # skip if megatron checkpoint not found
  if [[ ! -d "${MEGATRON_CHECKPOINT_DIR}" ]]; then
    echo "WARN: ${MEGATRON_CHECKPOINT_DIR} not found. Skipping..."
    continue
  fi

  export CUDA_DEVICE_MAX_CONNECTIONS=1
  MEGATRON_LM_PATH=/home/group_9d80ef/kazuki_fujii/src/Megatron-LM-v0.15.0
  MEGATRON_BRIDGE_PATH=/home/group_9d80ef/kazuki_fujii/src/Megatron-Bridge-v0.15.0/src

  # convert
  singularity exec \
  --nv \
  --bind /home/user_00001_9d80ef:/home/user_00001_9d80ef \
  --bind /home/group_9d80ef:/home/group_9d80ef \
  --bind /dev/shm:/dev/shm \
  --bind /tmp:/tmp \
  /home/group_9d80ef/kazuki_fujii/container/nemo-25.11.sif \
  bash -c "export PYTHONPATH=${MEGATRON_BRIDGE_PATH}:${MEGATRON_LM_PATH}:\${PYTHONPATH:-} && \
    python examples/conversion/convert_checkpoints.py export \
      --hf-model ${HF_CHECKPOINT_DIR} \
      --megatron-path ${MEGATRON_CHECKPOINT_DIR} \
      --hf-path ${HF_CHECKPOINT_SAVE_DIR} \
      --model-type gpt"
done

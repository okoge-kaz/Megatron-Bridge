#!/bin/bash
#SBATCH --job-name=hf-to-mcore
#SBATCH --time=1:00:00
#SBATCH --partition=part-group_9d80ef
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --output=outputs/hf-to-mcore/%x-%j.out
#SBATCH --error=outputs/hf-to-mcore/%x-%j.out

set -eu -o pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "Node name: $SLURMD_NODENAME"
echo "----------------------------------------"

module load singularitypro/4.1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCH_CUDA_ARCH_LIST="9.0"

# model config
HF_CHECKPOINT_DIR=/home/group_9d80ef/kazuki_fujii/hf_checkpoints/Qwen3-0.6B-Base
MEGATRON_CHECKPOINT_DIR=/home/group_9d80ef/kazuki_fujii/checkpoints/hf-to-megatron/Megatron-Bridge/Qwen3-0.6B-Base

mkdir -p ${MEGATRON_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL=/home/group_9d80ef/kazuki_fujii/hf_checkpoints/Qwen3-0.6B-Base

export CUDA_DEVICE_MAX_CONNECTIONS=1
MEGATRON_LM_PATH=/home/group_9d80ef/kazuki_fujii/src/Megatron-LM-v0.17.0-dev
MEGATRON_BRIDGE_PATH=$(pwd)/src
export PYTHONPATH="$MEGATRON_BRIDGE_PATH:$MEGATRON_LM_PATH:${PYTHONPATH:-}"

nvidia-smi --query-gpu=index,name,memory.free --format=csv

# convert
singularity exec \
  --nv \
  --bind /home/user_00001_9d80ef:/home/user_00001_9d80ef \
  --bind /home/group_9d80ef:/home/group_9d80ef \
  --bind /dev/shm:/dev/shm \
  --bind /tmp:/tmp \
  /home/group_9d80ef/kazuki_fujii/container/ngc-pytorch-26.01.sif \
  python examples/conversion/convert_checkpoints.py import \
  --hf-model ${HF_CHECKPOINT_DIR} \
  --megatron-path ${MEGATRON_CHECKPOINT_DIR} \
  --torch-dtype bfloat16

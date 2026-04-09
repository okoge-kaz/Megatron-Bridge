#!/bin/bash
#SBATCH --job-name=convert_hf_megatron
#SBATCH --time=1:00:00
#SBATCH --partition=part-group_821839-9-w1j7
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --output=outputs/convert/hf-to-megatron/%x-%j.out
#SBATCH --error=outputs/convert/hf-to-megatron/%x-%j.out

set -eu -o pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "Node name: $SLURMD_NODENAME"
echo "----------------------------------------"

module load singularitypro/4.1
module load hpcx/v2.18.1-cuda12

# distributed settings
export MASTER_ADDR=$SLURMD_NODENAME
export MASTER_PORT=$(( 10000 + SLURM_JOB_ID % 50000 ))
echo "MASTER_ADDR=${MASTER_ADDR}"

WORKSPACE=/home/user_00024_821839/workspace/checkpoints/megatron-bridge/megatron
mkdir -p ${WORKSPACE}

singularity exec \
  --nv \
  --bind /home/user_00024_821839:/home/user_00024_821839 \
  --bind /home/group_821839:/home/group_821839 \
  --bind /dev/shm:/dev/shm \
  --bind /tmp:/tmp \
  /home/user_00024_821839/workspace/container/nemo-26.0.2.sif \
  /opt/venv/bin/python examples/conversion/convert_checkpoints.py import \
    --hf-model Qwen/Qwen3-VL-8B-Instruct \
    --megatron-path ${WORKSPACE}/Qwen3-VL-8B-Instruct

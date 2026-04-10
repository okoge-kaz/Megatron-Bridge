#!/bin/bash
#SBATCH --job-name=pretrain_qwen3_vl_8b
#SBATCH --time=1:00:00
#SBATCH --partition=part-group_821839-9-w1j7
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=outputs/pretrain/qwen3-vl-8b/%x-%j.out
#SBATCH --error=outputs/pretrain/qwen3-vl-8b/%x-%j.out

set -eu -o pipefail

echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"
echo "Node name: $SLURMD_NODENAME"
echo "----------------------------------------"

module load singularitypro/4.1

# distributed settings
export MASTER_ADDR=$SLURMD_NODENAME
export MASTER_PORT=$(( 10000 + SLURM_JOB_ID % 50000 ))
echo "MASTER_ADDR=${MASTER_ADDR}"

nvidia-smi --query-gpu=index,name,memory.free --format=csv

NUM_GPU_PER_NODE=8
NUM_NODES=$SLURM_JOB_NUM_NODES

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=22

TENSOR_PARALLEL_SIZE=1
PIPELINE_PARALLEL_SIZE=1
CONTEXT_PARALLEL_SIZE=1
SEQUENCE_PARALLEL=False

TRAIN_ITERS=1000
SEQ_LENGTH=8192
GLOBAL_BATCH_SIZE=64
MICRO_BATCH_SIZE=4

LR=3.0e-4
MIN_LR=3.0e-5
LR_WARMUP_ITERS=500
LR_DECAY_ITERS=${TRAIN_ITERS}

# dataset: mock / hf / preloaded / energon
DATASET_TYPE="hf"

FREEZE_ARGS=""
FREEZE_ARGS="${FREEZE_ARGS} --freeze-language-model"
FREEZE_ARGS="${FREEZE_ARGS} --freeze-vision-model"
# FREEZE_ARGS="${FREEZE_ARGS} --freeze-vision-projection"


LOG_INTERVAL=1
TIMING_LOG_LEVEL=0
THROUGHPUT_WINDOW_SIZE=1
WANDB_ARGS=""
WANDB_ARGS="${WANDB_ARGS} --wandb-project Qwen3-VL-8B"
WANDB_ARGS="${WANDB_ARGS} --wandb-exp-name pretrain-benchmark"
WANDB_ARGS="${WANDB_ARGS} --wandb-entity okoge"

SAVE_INTERVAL=500
EVAL_INTERVAL=500
CHECKPOINT_DIR=/home/user_00024_821839/workspace/checkpoints/qwen3-vl-8b

SEQUENCE_PARALLEL_ARGS=""
if [[ "${SEQUENCE_PARALLEL}" == "True" ]]; then
  SEQUENCE_PARALLEL_ARGS="--sequence-parallel"
fi

CHECKPOINT_ARGS=""
if [[ -n "${CHECKPOINT_DIR:-}" ]]; then
  mkdir -p ${CHECKPOINT_DIR}
  CHECKPOINT_ARGS="--checkpoint-dir ${CHECKPOINT_DIR}"
fi

export SINGULARITYENV_PYTHONPATH=/home/group_821839/home/kazuki_fujii/src/infra/Megatron-Bridge-v0.3.1/src${PYTHONPATH:+:$PYTHONPATH}

srun singularity exec \
  --nv \
  --bind /home/user_00024_821839:/home/user_00024_821839 \
  --bind /home/group_821839:/home/group_821839 \
  --bind /dev/shm:/dev/shm \
  --bind /tmp:/tmp \
  /home/user_00024_821839/workspace/container/nemo-26.0.2.sif \
  torchrun \
    --nproc_per_node=$NUM_GPU_PER_NODE \
    --nnodes=$NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    experiments/slurm/qwen3-vl/pretrain_qwen3_vl_8b.py \
    --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE} \
    --context-parallel-size ${CONTEXT_PARALLEL_SIZE} \
    ${SEQUENCE_PARALLEL_ARGS} \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
    --global-batch-size ${GLOBAL_BATCH_SIZE} \
    --micro-batch-size ${MICRO_BATCH_SIZE} \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-warmup-iters ${LR_WARMUP_ITERS} \
    --lr-decay-iters ${LR_DECAY_ITERS} \
    --dataset-type ${DATASET_TYPE} \
    ${FREEZE_ARGS} \
    --log-interval ${LOG_INTERVAL} \
    --log-throughput \
    --log-throughput-to-tensorboard \
    --throughput-window-size ${THROUGHPUT_WINDOW_SIZE} \
    --log-timers-to-tensorboard \
    --log-memory-to-tensorboard \
    --log-runtime-to-tensorboard \
    --log-world-size-to-tensorboard \
    --log-params-norm \
    --log-progress \
    --timing-log-level ${TIMING_LOG_LEVEL} \
    --save-interval ${SAVE_INTERVAL} \
    --eval-interval ${EVAL_INTERVAL} \
    ${WANDB_ARGS} \
    ${CHECKPOINT_ARGS}

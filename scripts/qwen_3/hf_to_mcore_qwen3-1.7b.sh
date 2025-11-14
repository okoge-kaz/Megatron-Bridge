#!/bin/sh
#PBS -q rt_HF
#PBS -N hf-to-megatron
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/convert/hf-to-megatron/

cd $PBS_O_WORKDIR
mkdir -p outputs/convert/hf-to-megatron

echo "Nodes allocated to this job:"
cat $PBS_NODEFILE

source /etc/profile.d/modules.sh
module use /home/acf15649kv/modules/modulefiles

module load cuda/12.9.1
module load cudnn/9.10.2
module load nccl/2.27.5-cuda12.9
module load hpcx/2.23.0

MEGATRON_LM_PATH=/home/acf15649kv/src/Megatron-LM-v0.14.0rc7
MEGATRON_BRIDGE_PATH=$(pwd)/src
export PYTHONPATH=$PYTHONPATH:$MEGATRON_LM_PATH
export PYTHONPATH=$PYTHONPATH:$MEGATRON_BRIDGE_PATH

source $MEGATRON_LM_PATH/.venv/bin/activate

# model config
HF_CHECKPOINT_DIR=/groups/gag51395/hf_checkpoints/Qwen3-1.7B-Base
MEGATRON_CHECKPOINT_DIR=/groups/gag51395/checkpoints/hf-to-megatron/Megatron-Bridge/Qwen3-1.7B-Base

mkdir -p ${MEGATRON_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL=/groups/gag51395/hf_checkpoints/Qwen3-1.7B-Base

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MEGATRON_ARGS=1

# convert
python examples/conversion/convert_checkpoints.py import \
  --hf-model ${HF_CHECKPOINT_DIR} \
  --megatron-path ${MEGATRON_CHECKPOINT_DIR} \
  --torch-dtype bfloat16

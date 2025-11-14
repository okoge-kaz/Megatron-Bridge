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
module load hpcx/2.20

MODEL_NAME=Qwen3-Next-80B-A3B-Instruct
# MODEL_NAME=Qwen3-Next-80B-A3B-Thinking

MEGATRON_LM_PATH=/home/acf15649kv/src/Megatron-LM-v0.14.0rc7
MEGATRON_BRIDGE_PATH=$(pwd)/src
export PYTHONPATH=$PYTHONPATH:$MEGATRON_LM_PATH
export PYTHONPATH=$PYTHONPATH:$MEGATRON_BRIDGE_PATH

# model config
HF_CHECKPOINT_DIR=/groups/gch51639/fujii/hf_checkpoints/${MODEL_NAME}
MEGATRON_CHECKPOINT_DIR=/groups/gch51639/fujii/checkpoints/hf-to-megatron/Megatron-Bridge-r0.2.0/${MODEL_NAME}
mkdir -p ${MEGATRON_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL=/groups/gch51639/fujii/hf_checkpoints/${MODEL_NAME}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MEGATRON_ARGS=1

# convert
singularity exec \
  --nv \
  --bind /groups/gag51395:/groups/gag51395 \
  --bind /groups/gch51639:/groups/gch51639 \
  --bind /home/acf15649kv:/home/acf15649kv \
  --bind /dev/shm:/dev/shm \
  --bind /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:/etc/ssl/certs/ca-certificates.crt \
  --bind /tmp:/tmp \
  /groups/gch51639/fujii/container/ngc-pytorch-25.09.sif \
  python examples/conversion/convert_checkpoints.py import \
    --hf-model ${HF_CHECKPOINT_DIR} \
    --megatron-path ${MEGATRON_CHECKPOINT_DIR} \
    --torch-dtype bfloat16 \
    --iteration 1

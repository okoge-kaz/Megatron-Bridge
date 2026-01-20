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

# model config
HF_CHECKPOINT_DIR=/groups/gag51395/hf_checkpoints/Qwen3-32B
MEGATRON_CHECKPOINT_DIR=/groups/gag51395/checkpoints/hf-to-megatron/Megatron-Bridge/Qwen3-32B

mkdir -p ${MEGATRON_CHECKPOINT_DIR}

# tokenizer config
TOKENIZER_MODEL=/groups/gag51395/hf_checkpoints/Qwen3-32B

export CUDA_DEVICE_MAX_CONNECTIONS=1
MEGATRON_LM_PATH=/home/acf15649kv/src/Megatron-LM-v0.15.0
MEGATRON_BRIDGE_PATH=$(pwd)/src
export PYTHONPATH="$PYTHONPATH:$MEGATRON_LM_PATH:$MEGATRON_BRIDGE_PATH"

# convert
singularity exec \
  --nv \
  --bind /groups/gag51395:/groups/gag51395 \
  --bind /groups/gch51639:/groups/gch51639 \
  --bind /home/acf15649kv:/home/acf15649kv \
  --bind /dev/shm:/dev/shm \
  --bind /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:/etc/ssl/certs/ca-certificates.crt \
  --bind /tmp:/tmp \
  /groups/gch51639/fujii/container/ngc-pytorch-25.10.sif \
  python examples/models/checkpoint_conversion.py import \
  --hf-model ${HF_CHECKPOINT_DIR} \
  --megatron-path ${MEGATRON_CHECKPOINT_DIR} \
  --torch-dtype bfloat16

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

# model config
HF_CHECKPOINT_DIR=/groups/gag51395/hf_checkpoints/gpt-oss-20b
MEGATRON_CHECKPOINT_DIR=/groups/gch51639/fujii/checkpoints/hf-to-megatron/Megatron-Bridge/2025-11/gpt-oss-20b
mkdir -p ${MEGATRON_CHECKPOINT_DIR}

export CUDA_DEVICE_MAX_CONNECTIONS=1
MEGATRON_LM_PATH=/groups/gch51639/fujii/src/library/Megatron-LM-core-0.15.0-dev
MEGATRON_BRIDGE_PATH=$(pwd)/src
export PYTHONPATH=$PYTHONPATH:$MEGATRON_LM_PATH
export PYTHONPATH=$PYTHONPATH:$MEGATRON_BRIDGE_PATH

# ngc-pytorch-25.10: transformers==4.55.4
# local: ~/.local/lib/python3.12/site-packages/ (transformers, torchao)

singularity exec \
  --nv \
  --bind /groups/gag51395:/groups/gag51395 \
  --bind /groups/gch51639:/groups/gch51639 \
  --bind /home/acf15649kv:/home/acf15649kv \
  --bind /dev/shm:/dev/shm \
  --bind /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:/etc/ssl/certs/ca-certificates.crt \
  --bind /tmp:/tmp \
  /groups/gch51639/fujii/container/ngc-pytorch-25.10.sif \
  python examples/conversion/convert_checkpoints.py import \
    --hf-model ${HF_CHECKPOINT_DIR} \
    --megatron-path ${MEGATRON_CHECKPOINT_DIR} \
    --torch-dtype bfloat16 \
    --trust-remote-code

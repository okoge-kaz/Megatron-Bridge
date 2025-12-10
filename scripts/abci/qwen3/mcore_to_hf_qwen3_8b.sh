#!/bin/sh
#PBS -q rt_HF
#PBS -N megatron-to-hf
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -V
#PBS -o outputs/convert/megatron-to-hf/

cd $PBS_O_WORKDIR
mkdir -p outputs/convert/megatron-to-hf/

echo "Nodes allocated to this job:"
cat $PBS_NODEFILE

source /etc/profile.d/modules.sh
module use /home/acf15649kv/modules/modulefiles
module load hpcx/2.20

MODEL_NAME=Qwen3-8B-Base

MEGATRON_LM_PATH=/home/acf15649kv/src/Megatron-LM-core_dev_v0.15.0
MEGATRON_BRIDGE_PATH=$(pwd)/src
export PYTHONPATH=$PYTHONPATH:$MEGATRON_LM_PATH
export PYTHONPATH=$PYTHONPATH:$MEGATRON_BRIDGE_PATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

ITERATIONS=(10 100 500)

MEGATRON_DIR=/groups/gch51639/fujii/checkpoints/Qwen-3-Swallow-8B-v0.1-fp8-param-test/tp2-pp1-ct1/LR1.50E-5-MINLR1.50E-6-WD0.1
HF_DIR=/groups/gch51639/fujii/checkpoints/megatron-to-hf/Qwen3-Swallow-8B-fp8-param-test/tp2-pp1-ct1/LR1.50E-5-MINLR1.50E-6-WD0.1

for ITERATION in "${ITERATIONS[@]}"; do
  FORMATTED_ITERATION="$(printf "%07d" "${ITERATION}")"
  echo -e "Converting iteration ${ITERATION}\n"

  HF_CHECKPOINT_DIR=/groups/gag51395/hf_checkpoints/${MODEL_NAME}
  MEGATRON_CHECKPOINT_DIR=${MEGATRON_DIR}/iter_${FORMATTED_ITERATION}
  HF_CHECKPOINT_SAVE_DIR=${HF_DIR}/iteration_${FORMATTED_ITERATION}
  mkdir -p "${HF_CHECKPOINT_SAVE_DIR}"

  if [[ ! -d "${MEGATRON_CHECKPOINT_DIR}" ]]; then
    echo "WARN: ${MEGATRON_CHECKPOINT_DIR} not found. Skipping..."
    continue
  fi
  singularity exec \
    --nv \
    --bind /groups/gag51395:/groups/gag51395 \
    --bind /groups/gch51639:/groups/gch51639 \
    --bind /home/acf15649kv:/home/acf15649kv \
    --bind /dev/shm:/dev/shm \
    --bind /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem:/etc/ssl/certs/ca-certificates.crt \
    --bind /tmp:/tmp \
    /groups/gch51639/fujii/container/ngc-pytorch-25.10.sif \
    python examples/conversion/convert_checkpoints.py export \
      --hf-model "${HF_CHECKPOINT_DIR}" \
      --megatron-path "${MEGATRON_CHECKPOINT_DIR}" \
      --hf-path "${HF_CHECKPOINT_SAVE_DIR}" \
      --model-type gpt
done

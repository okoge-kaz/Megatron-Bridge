#!/bin/sh
#PBS -q rt_HF
#PBS -N megatron-to-hf
#PBS -l select=1:ncpus=192:ngpus=8
#PBS -l walltime=1:30:00
#PBS -j oe
#PBS -m n
#PBS -v USE_SSH=1
#PBS -koed
#PBS -o outputs/convert/megatron-to-hf/

cd $PBS_O_WORKDIR
mkdir -p outputs/convert/megatron-to-hf/

echo "Nodes allocated to this job:"
cat $PBS_NODEFILE

ITERATIONS=(3200 6400 9000)

for ITERATION in "${ITERATIONS[@]}"; do
  FORMATTED_ITERATION="$(printf "%07d" "${ITERATION}")"
  echo -e "Converting iteration ${ITERATION}\n"

  # model config
  HF_CHECKPOINT_DIR=/groups/gag51395/hf_checkpoints/Qwen3-30B-A3B
  MEGATRON_CHECKPOINT_DIR=/groups/gch51639/fujii/checkpoints/Qwen-3-Swallow-30B-A3B-v0.2/tp1-pp1-ct2-ep8/LR2.50E-5-MINLR2.50E-6-WD0.1/iter_${FORMATTED_ITERATION}
  HF_CHECKPOINT_SAVE_DIR=/groups/gch51639/fujii/checkpoints/megatron-to-hf/Qwen-3-Swallow-30B-A3B-v0.2/tp1-pp1-ct2-ep8/LR2.50E-5-MINLR2.50E-6-WD0.1/iteration_${FORMATTED_ITERATION}
  mkdir -p "${HF_CHECKPOINT_SAVE_DIR}"

  # skip if megatron checkpoint not found
  if [[ ! -d "${MEGATRON_CHECKPOINT_DIR}" ]]; then
    echo "WARN: ${MEGATRON_CHECKPOINT_DIR} not found. Skipping..."
    continue
  fi

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
  python examples/models/checkpoint_conversion.py export \
    --hf-model "${HF_CHECKPOINT_DIR}" \
    --megatron-path "${MEGATRON_CHECKPOINT_DIR}" \
    --hf-path "${HF_CHECKPOINT_SAVE_DIR}" \
    --model-type gpt
done

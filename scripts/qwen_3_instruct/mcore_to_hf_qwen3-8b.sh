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

module load cuda/12.9.1
module load cudnn/9.10.2
module load nccl/2.27.5-cuda12.9
module load hpcx/2.23.0

source /home/acf15649kv/src/Megatron-LM-v0.13.0rc2/.venv/bin/activate

ITERATIONS=(23500)

for ITERATION in "${ITERATIONS[@]}"; do
  FORMATTED_ITERATION="$(printf "%07d" "${ITERATION}")"
  echo -e "Converting iteration ${ITERATION}\n"

  # model config
  HF_CHECKPOINT_DIR=/groups/gag51395/hf_checkpoints/Qwen3-8B
  MEGATRON_CHECKPOINT_DIR=/groups/gch51639/fujii/checkpoints/Qwen-3-Swallow-8B-v0.1-SFT/swallow-reasoning/exp13/LR1.50E-5-MINLR1.50E-6-WD0.1/iter_${FORMATTED_ITERATION}
  HF_CHECKPOINT_SAVE_DIR=/groups/gch51639/fujii/checkpoints/megatron-to-hf/Qwen3-Swallow-8B-v0.1-SFT/swallow-reasoning/exp13/iteration_${FORMATTED_ITERATION}
  mkdir -p "${HF_CHECKPOINT_SAVE_DIR}"

  # skip if megatron checkpoint not found
  if [[ ! -d "${MEGATRON_CHECKPOINT_DIR}" ]]; then
    echo "WARN: ${MEGATRON_CHECKPOINT_DIR} not found. Skipping..."
    continue
  fi

  export CUDA_DEVICE_MAX_CONNECTIONS=1
  MEGATRON_LM_PATH=/home/acf15649kv/src/Megatron-LM-v0.13.0rc2
  MEGATRON_BRIDGE_PATH=$(pwd)/src
  export PYTHONPATH="$PYTHONPATH:$MEGATRON_LM_PATH:$MEGATRON_BRIDGE_PATH"

  # convert
  python examples/models/checkpoint_conversion.py export \
    --hf-model "${HF_CHECKPOINT_DIR}" \
    --megatron-path "${MEGATRON_CHECKPOINT_DIR}" \
    --hf-path "${HF_CHECKPOINT_SAVE_DIR}" \
    --model-type gpt
done

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

START_ITERATION=500
END_ITERATION=500
INCREMENT=2500

for ITERATION in $(seq $START_ITERATION $INCREMENT $END_ITERATION); do
  FORMATTED_ITERATION=$(printf "%07d" $ITERATION)
  echo -e "Converting iteration ${ITERATION}\n"

  # model config
  HF_CHECKPOINT_DIR=/groups/gag51395/hf_checkpoints/Qwen3-8B
  MEGATRON_CHECKPOINT_DIR=/groups/gag51395/fujii/checkpoints/Qwen-3-8B/exp1/tp1-pp1-ct1/LR1.00E-5-MINLR1.00E-6-WD0.1/iter_${FORMATTED_ITERATION}
  HF_CHECKPOINT_SAVE_DIR=/groups/gag51395/fujii/checkpoints/megatron-to-hf/Qwen3-8B/iteration_${FORMATTED_ITERATION}
  mkdir -p ${HF_CHECKPOINT_SAVE_DIR}

  export CUDA_DEVICE_MAX_CONNECTIONS=1
  MEGATRON_LM_PATH=/home/acf15649kv/src/Megatron-LM-v0.13.0rc2
  MEGATRON_BRIDGE_PATH=/home/acf15649kv/src/Megatron-Bridge/src
  export PYTHONPATH=$PYTHONPATH:$MEGATRON_LM_PATH
  export PYTHONPATH=$PYTHONPATH:$MEGATRON_BRIDGE_PATH

  # convert
  python examples/models/checkpoint_conversion.py export \
    --hf-model ${HF_CHECKPOINT_DIR} \
    --megatron-path ${MEGATRON_CHECKPOINT_DIR} \
    --hf-path ${HF_CHECKPOINT_SAVE_DIR} \
    --model-type gpt
done

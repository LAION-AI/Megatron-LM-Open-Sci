#!/bin/bash
#SBATCH --job-name=0080_convert
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=8
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.out

set -eu -o pipefail

module load cuda/12.1
module load /data/cudnn-tmp-install/modulefiles/8.9.4
module load hpcx/2.17.1-gcc-cuda12/hpcx
module load nccl/2.20.5
source scripts/ckpt/sakura/mpi_variables.sh
source venv/bin/activate

export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS_PER_NODE=$(echo $SLURM_TASKS_PER_NODE | cut -d '(' -f 1)
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))

echo NUM_NODES=$NUM_NODES
echo NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE
echo NUM_GPUS=$NUM_GPUS

# open file limit
ulimit -n 65536 1048576

export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_LOGDEST_DBG=stderr
export CUDNN_LOGERR_DBG=1

MEGATRON_PATH="/home/taishi/workspace/Open-Sci/Megatron-LM"
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH:-}

TARGET_TP_SIZE=1
TARGET_PP_SIZE=1
WORLD_SIZE=$((TARGET_TP_SIZE * TARGET_PP_SIZE))

# model config
LOAD_CHECKPOINT_PATH=/home/taishi/Open-Sci_sample_checkpoint/iter_0012406
SAVE_CHECKPOINT_PATH=/home/taishi/Open-Sci_sample_checkpoint_hf/iter_0012406
SOURCE_MODEL_PATH=Open-Sci-hf/sample

mkdir -p ${SAVE_CHECKPOINT_PATH}

python scripts/ckpt/mcore_to_hf_opensci.py \
    --load_path "${LOAD_CHECKPOINT_PATH}" \
    --save_path "${SAVE_CHECKPOINT_PATH}" \
    --source_model "${SOURCE_MODEL_PATH}" \
    --target_tensor_model_parallel_size ${TARGET_TP_SIZE} \
    --target_pipeline_model_parallel_size ${TARGET_PP_SIZE} \
    --target_params_dtype "bf16" \
    --world_size ${WORLD_SIZE} \
    --convert_checkpoint_from_megatron_to_transformers \
    --print-checkpoint-structure

cp -r Open-Sci-hf/sample/modeling_opensci.py "${SAVE_CHECKPOINT_PATH}"

# Tokenizer
cp -r Open-Sci-hf/sample/tokenizer* "${SAVE_CHECKPOINT_PATH}"
cp -r Open-Sci-hf/sample/special_tokens_map.json "${SAVE_CHECKPOINT_PATH}"
cp -r Open-Sci-hf/sample/vocab.json "${SAVE_CHECKPOINT_PATH}"

python scripts/ckpt/inference.py \
    --model_path "${SAVE_CHECKPOINT_PATH}" \
    --num_generations 5 \
    --max_length 512 \
    --temperature 0.7 \
    --top_p 0.9

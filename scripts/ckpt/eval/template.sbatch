#!/bin/bash -x
#SBATCH --account={account}
#SBATCH --partition={partition} 
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output={eval_logs_dir}/%x_%j.out
#SBATCH --job-name=eval


CONDA_ENV="$WORK/marianna/envs/vllm"
MINICONDA_PATH="$WORK/marianna/miniconda/miniforge"
source ${MINICONDA_PATH}/bin/activate ${CONDA_ENV}

export HF_HOME="{hf_cache_dir}"
CACHE_DIR="{cache_dir}"

export HF_HUB_OFFLINE=1

OUTPUT_DIR='{eval_save_dir}'
mkdir -p ${OUTPUT_DIR}  

TASKS={tasks}


MODEL_PATH={model_path}

srun accelerate launch -m lm_eval --model hf \
    --model_args pretrained=$MODEL_PATH,trust_remote_code=True\
    --tasks $TASKS \
    --output_path $OUTPUT_DIR \
    --use_cache $CACHE_DIR \
    --batch_size {batch_size}
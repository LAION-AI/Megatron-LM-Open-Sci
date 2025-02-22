#!/bin/bash
#SBATCH --job-name=convert
#SBATCH --account=cbrdg24
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --output=slurm-output/%x-%j.out


source env


MEGATRON_OPEN_SCI_PATH="Megatron-LM-Open-Sci" # Path to Megatron-LM-Open-Sci
OPEN_SCI_HF_PATH="Open-Sci-hf" # Path to Open-Sci-hf
MEGATRON_PATH="Megatron-LM" # Path to Megatron-LM
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH:-}


cd $MEGATRON_OPEN_SCI_PATH

export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=$((10000 + ($SLURM_JOBID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS_PER_NODE=4
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))

echo NUM_NODES=$NUM_NODES
echo NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE
echo NUM_GPUS=$NUM_GPUS



TARGET_TP_SIZE=1
TARGET_PP_SIZE=1
WORLD_SIZE=$((TARGET_TP_SIZE * TARGET_PP_SIZE))


# model config
MODEL_NAME="open-sci-ref_model-1.7b_data-FineWeb-Edu-1.4T_samples-300B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO"
ITER="0072661"
LOAD_CHECKPOINT_PATH=/converted_torch/${MODEL_NAME}/iter_$ITER # This is the path where the original checkpoint is stored
SAVE_CHECKPOINT_PATH=/checkpoints/converted-hf/${MODEL_NAME} # This is the path where the converted checkpoint will be saved
mkdir -p ${SAVE_CHECKPOINT_PATH}

intermediate_size=8192
max_position_embeddings=4096
NUM_KEY_VALUE_HEADS=32 # need to pass it separately as it is stored as 1 in the checkpoint


cat <<EOF > ${SAVE_CHECKPOINT_PATH}/config.json
{
    "_name_or_path": "",
    "architectures": [
      "OpensciForCausalLM"
    ],
    "attention_bias": true,
    "attention_dropout": 0.0,
    "auto_map": {
        "AutoConfig": "configuration_opensci.OpensciConfig",
        "AutoModel": "modeling_opensci.OpensciPreTrainedModel",
        "AutoModelForCausalLM": "modeling_opensci.OpensciForCausalLM"
      },
    "bos_token_id": 0,
    "eos_token_id": 0,
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": $intermediate_size,
    "max_position_embeddings": $max_position_embeddings,
    "mlp_bias": true,
    "model_type": "opensci",
    "num_attention_heads": 32,
    "num_hidden_layers": 24,
    "num_key_value_heads": $NUM_KEY_VALUE_HEADS,
    "pretraining_tp": 1,
    "qk_layernorm": true,
    "rms_norm_eps": 1e-05,
    "rope_scaling": null,
    "rope_theta": 10000,
    "tie_word_embeddings": true,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.48.3",
    "use_cache": true,
    "vocab_size": 50304
  }
EOF

cp -r $OPEN_SCI_HF_PATH/sample/modeling_opensci.py "${SAVE_CHECKPOINT_PATH}"
cp -r $OPEN_SCI_HF_PATH/sample/configuration_opensci.py "${SAVE_CHECKPOINT_PATH}"

# Tokenizer
cp -r $OPEN_SCI_HF_PATH/sample/tokenizer* "${SAVE_CHECKPOINT_PATH}"
cp -r $OPEN_SCI_HF_PATH/sample/special_tokens_map.json "${SAVE_CHECKPOINT_PATH}"
cp -r $OPEN_SCI_HF_PATH/sample/vocab.json "${SAVE_CHECKPOINT_PATH}"


mkdir -p ${SAVE_CHECKPOINT_PATH}

python -u scripts/ckpt/mcore_to_hf_opensci.py \
    --load_path "${LOAD_CHECKPOINT_PATH}" \
    --save_path "${SAVE_CHECKPOINT_PATH}" \
    --source_model "${SAVE_CHECKPOINT_PATH}" \
    --target_tensor_model_parallel_size ${TARGET_TP_SIZE} \
    --target_pipeline_model_parallel_size ${TARGET_PP_SIZE} \
    --target_params_dtype "bf16" \
    --world_size ${WORLD_SIZE} \
    --convert_checkpoint_from_megatron_to_transformers \
    --num_key_value_heads ${NUM_KEY_VALUE_HEADS} \
    --print-checkpoint-structure



# Test the converted model
python -u scripts/ckpt/inference.py \
    --model_path "${SAVE_CHECKPOINT_PATH}" \
    --num_generations 5 \
    --max_length 512 \
    --temperature 0 

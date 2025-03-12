ACCOUNT=EUHPC_E03_068
PARTITION=boost_usr_prod
CONTAINER_IMAGE=/leonardo_work/EUHPC_E03_068/shared/container_images/pytorch_24.09-py3_leonardo.sif
TRAIN_LOGS_DIr_OR_PATH="/leonardo_work/EUHPC_E03_068/jjitsev0/megatron_lm_reference/slurm_output/open-sci-ref_model-1.7b_data-Nemotron-cc-2024-HQ-real-synth-mix_tokenizer-GPT-NeoX_samples-300B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_13715533.out::/leonardo_work/EUHPC_E03_068/jjitsev0/megatron_lm_reference/slurm_output/open-sci-ref_model-1.3b_data-Nemotron-cc-2024-HQ-real-synth-mix_tokenizer-GPT-NeoX_samples-300B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_13661750.out::/leonardo_work/EUHPC_E03_068/jjitsev0/megatron_lm_reference/slurm_output/open-sci-ref_model-1.7b_data-HPLT-2.0_tokenizer-GPT-NeoX_samples-300B_global_bs-1008_context-4096_schedule-WSD_lr-4e-3_warmup-25000_machine-LEONARDO_13686312.out"
CONVERT_LOGS_DIR=/leonardo/home/userexternal/mnezhuri/megatron/convert_logs
mkdir -p $CONVERT_LOGS_DIR

OPENSCI_MEGATRON_PATH="/leonardo/home/userexternal/mnezhuri/megatron/Megatron-LM-Open-Sci"
MEGATRON_PATH="/leonardo_work/EUHPC_E03_068/shared/repos/Megatron-LM"
OPEN_SCI_HF_PATH=/leonardo/home/userexternal/mnezhuri/megatron/Open-Sci-hf
SAVE_CHECKPOINTS_DIR=/leonardo_work/EUHPC_E03_068/marianna/megatron_lm_reference/checkpoints

SCRIPT=${OPENSCI_MEGATRON_PATH}/scripts/ckpt/convert_full/converter.py

CMD="python3 $SCRIPT \
    --container_image $CONTAINER_IMAGE \
    --train_logs_dir_or_path $TRAIN_LOGS_DIr_OR_PATH \
    --convert_logs_dir $CONVERT_LOGS_DIR \
    --save_checkpoints_dir $SAVE_CHECKPOINTS_DIR
    --account $ACCOUNT \
    --partition $PARTITION \
    --megatron_path $MEGATRON_PATH \
    --open_sci_hf_path $OPEN_SCI_HF_PATH
    --opensci_megatron_path $OPENSCI_MEGATRON_PATH"

echo $CMD
$CMD
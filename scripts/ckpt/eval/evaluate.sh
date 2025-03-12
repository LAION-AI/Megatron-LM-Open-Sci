ACCOUNT=EUHPC_E03_068
PARTITION=boost_usr_prod
EVAL_LOGS_DIR=/leonardo/home/userexternal/mnezhuri/megatron/eval_logs
mkdir -p $EVAL_LOGS_DIR
EVAl_SAVE_DIR=/leonardo/home/userexternal/mnezhuri/megatron/eval_results
mkdir -p $EVAl_SAVE_DIR
TASKS="commonsense_qa,piqa,social_iqa,winogrande,arc_challenge,arc_easy,mmlu"
BATCH_SIZE=64
CHECKPOINTS_DIR_OR_PATH=/leonardo_work/EUHPC_E03_068/marianna/megatron_lm_reference/checkpoints/hf
OPENSCI_MEGATRON_PATH="/leonardo/home/userexternal/mnezhuri/megatron/Megatron-LM-Open-Sci"
export HF_HOME=$SCRATCH/HF_cache
HF_CACHR_DIR=$HF_HOME
EVAL_CACHE_DIR=$SCRATCH/eval_cache
mkdir -p $HF_CACHR_DIR

SCRIPT=${OPENSCI_MEGATRON_PATH}/scripts/ckpt/eval/evaluator.py

CMD="python3 $SCRIPT \
    --account $ACCOUNT \
    --partition $PARTITION \
    --eval_logs_dir $EVAL_LOGS_DIR \
    --eval_save_dir $EVAl_SAVE_DIR \
    --tasks $TASKS \
    --batch_size $BATCH_SIZE \
    --checkpoints_dir_or_path $CHECKPOINTS_DIR_OR_PATH \
    --opensci_megatron_path $OPENSCI_MEGATRON_PATH \
    --eval_cache_dir $EVAL_CACHE_DIR \
    --hf_cache_dir $HF_CACHR_DIR"

echo $CMD
$CMD

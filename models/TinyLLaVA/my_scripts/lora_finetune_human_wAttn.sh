#!/bin/bash

# This script takes two parameters:
# $1: Number of epochs
# $2: Guided attention (True/False)

# Ensure that the result is deterministic:
PYTHONHASHSEED=2002

# Ensure parameters are provided
DEFAULT_EPOCHS=10
DEFAULT_GUIDED_ATTN="True"

# Use default values if parameters are not provided
EPOCHS=${1:-$DEFAULT_EPOCHS}
GUIDED_ATTN=${2:-$DEFAULT_GUIDED_ATTN}

echo "Using epochs: $EPOCHS, guided_attn_map: $GUIDED_ATTN"

DATA_PATH="./data/train/data.json"
IMAGE_PATH="./data/train"
MODEL_MAX_LENGTH=2
CONV_MODE="llama" # depending on the model used, it can be llama or phi ...

LORA_SIGLIP="lora-finetune-TinyLLaVA-OpenELM-450M-SigLIP-0.89B"
EXP_NAME="S1_K0_${EPOCHS}epoch_guided_${GUIDED_ATTN}_seed2002_non_binary"
OUTPUT_DIR="OUTPUTS/${LORA_SIGLIP}_${EXP_NAME}"

echo "=================================================="
echo "Starting experiment: $EXP_NAME"
echo "Epochs: $EPOCHS, Guided Attention Map: $GUIDED_ATTN"
echo "Output directory: $OUTPUT_DIR"
echo "=================================================="


export PYTHONPATH=$PYTHONPATH:./

deepspeed --include localhost:0 --master_port 29502 tinyllava/train/custom_finetune.py \
    --deepspeed ./my_scripts/zero2.json \
    --data_path  $DATA_PATH \
    --image_folder $IMAGE_PATH \
    --is_multimodal True \
    --conv_version llama \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --fp16 False \
    --training_recipe lora \
    --tune_type_llm lora \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --lora_r 32 \
    --lora_alpha 64 \
    --group_by_modality_length False \
    --pretrained_model_path "my_modeling/TinyLLaVA-OpenELM-450M-SigLIP-0.89B_wAttn" \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --seed $PYTHONHASHSEED \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.3 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --run_name $EXP_NAME \
    --guided_attn_map $GUIDED_ATTN \
    --logging_dir "${OUTPUT_DIR}/log_files"
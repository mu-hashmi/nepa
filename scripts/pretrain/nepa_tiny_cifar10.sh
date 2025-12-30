#!/bin/bash
# ========================
# NEPA Tiny Model - CIFAR-10 Pretraining
# Optimized for single GPU training
# ========================

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")

EXPERIMENT_NAME="nepa-tiny-patch14-224-cifar10"
WANDB_PROJECT="Nepa-Pretrain-CIFAR10"

CONFIG_NAME="configs/pretrain/nepa-tiny-patch14-224-cifar10"
OUTPUT_DIR="outputs/${EXPERIMENT_NAME}"

# Reduced batch sizes for single GPU
TOTAL_BATCH_SIZE=512
PER_DEVICE_BATCH_SIZE=128
GRAD_ACCUM_STEPS=$((TOTAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NGPU)))

# Reduced epochs for faster iteration
NUM_EPOCHS=100

BASE_LEARNING_RATE=1e-4
LEARNING_RATE=$(python -c "print(${BASE_LEARNING_RATE} * ${TOTAL_BATCH_SIZE} / 256)")

DATALOADER_NUM_WORKERS=4

# ========================
export WANDB_PROJECT=$WANDB_PROJECT

# ========================
torchrun \
    --nproc_per_node $NGPU run_nepa.py \
    \
    --ddp_backend nccl \
    --ddp_find_unused_parameters False \
    \
    --config_name $CONFIG_NAME \
    --image_processor_name $CONFIG_NAME \
    --dataset_name "cifar10" \
    --load_from_disk False \
    --dataloader_drop_last True \
    \
    --do_train \
    --output_dir $OUTPUT_DIR \
    --remove_unused_columns False \
    \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --optim adamw_torch \
    \
    --logging_strategy steps \
    --logging_steps 50 \
    --save_strategy steps \
    --save_steps 5000 \
    \
    --seed 1337 \
    --bf16 True \
    \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --dataloader_persistent_workers True \
    --dataloader_pin_memory False \
    \
    --report_to wandb \
    --run_name $EXPERIMENT_NAME

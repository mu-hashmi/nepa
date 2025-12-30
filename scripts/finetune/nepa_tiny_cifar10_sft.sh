#!/bin/bash
# ========================
# NEPA Tiny Model - CIFAR-10 Fine-tuning
# Optimized for single GPU training
# ========================

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")

EXPERIMENT_NAME="nepa-tiny-patch14-224-cifar10-sft"
WANDB_PROJECT="Nepa-SFT-CIFAR10"

# Use pretrained model from local output or specify path
MODEL_NAME="outputs/nepa-tiny-patch14-224-cifar10"
CONFIG_NAME="configs/finetune/nepa-tiny-patch14-224-cifar10-sft"
OUTPUT_DIR="outputs/${EXPERIMENT_NAME}"

# Reduced batch sizes for single GPU
TOTAL_BATCH_SIZE=256
PER_DEVICE_BATCH_SIZE=64
GRAD_ACCUM_STEPS=$((TOTAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NGPU)))

# Reduced epochs for faster iteration
NUM_EPOCHS=50

BASE_LEARNING_RATE=5e-4
LEARNING_RATE=$(python -c "print(${BASE_LEARNING_RATE} * ${TOTAL_BATCH_SIZE} / 256)")

DATALOADER_NUM_WORKERS=4

# ========================
export WANDB_PROJECT=$WANDB_PROJECT

# ========================
torchrun \
    --nproc_per_node $NGPU run_image_classification.py \
    \
    --ddp_backend nccl \
    --ddp_find_unused_parameters False \
    \
    --model_name_or_path $MODEL_NAME \
    --config_name $CONFIG_NAME \
    --freeze_vit False \
    --freeze_embed True \
    \
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
    --lr_scheduler_kwargs '{"custom_scheduler_type": "llrd_cosine_warmup"}' \
    --warmup_ratio 0.10 \
    --weight_decay 0.05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --optim adamw_torch \
    --llrd 0.65 \
    --ema_decay 0.9999 \
    \
    --logging_strategy steps \
    --logging_steps 50 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_ema_accuracy \
    --save_total_limit 2 \
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

#!/bin/bash
# ========================
# NEPA Diffusion Decoder - CIFAR-10 Training
# Trains UNet diffusion decoder conditioned on NEPA embeddings
# ========================

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")

EXPERIMENT_NAME="nepa-tiny-diffusion-cifar10"
WANDB_PROJECT="Nepa-Diffusion-CIFAR10"

# Path to pretrained NEPA encoder (must be trained first)
NEPA_MODEL_PATH="outputs/nepa-tiny-patch14-224-cifar10"
CONFIG_NAME="configs/diffusion/nepa-tiny-diffusion-cifar10"
OUTPUT_DIR="outputs/${EXPERIMENT_NAME}"

# Batch sizes for single GPU
TOTAL_BATCH_SIZE=64
PER_DEVICE_BATCH_SIZE=32
GRAD_ACCUM_STEPS=$((TOTAL_BATCH_SIZE / (PER_DEVICE_BATCH_SIZE * NGPU)))

# Training settings
NUM_EPOCHS=200
LEARNING_RATE=1e-4

DATALOADER_NUM_WORKERS=4

# ========================
export WANDB_PROJECT=$WANDB_PROJECT

# ========================
torchrun \
    --nproc_per_node $NGPU run_diffusion_reconstruction.py \
    \
    --ddp_backend nccl \
    --ddp_find_unused_parameters False \
    \
    --nepa_model_path $NEPA_MODEL_PATH \
    --config_name $CONFIG_NAME \
    --use_ema True \
    --ema_decay 0.9999 \
    \
    --dataset_name "cifar10" \
    --image_column_name "img" \
    --load_from_disk False \
    --image_size 224 \
    --resize_size 256 \
    \
    --do_train \
    --do_eval \
    --output_dir $OUTPUT_DIR \
    --remove_unused_columns False \
    \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --optim adamw_torch \
    \
    --logging_strategy steps \
    --logging_steps 50 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 3 \
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

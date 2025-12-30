#!/bin/bash
# ========================
# NEPA Tiny Model - CIFAR-10 Evaluation
# ========================

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")

MODEL_NAME="outputs/nepa-tiny-patch14-224-cifar10-sft"

DATALOADER_NUM_WORKERS=4

# ========================
torchrun \
    --nproc_per_node $NGPU run_image_classification.py \
    \
    --ddp_backend nccl \
    --ddp_find_unused_parameters False \
    \
    --model_name_or_path $MODEL_NAME \
    --dataset_name "cifar10" \
    --image_column_name "img" \
    --load_from_disk False \
    --do_eval True \
    --per_device_eval_batch_size 128 \
    --remove_unused_columns False \
    \
    --bf16 True \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS \
    --dataloader_persistent_workers True \
    --dataloader_pin_memory False

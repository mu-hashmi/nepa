# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NEPA (Next-Embedding Prediction Makes Strong Vision Learners) is a PyTorch implementation of a vision model that uses autoregressive next-embedding prediction as a self-supervised pretraining method. The project supports pretraining from scratch, fine-tuning, and evaluation on ImageNet-1k.

Paper: https://arxiv.org/abs/2512.16922

## Branch: extended

This branch contains experimental extensions to the base NEPA implementation.

**Planned changes:**
- Custom Fast-RoPE implementation (replacing standard RoPE)
- Decoder/diffusion-based generator integration
- CIFAR-10 dataset for faster iteration (instead of ImageNet-1k)

## Common Commands

### Environment Setup
```bash
conda env create -f environment.yml && conda activate nepa
hf auth login  # Required for HuggingFace model/dataset access
```

### Dataset Preparation
```bash
python download_dataset.py  # Downloads ImageNet-1k (~150GB) to data/imagenet-1k-hf/
```

### Pretraining
```bash
bash scripts/pretrain/nepa_b.sh  # Base model (8 A100 GPUs required)
bash scripts/pretrain/nepa_l.sh  # Large model
```

### Fine-tuning
```bash
bash scripts/finetune/nepa_b_sft.sh  # Base model
bash scripts/finetune/nepa_l_sft.sh  # Large model
```

### Evaluation
```bash
bash scripts/eval/nepa_b_sft_eval.sh  # Base: expects 83.75% accuracy
bash scripts/eval/nepa_l_sft_eval.sh  # Large: expects 85.40% accuracy
```

### Model Conversion (pretrained to classification)
```bash
python init_nepa_cls_from_pretrain.py \
  --pretrained_model_id SixAILab/nepa-base-patch14-224 \
  --config_model_id configs/finetune/nepa-base-patch14-224-sft \
  --save_local --local_dir ./nepa-base-patch14-224-sft
```

## Architecture

### Key Entry Points
- `run_nepa.py` - Pretraining script using `ViTNepaForPreTraining`
- `run_image_classification.py` - Fine-tuning and evaluation using `ViTNepaForImageClassification`
- `run_visualization.py` - Attention/prediction visualization utilities

### Model Implementation
Located in `models/vit_nepa/`:
- `modeling_vit_nepa.py` - Core model classes: `ViTNepaModel`, `ViTNepaForPreTraining`, `ViTNepaForImageClassification`
- `configuration_vit_nepa.py` - Configuration class with architecture parameters

### Custom Components
- **schedulers.py** - `LayerLambdaLR` for layer-wise learning rate decay (LLRD)
- **Next Embedding Prediction Loss** - Cosine similarity loss between predicted and actual next hidden states (`prediction_loss()` in modeling)
- **EMA Support** - Exponential Moving Average model tracking in both training scripts

### Configuration
Model configs in `configs/` contain architecture parameters, classification head settings, and ImageNet-1k label mappings.

### Training Infrastructure
- Uses `torchrun` with NCCL backend for distributed training
- Hugging Face Trainer with custom `EnhancedTrainer` class (EMA, layer-specific LR)
- Weights & Biases integration for experiment tracking

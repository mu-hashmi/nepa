# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NEPA (Next-Embedding Prediction Makes Strong Vision Learners) is a PyTorch implementation of a vision model that uses autoregressive next-embedding prediction as a self-supervised pretraining method. The project supports pretraining from scratch, fine-tuning, and evaluation on ImageNet-1k.

Paper: https://arxiv.org/abs/2512.16922

## Branch: extended

This branch contains experimental extensions to the base NEPA implementation.

**Planned changes:**
- Custom Fast-RoPE implementation (replacing standard RoPE)

**Available:**
- Tiny model config + CIFAR-10 dataset for faster iteration
- Diffusion decoder for image reconstruction from NEPA embeddings

## CIFAR-10 Training (Lightweight)

Tiny model (~15M params) optimized for single GPU training on CIFAR-10.

### Configs
- Pretrain: `configs/pretrain/nepa-tiny-patch14-224-cifar10/`
- Finetune: `configs/finetune/nepa-tiny-patch14-224-cifar10-sft/`

### Scripts
```bash
bash scripts/pretrain/nepa_tiny_cifar10.sh      # Pretrain (100 epochs)
bash scripts/finetune/nepa_tiny_cifar10_sft.sh  # Fine-tune (50 epochs)
bash scripts/eval/nepa_tiny_cifar10_sft_eval.sh # Evaluate
```

### Google Colab
Use `notebooks/nepa_cifar10_colab.ipynb` for training on Colab with GPU.

## Diffusion Decoder

UNet diffusion decoder that reconstructs images from NEPA embeddings. Uses cross-attention conditioning with frozen NEPA encoder.

### Architecture
- **Encoder**: Frozen NEPA (384-D, 6 layers, 16x16 patches)
- **Decoder**: UNet with cross-attention to NEPA embeddings (~40M params)
- **Training**: DDPM with cosine schedule, MSE loss on predicted noise
- **Inference**: DDIM for fast sampling (50 steps)

### Config
- `configs/diffusion/nepa-tiny-diffusion-cifar10/`

### Scripts
```bash
# Train diffusion decoder (requires pretrained NEPA encoder)
bash scripts/diffusion/train_diffusion_tiny_cifar10.sh
```

### Google Colab
Use `notebooks/nepa_diffusion_colab.ipynb` for training on Colab.

### Model Code
Located in `models/diffusion_decoder/`:
- `configuration_diffusion_decoder.py` - Config class
- `modeling_diffusion_decoder.py` - UNet wrapper with NEPA conditioning

### Training Script
- `run_diffusion_reconstruction.py` - Main entry point

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

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Training script for NEPA-conditioned diffusion decoder.

This script trains a UNet diffusion model to reconstruct images from NEPA embeddings.
The NEPA encoder is frozen and used only for feature extraction.
"""

import logging
import os
import sys
import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

import transformers
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

from models.diffusion_decoder import (
    DiffusionDecoderConfig,
    NepaConditionedDiffusionForReconstruction,
)

logger = logging.getLogger(__name__)

require_version("datasets>=2.14.0")
require_version("diffusers>=0.20.0", "To fix: pip install diffusers>=0.20.0")


def pil_loader(path: str):
    with open(path, "rb") as f:
        im = Image.open(f)
        return im.convert("RGB")


class DiffusionTrainer(Trainer):
    """Trainer for diffusion models with EMA support."""

    def __init__(
        self,
        *args,
        ema_decay=0.9999,
        use_ema=True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ema_decay = ema_decay
        self.use_ema = use_ema
        self.ema_model = None

    def _init_ema_model(self):
        if self.ema_model is None:
            # Only EMA the decoder, not the frozen NEPA encoder
            self.ema_model = copy.deepcopy(self.model.decoder)
            self.ema_model.eval()
            self.ema_model = self.ema_model.float()
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

    def _update_ema(self):
        if not self.use_ema:
            return
        if self.ema_model is None:
            self._init_ema_model()
        with torch.no_grad():
            msd = self.model.decoder.state_dict()
            for k, v in self.ema_model.state_dict().items():
                if k in msd:
                    model_param = msd[k].float()
                    v.mul_(self.ema_decay).add_(model_param, alpha=1.0 - self.ema_decay)

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        if self.state.global_step > getattr(self, "_ema_global_step", 0):
            self._update_ema()
            self._ema_global_step = self.state.global_step
        super()._maybe_log_save_evaluate(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute diffusion loss."""
        pixel_values = inputs["pixel_values"]
        outputs = model(pixel_values)
        loss = outputs.loss

        if return_outputs:
            return loss, outputs
        return loss

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)
        output_dir = output_dir if output_dir is not None else self.args.output_dir

        if self.use_ema and self.ema_model is not None and self.args.should_save:
            ema_path = f"{output_dir}/decoder_ema.bin"
            os.makedirs(os.path.dirname(ema_path), exist_ok=True)
            torch.save(self.ema_model.state_dict(), ema_path)
            logger.info(f"EMA decoder saved to {ema_path}")


@dataclass
class DataTrainingArguments:
    """Arguments for data loading."""

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of a dataset from the hub."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset."},
    )
    train_dir: Optional[str] = field(
        default=None,
        metadata={"help": "A folder containing the training data."},
    )
    validation_dir: Optional[str] = field(
        default=None,
        metadata={"help": "A folder containing the validation data."},
    )
    train_val_split: Optional[float] = field(
        default=0.15,
        metadata={"help": "Percent to split off of train for validation."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate training examples for debugging."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate evaluation examples for debugging."},
    )
    image_column_name: str = field(
        default="image",
        metadata={"help": "Dataset column containing images. Use 'img' for CIFAR-10."},
    )
    load_from_disk: bool = field(
        default=False,
        metadata={"help": "Load dataset from disk instead of hub."},
    )
    image_size: int = field(
        default=224,
        metadata={"help": "Size to resize images to."},
    )
    resize_size: int = field(
        default=256,
        metadata={"help": "Size to resize images to before cropping."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_dir is None:
            raise ValueError("Specify dataset_name or train_dir.")


@dataclass
class ModelArguments:
    """Arguments for model configuration."""

    nepa_model_path: str = field(
        metadata={"help": "Path to pretrained NEPA model."},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path to diffusion decoder config."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store downloaded models."},
    )
    use_ema: bool = field(
        default=True,
        metadata={"help": "Use EMA for decoder weights."},
    )
    ema_decay: float = field(
        default=0.9999,
        metadata={"help": "EMA decay rate."},
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    # Set seed
    set_seed(training_args.seed)

    # Load dataset
    if data_args.load_from_disk:
        dataset = load_from_disk(data_args.dataset_name)
    elif data_args.train_dir is not None:
        dataset = load_dataset(
            "imagefolder",
            data_dir=data_args.train_dir,
            cache_dir=model_args.cache_dir,
        )
    else:
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )

    # Handle CIFAR-10 column naming (uses 'img' instead of 'image')
    if data_args.dataset_name == "cifar10":
        data_args.image_column_name = "img"

    # Create train/validation split if needed
    if "validation" not in dataset.keys():
        split = dataset["train"].train_test_split(test_size=data_args.train_val_split, seed=training_args.seed)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]

    # Load config
    if model_args.config_name:
        config = DiffusionDecoderConfig.from_pretrained(model_args.config_name)
    else:
        # Default config for tiny model
        config = DiffusionDecoderConfig(
            nepa_model_path=model_args.nepa_model_path,
            nepa_hidden_size=384,
            image_size=data_args.image_size,
            use_ema=model_args.use_ema,
            ema_decay=model_args.ema_decay,
        )

    # Create model
    model = NepaConditionedDiffusionForReconstruction(config)
    model.load_nepa_encoder(model_args.nepa_model_path)

    logger.info(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    logger.info(f"NEPA encoder has {sum(p.numel() for p in model.nepa_encoder.parameters()):,} parameters (frozen)")

    # Data transforms - normalize to [-1, 1] for diffusion
    train_transforms = Compose([
        RandomResizedCrop(
            data_args.image_size,
            scale=(0.8, 1.0),
            interpolation=Image.BICUBIC,
        ),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_transforms = Compose([
        Resize(data_args.resize_size, interpolation=Image.BICUBIC),
        CenterCrop(data_args.image_size),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    def preprocess_train(example_batch):
        example_batch["pixel_values"] = [
            train_transforms(img.convert("RGB"))
            for img in example_batch[data_args.image_column_name]
        ]
        return example_batch

    def preprocess_val(example_batch):
        example_batch["pixel_values"] = [
            val_transforms(img.convert("RGB"))
            for img in example_batch[data_args.image_column_name]
        ]
        return example_batch

    # Apply transforms
    if training_args.do_train:
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset.set_transform(preprocess_train)

    if training_args.do_eval:
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset.set_transform(preprocess_val)

    # Data collator
    def collate_fn(examples):
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
        return {"pixel_values": pixel_values}

    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
        use_ema=model_args.use_ema,
        ema_decay=model_args.ema_decay,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()

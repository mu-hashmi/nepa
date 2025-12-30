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
NEPA-Conditioned Diffusion Decoder for Image Reconstruction

This module implements a UNet-based diffusion model conditioned on NEPA embeddings
for reconstructing images from NEPA's patch representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from transformers import PreTrainedModel
from transformers.utils import ModelOutput, logging
from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler

from .configuration_diffusion_decoder import DiffusionDecoderConfig


logger = logging.get_logger(__name__)


@dataclass
class DiffusionDecoderOutput(ModelOutput):
    """Output class for diffusion decoder training."""
    loss: Optional[torch.FloatTensor] = None
    pred_noise: Optional[torch.FloatTensor] = None
    noisy_images: Optional[torch.FloatTensor] = None
    timesteps: Optional[torch.LongTensor] = None


@dataclass
class DiffusionGenerationOutput(ModelOutput):
    """Output class for diffusion decoder generation."""
    images: Optional[torch.FloatTensor] = None
    nepa_embeddings: Optional[torch.FloatTensor] = None


class NepaConditionedDiffusion(nn.Module):
    """
    UNet diffusion model conditioned on NEPA embeddings via cross-attention.

    This is the core decoder component without the NEPA encoder.
    Use NepaConditionedDiffusionForReconstruction for the full model.
    """

    def __init__(self, config: DiffusionDecoderConfig):
        super().__init__()
        self.config = config

        # Create UNet from diffusers with cross-attention for NEPA conditioning
        self.unet = UNet2DConditionModel(
            sample_size=config.image_size,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            cross_attention_dim=config.cross_attention_dim,
            attention_head_dim=config.attention_head_dim,
        )

    def forward(
        self,
        noisy_images: torch.FloatTensor,
        timesteps: torch.LongTensor,
        encoder_hidden_states: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Forward pass of the UNet denoiser.

        Args:
            noisy_images: Noisy input images (B, C, H, W)
            timesteps: Diffusion timesteps (B,)
            encoder_hidden_states: NEPA patch embeddings (B, num_patches, hidden_size)

        Returns:
            Predicted noise (B, C, H, W)
        """
        return self.unet(
            noisy_images,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
        ).sample


class NepaConditionedDiffusionForReconstruction(PreTrainedModel):
    """
    Full model for image reconstruction from NEPA embeddings.

    Combines:
    - Frozen NEPA encoder for extracting patch embeddings
    - Trainable UNet diffusion decoder for reconstruction
    - DDPM/DDIM schedulers for training and inference
    """

    config_class = DiffusionDecoderConfig

    def __init__(self, config: DiffusionDecoderConfig):
        super().__init__(config)
        self.config = config

        # Load NEPA encoder (will be frozen)
        self.nepa_encoder = None  # Loaded separately via load_nepa_encoder()

        # Create diffusion decoder
        self.decoder = NepaConditionedDiffusion(config)

        # Create noise schedulers
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type,
        )

        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule=config.beta_schedule,
            prediction_type=config.prediction_type,
        )

    def load_nepa_encoder(self, nepa_model_path: str):
        """
        Load and freeze the NEPA encoder.

        Args:
            nepa_model_path: Path to pretrained NEPA model
        """
        # Import here to avoid circular imports
        import sys
        sys.path.insert(0, '.')
        from models.vit_nepa import ViTNepaModel

        logger.info(f"Loading NEPA encoder from {nepa_model_path}")
        self.nepa_encoder = ViTNepaModel.from_pretrained(nepa_model_path)

        # Freeze NEPA encoder
        if self.config.freeze_nepa:
            for param in self.nepa_encoder.parameters():
                param.requires_grad = False
            self.nepa_encoder.eval()
            logger.info("NEPA encoder frozen")

    def get_nepa_embeddings(
        self,
        pixel_values: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Extract NEPA embeddings from images.

        Args:
            pixel_values: Input images (B, C, H, W)

        Returns:
            NEPA patch embeddings (B, num_patches, hidden_size) - excludes CLS token
        """
        if self.nepa_encoder is None:
            raise ValueError("NEPA encoder not loaded. Call load_nepa_encoder() first.")

        with torch.no_grad():
            nepa_outputs = self.nepa_encoder(pixel_values)
            # Get patch embeddings, excluding CLS token (position 0)
            encoder_hidden_states = nepa_outputs.last_hidden_state[:, 1:, :]

        return encoder_hidden_states

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        timesteps: Optional[torch.LongTensor] = None,
        noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ) -> Union[DiffusionDecoderOutput, Tuple]:
        """
        Training forward pass.

        Args:
            pixel_values: Clean input images (B, C, H, W), normalized to [-1, 1]
            timesteps: Optional diffusion timesteps (B,). Random if not provided.
            noise: Optional noise tensor (B, C, H, W). Random if not provided.
            return_dict: Whether to return a dataclass or tuple.

        Returns:
            DiffusionDecoderOutput with loss, predicted noise, etc.
        """
        batch_size = pixel_values.shape[0]
        device = pixel_values.device

        # Get NEPA embeddings (frozen encoder)
        encoder_hidden_states = self.get_nepa_embeddings(pixel_values)

        # Sample random timesteps if not provided
        if timesteps is None:
            timesteps = torch.randint(
                0, self.config.num_train_timesteps,
                (batch_size,), device=device
            ).long()

        # Sample noise if not provided
        if noise is None:
            noise = torch.randn_like(pixel_values)

        # Add noise to images
        noisy_images = self.noise_scheduler.add_noise(pixel_values, noise, timesteps)

        # Predict noise
        pred_noise = self.decoder(noisy_images, timesteps, encoder_hidden_states)

        # Compute loss
        loss = F.mse_loss(pred_noise, noise)

        if not return_dict:
            return (loss, pred_noise, noisy_images, timesteps)

        return DiffusionDecoderOutput(
            loss=loss,
            pred_noise=pred_noise,
            noisy_images=noisy_images,
            timesteps=timesteps,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[DiffusionGenerationOutput, Tuple]:
        """
        Generate/reconstruct images from NEPA embeddings.

        Args:
            pixel_values: Input images to reconstruct (B, C, H, W). Used to get NEPA embeddings.
            encoder_hidden_states: Pre-computed NEPA embeddings (B, num_patches, hidden_size).
                                   If provided, pixel_values is ignored for encoding.
            num_inference_steps: Number of denoising steps (default: config.num_inference_timesteps)
            generator: Random generator for reproducibility.
            return_dict: Whether to return a dataclass or tuple.

        Returns:
            DiffusionGenerationOutput with generated images and NEPA embeddings.
        """
        if num_inference_steps is None:
            num_inference_steps = self.config.num_inference_timesteps

        # Get NEPA embeddings
        if encoder_hidden_states is None:
            if pixel_values is None:
                raise ValueError("Either pixel_values or encoder_hidden_states must be provided")
            encoder_hidden_states = self.get_nepa_embeddings(pixel_values)
            batch_size = pixel_values.shape[0]
            device = pixel_values.device
        else:
            batch_size = encoder_hidden_states.shape[0]
            device = encoder_hidden_states.device

        # Initialize with random noise
        image_shape = (batch_size, self.config.in_channels, self.config.image_size, self.config.image_size)
        images = torch.randn(image_shape, device=device, generator=generator)

        # Set up inference scheduler
        self.inference_scheduler.set_timesteps(num_inference_steps, device=device)

        # Denoising loop
        for t in self.inference_scheduler.timesteps:
            # Expand timesteps to batch
            timesteps = t.expand(batch_size)

            # Predict noise
            pred_noise = self.decoder(images, timesteps, encoder_hidden_states)

            # Denoise step
            images = self.inference_scheduler.step(pred_noise, t, images).prev_sample

        # Clamp to valid range
        images = images.clamp(-1, 1)

        if not return_dict:
            return (images, encoder_hidden_states)

        return DiffusionGenerationOutput(
            images=images,
            nepa_embeddings=encoder_hidden_states,
        )


__all__ = [
    "DiffusionDecoderOutput",
    "DiffusionGenerationOutput",
    "NepaConditionedDiffusion",
    "NepaConditionedDiffusionForReconstruction",
]

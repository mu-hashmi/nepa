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
"""Diffusion Decoder configuration for NEPA-conditioned image reconstruction"""

from typing import Optional, Tuple
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class DiffusionDecoderConfig(PretrainedConfig):
    r"""
    Configuration class for the diffusion decoder that reconstructs images from NEPA embeddings.

    This decoder uses a UNet architecture conditioned on NEPA's patch embeddings via cross-attention.
    It follows the denoising diffusion probabilistic model (DDPM) framework.

    Args:
        nepa_model_path (`str`, *optional*):
            Path to the pretrained NEPA model to use as encoder.
        nepa_hidden_size (`int`, *optional*, defaults to 384):
            Hidden size of the NEPA encoder (must match pretrained model).
        nepa_num_patches (`int`, *optional*, defaults to 256):
            Number of patches from NEPA (16x16 grid for 224x224 images with patch_size=14).
        image_size (`int`, *optional*, defaults to 224):
            Size of the input/output images.
        in_channels (`int`, *optional*, defaults to 3):
            Number of input channels (RGB).
        out_channels (`int`, *optional*, defaults to 3):
            Number of output channels (RGB).
        block_out_channels (`Tuple[int]`, *optional*, defaults to (64, 128, 256, 512)):
            Channel dimensions for each UNet block.
        layers_per_block (`int`, *optional*, defaults to 2):
            Number of residual blocks per UNet level.
        attention_head_dim (`int`, *optional*, defaults to 8):
            Dimension of each attention head.
        cross_attention_dim (`int`, *optional*, defaults to 384):
            Dimension for cross-attention (should match nepa_hidden_size).
        num_train_timesteps (`int`, *optional*, defaults to 1000):
            Number of diffusion timesteps for training.
        num_inference_timesteps (`int`, *optional*, defaults to 50):
            Number of diffusion timesteps for inference (with DDIM).
        beta_schedule (`str`, *optional*, defaults to "scaled_linear"):
            Noise schedule type: "linear", "scaled_linear", or "squaredcos_cap_v2".
        prediction_type (`str`, *optional*, defaults to "epsilon"):
            What the model predicts: "epsilon" (noise) or "v_prediction".
        use_ema (`bool`, *optional*, defaults to True):
            Whether to use exponential moving average for the model weights.
        ema_decay (`float`, *optional*, defaults to 0.9999):
            Decay rate for EMA.
        freeze_nepa (`bool`, *optional*, defaults to True):
            Whether to freeze the NEPA encoder during training.

    Example:
        ```python
        >>> from models.diffusion_decoder import DiffusionDecoderConfig, NepaConditionedDiffusion

        >>> config = DiffusionDecoderConfig(
        ...     nepa_model_path="outputs/nepa-tiny-patch14-224-cifar10",
        ...     nepa_hidden_size=384,
        ... )
        >>> model = NepaConditionedDiffusion(config)
        ```
    """

    model_type = "diffusion_decoder"

    def __init__(
        self,
        nepa_model_path: Optional[str] = None,
        nepa_hidden_size: int = 384,
        nepa_num_patches: int = 256,
        image_size: int = 224,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64, 128, 256, 512),
        layers_per_block: int = 2,
        attention_head_dim: int = 8,
        cross_attention_dim: int = 384,
        num_train_timesteps: int = 1000,
        num_inference_timesteps: int = 50,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "epsilon",
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        freeze_nepa: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.nepa_model_path = nepa_model_path
        self.nepa_hidden_size = nepa_hidden_size
        self.nepa_num_patches = nepa_num_patches
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_out_channels = block_out_channels
        self.layers_per_block = layers_per_block
        self.attention_head_dim = attention_head_dim
        self.cross_attention_dim = cross_attention_dim
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.freeze_nepa = freeze_nepa


__all__ = ["DiffusionDecoderConfig"]

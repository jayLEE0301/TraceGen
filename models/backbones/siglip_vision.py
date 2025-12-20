import logging

import torch
import torch.nn as nn
from transformers import SiglipImageProcessor, SiglipVisionModel

logger = logging.getLogger(__name__)


class SigLIPVisionEncoder(nn.Module):
    """
    SigLIP Vision Encoder that extracts visual features for point detection.

    Args:
        model_name: HuggingFace model identifier (e.g., 'google/siglip-base-patch16-384')
        freeze: Whether to freeze the backbone parameters
        d_model: Output dimension (will add projection if different from SigLIP dim)
        image_size: Expected input image size
        patch_size: Patch size for vision transformer
    """

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-384",
        freeze: bool = True,
        d_model: int = 768,
        image_size: int = 384,
        patch_size: int = 16,
        add_stem_adapter: bool = False,
    ):
        super().__init__()

        self.model_name = model_name
        self.freeze = freeze
        self.d_model = d_model
        self.image_size = image_size
        self.patch_size = patch_size
        self.add_stem_adapter = add_stem_adapter
        # Load SigLIP vision model
        logger.info(f"Loading SigLIP vision model: {model_name}")
        self.vision_model = SiglipVisionModel.from_pretrained(model_name)
        self.processor = SiglipImageProcessor.from_pretrained(model_name)

        self.stem_adapter = nn.Conv2d(1, 3, kernel_size=1) if self.add_stem_adapter else nn.Identity()

        # Get the actual hidden size from the model
        self.siglip_dim = self.vision_model.config.hidden_size

        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2

        self.final_layernorm = nn.LayerNorm(self.siglip_dim)

        # Freeze parameters if requested
        if freeze:
            self._freeze_backbone()

        logger.info("SigLIP Vision Encoder initialized:")
        logger.info(f"  - Model: {model_name}")
        logger.info(f"  - SigLIP dim: {self.siglip_dim}")
        logger.info(f"  - Output dim: {d_model}")
        logger.info(f"  - Image size: {image_size}")
        logger.info(f"  - Num patches: {self.num_patches}")
        logger.info(f"  - Frozen: {freeze}")

    def _freeze_backbone(self):
        """Freeze all backbone parameters, only except stem adapter"""
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.stem_adapter.parameters():
            param.requires_grad = True
        logger.info("SigLIP vision backbone frozen")

    def unfreeze_last_blocks(self, num_blocks: int = 2):
        """
        Unfreeze the last N transformer blocks for fine-tuning.

        Args:
            num_blocks: Number of last blocks to unfreeze
        """
        if not self.freeze:
            logger.warning("Backbone is not frozen, nothing to unfreeze")
            return

        # Unfreeze the last num_blocks encoder layers
        total_layers = len(self.vision_model.vision_model.encoder.layers)
        start_idx = max(0, total_layers - num_blocks)

        for i in range(start_idx, total_layers):
            for param in self.vision_model.vision_model.encoder.layers[i].parameters():
                param.requires_grad = True

        # Also unfreeze the layer norm at the end
        for param in self.vision_model.vision_model.post_layernorm.parameters():
            param.requires_grad = True

        logger.info(f"Unfroze last {num_blocks} vision transformer blocks (layers {start_idx}-{total_layers-1})")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the vision encoder.

        Args:
            pixel_values: Input images [B, 3, H, W]

        Returns:
            features: Visual features [B, N, d_model] where N = num_patches
        """
        # Standard forward pass
        if self.add_stem_adapter:
            pixel_values = self.stem_adapter(pixel_values)
            features = self.vision_model(pixel_values).last_hidden_state
        else:
            inputs = self.processor(images=pixel_values, return_tensors="pt", do_rescale=False)
            features = self.vision_model(inputs['pixel_values'].cuda()).last_hidden_state

        # Post layer norm
        features = self.final_layernorm(features)

        return features

    def get_patch_info(self) -> dict:
        """Get information about patch organization."""
        patches_per_side = self.image_size // self.patch_size
        return {
            'num_patches': self.num_patches,
            'patches_per_side': patches_per_side,
            'patch_size': self.patch_size,
            'image_size': self.image_size
        }

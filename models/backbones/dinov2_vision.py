import logging
from typing import Dict

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DinoV2VisionEncoder(nn.Module):
    """DINOv2 Vision Transformer encoder."""
    
    def __init__(
        self,
        model_name: str = "vit_large_patch16_dinov3.lvd1689m",  # vit_base_patch16_dinov3.lvd1689m
        freeze: bool = True,
        image_size: int = 384,
        patch_size: int = 14,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        self.image_size = image_size
        self.patch_size = patch_size
        
        # Load DINOv2 from TIMM
        logger.info(f"Loading DINOv2 model: {model_name}")
        self.vision_model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
        )
        data_config = timm.data.resolve_model_data_config(self.vision_model)
        self.data_mean = data_config['mean']
        self.data_std = data_config['std']
        # breakpoint()

        # CRITICAL: Get the ACTUAL output dimension by running a test forward pass
        # The embed_dim attribute is not reliable for intermediate layers
        logger.info("Detecting actual DINOv2 output dimension...")
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, image_size, image_size)
            try:
                # Try get_intermediate_layers (returns second-to-last layer by default)
                dummy_output = self.vision_model.get_intermediate_layers(dummy_input, n=1)[0]
                self.dinov2_embed_dim = dummy_output.shape[-1]
                logger.info(f"Using get_intermediate_layers, detected dim: {self.dinov2_embed_dim}")
            except:
                # Fallback to forward_features
                dummy_output = self.vision_model.forward_features(dummy_input)
                self.dinov2_embed_dim = dummy_output.shape[-1]
                logger.info(f"Using forward_features, detected dim: {self.dinov2_embed_dim}")
        
        logger.info(f"DINOv2 actual output dimension: {self.dinov2_embed_dim}")

        self.final_layernorm = nn.LayerNorm(self.dinov2_embed_dim)
        # Freeze if requested
        if freeze:
            self._freeze_encoder()
        
        logger.info("DINOv2 encoder initialized successfully")
    
    def _freeze_encoder(self):
        """Freeze all parameters in the vision model."""
        for param in self.vision_model.parameters():
            param.requires_grad = False
        logger.info("DINOv2 encoder frozen")
    
    def unfreeze_last_blocks(self, num_blocks: int = 2):
        """Unfreeze the last N transformer blocks for fine-tuning."""
        if not self.freeze:
            return
        
        total_blocks = len(self.vision_model.blocks)
        for i in range(total_blocks - num_blocks, total_blocks):
            for param in self.vision_model.blocks[i].parameters():
                param.requires_grad = True
        
        # Also unfreeze the norm layer
        for param in self.vision_model.norm.parameters():
            param.requires_grad = True
        
        logger.info(f"Unfroze last {num_blocks} blocks of DINOv2 encoder")
    
    @torch.no_grad()
    def _prep_tensor(self, images: torch.Tensor) -> torch.Tensor:
        """
          - uint8 → float32
          - [0,255] → [0,1] scale
          - H/W ≠ image_size → bicubic resize
          - channel-wise Normalize
        """
        # [B,3,H,W]
        x = images
        if x.dtype in (torch.uint8, torch.int16, torch.int32, torch.int64):
            x = x.float()
        # if range is greater than 1, scale to [0,255]
        if x.max() > 1.0:
            x = x / 255.0
        # resize to image_size
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(
                x, size=(self.image_size, self.image_size),
                mode="bicubic", align_corners=False, antialias=True
            )
        # channel-wise Normalize
        # breakpoint()
        mean = torch.tensor(self.data_mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(self.data_std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        x = (x - mean) / std

        return x

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through DINOv2 encoder.
        
        Args:
            images: Input images [B, 3, H, W]
            
        Returns:
            features: Encoded features [B, N, D] where N is number of patches
        """
        # Use get_intermediate_layers to get consistent features
        # n=1 means get the last layer only
        images_normalized = self._prep_tensor(images)
        features = self.vision_model.forward_features(images_normalized)
        # features is unpooled, a (B, 581, 768) shaped tensor

        # Remove CLS token if present (DINOv2 includes it as first token)
        # Expected number of patches (without CLS)
        expected_patches = (self.image_size // self.patch_size) ** 2
        
        if features.size(1) > expected_patches:
            # First token is CLS, remove it
            features = features[:, 1:, :]  # [B, N, D]
            # If still more than expected (e.g., reg tokens), keep last expected_patches tokens
            if features.size(1) > expected_patches:
                features = features[:, -expected_patches:, :]
        
        # Post layer norm
        features = self.final_layernorm(features)
        return features
    
    def get_patch_info(self) -> Dict:
        """Get information about the patch configuration."""
        num_patches = (self.image_size // self.patch_size) ** 2
        return {
            'image_size': self.image_size,
            'patch_size': self.patch_size,
            'num_patches': num_patches,
            'embed_dim': self.dinov2_embed_dim,
            'output_dim': self.dinov2_embed_dim
        }
    
    @property
    def embed_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.dinov2_embed_dim
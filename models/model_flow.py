import logging
from typing import Dict, List, Union

import torch
import torch.nn as nn

from .backbones.dinov2_vision import DinoV2VisionEncoder
from .backbones.siglip_vision import SigLIPVisionEncoder
from .backbones.t5_text import T5TextEncoder
from .decoder.cogvideox_flow import CogVideoXDecoder_flow

logger = logging.getLogger(__name__)


class TrajectoryFlow(nn.Module):
    """
    Complete TrajectoryFlow model for text-conditioned trajectory prediction.

    Combines (Prismatic-style):
    - DINOv2 vision encoder (RGB) - for spatial/geometric features
    - SigLIP vision encoder (RGB) - for vision-language features
    - SigLIP vision encoder (Depth) - for depth features
    - T5 text encoder for text embedding
    - Diffusion decoder for trajectory prediction

    Args:
        cfg: Configuration object containing all model parameters
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.trajectory_horizon = cfg.trajectory_horizon
        self.total_points = cfg.trajectory_horizon + 1
        self.d_model = cfg.d_model

        # Initialize DINOv2 vision encoder (for RGB)
        # from .backbones.dinov2_vision import DinoV2VisionEncoder
        self.dino_encoder = DinoV2VisionEncoder(
            model_name=getattr(cfg, 'dinov2_model', 'dinov2_vitl14'),
            freeze=cfg.freeze_vision,
            image_size=cfg.model.vision_encoder.image_size,
            patch_size=cfg.model.vision_encoder.patch_size
        )

        # Initialize SigLIP vision encoder (for RGB)
        self.siglip_encoder = SigLIPVisionEncoder(
            model_name=cfg.siglip_ckpt,
            freeze=cfg.freeze_vision,
            d_model=cfg.d_model,
            image_size=cfg.model.vision_encoder.image_size,
            patch_size=cfg.model.vision_encoder.patch_size
        )

        # Initialize Depth vision encoder (with stem adapter)
        self.depth_encoder = SigLIPVisionEncoder(
            model_name=cfg.siglip_ckpt,
            freeze=cfg.freeze_vision,
            d_model=cfg.d_model,
            image_size=cfg.model.vision_encoder.image_size,
            patch_size=cfg.model.vision_encoder.patch_size,
            add_stem_adapter=True
        )

        # Initialize T5 text encoder
        self.t5_encoder = T5TextEncoder(
            model_name=getattr(cfg, 't5_model', 't5-small'),
            freeze=getattr(cfg, 'freeze_t5', True),
            d_model=cfg.d_model,
            max_length=128
        )

        # Learnable depth mask token
        self.learnable_depth_mask_token = nn.Parameter(torch.randn(1, 1, self.depth_encoder.siglip_dim))

        vision_fusion_input = self.siglip_encoder.siglip_dim * 2 + self.dino_encoder.dinov2_embed_dim
        # Fuse concatenated vision streams (DINO + SigLIP + Depth) back to d_model
        self.vision_fusion = nn.Linear(vision_fusion_input, self.d_model)

        # Diffusion decoder
        decoder_config = {k: v for k, v in cfg.model.decoder.__dict__.items() 
                         if k != 'device'}
        
        logger.info(f"Initializing CogVideoXDecoder_flow without device parameter")
        self.diffusion_decoder = CogVideoXDecoder_flow(**decoder_config)

        # Store configuration
        self.freeze_vision = cfg.freeze_vision
        self.freeze_t5 = getattr(cfg, 'freeze_t5', True)

        logger.info(f"TrajectoryFlow model initialized (DINOv2+SigLIP+Depth, Prismatic-style):")
        logger.info(f"  - trajectory_horizon: {self.trajectory_horizon}")
        logger.info(f"  - total_points: {self.total_points}")
        logger.info(f"  - d_model: {self.d_model}")
        logger.info(f"  - Vision frozen: {self.freeze_vision}")
        logger.info(f"  - T5 frozen: {self.freeze_t5}")

    def unfreeze_backbones(self, num_blocks: int = 2):
        """Unfreeze the last few blocks of all encoders."""
        if self.freeze_vision:
            self.dino_encoder.unfreeze_last_blocks(num_blocks)
            self.siglip_encoder.unfreeze_last_blocks(num_blocks)
            self.depth_encoder.unfreeze_last_blocks(num_blocks)

        if self.freeze_t5:
            self.t5_encoder.unfreeze_last_blocks(num_blocks)

        logger.info(f"Unfroze last {num_blocks} blocks of all encoders")

    def encode_images(self, images: torch.Tensor, depth: torch.Tensor, 
                      is_depth_valid: torch.Tensor) -> torch.Tensor:
        """
        Encode images using DINOv2 + SigLIP + Depth (Prismatic-style).

        Args:
            images: Input RGB images [B, 3, H, W]
            depth: Input depth maps [B, 1, H, W]
            is_depth_valid: Depth validity mask [B]

        Returns:
            vision_features: Concatenated features [B, N, D_dino + D_siglip + D_depth]
        """
        # Process RGB with DINOv2
        dino_features = self.dino_encoder(images)  # [B, N_dino, D]
        
        # Process RGB with SigLIP
        siglip_features = self.siglip_encoder(images)  # [B, N_siglip, D]

        # Align token counts across encoders (DINOv2 vs SigLIP)
        # DINOv2 (reg tokens + removal of CLS) may yield non-square token counts (e.g., 728/733)
        # SigLIP yields a square grid (e.g., 24x24=576 for 384/16)
        B, Nd, D = dino_features.shape
        Ns = siglip_features.shape[1]

        if Nd != Ns:
            sd = int((Nd) ** 0.5)
            # If not a perfect square, try to drop 4 reg tokens or 1 CLS-equivalent remainder
            if sd * sd != Nd:
                if (Nd - 4) == int((Nd - 4) ** 0.5) ** 2:
                    dino_features = dino_features[:, 4:, :]
                    Nd = dino_features.shape[1]
                    sd = int(Nd ** 0.5)
                elif (Nd - 1) == int((Nd - 1) ** 0.5) ** 2:
                    dino_features = dino_features[:, 1:, :]
                    Nd = dino_features.shape[1]
                    sd = int(Nd ** 0.5)

            if sd * sd == Nd:
                # Resample DINO grid to SigLIP's grid size
                ss = int((Ns) ** 0.5)
                dino_grid = dino_features.view(B, sd, sd, D).permute(0, 3, 1, 2)  # [B, D, sd, sd]
                dino_grid = torch.nn.functional.interpolate(
                    dino_grid, size=(ss, ss), mode='bilinear', align_corners=False
                )
                dino_features = dino_grid.permute(0, 2, 3, 1).reshape(B, Ns, D)
            else:
                # Fallback: truncate to common minimum length
                min_len = min(Nd, Ns)
                dino_features = dino_features[:, :min_len, :]
                siglip_features = siglip_features[:, :min_len, :]
                Ns = min_len
        
        # Process depth
        valid_mask = is_depth_valid.bool().squeeze()
        N_final = siglip_features.shape[1]
        depth_features = torch.zeros(
            dino_features.shape[0], 
            N_final, 
            self.depth_encoder.siglip_dim,
            device=images.device
        )
        depth_features[:] = self.learnable_depth_mask_token
        
        if valid_mask.any():
            valid_depths = depth[valid_mask]
            # Ensure valid_depths is 4D: [B_valid, C, H, W]
            # Handle case where indexing might preserve extra dimensions
            # If 5D like [1, 1, 1, H, W], reshape to [1, 1, H, W]
            if valid_depths.dim() == 5:
                # Reshape to [B_valid, C, H, W] by flattening leading dims
                # Get last 4 dimensions: should be [B_valid, C, H, W] or [C, H, W]
                C, H, W = valid_depths.shape[-3:]
                # Flatten all leading dimensions into batch dimension
                valid_depths = valid_depths.view(-1, C, H, W)
            # If somehow we lost batch dimension, add it back
            elif valid_depths.dim() == 3:
                valid_depths = valid_depths.unsqueeze(0)
            elif valid_depths.dim() != 4:
                raise ValueError(f"Expected depth tensor to be 4D [B, C, H, W], got {valid_depths.dim()}D with shape {valid_depths.shape}")
            encoded_depths = self.depth_encoder(valid_depths)
            # Ensure depth stream has same token count
            if encoded_depths.shape[1] != N_final:
                # If mismatch, truncate or resample minimally (truncate is sufficient typically)
                if encoded_depths.shape[1] > N_final:
                    encoded_depths = encoded_depths[:, :N_final, :]
                else:
                    # Pad with mask token if fewer (rare); keep gradients consistent
                    pad_len = N_final - encoded_depths.shape[1]
                    pad = self.learnable_depth_mask_token.expand(encoded_depths.size(0), pad_len, -1)
                    encoded_depths = torch.cat([encoded_depths, pad], dim=1)
            depth_features[valid_mask] = encoded_depths
        
        # Concatenate along FEATURE dimension (Prismatic-style)
        # [B, N, D_dino + D_siglip + D_depth]
        vision_features = torch.cat([dino_features, siglip_features, depth_features], dim=2)
        # Project concatenated features back to d_model for downstream modules
        vision_features = self.vision_fusion(vision_features)
        
        return vision_features

    def encode_t5_texts(self, texts: Union[str, List[str]], device: torch.device) -> torch.Tensor:
        """Encode texts using T5 text encoder."""
        if isinstance(texts, str):
            texts = [texts]

        t5_features = self.t5_encoder.encode_texts(texts, device)  # [B, 128, D]

        return t5_features

    def forward_diffusion_training(
        self,
        images: torch.Tensor,
        texts: Union[str, List[str]],
        depth: torch.Tensor,
        is_depth_valid: torch.Tensor,
        target_trajectory: torch.Tensor,
        first_keypoint: torch.Tensor,
        diffusion_loss,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for diffusion training."""
        device = images.device

        
        # Encode vision (DINOv2 + SigLIP + Depth)
        vision_features = self.encode_images(images, depth, is_depth_valid)  # [B, N, D_combined]
        
        # Encode text
        text_features = self.encode_t5_texts(texts, device)  # [B, 128, D]
        
        # Concatenate along sequence dimension
        combined_features = torch.cat([vision_features, text_features], dim=1)  # [B, N+128, D]
        
        # Forward through diffusion decoder
        outputs = self.diffusion_decoder.forward_diffusion_training(
            trunk_conditioning=combined_features,
            target_video=target_trajectory,
            diffusion_loss=diffusion_loss,
        )
        
        return outputs

    def predict_trajectory(
        self,
        images: torch.Tensor,
        texts: Union[str, List[str]],
        depth: torch.Tensor,
        is_depth_valid: torch.Tensor,
        first_keypoint: torch.Tensor,
        noise_scheduler,
        num_inference_steps: int = 100,
        guidance_scale: float = 2.0,
    ) -> torch.Tensor:
        """Predict future trajectory."""
        self.eval()
        
        with torch.no_grad():
            # Encode vision (DINOv2 + SigLIP + Depth)
            vision_features = self.encode_images(images, depth, is_depth_valid)  # [B, N, D_combined]
            
            # Encode text
            text_features = self.encode_t5_texts(texts, images.device)  # [B, 128, D]
            
            # Concatenate along sequence dimension
            combined_features = torch.cat([vision_features, text_features], dim=1)
            
            # Generate trajectory
            complete_trajectory = self.diffusion_decoder.predict_trajectory(
                trunk_conditioning=combined_features,
                scheduler=noise_scheduler,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
        
        return complete_trajectory

    def get_model_info(self) -> Dict:
        """Get model configuration and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'trajectory_horizon': self.trajectory_horizon,
            'total_points': self.total_points,
            'd_model': self.d_model,
            'vision_frozen': self.freeze_vision,
            't5_frozen': self.freeze_t5,
            'dino_patch_info': self.dino_encoder.get_patch_info(),
            'siglip_patch_info': self.siglip_encoder.get_patch_info(),
            'depth_patch_info': self.depth_encoder.get_patch_info(),
            't5_config': self.t5_encoder.get_text_config()
        }
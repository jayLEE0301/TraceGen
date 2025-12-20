import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


class TrajectoryLoss(nn.Module):
    """
    Combined loss function for trajectory diffusion training.

    Includes:
    - Diffusion loss for the denoising process
    - Noise scheduler for diffusion process
    - Optional regularization terms
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # Loss weights - only diffusion loss
        self.trajectory_horizon = getattr(cfg, 'trajectory_horizon', 32)

        # Initialize noise scheduler for diffusion training
        from diffusers import CogVideoXDDIMScheduler

        self.noise_scheduler = CogVideoXDDIMScheduler(
            num_train_timesteps=100,
            beta_schedule="linear",
            beta_start=0.00085,
            beta_end=0.012,
        )
        # breakpoint()

        # Loss functions for metrics only
        self.mse_loss = nn.MSELoss(reduction='none')

        logger.info("TrajectoryLoss initialized:")
        logger.info("  - Using only diffusion loss for training")
        logger.info(f"  - Noise scheduler: {self.noise_scheduler.__class__.__name__}")

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        trajectory_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute trajectory losses.

        Args:
            predictions: Dictionary containing model predictions
            targets: Dictionary containing ground truth targets
            trajectory_mask: Mask for valid trajectory points [B, H+1]

        Returns:
            loss_dict: Dictionary containing individual losses and total loss
        """
        loss_dict = {}

        # breakpoint()

        if 'noise_pred' in predictions:
            # Handle diffusion policy output format
            noise_pred = predictions['noise_pred']  # [B, H, 2]
            noise_target = predictions['noise_target']  # [B, H, 2]

            noise_pred = rearrange(noise_pred, 'b c t h w -> b (h w) t c')
            noise_target = rearrange(noise_target, 'b c t h w -> b (h w) t c')
            traj_mask = trajectory_mask[:, :, 1:, :]  # only care about the non-keypoint mask (not horizon-wise mask)

            diffusion_loss = F.mse_loss(noise_pred * traj_mask, noise_target * traj_mask)  # [B, H, 2]
        else:
            traj_pred = predictions['trajectory_pred']  # [B, N, H, 2]
            traj_target = targets['trajectory'][:, :, 1:]  # [B, N, H, 2]

            traj_mask = trajectory_mask[:, :, 1:]  # exclude first keypoint mask
            traj_mask = rearrange(traj_mask, 'b n h d -> b h (n d)')

            diffusion_loss = F.mse_loss(traj_pred * traj_mask, traj_target * traj_mask)  # [B, H, 2]

        loss_dict['diffusion_loss'] = diffusion_loss
        loss_dict['total_loss'] = diffusion_loss

        return loss_dict

    def add_noise(self, clean_actions, noise, timesteps):
        """Add noise to clean actions using the noise scheduler."""
        return self.noise_scheduler.add_noise(clean_actions, noise, timesteps)

    def compute_trajectory_metrics(
        self,
        pred_trajectory: torch.Tensor,
        target_trajectory: torch.Tensor,
        trajectory_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute evaluation metrics for trajectory prediction.

        Args:
            pred_trajectory: Predicted trajectory [B, N, H+1, 2]
            target_trajectory: Target trajectory [B, N, H+1, 2]
            trajectory_mask: Valid trajectory mask [B, N, H+1, 2]

        Returns:
            metrics: Dictionary containing evaluation metrics
        """
        metrics = {}

        target_trajectory = target_trajectory[:, :, 1:]
        trajectory_mask = trajectory_mask[:, :, 1:]

        # MSE loss per trajectory point
        mse_per_point = self.mse_loss(pred_trajectory, target_trajectory)  # [B, N, H+1, 2]
        mse_per_point = (mse_per_point * trajectory_mask).sum(dim=-1)  # [B, N, H+1]

        trajectory_mse = mse_per_point.sum() / (trajectory_mask.sum() + 1e-8)

        metrics['trajectory_mse'] = trajectory_mse

        return metrics


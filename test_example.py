import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml
from PIL import Image
from scipy.interpolate import interp1d
from scipy.ndimage import generic_filter, gaussian_filter, gaussian_filter1d

from test_helpers import (
    create_trajectory_visualization,
    load_depth,
    load_images_and_texts,
    predict_trajectory_simple,
)
from trainer.trainer import DictToNamespace, TrajectoryDiffusionTrainer
from utils.misc import setup_logging

# Add project root to path
sys.path.append(str(Path(__file__).parent))

logger = logging.getLogger(__name__)


class TrajectoryDiffusionTest(TrajectoryDiffusionTrainer):
    """
    Main trainer class for trajectory diffusion system.
    """

    def __init__(self, cfg, output_dir=None):
        # Convert dict config to namespace for easier access
        if isinstance(cfg, dict):
            self.cfg = DictToNamespace(cfg)
        else:
            self.cfg = cfg
        self.device = torch.device(self.cfg.hardware.device if torch.cuda.is_available() else 'cpu')
        self.best_metric = 0.0
        self.start_epoch = 0

        # Metrics tracking for trajectory diffusion
        self.train_metrics = {
            'total_loss': [], 'diffusion_loss': []
        }
        self.val_metrics = {
            'total_loss': [], 'diffusion_loss': [],
            'trajectory_mse': [], 'trajectory_mae': []
        }

        # Initialize components
        self._setup_model()

        # Initialize checkpoint directory for visualizations
        if output_dir is not None:
            self.checkpoint_dir = Path(output_dir)
        else:
            checkpoint_path = getattr(self.cfg.logging, "checkpoint_dir", "./checkpoints")
            self.checkpoint_dir = Path(checkpoint_path)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _setup_model(self):
        """Initialize model and loss function."""
        logger.info("Setting up model...")

        # Create model
        # Initialize model
        from models.model_flow import TrajectoryFlow
        self.model = TrajectoryFlow(self.cfg)
        self.model = self.model.to(self.device)

        # Create loss function (diffusion loss will be handled by the decoder)
        # For now, we'll create a simple MSE loss for trajectory prediction
        from losses.trajectory_loss import TrajectoryLoss
        self.criterion = TrajectoryLoss(self.cfg)

        # Model compilation for PyTorch 2.0+
        if self.cfg.hardware.compile_model and hasattr(torch, 'compile'):
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        # Print model info
        model_info = self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
        logger.info(f"Model initialized: {model_info.get('total_parameters', 'Unknown')} total parameters")
        logger.info(f"Trainable parameters: {model_info.get('trainable_parameters', 'Unknown')}")

    def smooth_trajectory(
        self, trajectory, method="gaussian", sigma=1.0, window_size=5
    ):
        """Smooth trajectory to reduce noise and bumpiness.

        Args:
            trajectory: [N, T, 2] trajectory data
            method: smoothing method ('gaussian', 'moving_average', 'interpolation')
            sigma: standard deviation for Gaussian filter
            window_size: window size for moving average

        Returns:
            smoothed trajectory: [N, T, 2]
        """
        smoothed = trajectory.copy()

        if method == "gaussian":
            # Apply Gaussian filter along time dimension for each trajectory and coordinate
            for n in range(trajectory.shape[0]):
                for coord in range(2):  # x, y coordinates
                    smoothed[n, :, coord] = gaussian_filter1d(
                        trajectory[n, :, coord], sigma=sigma, mode="nearest"
                    )

        elif method == "moving_average":
            # Simple moving average
            for n in range(trajectory.shape[0]):
                for coord in range(2):
                    # Pad the signal to handle edges
                    padded = np.pad(
                        trajectory[n, :, coord],
                        (window_size // 2, window_size // 2),
                        mode="edge",
                    )
                    # Apply moving average
                    smoothed[n, :, coord] = np.convolve(
                        padded, np.ones(window_size) / window_size, mode="valid"
                    )

        elif method == "interpolation":
            # Cubic spline interpolation for very smooth curves
            for n in range(trajectory.shape[0]):
                t_original = np.arange(trajectory.shape[1])
                for coord in range(2):
                    # Skip if trajectory has too few points or no variation
                    if (
                        trajectory.shape[1] < 4
                        or np.std(trajectory[n, :, coord]) < 1e-6
                    ):
                        continue
                    try:
                        f = interp1d(
                            t_original,
                            trajectory[n, :, coord],
                            kind="cubic",
                            bounds_error=False,
                            fill_value="extrapolate",
                        )
                        smoothed[n, :, coord] = f(t_original)
                    except ValueError:
                        # Fallback to linear interpolation if cubic fails
                        f = interp1d(
                            t_original,
                            trajectory[n, :, coord],
                            kind="linear",
                            bounds_error=False,
                            fill_value="extrapolate",
                        )
                        smoothed[n, :, coord] = f(t_original)

        return smoothed

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        ckpt_state = checkpoint['model_state_dict']
        fixed_state = {}

        for k, v in ckpt_state.items():
            if ".module." in k:
                new_k = k.replace(".module.", ".")
            else:
                new_k = k

            fixed_state[new_k] = v

        load_result = self.model.load_state_dict(fixed_state, strict=False)

        print("=== After prefix-fix ===")
        print("missing_keys:")
        for k in load_result.missing_keys:
            print(" ", k)
        print("unexpected_keys:")
        for k in load_result.unexpected_keys:
            print(" ", k)

        model_state = self.model.state_dict()
        print("\nshape mismatches:")
        for k, v in fixed_state.items():
            if k in model_state and model_state[k].shape != v.shape:
                print(f"  {k}: ckpt {tuple(v.shape)} vs model {tuple(model_state[k].shape)}")

        # Load action statistics
        self.action_min = checkpoint.get('action_min', None)
        self.action_max = checkpoint.get('action_max', None)

        # breakpoint()

        # Load training state
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint.get('best_metric', 0.0)
        self.train_metrics = checkpoint.get('train_metrics', self.train_metrics)
        self.val_metrics = checkpoint.get('val_metrics', self.val_metrics)

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.start_epoch}")

        # Log action statistics if available
        if self.action_min is not None and self.action_max is not None:
            logger.info(f"Action statistics loaded - Min: {self.action_min}, Max: {self.action_max}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return config_dict


def main():
    """Simple example showing how to get predicted_trajectory from the model."""
    parser = argparse.ArgumentParser(description="Simple API example for trajectory prediction")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, required=True, help="Path to checkpoint to resume from")
    parser.add_argument("--test", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--output", type=str, default="./test_result", help="Output folder to save visualizations")
    parser.add_argument("--guidance_scale", type=float, default=None, help="Single guidance scale for inference (if provided, overrides multi-scale generation)")
    parser.add_argument("--movement_threshold", type=float, default=0.002, help="Minimum total movement to show trajectory (default: 0.2)")
    parser.add_argument("--max_trajectories", type=int, default=80, help="Maximum number of trajectories to show (default: 20)")
    parser.add_argument("--num_samples_per_scale", type=int, default=10, help="Number of samples to generate per guidance scale (default: 10)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)

    # ============================================================================
    # STEP 1: Initialize Model and Load Checkpoint
    # ============================================================================
    logger.info("Initializing model...")
    trainer = TrajectoryDiffusionTest(cfg, output_dir=args.output)
    trainer.load_checkpoint(args.resume)
    trainer.model.diffusion_decoder.set_data_act_statistics(trainer.action_max, trainer.action_min)
    trainer.model.eval()
    
    # ============================================================================
    # STEP 2: Load Input Data
    # ============================================================================
    logger.info("Loading input data...")
    
    # --- INPUT: Images and Texts ---
    image_size = cfg['model']['vision_encoder']['image_size']
    images, texts, image_files, original_size = load_images_and_texts(args.test, image_size, return_original_size=True)
    images = images.to(trainer.device)
    logger.info(f"Loaded {len(images)} images")
    logger.info(f"Image shape: {images.shape}")  # [B, 3, H, W]
    logger.info(f"Text examples: {texts[:3]}")  # Show first 3 texts
    logger.info(f"Using {len(image_files)} image files (sorted and filtered)")

    # --- OPTIONAL INPUT: Depth Maps ---
    # Use the same image_files returned from load_images_and_texts to ensure matching order
    depth_maps, is_depth_valid, sensored_depth_maps, is_sensored_depth_valid = load_depth(
        args.test, image_files, images.shape[2], images.shape[3]
    )

    sensored_depth_maps = sensored_depth_maps / 1000.0

    # 1. calculate ratio
    depth_ratio_map = sensored_depth_maps / (depth_maps + 1e-8)
    depth_ratio_map = depth_ratio_map.squeeze().cpu().numpy()  # [H, W]

    # 2. fill holes: local_mean_filter for inpainting
    def local_mean_filter(values):
        center = values[len(values)//2]
        if center == 0:
            valid = values[values > 0]
            return valid.mean() if len(valid) > 0 else 0
        else:
            return center

    depth_ratio_filled = generic_filter(depth_ratio_map, local_mean_filter, size=15)

    # 3. clip unreasonable ratios
    depth_ratio_filled = np.clip(depth_ratio_filled, 0.5, 2.0)

    # 4. now "extremely strong blur"
    blur_sigma = 30.0  # float OK here
    depth_ratio_filled = gaussian_filter(depth_ratio_filled, sigma=blur_sigma)

    # (optional) clip again
    depth_ratio_filled = np.clip(depth_ratio_filled, 0.5, 2.0)

    
    # Handle regular depth maps
    if depth_maps is not None:
        depth_maps = depth_maps.to(trainer.device)
        logger.info(f"Loaded regular depth maps: {depth_maps.shape}")  # [B, 1, H, W]
        if is_depth_valid is not None:
            logger.info(f"Regular depth valid flags: {is_depth_valid.sum().item()}/{len(is_depth_valid)} valid")
    else:
        logger.info("No regular depth maps found")
        depth_maps = None
        is_depth_valid = None
    
    # Handle sensored depth maps
    if sensored_depth_maps is not None:
        sensored_depth_maps = sensored_depth_maps.to(trainer.device)
        logger.info(f"Loaded sensored depth maps: {sensored_depth_maps.shape}")  # [B, 1, H, W]
        if is_sensored_depth_valid is not None:
            logger.info(f"Sensored depth valid flags: {is_sensored_depth_valid.sum().item()}/{len(is_sensored_depth_valid)} valid")
    else:
        logger.info("No sensored depth maps found")
        sensored_depth_maps = None
        is_sensored_depth_valid = None
    
    # ============================================================================
    # STEP 3: Generate multiple samples with different guidance scales
    # ============================================================================
    
    # Determine guidance scales to use
    if args.guidance_scale is not None:
        # Single guidance scale mode (backward compatibility)
        guidance_scales = [args.guidance_scale]
    else:
        # Multi-scale mode: generate 10 samples each for 0.5, 1.0, 2.0
        guidance_scales = [1.0, 2.0, 3.0]
    
    batch_size = images.size(0)
    # breakpoint()
    logger.info(f"Generating {args.num_samples_per_scale} samples for each guidance scale: {guidance_scales}")
    logger.info(f"Total visualizations to generate: {len(guidance_scales) * args.num_samples_per_scale} per image sample")
    logger.info(f"Processing {batch_size} image samples")
    
    # Organize by image sample (not by guidance scale or iteration)
    for img_idx in range(batch_size):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing image sample {img_idx}/{batch_size - 1}")
        logger.info(f"{'='*60}")
        
        # Extract single image sample
        single_image = images[img_idx:img_idx+1]  # Keep batch dimension
        single_text = [texts[img_idx]]
        single_depth = depth_maps[img_idx:img_idx+1] if depth_maps is not None else None
        single_depth_valid = is_depth_valid[img_idx:img_idx+1] if is_depth_valid is not None else None
        
        for guidance_scale in guidance_scales:
            for iter_idx in range(args.num_samples_per_scale):
                logger.info(f"  Guidance {guidance_scale}, iteration {iter_idx + 1}/{args.num_samples_per_scale}...")
                
                # Run inference with current guidance scale
                predicted_trajectory = predict_trajectory_simple(
                    trainer=trainer,
                    images=single_image,                    # [1, 3, H, W]
                    texts=single_text,                     # List[str] with 1 element
                    depth_maps=single_depth,               # [1, 1, H, W] or None
                    is_depth_valid=single_depth_valid,    # [1] or None
                    test_path=args.test,
                    image_files=image_files,
                    guidance_scale=guidance_scale
                )
                
                # Truncate to 20 timesteps if needed
                predicted_trajectory = predicted_trajectory[:, :, :35]
                
                # Create visualization organized by image sample
                # Directory: sample{img_idx}, filename includes guidance and iteration
                split_dir = f"sample{img_idx}"
                filename_suffix = f"guidance_{guidance_scale}_iter_{iter_idx}"
                # breakpoint()
                
                create_trajectory_visualization(
                    images=single_image,
                    texts=single_text,
                    predicted_trajectory=predicted_trajectory,
                    depth_maps=single_depth,
                    is_depth_valid=single_depth_valid,
                    output_dir=args.output,
                    split=split_dir,  # Directory name
                    filename_suffix=filename_suffix,  # Filename suffix
                    sample_index_in_filename=img_idx,  # Use actual image index in filename
                    absolute_action=trainer.cfg.absolute_action,
                    trainer=trainer,
                    test_path=args.test,
                    image_files=image_files,
                    movement_threshold=args.movement_threshold,
                    max_trajectories=args.max_trajectories
                )

                pred_traj = create_trajectory_visualization(
                    images=single_image,
                    texts=single_text,
                    predicted_trajectory=predicted_trajectory,
                    depth_maps=single_depth,
                    is_depth_valid=single_depth_valid,
                    output_dir=args.output,
                    split=split_dir,  # Directory name
                    filename_suffix=filename_suffix,  # Filename suffix
                    sample_index_in_filename=img_idx,  # Use actual image index in filename
                    absolute_action=trainer.cfg.absolute_action,
                    trainer=trainer,
                    test_path=args.test,
                    image_files=image_files,
                    movement_threshold=args.movement_threshold,
                    max_trajectories=args.max_trajectories,
                    get_reconstructed_trajectory=True,
                    original_size=original_size,
                    depth_ratio_filled=depth_ratio_filled
                )
                # breakpoint()
                # save the final_traj to a npz file with key 'traj', 'keypoints', 'valid_steps'
                final_traj_path = Path(args.output) / "trajectory_visualizations" / split_dir / f"final_traj_{filename_suffix}.npz"
                # valid_steps is 128-length boolean array, with True for valid steps
                valid_steps = np.zeros(128, dtype=bool)
                valid_steps[:36] = True
                np.savez(final_traj_path, traj=pred_traj, keypoints=pred_traj[:, 0, :2], valid_steps=valid_steps)

                # save the depth_ratio_filled * depth_maps to npz file with key 'depth_ratio_filled'
                # rescaled depth
                rescaled_depth = depth_ratio_filled * single_depth[0,0].detach().cpu().numpy()
                # resize it to be same with the image size
                depth_pil = Image.fromarray(rescaled_depth)

                depth_pil_resized = TF.resize(depth_pil, (400, 640), Image.BILINEAR)

                rescaled_depth = np.array(depth_pil_resized)

                rescaled_depth_path = Path(args.output) / "trajectory_visualizations" / split_dir / f"rescaled_depth_{filename_suffix}.npz"
                np.savez(rescaled_depth_path, depth=rescaled_depth)
                # breakpoint()
    
    logger.info(f"\n{'='*60}")
    logger.info("All visualizations completed!")
    logger.info(f"Visualizations saved to {Path(args.output) / 'trajectory_visualizations'}")
    logger.info(f"Generated {len(guidance_scales) * args.num_samples_per_scale} visualizations per sample")
    logger.info("Done!")


if __name__ == "__main__":
    main()


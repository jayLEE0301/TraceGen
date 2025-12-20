import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import wandb
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

matplotlib.use('Agg')  # Use non-interactive backend

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import dataio
from dataio.datasets import create_dataloaders
from dataio.transforms import create_train_transforms, create_val_transforms
from utils.misc import AverageMeter, create_optimizer

logger = logging.getLogger(__name__)

def should_show_progress_bar():
    """Only show tqdm progress bars in interactive terminals, not in log files."""
    return sys.stderr.isatty() and sys.stdout.isatty()

def load_images_and_texts(test_root, cfg):
    image_dir = os.path.join(test_root, "images")
    text_dir = os.path.join(test_root, "texts")
    label_file = os.path.join(text_dir, "label.json")

    with open(label_file, "r") as f:
        label_json = json.load(f)

    # collect image paths (sorted for consistent ordering)
    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    image_files.sort()  # Ensure consistent ordering
    
    logger.info(f"Loading images in order:")
    for i, img_file in enumerate(image_files):
        logger.info(f"  {i}: {os.path.basename(img_file)}")

    images = []
    texts = []

    # Get target image size from config (same as validation transforms)
    target_size = cfg.model.vision_encoder.image_size  # 384
    if isinstance(target_size, int):
        target_size = (target_size, target_size)

    for path in image_files:
        key = os.path.splitext(os.path.basename(path))[0]
        if key not in label_json:
            continue  # skip if no label

        # Load and process image same as datasets.py pipeline
        pil_image = Image.open(path).convert("RGB")

        # Apply ResizeTransform (same as validation)
        resized_image = TF.resize(pil_image, target_size, Image.BILINEAR)

        # Apply NormalizeTransform (convert to tensor, no ImageNet normalization)
        image_tensor = TF.to_tensor(resized_image)  # [3, H, W] in [0, 1]

        images.append(image_tensor)
        texts.append(label_json[key])

    # Stack tensors instead of numpy arrays
    images_tensor = torch.stack(images, dim=0)  # [frames, 3, h, w]

    return images_tensor, texts

class DictToNamespace:
    """Convert nested dictionary to namespace object for attribute access."""

    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, DictToNamespace(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

def namespace_to_dict(obj):
    """Convert DictToNamespace object (or nested structure) back to dict."""
    if isinstance(obj, DictToNamespace):
        return {key: namespace_to_dict(getattr(obj, key)) for key in dir(obj) if not key.startswith('_')}
    elif isinstance(obj, dict):
        return {key: namespace_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(namespace_to_dict(item) for item in obj)
    else:
        return obj

class TrajectoryDiffusionTrainer:
    """
    Main trainer class for trajectory diffusion system.
    """

    def __init__(self, cfg, rank=0, world_size=1, local_rank=0):  # ADD PARAMETERS
        # Convert dict config to namespace for easier access
        if isinstance(cfg, dict):
            self.cfg = DictToNamespace(cfg)
        else:
            self.cfg = cfg
        # self.device = torch.device(self.cfg.hardware.device if torch.cuda.is_available() else 'cpu')

        # ADD: Distributed training setup
        self.use_wandb = cfg.get("logging").get('use_wandb', False) if cfg.get("logging") is not None else False
        self.checkpoint_dir = Path(cfg.get("logging").get('checkpoint_dir', "checkpoints") if cfg.get("logging") is not None else "./data1")
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.is_main_process = rank == 0
        self.num_log_steps_per_epoch = cfg.get("train").get('num_log_steps_per_epoch', 10)
        # MODIFY: Set device based on distributed setup
        if world_size > 1:
            self.device = torch.device(f'cuda:{local_rank}')
        else:
            self.device = torch.device(self.cfg.hardware.device if torch.cuda.is_available() else 'cpu')
        
        self.best_metric = 0.0
        self.start_epoch = 0

        # Action statistics for normalization (saved with checkpoints)
        self.action_min = None
        self.action_max = None

        # Metrics tracking for trajectory diffusion
        self.train_metrics = {
            'total_loss': [], 'diffusion_loss': []
        }
        self.val_metrics = {
            'total_loss': [], 'diffusion_loss': [],
            'trajectory_mse': [], 'trajectory_mae': [],
            'endpoint_mse': [], 'first_keypoint_mse': []
        }

        # Initialize components
        self._setup_model()
        self._setup_data()
        self._setup_optimization()
        self._setup_logging()

    def _setup_model(self):
        """Initialize model and loss function."""
        try:
            if self.is_main_process:  # ADD: Only log on main process
                logger.info("Setting up model...")

            # Create model
            from models.model_flow import TrajectoryFlow
            self.model = TrajectoryFlow(self.cfg)
            logger.info(f"Rank {self.rank}: Moving to device {self.device}...")

            self.model = self.model.to(self.device)
            
            # ADD: Wrap model with DDP for multi-GPU training
            if self.world_size > 1:
                self.model = DDP(
                    self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    find_unused_parameters=False
                )
        except Exception as e:
            logger.error(f"Rank {self.rank}: Error setting up model: {e}")
            raise e

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
        # calculate the number of parameters in the model
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params}")
        logger.info(f"Trainable parameters: {trainable_params}")
        # breakpoint()

    def _setup_data(self):
        """Initialize data loaders."""
        logger.info("Setting up data loaders...")

        # Create transforms
        train_transform = create_train_transforms(self.cfg)
        val_transform = create_val_transforms(self.cfg)

        # Create data loaders
        self.train_loader, self.val_loader = create_dataloaders(
            self.cfg, train_transform, val_transform,
            world_size=self.world_size, rank=self.rank  # ADD
        )


        logger.info(f"Data loaders created:")
        logger.info(f"  - Train: {len(self.train_loader)} batches")
        logger.info(f"  - Val: {len(self.val_loader)} batches")

    def _setup_optimization(self):
        """Initialize optimizer."""
        logger.info("Setting up optimization...")

        # Create optimizer
        self.optimizer = create_optimizer(self.model, self.cfg)


        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.cfg.hardware.mixed_precision else None


    def _setup_logging(self):
        """Initialize logging and checkpointing."""
        # Create checkpoint directory
        from datetime import datetime
        if not self.is_main_process:  # ADD: Only main process does logging setup
            return

        timestamp1 = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.checkpoint_dir = Path(self.cfg.logging.checkpoint_dir) / self.cfg.logging.save_dir / f'{timestamp1}'

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb (only if enabled)
        self.use_wandb = getattr(self.cfg.logging, 'use_wandb', True)
        if (self.use_wandb and
            hasattr(self.cfg.logging, 'wandb_project') and
            self.cfg.logging.wandb_project):
            wandb.init(
                project=self.cfg.logging.wandb_project,
                config=self.cfg.__dict__,
                name=f"run_{self.cfg.seed}",
                tags=[f"num_kps_{self.cfg.num_kps}"]
            )
            logger.info("Wandb initialized")
        else:
            logger.info("Wandb disabled")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        if hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(epoch)

        # Meters for tracking metrics
        meters = {
            'total_loss': AverageMeter(),
            'diffusion_loss': AverageMeter()
        }
        # Create progress bar for training batches
        pbar = tqdm(enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch:3d}/{self.cfg.train.epochs} [Train]",
            leave=False,
            disable=not should_show_progress_bar(),  # Disable in log files
            file=sys.stderr)  # Write to stderr if enabled

        # Calculate checkpoint intervals
        if self.num_log_steps_per_epoch == 0:
            checkpoint_intervals = []
            viz_step_interval = []
        else:
            total_batches = len(self.train_loader)
            # Calculate checkpoint and visualization intervals
            checkpoint_intervals = [int(total_batches * i / (self.num_log_steps_per_epoch + 1)) for i in range(1, self.num_log_steps_per_epoch + 1)]
            checkpoint_intervals = [x for x in checkpoint_intervals if x > 0]  # Remove 0 if total_batches is small
            viz_step_interval = [int(total_batches * i / (self.num_log_steps_per_epoch + 1)) for i in range(1, self.num_log_steps_per_epoch + 1)]
            viz_step_interval = [x for x in viz_step_interval if x > 0]  # Remove 0 if total_batches is small
        for batch_idx, batch in pbar:
            # we filter out the data where valid_steps is smaller than 3, and  batch['movement_bool'].sum() is 0
            # Ensure all tensors are on the same device for the mask computation
            valid_mask = (batch['valid_steps'] > 3) & (batch['movement_bool'].sum(dim=1) > 0) & (batch['gt_mask'].sum(dim=1) > 0)
            # breakpoint()

            # if number of valid_mask is zero, skip the batch
            if valid_mask.sum() == 0:
                continue
            
            # Apply mask to filter batch, handling both tensors and non-tensor data
            filtered_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    # For tensors, apply the mask on the same device
                    filtered_batch[k] = v[valid_mask.to(v.device)]
                elif isinstance(v, list):
                    # For lists, apply the mask to get the corresponding elements
                    filtered_batch[k] = [v[i] for i in range(len(v)) if valid_mask[i].item()]
                else:
                    # For other data types, keep as is (they might be scalars or other types)
                    filtered_batch[k] = v
            batch = filtered_batch
            # breakpoint()
            # Move batch to device
            images = batch['image'].to(self.device)
            texts = batch['text']  # List of strings
            gt_coords = batch['gt_xy'].to(self.device)
            gt_mask = batch['gt_mask'].to(self.device)
            depth = batch['depth'].to(self.device)
            is_depth_valid = batch['is_depth_valid']
            movement_bool = batch['movement_bool'].to(self.device)

            # Extract trajectory data
            target_trajectory = batch['trajectory'].to(self.device)  # [B, N, H+1, 2]
            trajectory_mask = batch['trajectory_mask'].to(self.device)  # [B, N, H+1]

            # Prepare targets
            targets = {
                'gt_coords': gt_coords,
                'gt_mask': gt_mask,
                'trajectory': target_trajectory,
                'trajectory_mask': trajectory_mask
            }

            model_forward = self.model.module if hasattr(self.model, "module") else self.model
            outputs = model_forward.forward_diffusion_training(
                images=images,
                texts=texts,
                depth=depth,
                is_depth_valid=is_depth_valid,
                target_trajectory=target_trajectory,
                first_keypoint=gt_coords,
                diffusion_loss=self.criterion
            )

            losses = self.criterion(outputs, targets, trajectory_mask)
            total_loss = losses['total_loss']

            # Backward pass
            self.optimizer.zero_grad()

            if self.cfg.hardware.mixed_precision:
                self.scaler.scale(total_loss).backward()

                # Gradient clipping
                if self.cfg.train.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_grad_norm)
                    model_params = self.model.module.parameters() if isinstance(self.model, DDP) else self.model.parameters()
                    torch.nn.utils.clip_grad_norm_(model_params, self.cfg.train.clip_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()

                # Gradient clipping
                if self.cfg.train.clip_grad_norm > 0:
                    # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_grad_norm)
                    model_params = self.model.module.parameters() if isinstance(self.model, DDP) else self.model.parameters()
                    torch.nn.utils.clip_grad_norm_(model_params, self.cfg.train.clip_grad_norm)

                self.optimizer.step()

            # Update meters
            batch_size = images.size(0)
            for key in ['total_loss', 'diffusion_loss']:
                if key in losses:
                    meters[key].update(losses[key].item(), batch_size)

            # Logging
            if batch_idx % self.cfg.logging.log_every == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {meters['total_loss'].avg:.4f} "
                    f"Diffusion: {meters['diffusion_loss'].avg:.4f} "
                    f"LR: {current_lr:.2e}"
                )

                # Log to wandb
                if self.use_wandb and wandb.run:
                    wandb.log({
                        'train/total_loss': meters['total_loss'].val,
                        'train/learning_rate': current_lr,
                        'epoch': epoch,
                        'step': epoch * len(self.train_loader) + batch_idx
                    })

            # Save trajectory visualizations every 1/10th of total batches when visualize_every condition met
            # if self.is_main_process and (batch_idx == 0) and epoch % getattr(self.cfg.train, 'visualize_every', 20) == 0:
            if self.is_main_process and (batch_idx in viz_step_interval) and epoch % getattr(self.cfg.train, 'visualize_every', 20) == 0:
                with torch.no_grad():
                    guidance_scale = 1
                    if hasattr(self.model, "module"):
                        predicted_trajectory = self.model.module.predict_trajectory(
                        images=images,
                        texts=texts,
                        depth=depth,
                        is_depth_valid=is_depth_valid,
                        first_keypoint=gt_coords,
                        noise_scheduler=self.criterion.noise_scheduler,
                        guidance_scale=guidance_scale
                    )
                    else:
                        predicted_trajectory = self.model.predict_trajectory(
                        images=images,
                        texts=texts,
                        depth=depth,
                        is_depth_valid=is_depth_valid,
                        first_keypoint=gt_coords,
                        noise_scheduler=self.criterion.noise_scheduler,
                        guidance_scale=guidance_scale
                    )
                    start=time.time()
                    viz_save_dir = self._create_trajectory_visualizations_v2(
                        batch, predicted_trajectory, target_trajectory, trajectory_mask,
                        epoch, batch_idx, split=f"train_guidance_scale_{guidance_scale}",
                        depth=depth, is_depth_valid=is_depth_valid
                    )
                    end=time.time()
                    logger.info(f"Time taken to create trajectory visualizations: {end-start:.4f}s")
                    
                    # Upload visualizations to wandb
                    if self.use_wandb and wandb.run and viz_save_dir is not None:
                        import glob
                        viz_images = sorted(glob.glob(str(viz_save_dir / "*.png")))
                        for viz_img_path in viz_images[:5]:  # Upload first 5 images to avoid clutter
                            img_name = Path(viz_img_path).name
                            wandb.log({
                                f"train_visualizations/{img_name}": wandb.Image(viz_img_path),
                                'epoch': epoch,
                                'step': epoch * len(self.train_loader) + batch_idx
                            })

            if self.is_main_process and batch_idx in checkpoint_intervals:
                progress_fraction = checkpoint_intervals.index(batch_idx) + 1
                logger.info(f"Saving intermediate checkpoint at {progress_fraction}/{self.num_log_steps_per_epoch} of epoch {epoch} (batch {batch_idx}/{total_batches})")
                self.save_checkpoint(epoch, is_best=False, is_final=False, progress_fraction=progress_fraction)

        # Return epoch metrics
        return {key: meter.avg for key, meter in meters.items()}

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()

        meters = {
            'total_loss': AverageMeter(),
            'diffusion_loss': AverageMeter(),
        }

        # For computing validation metrics
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            pbar = tqdm(enumerate(self.val_loader),
                total=len(self.val_loader),
                desc=f"Epoch {epoch:3d}/{self.cfg.train.epochs} [Val]",
                leave=False,
                disable=not should_show_progress_bar(),  # Disable in log files
                file=sys.stderr)  # Write to stderr if enabled

            for batch_idx, batch in pbar:
                # we filter out the data where valid_steps is smaller than 3, and  batch['movement_bool'].sum() is 0
                # Ensure all tensors are on the same device for the mask computation
                valid_mask = (batch['valid_steps'] > 3) & (batch['movement_bool'].sum(dim=1) > 0) & (batch['gt_mask'].sum(dim=1) > 0)
                
                # if number of valid_mask is zero, skip the batch and continue
                if valid_mask.sum() == 0:
                    continue
                
                # Apply mask to filter batch, handling both tensors and non-tensor data
                filtered_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        # For tensors, apply the mask on the same device
                        filtered_batch[k] = v[valid_mask.to(v.device)]
                    elif isinstance(v, list):
                        # For lists, apply the mask to get the corresponding elements
                        filtered_batch[k] = [v[i] for i in range(len(v)) if valid_mask[i].item()]
                    else:
                        # For other data types, keep as is (they might be scalars or other types)
                        filtered_batch[k] = v
                batch = filtered_batch

                images = batch['image'].to(self.device)
                texts = batch['text']
                gt_coords = batch['gt_xy'].to(self.device)
                gt_mask = batch['gt_mask'].to(self.device)
                target_trajectory = batch['trajectory'].to(self.device)
                depth = batch['depth'].to(self.device)
                is_depth_valid = batch['is_depth_valid']
                trajectory_mask = batch['trajectory_mask'].to(self.device)
                depth = batch['depth'].to(self.device)
                is_depth_valid = batch['is_depth_valid']
                movement_bool = batch['movement_bool'].to(self.device)

                targets = {
                    'gt_coords': gt_coords,
                    'gt_mask': gt_mask,
                    'trajectory': target_trajectory,
                    'trajectory_mask': trajectory_mask
                }

                model_forward = self.model.module if hasattr(self.model, "module") else self.model
                outputs = model_forward.forward_diffusion_training(
                    images=images,
                    texts=texts,
                    depth=depth,
                    is_depth_valid=is_depth_valid,
                    target_trajectory=target_trajectory,
                    first_keypoint=gt_coords,
                    diffusion_loss=self.criterion,
                )
                losses = self.criterion(outputs, targets, trajectory_mask)
                total_loss = losses["total_loss"]
                diffusion_loss = losses.get("diffusion_loss", total_loss)

                batch_size = images.size(0)
                meters["total_loss"].update(total_loss.item(), batch_size)
                meters["diffusion_loss"].update(diffusion_loss.item(), batch_size)

                if self.cfg.train.visualize_during_validation:
                    # Generate predictions for metrics
                    model_predict = self.model.module if hasattr(self.model, "module") else self.model

                    predicted_trajectory = model_predict.predict_trajectory(
                        images=images,
                        texts=texts,
                        depth=depth,
                        is_depth_valid=is_depth_valid,
                        first_keypoint=gt_coords,
                        noise_scheduler=self.criterion.noise_scheduler,
                        guidance_scale=1.0
                    )
                    
                    all_predictions.append(predicted_trajectory.cpu())
                    all_targets.append(target_trajectory.cpu())

                    pbar.set_postfix({
                        "loss": f"{total_loss.item():.4f}",
                        "diff_loss": f"{diffusion_loss.item():.4f}",
                    })

                    # Visualization (keep existing code)
                    if self.is_main_process and batch_idx == 0 and epoch % getattr(self.cfg.train, "visualize_every", 20) == 0:
                        # Use the predicted_trajectory we already computed
                        start=time.time()
                        val_viz_save_dir = self._create_trajectory_visualizations_v2(
                            batch, predicted_trajectory, target_trajectory, 
                            trajectory_mask, epoch, batch_idx, split="val_guidance_scale_1",
                            depth=depth, is_depth_valid=is_depth_valid
                        )
                        end=time.time()
                        logger.info(f"Time taken to create trajectory visualizations: {end-start:.4f}s")
                        
                        # Upload validation visualizations to wandb
                        if self.use_wandb and wandb.run and val_viz_save_dir is not None:
                            import glob
                            viz_images = sorted(glob.glob(str(val_viz_save_dir / "*.png")))
                            for viz_img_path in viz_images[:5]:  # Upload first 5 images
                                img_name = Path(viz_img_path).name
                                wandb.log({
                                    f"val_visualizations/{img_name}": wandb.Image(viz_img_path),
                                    'epoch': epoch,
                                    'step': epoch * len(self.val_loader) + batch_idx
                                })
        # Now trajectory_metrics will have actual values
        val_metrics = {
            "total_loss": meters["total_loss"].avg,
            "diffusion_loss": meters["diffusion_loss"].avg,
        }

        trajectory_metrics = self._compute_trajectory_validation_metrics(
            all_predictions, all_targets
        )
        val_metrics.update(trajectory_metrics)

        return val_metrics

    def _create_trajectory_visualizations_v2(
        self, batch, predicted_trajectory, target_trajectory,
        trajectory_mask, epoch, batch_idx, split="val",
        depth=None, is_depth_valid=None, cmap_name="magma",
    ):
        """
        Fast 2D-only plotting:
        - If depth & valid: color line by sampled depth (per-trajectory min-max)
        - Else: simple solid line
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        from matplotlib.collections import LineCollection
        from matplotlib import cm, colors as mcolors

        # (B,N,T,2)
        pred_xy   = predicted_trajectory[:, :, :, :2].detach().cpu()
        gt_xy     = target_trajectory[:, :, :, :2].detach().cpu()
        mask_cpu  = trajectory_mask.detach().cpu()

        # first-step from GT + cumsum for relative actions
        pred_full = torch.cat((gt_xy[:, :, :1], pred_xy), dim=2)
        if not self.cfg.absolute_action:
            gt_xy_cum   = torch.cumsum(gt_xy,   dim=2)
            pred_xy_cum = torch.cumsum(pred_full, dim=2)
        else:
            gt_xy_cum   = gt_xy
            pred_xy_cum = pred_full

        # depth (optional)
        has_depth = (depth is not None) and (is_depth_valid is not None)
        if has_depth:
            depth_cpu = depth.detach().cpu()
            if depth_cpu.ndim == 4 and depth_cpu.size(1) == 1:  # [B,1,H,W] -> [B,H,W]
                depth_cpu = depth_cpu[:, 0]
            assert depth_cpu.ndim == 3, "depth must be [B,H,W] or [B,1,H,W]"
            depth_valid_cpu = (is_depth_valid.detach().cpu() > 0).view(-1)

        batch_size = min(10, batch["image"].size(0))
        save_dir = self.checkpoint_dir / "trajectory_visualizations" / f"epoch_{epoch:03d}" / split
        save_dir.mkdir(parents=True, exist_ok=True)

        base_colors = ("g")
        cmap = cm.get_cmap(cmap_name)

        def add_depth_colored_line(ax, traj_px, depth_map, absolute_action):
            # traj_px: (T,2) in pixel coords
            T = traj_px.shape[0]
            if T < 2:
                return
            # segments: (T-1, 2, 2)
            segs = np.concatenate([traj_px[:-1, None, :], traj_px[1:, None, :]], axis=1)

            if depth_map is None:
                lc = LineCollection(segs, linewidths=1.6, alpha=0.95)
                ax.add_collection(lc)
                return

            # sample z at integer pixel coords
            xy = np.rint(traj_px).astype(np.int64)
            xs = np.clip(xy[:, 0], 0, depth_map.shape[1]-1)
            ys = np.clip(xy[:, 1], 0, depth_map.shape[0]-1)
            z  = depth_map[ys, xs].astype(np.float32)  # (T,)

            if not absolute_action:
                z = np.cumsum(z, axis=0)

            # segment depth = mean of endpoints (no sort/percentile)
            z_seg = (z[:-1] + z[1:]) * 0.5  # (T-1,)

            zmin = float(np.min(z_seg))
            zmax = float(np.max(z_seg))
            if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
                # fallback: solid line if degenerate
                lc = LineCollection(segs, linewidths=1.6, alpha=0.95)
                ax.add_collection(lc)
                return

            norm = mcolors.Normalize(vmin=zmin, vmax=zmax)
            colors = cmap(norm(z_seg))
            lc = LineCollection(segs, colors=colors, linewidths=1.8, alpha=0.98)
            ax.add_collection(lc)

        for i in range(batch_size):
            # background image (no copies beyond necessary)
            img_t = batch["image"][i].detach().cpu()
            if isinstance(img_t, torch.Tensor):
                img = img_t.permute(1, 2, 0).numpy()
                if img.min() < 0.0 or img.max() > 1.0:
                    img = img * np.array([0.229, 0.224, 0.225], dtype=img.dtype) + np.array([0.485, 0.456, 0.406], dtype=img.dtype)
                img = np.clip(img, 0.0, 1.0)
            else:
                img = img_t
            H, W = img.shape[:2]

            gt_px   = gt_xy_cum[i].numpy()
            pred_px = pred_xy_cum[i].numpy()
            gt_px[..., 0]   *= W; gt_px[..., 1]   *= H
            pred_px[..., 0] *= W; pred_px[..., 1] *= H

            depth_map_i = None
            if has_depth and bool(depth_valid_cpu[i]):
                depth_map_i = depth_cpu[i].numpy()

            N = min(gt_px.shape[0], pred_px.shape[0])

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.8))
            ax1.imshow(img); ax1.set_title(f"GT (E{epoch})", fontsize=10)
            ax2.imshow(img); ax2.set_title(f"Pred (E{epoch})", fontsize=10)

            for n in range(N):
                if mask_cpu[i, n, :, 0].sum() == 0:
                    break
                T = int(mask_cpu[i, n, :, 0].sum().item())
                if T < 2:
                    continue

                gt_n   = gt_px[n, :T]
                pred_n = pred_px[n, :T]

                
                c = base_colors[n % len(base_colors)]  # Normal color scheme

                if depth_map_i is None:
                    ax1.plot(gt_n[:, 0],   gt_n[:, 1],   "-", color=c, linewidth=1.6, alpha=0.95)
                    ax2.plot(pred_n[:, 0], pred_n[:, 1], "-", color=c, linewidth=1.6, alpha=0.95)
                else:
                    # For depth-colored lines, use the determined color
                    if c == "r":
                        # Force red color for no-movement trajectories
                        ax1.plot(gt_n[:, 0],   gt_n[:, 1],   "-", color=c, linewidth=1.6, alpha=0.95)
                        ax2.plot(pred_n[:, 0], pred_n[:, 1], "-", color=c, linewidth=1.6, alpha=0.95)
                    else:
                        # Use depth coloring for movement trajectories
                        add_depth_colored_line(ax1, gt_n,   depth_map_i, self.cfg.absolute_action)
                        add_depth_colored_line(ax2, pred_n, depth_map_i, self.cfg.absolute_action)

            for ax in (ax1, ax2):
                ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

            out = save_dir / f"traj_batch{batch_idx}_sample{i}_{split}.png"
            plt.savefig(out, dpi=70, bbox_inches="tight")
            plt.close('all')  # Explicitly close all figures
            plt.clf()  # Clear the current figure
            plt.cla()  # Clear the current axes
            import gc
            gc.collect()  # Force garbage collection after visualization

        logger.info(f"Saved {batch_size} {split} trajectory visualizations to {save_dir}")
        return save_dir
        
    def _compute_trajectory_validation_metrics(
        self, pred_trajectories_list, target_trajectories_list
    ) -> Dict[str, float]:
        """Compute trajectory-specific validation metrics."""
        if not pred_trajectories_list or not target_trajectories_list:
            logger.warning("No predictions or targets to compute metrics")
            return {}

        # Concatenate all predictions and targets
        all_pred_trajectories = torch.cat(pred_trajectories_list, dim=0)
        all_target_trajectories = torch.cat(target_trajectories_list, dim=0)

        # Debug shapes
        logger.info(f"Pred shape before processing: {all_pred_trajectories.shape}")
        logger.info(f"Target shape before processing: {all_target_trajectories.shape}")

        # RESHAPE predictions from [Total_B, 400, T, 2] to [Total_B, N, T, 2]
        # Assuming spatial grid is 20x20 = 400
        if all_pred_trajectories.dim() == 4 and all_pred_trajectories.size(1) == 400:
            # Predictions are in spatial grid format, need to reshape
            B = all_pred_trajectories.size(0)
            T = all_pred_trajectories.size(2)
            C = all_pred_trajectories.size(3)
            
            # Reshape from [B, 400, T, 2] to [B, 20, 20, T, 2] then to [B*400, T, 2]
            all_pred_trajectories = all_pred_trajectories.reshape(B * 400, T, C)
            
            # Do the same for targets - flatten N dimension
            if all_target_trajectories.dim() == 4:
                B_t = all_target_trajectories.size(0)
                N = all_target_trajectories.size(1)
                T_t = all_target_trajectories.size(2)
                C_t = all_target_trajectories.size(3)
                all_target_trajectories = all_target_trajectories.reshape(B_t * N, T_t, C_t)
        
        logger.info(f"Pred shape after reshape: {all_pred_trajectories.shape}")
        logger.info(f"Target shape after reshape: {all_target_trajectories.shape}")

        # Handle trajectory dimension - remove first keypoint if needed
        if all_target_trajectories.size(1) > all_pred_trajectories.size(1):
            # Remove first keypoint from targets
            all_target_trajectories = all_target_trajectories[:, 1:, :]

        # Ensure TEMPORAL dimension (dim 1) matches
        min_seq_len = min(
            all_pred_trajectories.size(1), all_target_trajectories.size(1)
        )
        all_pred_trajectories = all_pred_trajectories[:, :min_seq_len, :]
        all_target_trajectories = all_target_trajectories[:, :min_seq_len, :]

        # Ensure same number of samples
        min_samples = min(all_pred_trajectories.size(0), all_target_trajectories.size(0))
        all_pred_trajectories = all_pred_trajectories[:min_samples]
        all_target_trajectories = all_target_trajectories[:min_samples]

        logger.info(f"Final pred shape: {all_pred_trajectories.shape}")
        logger.info(f"Final target shape: {all_target_trajectories.shape}")

        # Compute overall trajectory MSE and MAE
        trajectory_mse = F.mse_loss(all_pred_trajectories, all_target_trajectories).item()
        trajectory_mae = F.l1_loss(all_pred_trajectories, all_target_trajectories).item()

        # Compute endpoint accuracy (last trajectory point)
        pred_endpoints = all_pred_trajectories[:, -1, :]
        target_endpoints = all_target_trajectories[:, -1, :]
        endpoint_mse = F.mse_loss(pred_endpoints, target_endpoints).item()

        # Compute first keypoint accuracy
        first_keypoint_mse = F.mse_loss(
            all_pred_trajectories[:, 0, :],
            all_target_trajectories[:, 0, :]
        ).item()

        metrics = {
            'trajectory_mse': trajectory_mse,
            'trajectory_mae': trajectory_mae,
            'endpoint_mse': endpoint_mse,
            'first_keypoint_mse': first_keypoint_mse
        }

        logger.info(f"Computed trajectory metrics from {min_samples} samples")

        return metrics

    def get_action_statistics(self):
        """Get statistics for action."""
        all_actions_min = torch.tensor([-0.05, -0.05, -0.04])
        all_actions_max = torch.tensor([0.05, 0.05, 0.04])
        return all_actions_min, all_actions_max
    
    def test_epoch(self, epoch: int, split: int = 999) -> Dict[str, float]:
        """Test on OOD dataset for one epoch."""
        
        # Check if test dataset path is configured
        test_path = getattr(self.cfg, 'test_path', None)
        if test_path is None:
            logger.warning("No test dataset path configured, skipping test epoch")
            return {}
        
        logger.info(f"Running test epoch on OOD dataset: {test_path}")
        self.model.eval()

        # Load test images and texts
        try:
            images, texts = load_images_and_texts(test_path, self.cfg)
            images = images.to(self.device)
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return {}
        
        # Use actual image dimensions for grid
        img_h, img_w = images.shape[2], images.shape[3]
        grid_size = 20
        
        # Create uniform grid queries in normalized [0, 1] space
        grid_queries = self._create_uniform_grid(grid_size)
        grid_queries = grid_queries.repeat(images.size(0), 1, 1, 1)  # [B, 400, 1, 2]

        is_depth_valid = torch.zeros(images.size(0), 1, 1, 1)
        depth = torch.zeros(images.size(0), 1, img_h, img_w)

        guidance_scale = 1.0
        
        with torch.no_grad():
            # Generate predictions
            if hasattr(self.model, "module"):
                predicted_trajectory = self.model.module.predict_trajectory(
                    images=images,
                    texts=texts,
                    depth=depth,
                    is_depth_valid=is_depth_valid,
                    first_keypoint=grid_queries.squeeze(2),  # [B, 400, 2]
                    noise_scheduler=self.criterion.noise_scheduler,
                    guidance_scale=guidance_scale
                )
            else:
                predicted_trajectory = self.model.predict_trajectory(
                    images=images,
                    texts=texts,
                    depth=depth,
                    is_depth_valid=is_depth_valid,
                    first_keypoint=grid_queries.squeeze(2),  # [B, 400, 2]
                    noise_scheduler=self.criterion.noise_scheduler,
                    guidance_scale=guidance_scale
                )
            
            # Create batch dict for visualization
            batch = {
                'image': images,
                'text': texts,
            }
            
            # Visualize
            save_dir = self.checkpoint_dir / "test_visualizations" / f"epoch_{epoch:03d}_{split:03d}"
            self._create_test_visualizations(
                batch, predicted_trajectory, grid_queries, 
                epoch, split=f"test_guidance_{guidance_scale}",
                save_dir=save_dir
            )

        return {} 

    def _create_test_visualizations(self, batch, predicted_trajectory, grid_queries, 
                                    epoch, split="test", save_dir=None):
        """Minimal plotting for test data (no ground truth)."""
        
        import matplotlib.pyplot as plt
        import numpy as np
        import torch
        
        pred_traj_cpu = predicted_trajectory[..., :2].cpu()
        grid_cpu = grid_queries.squeeze(2).cpu()  # [B, 400, 2]
        
        # Add first keypoint to predictions
        predicted_trajectory_full = torch.cat((grid_cpu.unsqueeze(2), pred_traj_cpu), dim=2)
        
        # Apply cumsum if using relative actions
        if not self.cfg.absolute_action:
            predicted_trajectory_cumsum = torch.cumsum(predicted_trajectory_full, dim=2)
        else:
            predicted_trajectory_cumsum = predicted_trajectory_full

        batch_size = min(10, batch['image'].size(0))
        
        # Create save directory
        if save_dir is None:
            save_dir = self.checkpoint_dir / "test_visualizations" / split
        save_dir.mkdir(parents=True, exist_ok=True)

        for i in range(batch_size):
            image = batch['image'][i].cpu()
            if isinstance(image, torch.Tensor):
                image_np = image.permute(1, 2, 0).detach().numpy()
                if image_np.min() < 0.0 or image_np.max() > 1.0:
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    image_np = image_np * std + mean
                image_np = np.clip(image_np, 0, 1)

            img_h, img_w = image_np.shape[:2]

            pred_traj = predicted_trajectory_cumsum[i].detach().numpy()
            
            # Convert to pixel space
            pred_traj_px = pred_traj.copy()
            pred_traj_px[..., 0] *= img_w
            pred_traj_px[..., 1] *= img_h

            N = pred_traj_px.shape[0]

            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

            ax.imshow(image_np)
            ax.set_title(f'Test Predictions (N={N})\nEpoch {epoch}, Sample {i}', fontsize=12)

            # Plot all trajectories
            for n in range(N):
                c = "g"  # Default to green
                
                traj = pred_traj_px[n]  # [T, 2]
                
                if traj.shape[0] >= 2:
                    # Only plot if trajectory has movement
                    movement = np.linalg.norm(traj[-1] - traj[0])
                    if movement > 1.0:  # Skip if less than 1 pixel movement
                        ax.plot(traj[:, 0], traj[:, 1], '-', 
                               color=c, linewidth=1.5, alpha=0.6)

            ax.set_xlim(0, img_w)
            ax.set_ylim(img_h, 0)
            ax.axis('off')
            
            # Add text
            if 'text' in batch and i < len(batch['text']):
                text = batch['text'][i]
                ax.text(10, 10, text, fontsize=12, color='white',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

            save_path = save_dir / f"test_sample{i}_epoch{epoch}.png"
            plt.savefig(save_path, dpi=100)
            plt.close('all')  # Explicitly close all figures
            plt.clf()  # Clear the current figure
            plt.cla()  # Clear the current axes
            import gc
            gc.collect()  # Force garbage collection after visualization
        
        logger.info(f"Saved {batch_size} test visualizations to {save_dir}")

    def _create_uniform_grid(self, grid_size=20):
        """Create uniform grid in normalized [0, 1] space."""
        # Create uniform grid
        y_coords = np.linspace(0, 1, grid_size)
        x_coords = np.linspace(0, 1, grid_size)
        
        # Create meshgrid
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # Flatten and create points [N, 2]
        grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)
        
        # Convert to tensor [1, N, 1, 2]
        grid_tensor = torch.tensor(
            grid_points, 
            dtype=torch.float32, 
            device=self.device
        ).unsqueeze(0).unsqueeze(2)
        
        return grid_tensor

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"Training for {self.cfg.train.epochs} epochs")
        logger.info(f"Device: {self.device}")
        
        for epoch in range(self.start_epoch, self.cfg.train.epochs):
            # Train epoch
            train_metrics = self.train_epoch(epoch)

            # Update train metrics history
            for key, value in train_metrics.items():
                if key in self.train_metrics:
                    self.train_metrics[key].append(value)

            # Validation
            if epoch % self.cfg.train.eval_every == 0 or epoch == self.cfg.train.epochs - 1:
                val_metrics = self.validate_epoch(epoch)
                
                # Update validation metrics history
                for key, value in val_metrics.items():
                    if key in self.val_metrics:
                        self.val_metrics[key].append(value)

                # Log main metrics
                logger.info(
                    f"Epoch {epoch} - Val Total Loss: {val_metrics['total_loss']:.4f}"
                )
                logger.info(f"  Diffusion Loss: {val_metrics['diffusion_loss']:.4f}")

                # Log trajectory metrics if available
                if "trajectory_mse" in val_metrics:
                    logger.info(
                        f"  Trajectory MSE: {val_metrics['trajectory_mse']:.6f}"
                    )
                if "trajectory_mae" in val_metrics:
                    logger.info(
                        f"  Trajectory MAE: {val_metrics['trajectory_mae']:.6f}"
                    )
                if "endpoint_mse" in val_metrics:
                    logger.info(f"  Endpoint MSE: {val_metrics['endpoint_mse']:.6f}")

                # Log to wandb
                if self.use_wandb and wandb.run:
                    log_dict = {f"val/{key}": value for key, value in val_metrics.items()}
                    log_dict['epoch'] = epoch
                    wandb.log(log_dict)

                # Check for best model using diffusion loss
                current_metric = -val_metrics["diffusion_loss"]
                is_best = current_metric > self.best_metric
                if is_best:
                    self.best_metric = current_metric
                    logger.info(
                        f"New best model! Diffusion Loss: {val_metrics['diffusion_loss']:.6f}"
                    )

                # Save checkpoint
                if epoch % self.cfg.train.save_every == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)

        logger.info("Training completed!")

        # Final checkpoint
        self.save_checkpoint(epoch, is_final=True)

    def save_checkpoint(self, epoch: int, is_best: bool = False, is_final: bool = False, progress_fraction: int = None):
        """Save model checkpoint."""
        if not self.is_main_process:  # ADD
            return  
        model_state_dict = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()

        # Convert DictToNamespace to dict for checkpoint saving (to avoid pickle issues)
        if isinstance(self.cfg, DictToNamespace):
            config_dict = namespace_to_dict(self.cfg)
        else:
            # If cfg is already a dict or other type, try to convert
            try:
                config_dict = namespace_to_dict(self.cfg)
            except:
                config_dict = self.cfg.__dict__ if hasattr(self.cfg, '__dict__') else self.cfg
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': config_dict,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'action_min': self.action_min,
            'action_max': self.action_max
        }

        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        # Save regular checkpoint
        if progress_fraction is not None:
            # Intermediate checkpoint during epoch
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}_progress_{progress_fraction}of{self.num_log_steps_per_epoch}.pth"
        else:
            # Regular checkpoint at end of epoch
            checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")

        # Save final model
        if is_final:
            final_path = self.checkpoint_dir / "final_model.pth"
            torch.save(checkpoint, final_path)
            logger.info(f"Final model saved to {final_path}")

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Temporarily add DictToNamespace to __main__ module for unpickling old checkpoints
        # This is needed because old checkpoints may contain DictToNamespace objects
        import sys
        if '__main__' in sys.modules:
            main_module = sys.modules['__main__']
            if not hasattr(main_module, 'DictToNamespace'):
                main_module.DictToNamespace = DictToNamespace
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Convert config from DictToNamespace to dict if needed (for compatibility)
        if 'config' in checkpoint and isinstance(checkpoint['config'], DictToNamespace):
            checkpoint['config'] = namespace_to_dict(checkpoint['config'])

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scaler state
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Load action statistics
        self.action_min = checkpoint.get('action_min', None)
        self.action_max = checkpoint.get('action_max', None)

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



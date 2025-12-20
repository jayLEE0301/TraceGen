import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

logger = logging.getLogger(__name__)


def load_images_and_texts(test_root: str, image_size: int, return_original_size: bool = False) -> Tuple[torch.Tensor, List[str], List[str]]:
    """Load images and texts from test dataset directory.
    
    Args:
        test_root: Root directory containing 'images' and 'texts' subdirectories
        image_size: Target image size (assumed square, e.g., 384)
        return_original_size: Whether to return the original size of the image before resizing
    Returns:
        images: torch.Tensor [B, 3, H, W] - normalized images in [0, 1]
        texts: List[str] - text descriptions for each image
        image_files: List[str] - paths to image files that were actually loaded (sorted, filtered)
        original_size: Tuple[int, int] - original size of the image before resizing
    """
    image_dir = os.path.join(test_root, "images")
    text_dir = os.path.join(test_root, "texts")
    label_file = os.path.join(text_dir, "label.json")

    with open(label_file, "r") as f:
        label_json = json.load(f)

    # Collect image paths and sort to ensure consistent ordering
    image_files = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    image_files.sort()  # Ensure consistent ordering

    images = []
    texts = []
    loaded_image_files = []  # Track which files were actually loaded

    # Get target image size
    if isinstance(image_size, int):
        target_size = (image_size, image_size)
    else:
        target_size = image_size

    for path in image_files:
        key = os.path.splitext(os.path.basename(path))[0]
        if key not in label_json:
            continue  # skip if no label

        # Load and process image
        pil_image = Image.open(path).convert("RGB")
        resized_image = TF.resize(pil_image, target_size, Image.BILINEAR)
        image_tensor = TF.to_tensor(resized_image)  # [3, H, W] in [0, 1]

        images.append(image_tensor)
        texts.append(label_json[key])
        loaded_image_files.append(path)  # Track this file as loaded

    # Stack tensors
    images_tensor = torch.stack(images, dim=0)  # [B, 3, h, w]
    if return_original_size:
        original_size = (pil_image.width, pil_image.height)
        # breakpoint()
        return images_tensor, texts, loaded_image_files, original_size
    else:
        return images_tensor, texts, loaded_image_files


def load_depth(test_root: str, image_files: List[str], img_h: int, img_w: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load depth maps for test images.
    
    Loads both sensored_depth/ (PNG files) and depth/ (NPZ files) directories separately.
    
    Args:
        test_root: Root directory of test dataset
        image_files: List of image file paths (sorted)
        img_h: Target image height
        img_w: Target image width
        
    Returns:
        depth_tensor: torch.Tensor [B, 1, H, W] or None - regular depth from depth/ directory (NPZ)
        is_depth_valid: torch.Tensor [B] or None - validity flags for regular depth
        sensored_depth_tensor: torch.Tensor [B, 1, H, W] or None - sensored depth from sensored_depth/ directory (PNG)
        is_sensored_depth_valid: torch.Tensor [B] or None - validity flags for sensored depth
    """
    sensored_depth_dir = os.path.join(test_root, "sensored_depth")
    depth_dir = os.path.join(test_root, "depth")
    
    # Check which depth directories exist
    use_sensored_depth = os.path.exists(sensored_depth_dir)
    use_regular_depth = os.path.exists(depth_dir)
    
    if not use_sensored_depth and not use_regular_depth:
        logger.info("No depth directory found (neither sensored_depth nor depth), returning None")
        return None, None, None, None
    
    target_size = (img_h, img_w)
    
    # Lists for regular depth
    depth_list = []
    is_depth_valid_list = []
    
    # Lists for sensored depth
    sensored_depth_list = []
    is_sensored_depth_valid_list = []
    
    for img_path in image_files:
        key = os.path.splitext(os.path.basename(img_path))[0]
        
        # Load sensored_depth (PNG files with same name as image)
        if use_sensored_depth:
            sensored_depth_path = os.path.join(sensored_depth_dir, f"{key}.npz")
            
            if os.path.exists(sensored_depth_path):
                try:
                    depth_data = np.load(sensored_depth_path)
                    
                    # Get the depth array (might be in different keys)
                    if 'depth' in depth_data:
                        depth_array = depth_data['depth']
                        # breakpoint()
                    else:
                        logger.warning(f"Depth key not found in {sensored_depth_path}, using dummy depth")
                        resize_depth_tensor = torch.zeros(1, img_h, img_w)
                        sensored_depth_list.append(resize_depth_tensor)
                        is_sensored_depth_valid_list.append(False)
                        continue
                    
                    # Convert numpy array to tensor
                    depth_tensor = torch.from_numpy(depth_array).float()
                    
                    # Add channel dimension if needed to make it [1, H, W]
                    if depth_tensor.ndim == 2:
                        depth_tensor = depth_tensor.unsqueeze(0)  # [1, H, W]
                    
                    # Resize
                    resize_depth_tensor = TF.resize(depth_tensor, target_size, Image.BILINEAR)
                    sensored_depth_list.append(resize_depth_tensor)
                    is_sensored_depth_valid_list.append(True)
                    
                except Exception as e:
                    logger.warning(f"Failed to load depth from {sensored_depth_path}: {e}")
                    resize_depth_tensor = torch.zeros(1, img_h, img_w)
                    sensored_depth_list.append(resize_depth_tensor)
                    is_sensored_depth_valid_list.append(False)
            else:
                logger.debug(f"Depth file not found: {sensored_depth_path}")
                resize_depth_tensor = torch.zeros(1, img_h, img_w)
                sensored_depth_list.append(resize_depth_tensor)
                is_sensored_depth_valid_list.append(False)
        
        # Load regular depth (NPZ files)
        if use_regular_depth:
            depth_path = os.path.join(depth_dir, f"{key}.npz")
            
            if os.path.exists(depth_path):
                try:
                    depth_data = np.load(depth_path)
                    
                    # Get the depth array (might be in different keys)
                    if 'depth' in depth_data:
                        depth_array = depth_data['depth']
                    else:
                        logger.warning(f"Depth key not found in {depth_path}, using dummy depth")
                        resize_depth_tensor = torch.zeros(1, img_h, img_w)
                        depth_list.append(resize_depth_tensor)
                        is_depth_valid_list.append(False)
                        continue
                    
                    # Convert numpy array to tensor
                    depth_tensor = torch.from_numpy(depth_array).float()
                    
                    # Add channel dimension if needed to make it [1, H, W]
                    if depth_tensor.ndim == 2:
                        depth_tensor = depth_tensor.unsqueeze(0)  # [1, H, W]
                    
                    # Resize
                    resize_depth_tensor = TF.resize(depth_tensor, target_size, Image.BILINEAR)
                    depth_list.append(resize_depth_tensor)
                    is_depth_valid_list.append(True)
                    
                except Exception as e:
                    logger.warning(f"Failed to load depth from {depth_path}: {e}")
                    resize_depth_tensor = torch.zeros(1, img_h, img_w)
                    depth_list.append(resize_depth_tensor)
                    is_depth_valid_list.append(False)
            else:
                logger.debug(f"Depth file not found: {depth_path}")
                resize_depth_tensor = torch.zeros(1, img_h, img_w)
                depth_list.append(resize_depth_tensor)
                is_depth_valid_list.append(False)
    
    # Stack tensors (we always process all images, so lists will always have length > 0)
    depth_tensor = torch.stack(depth_list, dim=0)  # [B, 1, H, W]
    is_depth_valid = torch.tensor(is_depth_valid_list, dtype=torch.float32)  # [B]
    
    sensored_depth_tensor = torch.stack(sensored_depth_list, dim=0)  # [B, 1, H, W]
    is_sensored_depth_valid = torch.tensor(is_sensored_depth_valid_list, dtype=torch.float32)  # [B]
    
    # Log results
    if use_regular_depth:
        valid_count = is_depth_valid.sum().item() if is_depth_valid is not None else 0
        logger.info(f"Loaded regular depth maps: {valid_count}/{len(image_files)} valid")
    
    if use_sensored_depth:
        sensored_valid_count = is_sensored_depth_valid.sum().item() if is_sensored_depth_valid is not None else 0
        logger.info(f"Loaded sensored_depth maps: {sensored_valid_count}/{len(image_files)} valid")
    
    return depth_tensor, is_depth_valid, sensored_depth_tensor, is_sensored_depth_valid


def create_uniform_grid_points(height: int, width: int, grid_size: int = 20, device: str = 'cuda') -> torch.Tensor:
    """Create uniform grid points across the image.

    Args:
        height: Image height
        width: Image width
        grid_size: Grid size (grid_size x grid_size)
        device: Device for tensor

    Returns:
        torch.Tensor: Grid points [1, grid_size*grid_size, 3] where each point is [t, x, y]
    """
    # Create uniform grid
    y_coords = np.linspace(0, height - 1, grid_size)
    x_coords = np.linspace(0, width - 1, grid_size)

    # Create meshgrid
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Flatten and create points [N, 2]
    grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)

    # Add time dimension (t=0 for all points) -> [N, 3]
    time_col = np.zeros((grid_points.shape[0], 1))
    grid_points_3d = np.concatenate([time_col, grid_points], axis=1)

    # Convert to tensor and add batch dimension -> [1, N, 3]
    grid_tensor = torch.tensor(grid_points_3d, dtype=torch.float32, device=device).unsqueeze(0)

    return grid_tensor


def sample_depth_from_grid(grid_points: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
    """Sample depth values from depth map using grid coordinates.
    
    Args:
        grid_points: [B, N, 1, 2] normalized coordinates in [0, 1] range
        depth_map: [B, 1, H, W] depth map
        
    Returns:
        depth_values: [B, N, 1] sampled depth values
    """
    # Convert from [0, 1] to [-1, 1] for grid_sample
    grid_normalized = (grid_points * 2.0 - 1.0).clone()
    
    # Sample depth values using bilinear interpolation
    depth_values = F.grid_sample(
        depth_map,
        grid_normalized,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    )  # [B, 1, N, 1]
    
    # Squeeze to [B, N, 1]
    depth_values = depth_values.squeeze(1).squeeze(-1).unsqueeze(-1)  # [B, N, 1]
    
    return depth_values


def prepare_grid_queries_with_depth(
    images: torch.Tensor,
    depth_maps: Optional[torch.Tensor],
    grid_size: int = 20,
    device: str = 'cuda'
) -> torch.Tensor:
    """Prepare grid queries with optional depth sampling.
    
    Args:
        images: [B, 3, H, W] input images
        depth_maps: [B, 1, H, W] depth maps or None
        grid_size: Grid size (default 20x20 = 400 points)
        device: Device for tensors
        
    Returns:
        grid_queries: [B, N, 1, 3] grid queries with [x, y, depth] normalized to [0, 1]
    """
    img_h, img_w = images.shape[2], images.shape[3]
    batch_size = images.shape[0]
    
    # Create 2D grid queries first (normalized [0, 1])
    grid_full = create_uniform_grid_points(img_h, img_w, grid_size=grid_size, device=device)  # [1, N, 3]
    grid_xy = grid_full[:, :, 1:]  # [1, N, 2] - x, y coordinates
    # Normalize x by width and y by height
    grid_xy_normalized = grid_xy.clone()
    grid_xy_normalized[:, :, 0] = grid_xy[:, :, 0] / img_w  # normalize x
    grid_xy_normalized[:, :, 1] = grid_xy[:, :, 1] / img_h  # normalize y
    grid_queries_2d = grid_xy_normalized.unsqueeze(2)  # [1, N, 1, 2] normalized to [0, 1]
    
    # Repeat for batch
    grid_queries_2d = grid_queries_2d.repeat(batch_size, 1, 1, 1)  # [B, N, 1, 2]
    
    # Sample depth if available
    if depth_maps is not None:
        grid_with_depth = sample_depth_from_grid(grid_queries_2d, depth_maps)  # [B, N, 1]
        # Concatenate depth as third dimension
        grid_queries = torch.cat([grid_queries_2d.squeeze(2), grid_with_depth], dim=-1).unsqueeze(2)  # [B, N, 1, 3]
    else:
        # No depth, use zeros for depth dimension
        zeros_depth = torch.zeros(batch_size, grid_size * grid_size, 1, device=device)
        grid_queries = torch.cat([grid_queries_2d.squeeze(2), zeros_depth], dim=-1).unsqueeze(2)  # [B, N, 1, 3]
    
    return grid_queries


def predict_trajectory_simple(
    trainer,
    images: torch.Tensor,
    texts: List[str],
    depth_maps: Optional[torch.Tensor] = None,
    is_depth_valid: Optional[torch.Tensor] = None,
    test_path: Optional[str] = None,
    image_files: Optional[List[str]] = None,
    guidance_scale: float = 1.0,
    grid_size: int = 20
) -> torch.Tensor:
    """Simple API to predict trajectory from images and texts.
    
    This function encapsulates:
    - Grid query preparation with optional depth
    - Movement bool computation (if needed)
    - Model inference
    
    Args:
        trainer: TrajectoryDiffusionTrainer instance with loaded model
        images: [B, 3, H, W] input images - REQUIRED
        texts: List[str] text descriptions - REQUIRED
        depth_maps: [B, 1, H, W] depth maps - OPTIONAL
        is_depth_valid: [B] depth validity flags - OPTIONAL
        test_path: Path to test dataset (for movement bool computation) - OPTIONAL
        image_files: List of image file paths (for movement bool computation) - OPTIONAL
        guidance_scale: Guidance scale for inference (default: 1.0)
        grid_size: Grid size for trajectory points (default: 20x20 = 400 points)
        
    Returns:
        predicted_trajectory: [B, N, T, 2] predicted trajectory
    """
    # Run model inference
    with torch.no_grad():
        predicted_trajectory = trainer.model.predict_trajectory(
            images=images,
            texts=texts,
            depth=depth_maps,
            is_depth_valid=is_depth_valid,
            first_keypoint=None,
            noise_scheduler=trainer.criterion.noise_scheduler,
            guidance_scale=guidance_scale
        )
    
    return predicted_trajectory


def create_trajectory_visualization(
    images: torch.Tensor,
    texts: list,
    predicted_trajectory: torch.Tensor,
    depth_maps: torch.Tensor = None,
    is_depth_valid: torch.Tensor = None,
    output_dir: str = "./test_result",
    split: str = "test",
    absolute_action: bool = False,
    trainer=None,
    test_path: str = None,
    image_files: list = None,
    grid_size: int = 20,
    movement_threshold: float = 0.2,
    max_trajectories: int = 50,
    filename_suffix: str = None,
    sample_index_in_filename: int = None,
    get_reconstructed_trajectory: bool = False,
    original_size: Tuple[int, int] = None,
    depth_ratio_filled: np.ndarray = None
):
    """Create trajectory visualization showing filtered trajectories (traces only).
    
    This function handles all internal preparation (grid queries, etc.).
    Only trajectories with total movement > movement_threshold are shown, and a random
    sample of up to max_trajectories is displayed.
    
    Args:
        images: [B, 3, H, W] input images - REQUIRED
        texts: List[str] text descriptions - REQUIRED
        predicted_trajectory: [B, N, T, 2] predicted trajectory - REQUIRED
        depth_maps: [B, 1, H, W] optional depth maps
        is_depth_valid: [B] optional depth validity flags
        output_dir: Output directory for visualizations
        split: Split name for saving
        absolute_action: Whether to use absolute action
        trainer: TrajectoryDiffusionTrainer instance
        test_path: Path to test dataset
        image_files: List of image file paths
        grid_size: Grid size for trajectory points (default: 20x20 = 400 points)
        movement_threshold: Minimum total movement to include trajectory (default: 0.2)
        max_trajectories: Maximum number of trajectories to show (default: 50)
        filename_suffix: Optional custom suffix to append to filename (default: None, uses split name)
        sample_index_in_filename: Optional sample index to use in filename (default: None, uses batch index i)
    """
    import random
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm, colors as mcolors
    from matplotlib.collections import LineCollection
    
    device = images.device
    
    # Prepare grid queries with optional depth
    grid_queries = prepare_grid_queries_with_depth(
        images, depth_maps, grid_size=grid_size, device=device
    )

    # Combine grid with predicted trajectory
    # grid_queries is [B, N, 1, 3], extract first time step and x,y coordinates: [B, N, 1, 2]
    grid_xy = grid_queries[:, :, :1, :2]  # [B, N, 1, 2]
    # predicted_trajectory might be [B, N, T, 2] or [B, N, T, 3], extract only x,y coordinates
    if predicted_trajectory.shape[-1] > 2:
        predicted_trajectory_xy = predicted_trajectory[..., :2]  # [B, N, T, 2]
    else:
        predicted_trajectory_xy = predicted_trajectory  # [B, N, T, 2]
    # Concatenate along time dimension (dim=2)
    pred_traj_with_grid = torch.cat((grid_xy, predicted_trajectory_xy), dim=2)  # [B, N, 1+T, 2]
    
    if not absolute_action:
        pred_traj_with_grid = torch.cumsum(pred_traj_with_grid, dim=2)

    if get_reconstructed_trajectory:
        # breakpoint()

        img_h, img_w = images[0].shape[1], images[0].shape[2]

        # Denormalize trajectories to pixel space for visualization
        pred_traj_px = pred_traj_with_grid[0].detach().cpu().numpy()
        pred_traj_px[..., 0] *= img_w  # x
        pred_traj_px[..., 1] *= img_h  # y

        # Sample depth at trajectory points
        # breakpoint()
        xy = np.rint(pred_traj_px).astype(np.int64)
        xs = np.clip(xy[:, 0, 0], 0, img_w - 1)
        ys = np.clip(xy[:, 0, 1], 0, img_h - 1)
        z = depth_maps.detach().cpu().numpy()[0, 0, ys, xs].astype(np.float32)

        final_traj_z = np.cumsum(np.concatenate([z.reshape(1, -1, 1), predicted_trajectory[..., 2].cpu().numpy()], axis=2), axis=2)
        final_traj = np.concatenate([pred_traj_px, np.expand_dims(final_traj_z[0], axis=2)], axis=2)

        ########## z scaling ##########
        # 1. make all tensors numpy
        if isinstance(final_traj, torch.Tensor):
            traj_np = final_traj.detach().cpu().numpy()
        else:
            traj_np = final_traj  # already numpy

        if isinstance(depth_ratio_filled, torch.Tensor):
            ratio_np = depth_ratio_filled.detach().cpu().numpy()
        else:
            ratio_np = depth_ratio_filled  # already numpy

        H, W = ratio_np.shape  # 384, 384

        # 2. extract x,y (shape: (400, 21))
        xs = traj_np[:, :, 0]
        ys = traj_np[:, :, 1]

        # 3. create pixel indices (rounded to int)
        ix = np.rint(xs).astype(np.int32)
        iy = np.rint(ys).astype(np.int32)

        # 4. clamp if boundary is exceeded
        ix = np.clip(ix, 0, W - 1)
        iy = np.clip(iy, 0, H - 1)

        # 5. sample ratio values
        #    ratio_np[iy, ix] has shape (400, 21) (no broadcasting, integer array indexing)
        ratio_at_points = ratio_np[iy, ix]  # shape (400, 21)

        # 6. extract z
        z_orig = traj_np[:, :, 2]  # (400, 21)

        # 7. new z = original z * ratio
        z_scaled = z_orig * ratio_at_points  # (400, 21)

        # 8. to reflect the original traj:
        traj_new = traj_np.copy()
        traj_new[:, :, 2] = z_scaled
        ########## z scaling ##########

        traj_new[:, :, 0] = traj_new[:, :, 0] * original_size[0] / img_w
        traj_new[:, :, 1] = traj_new[:, :, 1] * original_size[1] / img_h
        
        return traj_new
    
    # Depth handling
    has_depth = (depth_maps is not None) and (is_depth_valid is not None)
    if has_depth:
        depth_cpu = depth_maps.detach().cpu()
        if depth_cpu.ndim == 4 and depth_cpu.size(1) == 1:  # [B,1,H,W] -> [B,H,W]
            depth_cpu = depth_cpu[:, 0]
        depth_valid_cpu = (is_depth_valid.detach().cpu() > 0).view(-1)
    
    # Setup colormap for depth visualization
    cmap_name = "magma"
    cmap = cm.get_cmap(cmap_name)
    
    def add_depth_colored_line(ax, traj_px, depth_map, absolute_action):
        """Add depth-colored line to axis."""
        T = traj_px.shape[0]
        if T < 2:
            return
        segs = np.concatenate([traj_px[:-1, None, :], traj_px[1:, None, :]], axis=1)
        
        if depth_map is None:
            lc = LineCollection(segs, linewidths=1.6, alpha=0.95)
            ax.add_collection(lc)
            return
        
        # Sample depth at trajectory points
        xy = np.rint(traj_px).astype(np.int64)
        xs = np.clip(xy[:, 0], 0, depth_map.shape[1] - 1)
        ys = np.clip(xy[:, 1], 0, depth_map.shape[0] - 1)
        z = depth_map[ys, xs].astype(np.float32)
        
        if not absolute_action:
            z = np.cumsum(z, axis=0)
        
        z_seg = (z[:-1] + z[1:]) * 0.5
        
        zmin = float(np.min(z_seg))
        zmax = float(np.max(z_seg))
        if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
            lc = LineCollection(segs, linewidths=1.6, alpha=0.95)
            ax.add_collection(lc)
            return
        
        norm = mcolors.Normalize(vmin=zmin, vmax=zmax)
        colors = cmap(norm(z_seg))
        lc = LineCollection(segs, colors=colors, linewidths=1.8, alpha=0.98)
        ax.add_collection(lc)
    
    # Handle split path: if it contains "/", use first part as directory
    if "/" in split:
        split_parts = split.split("/", 1)
        split_dir = split_parts[0]
        split_name_for_file = split_parts[1] if len(split_parts) > 1 else split_parts[0]
    else:
        split_dir = split
        split_name_for_file = split
    
    # Create save directory
    save_dir = Path(output_dir) / "trajectory_visualizations" / split_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    
    batch_size = images.size(0)
    
    for i in range(batch_size):
        # Image processing
        image = images[i]  # [3, H, W]
        text = texts[i]
        
        if isinstance(image, torch.Tensor):
            image_np = image.permute(1, 2, 0).detach().cpu().numpy()
            if image_np.min() < 0.0 or image_np.max() > 1.0:
                mean = np.array([0.485, 0.456, 0.406], dtype=image_np.dtype)
                std = np.array([0.229, 0.224, 0.225], dtype=image_np.dtype)
                image_np = image_np * std + mean
            image_np = np.clip(image_np, 0, 1)
        else:
            image_np = image
        
        img_h, img_w = image_np.shape[:2]
        
        # Extract trajectories
        pred_traj = pred_traj_with_grid[i].detach().cpu().numpy()  # [N, T, 2]
        if pred_traj.shape[-1] > 2:
            pred_traj = pred_traj[..., :2]
        
        N = pred_traj.shape[0]
        
        # Calculate total movement for each trajectory in normalized space
        # Movement is the sum of distances between consecutive points
        trajectory_movements = []
        valid_trajectory_indices = []
        
        for n in range(N):
            traj_length = pred_traj.shape[1]
            traj = pred_traj[n, :traj_length]  # [T, 2] in normalized [0, 1] space
            
            if traj.shape[0] >= 2:
                # Calculate total movement as sum of distances between consecutive points
                if absolute_action:
                    # For absolute action, movement is distance from start to end
                    total_movement = np.linalg.norm(traj[-1] - traj[0])
                else:
                    # For relative action, sum all step distances
                    diffs = np.diff(traj, axis=0)  # [T-1, 2]
                    distances = np.linalg.norm(diffs, axis=1)  # [T-1]
                    total_movement = np.sum(distances)
                
                trajectory_movements.append(total_movement)
                valid_trajectory_indices.append(n)
        
        # Filter trajectories with movement > threshold
        filtered_indices = [
            idx for idx, mov in zip(valid_trajectory_indices, trajectory_movements)
            if mov > movement_threshold
        ]
        
        # Randomly sample from filtered trajectories
        if len(filtered_indices) > max_trajectories:
            filtered_indices = random.sample(filtered_indices, max_trajectories)
        
        num_shown = len(filtered_indices)
        
        # Get depth map for this sample
        depth_map_i = None
        if has_depth and i < len(depth_valid_cpu) and bool(depth_valid_cpu[i]):
            depth_map_i = depth_cpu[i].numpy()
        
        # Create figure (only left panel)
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        
        ax.imshow(image_np)
        ax.set_title(f"Filtered Trajectories (N={num_shown}/{N})\nSample {i} ({split.upper()})")
        
        # Denormalize trajectories to pixel space for visualization
        pred_traj_px = pred_traj.copy()
        pred_traj_px[..., 0] *= img_w  # x
        pred_traj_px[..., 1] *= img_h  # y
        
        # Plot filtered and sampled trajectories (traces only, no markers)
        for n in filtered_indices:
            traj_length = pred_traj_px.shape[1]
            traj = pred_traj_px[n, :traj_length]  # [T, 2] in pixel space
            
            if traj.shape[0] >= 2:
                # Plot trajectory line only (no markers)
                if depth_map_i is not None:
                    add_depth_colored_line(ax, traj, depth_map_i, absolute_action)
                else:
                    c = "g"  # Default to green
                    ax.plot(traj[:, 0], traj[:, 1], '-', color=c, linewidth=2, alpha=0.8)
        
        # Axes setup
        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)  # image coords
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Simplified legend (trajectory line only)
        import matplotlib.lines as mlines
        line_legend = mlines.Line2D([], [], color='black', linestyle='-', linewidth=2, 
                                    label='Trajectory')
        ax.legend(handles=[line_legend], loc='lower right')
        
        # Visualize text on the image
        ax.text(10, 10, text, fontsize=12, color='white', 
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        plt.tight_layout()
        
        # Use sample_index_in_filename if provided, otherwise use batch index i
        sample_idx_for_file = sample_index_in_filename if sample_index_in_filename is not None else i
        
        # Use filename_suffix if provided, otherwise use split_name_for_file
        name_suffix = filename_suffix if filename_suffix is not None else split_name_for_file
        save_path = save_dir / f"traj_comparison_sample{sample_idx_for_file}_N{N}_{name_suffix}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization: {save_path}")

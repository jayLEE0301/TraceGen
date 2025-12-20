import hashlib
import json
import logging
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

DEFAULT_TEXT_DESCRIPTION = "Can you move the robot gripper or human hand to solve the task?"


class EpisodePointDataset(Dataset):
    """
    Dataset with metadata caching for faster initialization.
    
    New args:
        use_cache: Whether to use cached metadata (default: True)
        cache_dir: Directory to store cache files (default: ./cache)
        force_refresh: Force rebuild cache even if it exists
    """

    def __init__(
        self,
        dataset_dirs: List[str],
        split: str,
        transform: Optional[callable] = None,
        cfg=None,
        max_episodes: Optional[int] = None,
        cache_text: bool = True,
        trajectory_horizon: int = 32,
        absolute_action: bool = False,
        val_split: float = 0.1,
        random_seed: int = 42,
        use_cache: bool = True,
        cache_dir: str = "./cache",
        force_refresh: bool = False
    ):
        self.dataset_dirs = [Path(d) for d in dataset_dirs]
        self.split = split
        self.transform = transform
        self.cfg = cfg
        self.cache_text = cache_text
        self.trajectory_horizon = trajectory_horizon
        self.absolute_action = absolute_action
        self.val_split = val_split
        self.random_seed = random_seed
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.force_refresh = force_refresh

        # Create cache directory
        if self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Validate dataset directories
        for dataset_dir in self.dataset_dirs:
            if not dataset_dir.exists():
                raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

        # Episode metadata for lazy loading
        self.episode_metadata = []
        self.sample_indices = []
        self.text_cache = {}

        # Generate cache key based on dataset directories and split settings
        cache_key = self._generate_cache_key()
        cache_file = self.cache_dir / f"metadata_{cache_key}.pkl"

        # Try to load from cache
        if self.use_cache and not self.force_refresh and cache_file.exists():
            logger.info(f"Loading metadata from cache: {cache_file}")
            try:
                cache_data = self._load_cache(cache_file)
                
                # Validate cache is still valid
                if self._validate_cache(cache_data):
                    self._restore_from_cache(cache_data)
                    logger.info("Cache loaded successfully!")
                    logger.info("Loaded %d episodes for %s split", len(self.episodes), split)
                    logger.info("Total samples: %d", len(self.sample_indices))
                    
                    # Validate num_kps
                    if cfg is not None:
                        self.num_kps = cfg.num_kps
                    else:
                        self.num_kps = 400
                    return
                else:
                    logger.warning("Cache validation failed, rebuilding...")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, rebuilding...")

        # Build metadata from scratch
        logger.info(f"Scanning episodes from {len(self.dataset_dirs)} dataset directories")
        
        # Collect all episodes and create train/val split
        all_episodes = self._collect_all_episodes()
        if max_episodes is not None:
            all_episodes = all_episodes[:max_episodes]

        # Split episodes into train/val
        train_episodes, val_episodes = self._split_episodes(all_episodes)

        if self.split == 'train':
            self.episodes = train_episodes
        elif self.split == 'val':
            self.episodes = val_episodes
        else:
            raise ValueError(f"Unsupported split: {self.split}")

        # Build episode metadata for lazy loading
        logger.info(f"Building metadata for {len(self.episodes)} episodes...")
        self._build_episode_metadata()

        logger.info(f"Loaded {len(self.episodes)} episodes for {split} split")
        logger.info(f"Total samples: {len(self.sample_indices)}")

        # Save to cache
        if self.use_cache:
            logger.info(f"Saving metadata to cache: {cache_file}")
            self._save_cache(cache_file, all_episodes, train_episodes, val_episodes)

        # Validate num_kps
        if cfg is not None:
            self.num_kps = cfg.num_kps
            self.trajectory_horizon = cfg.trajectory_horizon
        else:
            self.num_kps = 400
            self.trajectory_horizon = 64

    def _generate_cache_key(self) -> str:
        """Generate a unique cache key based on dataset configuration."""
        # Create a string representation of the configuration
        config_str = (
            f"{sorted([str(d) for d in self.dataset_dirs])}_"
            f"{self.split}_{self.val_split}_{self.random_seed}"
        )
        # Hash it for a shorter filename
        return hashlib.md5(config_str.encode()).hexdigest()

    def _save_cache(self, cache_file: Path, all_episodes: List[Path], 
                    train_episodes: List[Path], val_episodes: List[Path]):
        """Save metadata to cache file."""
        try:
            cache_data = {
                'version': '1.0',
                'timestamp': datetime.now().isoformat(),
                'dataset_dirs': [str(d) for d in self.dataset_dirs],
                'split': self.split,
                'val_split': self.val_split,
                'random_seed': self.random_seed,
                'all_episodes': [str(e) for e in all_episodes],
                'train_episodes': [str(e) for e in train_episodes],
                'val_episodes': [str(e) for e in val_episodes],
                'episode_metadata': self._serialize_metadata(),
                'sample_indices': self.sample_indices,
                'text_cache': self.text_cache
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.info(f"Cache saved successfully: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _load_cache(self, cache_file: Path) -> Dict:
        """Load metadata from cache file."""
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    def _validate_cache(self, cache_data: Dict) -> bool:
        """Validate that cached data is still valid."""
        try:
            # Check version
            if cache_data.get('version') != '1.0':
                logger.warning("Cache version mismatch")
                return False
            
            # Check configuration matches
            if cache_data['split'] != self.split:
                return False
            if cache_data['val_split'] != self.val_split:
                return False
            if cache_data['random_seed'] != self.random_seed:
                return False
            
            # Check dataset directories match
            cached_dirs = set(cache_data['dataset_dirs'])
            current_dirs = set(str(d) for d in self.dataset_dirs)
            if cached_dirs != current_dirs:
                logger.warning("Dataset directories changed")
                return False
            
            # Check if episode directories still exist (sample check)
            if self.split == 'train':
                episodes_to_check = cache_data['train_episodes'][:10]
            else:
                episodes_to_check = cache_data['val_episodes'][:10]
            
            for ep_path in episodes_to_check:
                if not Path(ep_path).exists():
                    logger.warning(f"Episode directory no longer exists: {ep_path}")
                    return False
            
            return True
        except Exception as e:
            logger.warning(f"Cache validation error: {e}")
            return False

    def _restore_from_cache(self, cache_data: Dict):
        """Restore dataset state from cache."""
        # Restore episodes
        if self.split == 'train':
            self.episodes = [Path(e) for e in cache_data['train_episodes']]
        else:
            self.episodes = [Path(e) for e in cache_data['val_episodes']]
        
        # Restore metadata
        self.episode_metadata = self._deserialize_metadata(cache_data['episode_metadata'])
        self.sample_indices = cache_data['sample_indices']
        self.text_cache = cache_data['text_cache']

    def _serialize_metadata(self) -> List[Dict]:
        """Convert metadata to serializable format."""
        serialized = []
        for metadata in self.episode_metadata:
            serialized.append({
                'episode_dir': str(metadata['episode_dir']),
                'img_dir': str(metadata['img_dir']),
                'npz_dir': str(metadata['npz_dir']),
                'all_data_dir': str(metadata['all_data_dir']) if metadata['all_data_dir'] is not None else None,
                'use_all_data': metadata['use_all_data'],
                'text_file': str(metadata['text_file']),
                'valid_pairs': [(str(img) if img is not None else img, str(npz), str(depth) if depth is not None else depth) for img, npz, depth in metadata['valid_pairs']],
                'num_samples': metadata['num_samples'],
                'text_descriptions': metadata['text_descriptions'],
                'episode_id': metadata['episode_id']
            })
        return serialized

    def _deserialize_metadata(self, serialized: List[Dict]) -> List[Dict]:
        """Convert serialized metadata back to original format."""
        deserialized = []
        for item in serialized:
            deserialized.append({
                'episode_dir': Path(item['episode_dir']),
                'img_dir': Path(item['img_dir']),
                'npz_dir': Path(item['npz_dir']),
                'all_data_dir': Path(item['all_data_dir']) if item['all_data_dir'] is not None else None,
                'use_all_data': item.get('use_all_data', False),  # Backward compatibility
                'text_file': Path(item['text_file']),
                'valid_pairs': [(Path(img) if img is not None else img, Path(npz), Path(depth) if depth is not None else depth) for img, npz, depth in item['valid_pairs']],
                'num_samples': item['num_samples'],
                'text_descriptions': item['text_descriptions'],
                'episode_id': item['episode_id']
            })
        return deserialized

    def _collect_all_episodes(self) -> List[Path]:
        """Collect all episode directories from all dataset directories."""
        all_episodes = []

        # Use progress bar for dataset scanning
        pbar = tqdm(self.dataset_dirs, desc="Scanning datasets", leave=False)
        for dataset_dir in pbar:
            pbar.set_description(f"Scanning {dataset_dir.name}")
            episode_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
            all_episodes.extend(episode_dirs)
            logger.info(f"Found {len(episode_dirs)} episodes in {dataset_dir}")

        logger.info(f"Total episodes collected: {len(all_episodes)}")
        return all_episodes

    def _split_episodes(self, all_episodes: List[Path]) -> Tuple[List[Path], List[Path]]:
        """Split episodes into train and validation sets."""
        # Set random seed for reproducible splits
        random.seed(self.random_seed)

        # Shuffle episodes
        shuffled_episodes = all_episodes.copy()
        random.shuffle(shuffled_episodes)

        # Calculate split point
        val_count = int(len(shuffled_episodes) * self.val_split)
        train_count = len(shuffled_episodes) - val_count

        train_episodes = shuffled_episodes[:train_count]
        val_episodes = shuffled_episodes[train_count:]

        logger.info(f"Split {len(all_episodes)} episodes: {len(train_episodes)} train, {len(val_episodes)} val")

        return train_episodes, val_episodes

    def _build_episode_metadata(self):
        """Build metadata for all episodes for efficient lazy loading."""
        self.episode_metadata = []
        self.sample_indices = []
        current_index = 0

        # Use progress bar for episode metadata building
        pbar = tqdm(self.episodes, desc="Building episode metadata", leave=False)
        for episode_dir in pbar:
            pbar.set_description(f"Scanning {episode_dir.name}")
            metadata = self._scan_episode(episode_dir)
            if metadata is not None:
                self.episode_metadata.append(metadata)
                # Add sample indices for this episode
                for i in range(metadata['num_samples']):
                    self.sample_indices.append((len(self.episode_metadata) - 1, i))
                current_index += metadata['num_samples']

    def _scan_episode(self, episode_dir: Path) -> Optional[Dict]:
        """Scan episode directory and return metadata without loading data."""
        img_dir = episode_dir / "images"
        npz_dir = episode_dir / "samples"
        depth_dir = episode_dir / "depth"
        all_data_dir = episode_dir / "all_data"
        text_file = episode_dir / "three_instructions.json"

        # Check if all_data directory exists with unified NPZ files
        use_all_data = all_data_dir.exists()
        
        if use_all_data:
            # Use unified NPZ files from all_data directory
            logger.info(f"Using unified NPZ files from all_data directory: {episode_dir}")
            npz_files = sorted(all_data_dir.glob("*.npz"))
            valid_pairs = []
            
            for npz_path in npz_files:
                stem = npz_path.stem
                # For all_data, we don't need separate image/depth paths since they're in the NPZ
                valid_pairs.append((None, npz_path, None))  # img_path=None, depth_path=None indicates unified loading
        else:
            # Fall back to separate files (original behavior)
            # Validate episode structure
            if not img_dir.exists():
                logger.warning(f"Images directory missing in {episode_dir}")
                return None
            if not npz_dir.exists():
                logger.warning(f"samples directory missing in {episode_dir}")
                return None
            
            # Count valid image-npz pairs
            img_files = sorted(img_dir.glob("*.png"))
            valid_pairs = []

            for img_path in img_files:
                stem = img_path.stem
                npz_path = npz_dir / f"{stem}.npz"
                if npz_path.exists():
                    if depth_dir.exists():
                        depth_path = depth_dir / f"{stem}_raw.npz"
                        if depth_path.exists():
                            valid_pairs.append((img_path, npz_path, depth_path))
                        else:
                            valid_pairs.append((img_path, npz_path, None))
                    else:
                        valid_pairs.append((img_path, npz_path, None))

        if len(valid_pairs) == 0:
            logger.warning(f"No valid image-npz pairs found in {episode_dir}")
            return None

        text_file_exists = text_file.exists()
        if not text_file_exists:
            logger.warning(
                f"three_instructions.json missing in {episode_dir}, using default text"
            )

        # Load and cache text descriptions if requested
        text_descriptions = None
        if self.cache_text:
            if text_file_exists:
                try:
                    with open(text_file, "r") as f:
                        text_data = json.load(f)

                    # Extract the three instructions
                    available_descriptions = []
                    for desc_key in ["instruction_1", "instruction_2", "instruction_3"]:
                        if desc_key in text_data:
                            desc_text = str(text_data[desc_key]).strip()
                            if desc_text:
                                available_descriptions.append(desc_text)

                    if not available_descriptions:
                        logger.warning(f"No valid instructions found in {text_file}")
                        available_descriptions = [DEFAULT_TEXT_DESCRIPTION]

                    text_descriptions = available_descriptions
                    self.text_cache[str(episode_dir)] = available_descriptions
                except Exception as e:
                    logger.warning(f"Failed to load text from {text_file}: {e}")
                    text_descriptions = [DEFAULT_TEXT_DESCRIPTION]
                    self.text_cache[str(episode_dir)] = text_descriptions
            else:
                # No text file exists, use default
                text_descriptions = [DEFAULT_TEXT_DESCRIPTION]
                self.text_cache[str(episode_dir)] = text_descriptions

        return {
            'episode_dir': episode_dir,
            'img_dir': img_dir,
            'npz_dir': npz_dir,
            'all_data_dir': all_data_dir if use_all_data else None,
            'use_all_data': use_all_data,
            'text_file': text_file,
            'valid_pairs': valid_pairs,
            'num_samples': len(valid_pairs),
            'text_descriptions': text_descriptions,
            'episode_id': episode_dir.name
        }

    def compute_movement_bool(self, trajectories, trajectory_mask, orig_size, threshold=0.1):
        """
        trajectories: (..., T, 2)  numpy or torch
        trajectory_mask: (..., T, 2)  numpy or torch (bool or 0/1)
        orig_size: (W, H)
        """
        traj_t = torch.as_tensor(trajectories, dtype=torch.float32)
        mask_t = torch.as_tensor(trajectory_mask, dtype=torch.float32)

        W, H = float(orig_size[0]), float(orig_size[1])

        dx = (traj_t[:, 1:, 0] / W - traj_t[:, :-1, 0] / W) * mask_t[:, 1:, 0]
        dy = (traj_t[:, 1:, 1] / H - traj_t[:, :-1, 1] / H) * mask_t[:, 1:, 1]

        movement_sum_x = dx.abs().sum(dim=1)           # shape: [batch]
        movement_sum_y = dy.abs().sum(dim=1)           # shape: [batch]
        movement_bool = (movement_sum_x + movement_sum_y) > threshold  # bool tensor

        return movement_bool

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset using lazy loading.

        Returns:
            sample: Dictionary containing:
                - 'image': Transformed image tensor [3, H, W]
                - 'text': Text description string
                - 'gt_xy': Ground truth coordinates [num_kps, 2] in [0, 1]
                - 'gt_mask': Valid ground truth mask [num_kps]
                - 'movement_bool': Movement indicator [num_kps] - True if trajectory movement > 0.05
                - 'orig_size': Original image size (H, W)
                - 'episode_id': Episode identifier
                - 'frame_id': Frame identifier
        """
        # Get episode and sample indices
        episode_idx, sample_idx = self.sample_indices[idx]
        episode_metadata = self.episode_metadata[episode_idx]

        # Get the specific image-npz pair
        img_path, npz_path, depth_path = episode_metadata['valid_pairs'][sample_idx]
        
        # Check if we're using unified NPZ files from all_data
        use_all_data = episode_metadata.get('use_all_data', False)
        
        if use_all_data:
            # img_path is None for all_data, get frame_id from npz_path
            frame_id = npz_path.stem
        else:
            # Original behavior: get frame_id from img_path
            frame_id = img_path.stem

        # Load image and depth
        keypoints = None
        trajectories = None
        try:
            if use_all_data:
                # Load from unified NPZ file
                npz_data = np.load(npz_path)
                
                # Load image from NPZ
                if 'image' in npz_data:
                    image_array = npz_data['image']  # Should be [H, W, 3] uint8
                    image = Image.fromarray(image_array)
                    orig_size = image.size  # (W, H)
                else:
                    raise ValueError(f"No 'image' key found in {npz_path}")
                
                # Load depth from NPZ if available
                if 'depth' in npz_data:
                    depth = npz_data['depth']
                else:
                    depth = None

                normalized_w, normalized_h = 384, 384
                original_w = npz_data['keypoints'][:, 0].max()
                original_h = npz_data['keypoints'][:, 1].max()
                keypoints = npz_data['keypoints'].copy()
                trajectories = npz_data['traj'].copy()
                keypoints[:, 0] = keypoints[:, 0] / original_w * normalized_w  # 400 by 2
                keypoints[:, 1] = keypoints[:, 1] / original_h * normalized_h  # 400 by 2
                trajectories[:, :, 0] = trajectories[:, :, 0] / original_w * normalized_w  # 400 by 128 by 2 or 3
                trajectories[:, :, 1] = trajectories[:, :, 1] / original_h * normalized_h  # 400 by 128 by 2 or 3
            else:
                # Original behavior: load separate files
                image = Image.open(img_path).convert("RGB")
                orig_size = image.size  # (W, H)

                if depth_path is not None:
                    depth_data = np.load(depth_path)
                    depth = depth_data['depth']
                else:
                    depth = None
        except Exception as e:
            logger.warning(f"Failed to load sample {idx}: {e}, skipping to next sample")
            return self.__getitem__(idx + 1)

        # Load keypoints and trajectories
        try:
            npz_data = np.load(npz_path)
            if keypoints is None:
                keypoints = npz_data['keypoints']  # [N, 2] in pixel coordinates

            # Check valid_steps if available
            valid_steps = torch.tensor(npz_data['valid_steps'].sum(), dtype=torch.int64)
            # if valid_steps.sum() < 4:
            #     # logger.warning(f"Sample {idx} has only {valid_steps.sum()} valid steps (< 5), skipping")
            #     return self.__getitem__(idx + 1)

            # Load trajectory data if available
            trajectory_mask = None
            movement_bool = torch.zeros(self.num_kps, dtype=torch.bool)  # Initialize default movement_bool
            if 'traj' in npz_data:
                if trajectories is None:
                    trajectories = npz_data['traj']  # [K, T, 2] future trajectories
                # trajectory_mask: False if trajectories is -inf
                trajectory_mask = trajectories != -np.inf
                # trajectories contains -inf -> we will pad -inf to the last valid trajectory component
                trajectories = self.fill_traj_with_last_valid(trajectories)

                movement_bool = self.compute_movement_bool(trajectories, trajectory_mask, orig_size, threshold=0.1)
            # some data has depth, and some data does not has depth - we will pad depth with zeros if there's no depth
            if depth_path is not None:
                depth_data = np.load(depth_path)
                depth = depth_data['depth']
                is_depth_valid = torch.tensor(1)
            else:
                depth = None
                is_depth_valid = torch.tensor(0)

            # Load text - randomly select from available descriptions
            if self.cache_text and episode_metadata['text_descriptions']:
                text_description = random.choice(episode_metadata['text_descriptions'])
            else:
                # Load text from file
                text_file = episode_metadata["text_file"]
                try:
                    with open(text_file, "r") as f:
                        text_data = json.load(f)

                    # Extract available descriptions
                    available_descriptions = []
                    for desc_key in ["instruction_1", "instruction_2", "instruction_3"]:
                        if desc_key in text_data:
                            desc_text = str(text_data[desc_key]).strip()
                            if desc_text:
                                available_descriptions.append(desc_text)

                    if not available_descriptions:
                        text_description = DEFAULT_TEXT_DESCRIPTION
                    else:
                        # randomly select one of the available descriptions
                        text_description = random.choice(available_descriptions)
                except Exception as e:
                    logger.error(f"Failed to load text from JSON {episode_metadata['text_file']}: {e}")
                    text_description = DEFAULT_TEXT_DESCRIPTION

            # Transform image and get transformation parameters
            transformed_data = self.transform(image, keypoints, trajectories, depth)

            # Transform returns both image and transformed keypoints
            image_tensor = transformed_data['image']
            keypoints_normalized = transformed_data['keypoints']
            trajectories_normalized = transformed_data['trajectories']
            depth_normalized = transformed_data['depth'] if transformed_data['depth'] is not None else torch.zeros_like(image_tensor[:1])

            # if trajectories has 2 dimensions, add a third dimension with -np.inf
            if trajectories_normalized.shape[-1] == 2:
                trajectories_normalized = np.concatenate([trajectories_normalized, np.full((trajectories_normalized.shape[0], trajectories_normalized.shape[1], 1), 0)], axis=-1)
                trajectory_mask = np.concatenate([trajectory_mask, np.full((trajectory_mask.shape[0], trajectory_mask.shape[1], 1), False)], axis=-1)

            # Prepare ground truth tensors (keep original for compatibility with system1)
            gt_xy, gt_mask, gt_trajectory, gt_trajectory_mask = self._prepare_gt_tensors(
                keypoints_normalized, trajectories_normalized, trajectory_mask
            )

            # first trajectory value is same with starting keypoint
            if not self.absolute_action and trajectories_normalized is not None:
                # the first element is always keypoint, but the future trajectory is relative to the previous keypoint
                gt_trajectory[:, 1:, :] = gt_trajectory[:, 1:, :] - gt_trajectory[:, :-1, :]

            return {
                'image': image_tensor,
                'text': text_description,
                'gt_xy': gt_xy,
                'gt_mask': gt_mask,
                'trajectory': gt_trajectory[:, :self.trajectory_horizon + 1, :],  # [N, H+1, 2] first keypoint + future trajectory
                'trajectory_mask': gt_trajectory_mask[:, :self.trajectory_horizon + 1],  # [N, H+1] mask for valid trajectory points
                'movement_bool': movement_bool,  # [K] boolean array indicating movement > 0.05
                'orig_size': torch.tensor(orig_size, dtype=torch.float32),  # (W, H)
                'episode_id': episode_metadata['episode_id'],
                'frame_id': frame_id,
                'depth': depth_normalized,
                'is_depth_valid': is_depth_valid,
                'valid_steps': valid_steps
            }
            
        except Exception as e:
            logger.warning(f"Failed to process sample {idx}: {e}, skipping to next sample")
            return self.__getitem__(idx + 1)

    def fill_traj_with_last_valid(self, trajectories):
        filled = trajectories.copy()
        K, T, D = trajectories.shape

        for k in range(K):
            traj = trajectories[k]  # [T, 2]
            mask = np.isfinite(traj[:, 0])  # first coordinate based on valid check
            if not mask.any():
                continue  # skip if all -inf
            last_idx = np.where(mask)[0][-1]  # last valid index
            traj[last_idx + 1:] = traj[last_idx]  # fill with last value
            filled[k] = traj
        return filled

    def _prepare_gt_tensors(
        self,
        keypoints_normalized: np.ndarray,
        trajectory_normalized: Optional[np.ndarray],
        trajectory_mask: Optional[np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare ground truth tensors with proper padding.

        Args:
            keypoints_normalized: Normalized keypoints [N, 2]
            trajectory_normalized: Optional normalized trajectories [N, T, 2]
            trajectory_mask: Optional trajectory mask [N, T]

        Returns:
            gt_xy: Ground truth coordinates [num_kps, 2]
            gt_mask: Valid ground truth mask [num_kps]
            trajectory: Trajectory sequence [num_kps, trajectory_horizon+1, 2]
            trajectory_mask: Valid trajectory mask [num_kps, trajectory_horizon+1]
        """
        num_keypoints = len(keypoints_normalized)

        # Initialize tensors
        gt_xy = torch.zeros(self.num_kps, 2, dtype=torch.float32)
        gt_mask = torch.zeros(self.num_kps, dtype=torch.bool)
        gt_trajectory = torch.zeros(self.num_kps, self.trajectory_horizon + 1, 3, dtype=torch.float32)
        gt_trajectory_mask = torch.zeros(self.num_kps, self.trajectory_horizon + 1, 3, dtype=torch.bool)

        # Fill with actual keypoints (up to num_kps)
        num_to_fill = min(num_keypoints, self.num_kps)
        max_seq_length = min(trajectory_normalized.shape[1], self.trajectory_horizon + 1)
        if num_to_fill > 0:
            gt_xy[:num_to_fill] = torch.from_numpy(keypoints_normalized[:num_to_fill])
            gt_mask[:num_to_fill] = True
            try:
                gt_trajectory[:num_to_fill, :max_seq_length] = torch.from_numpy(trajectory_normalized[:num_to_fill, :max_seq_length])
                gt_trajectory_mask[:num_to_fill, :max_seq_length] = torch.from_numpy(trajectory_mask[:num_to_fill, :max_seq_length])
            except Exception:
                logger.error(f"gt_trajectory shape: {gt_trajectory.shape}")
                logger.error(f"trajectory_normalized shape: {trajectory_normalized.shape}")
                logger.error(f"max_seq_length: {max_seq_length}")
        return gt_xy, gt_mask, gt_trajectory, gt_trajectory_mask

    def get_episode_info(self) -> Dict[str, int]:
        """Get information about episodes in the dataset."""
        episode_counts = {}
        for metadata in self.episode_metadata:
            episode_id = metadata["episode_id"]
            episode_counts[episode_id] = metadata['num_samples']

        return {
            'num_episodes': len(episode_counts),
            'num_samples': len(self.sample_indices),
            'samples_per_episode': episode_counts,
            'avg_samples_per_episode': sum(episode_counts.values()) / len(episode_counts) if episode_counts else 0
        }

    def get_text_descriptions(self) -> List[str]:
        """Get all unique text descriptions in the dataset."""
        if self.cache_text:
            all_descriptions = set()
            for metadata in self.episode_metadata:
                if metadata['text_descriptions']:
                    all_descriptions.update(metadata['text_descriptions'])
            return list(all_descriptions)
        else:
            descriptions = set()
            for metadata in self.episode_metadata:
                text_file = metadata['text_file']
                if text_file.exists():
                    try:
                        with open(text_file, 'r') as f:
                            text_data = json.load(f)

                        for desc_key in [
                            "instruction_1",
                            "instruction_2",
                            "instruction_3",
                        ]:
                            if desc_key in text_data:
                                desc_text = str(text_data[desc_key]).strip()
                                if desc_text:
                                    descriptions.add(desc_text)
                    except:
                        pass
            return list(descriptions)


class PointDatasetCollator:
    """
    Custom collator for batching point detection samples (compatible with trajectory data).
    Handles variable-length sequences and proper tensor stacking.
    """

    def __init__(self, pad_value: float = 0.0):
        self.pad_value = pad_value

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of samples.

        Args:
            batch: List of sample dictionaries

        Returns:
            collated_batch: Batched tensors
        """
        # Stack images
        images = torch.stack([sample['image'] for sample in batch])
        # Stack depth images if available
        depths = torch.stack([sample['depth'] for sample in batch])
        is_depth_valid = torch.stack([sample['is_depth_valid'] for sample in batch])
        # Collect texts (keep as list)
        texts = [sample['text'] for sample in batch]

        # Stack ground truth data
        gt_xy = torch.stack([sample['gt_xy'] for sample in batch])
        gt_mask = torch.stack([sample['gt_mask'] for sample in batch])
        movement_bool = torch.stack([sample['movement_bool'] for sample in batch])
        valid_steps = torch.stack([sample['valid_steps'] for sample in batch])
        # Stack trajectory data if available
        trajectories = None
        trajectory_masks = None
        if 'trajectory' in batch[0]:
            trajectories = torch.stack([sample['trajectory'] for sample in batch])
            trajectory_masks = torch.stack([sample['trajectory_mask'] for sample in batch])

        # Stack other metadata
        orig_sizes = torch.stack([sample['orig_size'] for sample in batch])

        # Keep episode and frame IDs as lists
        episode_ids = [sample['episode_id'] for sample in batch]
        frame_ids = [sample['frame_id'] for sample in batch]

        return {
            'image': images,          # [B, 3, H, W]
            'depth': depths,          # [B, 1, H, W]
            'is_depth_valid': is_depth_valid,  # [B, 1, H, W]
            'text': texts,            # List[str] of length B
            'gt_xy': gt_xy,          # [B, num_kps, 2]
            'gt_mask': gt_mask,      # [B, num_kps]
            'movement_bool': movement_bool,  # [B, num_kps] boolean array indicating movement > 0.05
            'valid_steps': valid_steps,  # [B, num_kps] boolean array indicating valid steps
            'orig_size': orig_sizes,  # [B, 2]
            'episode_id': episode_ids,
            'frame_id': frame_ids,
            'trajectory': trajectories,  # [B, trajectory_horizon, 3]
            'trajectory_mask': trajectory_masks  # [B, trajectory_horizon, 3]
        }


def create_dataloaders(
    cfg,
    transform_train=None,
    transform_val=None,
    world_size=1,
    rank=0,
    use_cache=True,
    force_refresh=False
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders with caching support.
    
    New args:
        use_cache: Whether to use metadata cache (default: True)
        force_refresh: Force rebuild cache (default: False)
    """
    # Get dataset directories from config
    if hasattr(cfg.data, 'dataset_dirs') and cfg.data.dataset_dirs:
        dataset_dirs = cfg.data.dataset_dirs
    elif hasattr(cfg.data, 'root_dir'):
        dataset_dirs = [cfg.data.root_dir]
    else:
        raise ValueError("Either cfg.data.dataset_dirs or cfg.data.root_dir must be specified")

    # Get validation split ratio
    val_split = getattr(cfg.data, 'val_split', 0.1)
    random_seed = getattr(cfg.data, 'random_seed', 42)
    
    # Get cache directory from config or use default
    cache_dir = getattr(cfg.data, 'cache_dir', './cache')

    # Create datasets with caching
    train_dataset = EpisodePointDataset(
        dataset_dirs=dataset_dirs,
        split='train',
        transform=transform_train,
        trajectory_horizon=cfg.trajectory_horizon,
        absolute_action=cfg.absolute_action,
        val_split=val_split,
        random_seed=random_seed,
        cfg=cfg,
        use_cache=use_cache,
        cache_dir=cache_dir,
        force_refresh=force_refresh
    )

    val_dataset = EpisodePointDataset(
        dataset_dirs=dataset_dirs,
        split='val',
        transform=transform_val,
        trajectory_horizon=cfg.trajectory_horizon,
        absolute_action=cfg.absolute_action,
        val_split=val_split,
        random_seed=random_seed,
        cfg=cfg,
        use_cache=use_cache,
        cache_dir=cache_dir,
        force_refresh=force_refresh
    )

    # Create collator
    collator = PointDatasetCollator()

    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    shuffle_train = True

    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
        shuffle_train = False  # DistributedSampler handles shuffling

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collator,
        drop_last=True  # For stable training
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collator,
        drop_last=False
    )

    logger.info("Created dataloaders:")
    logger.info(f"  - Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"  - Val: {len(val_dataset)} samples, {len(val_loader)} batches")

    return train_loader, val_loader

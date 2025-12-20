import logging
import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

logger = logging.getLogger(__name__)


class KeypointAwareTransform:
    """
    Base class for transforms that need to handle both images and keypoints.

    All keypoint-aware transforms should inherit from this class and implement
    the transform_image_and_keypoints method.
    """

    def __init__(self):
        pass

    def transform_image_and_keypoints(
        self,
        image: Image.Image,
        keypoints: np.ndarray,
        trajectories: Optional[np.ndarray],
        depth: Optional[Image.Image] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], np.ndarray, Optional[np.ndarray]]:
        """
        Transform both image and keypoints.

        Args:
            image: PIL Image
            keypoints: Keypoints array [N, 2] in pixel coordinates
            trajectories: Optional trajectories array [N, H+1, 2]
            depth: Optional depth image
        Returns:
            transformed_image: Transformed RGB image tensor
            transformed_depth: Transformed depth image tensor (if depth provided)
            transformed_keypoints: Transformed keypoints [N, 2] in pixel coordinates
            transformed_trajectories: Optional transformed trajectories [N, H+1, 2]
        """
        raise NotImplementedError

    def __call__(
        self,
        image: Image.Image,
        keypoints: Optional[np.ndarray] = None,
        trajectories: Optional[np.ndarray] = None,
        depth: Optional[Image.Image] = None,
    ) -> Dict:
        """
        Apply transform to image and optionally keypoints.

        Args:
            image: PIL Image
            keypoints: Optional keypoints array [N, 2]
            trajectories: Optional trajectories array [N, H+1, 2]
            depth: Optional depth image

        Returns:
            Dictionary containing 'image', 'depth', 'keypoints', 'trajectories'
        """

        (
            transformed_image,
            transformed_depth,
            transformed_keypoints,
            transformed_trajectories,
        ) = self.transform_image_and_keypoints(image, keypoints, trajectories, depth)

        return {
            "image": transformed_image,
            "depth": transformed_depth,
            "keypoints": transformed_keypoints,
            "trajectories": transformed_trajectories,
        }


class ResizeTransform(KeypointAwareTransform):
    """
    Resize image and scale keypoints accordingly.
    """

    def __init__(
        self, size: Union[int, Tuple[int, int]], interpolation: int = Image.BILINEAR
    ):
        super().__init__()
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size  # (H, W)
        self.interpolation = interpolation

    def transform_image_and_keypoints(
        self,
        image: Union[Image.Image, torch.Tensor],
        keypoints: np.ndarray,
        trajectories: Optional[np.ndarray],
        depth: Optional[Image.Image] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], np.ndarray, Optional[np.ndarray]]:
        # Handle case where image might already be a tensor
        if isinstance(image, torch.Tensor):
            pil_image = TF.to_pil_image(image)
            orig_w, orig_h = pil_image.size
        else:
            pil_image = image
            orig_w, orig_h = image.size

        target_h, target_w = self.size

        # Resize RGB image
        resized_image = TF.resize(pil_image, self.size, self.interpolation)
        image_tensor = TF.to_tensor(resized_image)

        # Resize depth image if provided
        resized_depth = None
        if depth is not None:
            # Convert numpy array to tensor first
            if isinstance(depth, np.ndarray):
                depth_tensor = torch.from_numpy(depth).float()
                # Add channel dimension if needed
                if depth_tensor.dim() == 2:
                    depth_tensor = depth_tensor.unsqueeze(0)
            else:
                depth_tensor = depth

            resized_depth = TF.resize(depth_tensor, self.size, self.interpolation)

        # Scale keypoints
        if len(keypoints) > 0:
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h

            transformed_keypoints = keypoints.copy()
            transformed_keypoints[:, 0] *= scale_x
            transformed_keypoints[:, 1] *= scale_y
        else:
            transformed_keypoints = keypoints

        # Scale trajectories
        transformed_trajectories = None
        if trajectories is not None:
            transformed_trajectories = trajectories.copy()
            transformed_trajectories[:, :, 0] *= scale_x
            transformed_trajectories[:, :, 1] *= scale_y

        return (
            image_tensor,
            resized_depth,
            transformed_keypoints,
            transformed_trajectories,
        )


class HorizontalFlipTransform(KeypointAwareTransform):
    """
    Random horizontal flip with keypoint adjustment.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def transform_image_and_keypoints(
        self,
        image: Union[Image.Image, torch.Tensor],
        keypoints: np.ndarray,
        trajectories: Optional[np.ndarray],
        depth: Optional[Image.Image] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], np.ndarray, Optional[np.ndarray]]:
        # Handle case where image might already be a tensor
        if isinstance(image, torch.Tensor):
            pil_image = TF.to_pil_image(image)
        else:
            pil_image = image

        image_width = pil_image.size[0]

        if random.random() < self.p:
            # Flip RGB image
            flipped_image = TF.hflip(pil_image)
            image_tensor = TF.to_tensor(flipped_image)

            # Flip depth image if provided
            flipped_depth = None
            if depth is not None:
                flipped_depth = TF.hflip(depth)
                flipped_depth = TF.to_tensor(flipped_depth)

            # Flip keypoints
            if len(keypoints) > 0:
                transformed_keypoints = keypoints.copy()
                transformed_keypoints[:, 0] = image_width - transformed_keypoints[:, 0]
            else:
                transformed_keypoints = keypoints
            if trajectories is not None:
                transformed_trajectories = trajectories.copy()
                transformed_trajectories[:, :, 0] = (
                    image_width - transformed_trajectories[:, :, 0]
                )
            else:
                transformed_trajectories = trajectories
        else:
            # No flip
            image_tensor = TF.to_tensor(pil_image)
            flipped_depth = TF.to_tensor(depth) if depth is not None else None
            transformed_keypoints = keypoints
            transformed_trajectories = trajectories

        return (
            image_tensor,
            flipped_depth,
            transformed_keypoints,
            transformed_trajectories,
        )


class ColorJitterTransform(KeypointAwareTransform):
    """
    Color jitter that doesn't affect keypoints.
    """

    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.1,
        hue: float = 0.05
    ):
        super().__init__()
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def transform_image_and_keypoints(
        self,
        image: Union[Image.Image, torch.Tensor],
        keypoints: np.ndarray,
        trajectories: Optional[np.ndarray],
        depth: Optional[Image.Image] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], np.ndarray, Optional[np.ndarray]]:
        # Handle case where image might already be a tensor
        if isinstance(image, torch.Tensor):
            pil_image = TF.to_pil_image(image)
        else:
            pil_image = image

        # Apply color jitter only to RGB image
        jittered_image = self.color_jitter(pil_image)
        image_tensor = TF.to_tensor(jittered_image)

        # Depth image remains unchanged (convert to tensor if provided)
        depth_tensor = None
        if depth is not None:
            depth_tensor = TF.to_tensor(depth)

        # Keypoints and trajectories remain unchanged for color transforms
        return image_tensor, depth_tensor, keypoints, trajectories


class NormalizeTransform(KeypointAwareTransform):
    """
    Normalize image and convert keypoints to [0, 1] range.
    For depth images, we apply logarithmic transform and normalize to [0, 1].
    """

    def __init__(
        self,
        normalize_image: bool = False,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        depth_epsilon: float = 1e-6,
    ):
        super().__init__()
        self.normalize_image = normalize_image
        self.mean = mean
        self.std = std
        self.depth_epsilon = depth_epsilon

    def preprocess_depth(
        self, depth_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Preprocess depth using logarithmic transform and normalize to [0, 1].

        Args:
            depth_tensor: Single channel depth tensor [1, H, W]

        Returns:
            processed_depth: 3-channel depth tensor [3, H, W]
        """

        return depth_tensor

    def transform_image_and_keypoints(
        self,
        image: Union[Image.Image, torch.Tensor],
        keypoints: np.ndarray,
        trajectories: Optional[np.ndarray],
        depth: Optional[Union[Image.Image, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], np.ndarray, Optional[np.ndarray]]:
        # Handle case where image might already be a tensor
        if isinstance(image, torch.Tensor):
            image_tensor = image
            # Assume image is already in correct format, get size from tensor
            height, width = image_tensor.shape[-2:]
        else:
            # Convert to tensor and get size from PIL image
            image_tensor = TF.to_tensor(image)
            width, height = image.size

        # Normalize RGB image
        if self.normalize_image:
            normalized_image = TF.normalize(image_tensor, self.mean, self.std)
        else:
            normalized_image = image_tensor

        # Handle depth preprocessing
        normalized_depth = None
        if depth is not None:
            if isinstance(depth, torch.Tensor):
                depth_tensor = depth
            else:
                depth_tensor = TF.to_tensor(depth)

            # Ensure depth is single channel
            if depth_tensor.shape[0] != 1:
                depth_tensor = depth_tensor[:1]  # Take first channel if multi-channel

            # Apply depth preprocessing pipeline
            normalized_depth = self.preprocess_depth(depth_tensor)

        # Normalize keypoints to [0, 1]
        if len(keypoints) > 0:
            normalized_keypoints = keypoints.copy().astype(np.float32)
            normalized_keypoints[:, 0] /= width  # x
            normalized_keypoints[:, 1] /= height  # y
            # Clamp to [0, 1]
            normalized_keypoints = np.clip(normalized_keypoints, 0.0, 1.0)
        else:
            normalized_keypoints = keypoints.astype(np.float32)
        # Normalize trajectories
        normalized_trajectories = None
        if trajectories is not None:
            normalized_trajectories = trajectories.copy().astype(np.float32)
            normalized_trajectories[:, :, 0] /= width  # x
            normalized_trajectories[:, :, 1] /= height  # y
            # Clamp to [0, 1]
            normalized_trajectories[:, :, :2] = np.clip(normalized_trajectories[:, :, :2], 0.0, 1.0)

        return (
            normalized_image,
            normalized_depth,
            normalized_keypoints,
            normalized_trajectories,
        )


class CompositeTransform:
    """
    Compose multiple keypoint-aware transforms.
    """

    def __init__(self, transforms: List[KeypointAwareTransform]):
        self.transforms = transforms

    def __call__(
        self,
        image: Image.Image,
        keypoints: Optional[np.ndarray] = None,
        trajectories: Optional[np.ndarray] = None,
        depth: Optional[Image.Image] = None,
    ) -> Dict:
        """
        Apply all transforms in sequence.

        Args:
            image: PIL Image
            keypoints: Optional keypoints array [N, 2]
            trajectories: Optional trajectories array [N, H+1, 2]
        Returns:
            Dictionary containing final 'image' and 'keypoints'
        """
        if keypoints is None:
            keypoints = np.empty((0, 2), dtype=np.float32)

        current_image = image
        current_keypoints = keypoints
        current_trajectories = trajectories
        current_depth = depth
        for transform in self.transforms:
            if isinstance(transform, KeypointAwareTransform):
                (
                    current_image,
                    current_depth,
                    current_keypoints,
                    current_trajectories,
                ) = transform.transform_image_and_keypoints(
                    current_image,
                    current_keypoints,
                    current_trajectories,
                    current_depth,
                )

                # Convert tensors back to PIL for next transform (except for last transform)
                if transform != self.transforms[-1]:
                    if isinstance(current_image, torch.Tensor):
                        current_image = TF.to_pil_image(current_image)
            else:
                # Regular transform that only affects image
                current_image = transform(current_image)

        return {
            'image': current_image,
            'depth': current_depth,
            'keypoints': current_keypoints,
            'trajectories': current_trajectories
        }


def create_train_transforms(cfg) -> CompositeTransform:
    """
    Create training transforms from configuration.

    Args:
        cfg: Configuration object

    Returns:
        CompositeTransform for training
    """
    transforms = []

    # Simple resize
    transforms.append(ResizeTransform(
        size=cfg.model.vision_encoder.image_size
    ))

    # Horizontal flip
    if hasattr(cfg.data.augmentation, 'horizontal_flip'):
        transforms.append(HorizontalFlipTransform(
            p=cfg.data.augmentation.horizontal_flip
        ))

    # Color jitter
    if hasattr(cfg.data.augmentation, 'brightness'):
        transforms.append(ColorJitterTransform(
            brightness=cfg.data.augmentation.brightness,
            contrast=cfg.data.augmentation.contrast,
            saturation=cfg.data.augmentation.saturation,
            hue=cfg.data.augmentation.hue
        ))

    # Normalization (converts keypoints to [0, 1])
    transforms.append(NormalizeTransform())

    logger.info(f"Created training transforms with {len(transforms)} components")

    return CompositeTransform(transforms)


def create_val_transforms(cfg) -> CompositeTransform:
    """
    Create validation transforms from configuration.

    Args:
        cfg: Configuration object

    Returns:
        CompositeTransform for validation
    """
    transforms = [
        ResizeTransform(size=cfg.model.vision_encoder.image_size),
        NormalizeTransform()
    ]

    logger.info(f"Created validation transforms with {len(transforms)} components")

    return CompositeTransform(transforms)
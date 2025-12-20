import logging
from typing import Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


def normalize_coordinates(
    coords: np.ndarray,
    image_size: Tuple[int, int],
    output_range: str = "01"
) -> np.ndarray:
    """
    Normalize coordinates to specified range.

    Args:
        coords: Coordinates array [N, 2] in pixel space (x, y)
        image_size: Image size (width, height)
        output_range: Output range ("01" for [0,1], "11" for [-1,1])

    Returns:
        normalized_coords: Normalized coordinates [N, 2]
    """
    if len(coords) == 0:
        return coords.astype(np.float32)

    normalized = coords.astype(np.float32)
    width, height = image_size

    # Normalize to [0, 1]
    normalized[:, 0] /= width   # x
    normalized[:, 1] /= height  # y

    # Clamp to valid range
    normalized = np.clip(normalized, 0.0, 1.0)

    if output_range == "11":
        # Convert to [-1, 1]
        normalized = normalized * 2.0 - 1.0

    return normalized


def denormalize_coordinates(
    coords: Union[np.ndarray, torch.Tensor],
    image_size: Tuple[int, int],
    input_range: str = "01"
) -> Union[np.ndarray, torch.Tensor]:
    """
    Denormalize coordinates back to pixel space.

    Args:
        coords: Normalized coordinates [N, 2] or [B, N, 2]
        image_size: Image size (width, height)
        input_range: Input range ("01" for [0,1], "11" for [-1,1])

    Returns:
        pixel_coords: Coordinates in pixel space
    """
    if isinstance(coords, torch.Tensor):
        is_tensor = True
        device = coords.device
        coords_np = coords.detach().cpu().numpy()
    else:
        is_tensor = False
        coords_np = coords

    if coords_np.size == 0:
        return coords if is_tensor else coords_np

    denormalized = coords_np.copy()
    width, height = image_size

    if input_range == "11":
        # Convert from [-1, 1] to [0, 1]
        denormalized = (denormalized + 1.0) / 2.0

    # Scale to pixel coordinates
    denormalized[..., 0] *= width   # x
    denormalized[..., 1] *= height  # y

    if is_tensor:
        return torch.from_numpy(denormalized).to(device)
    else:
        return denormalized


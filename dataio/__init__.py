from .datasets import (
    EpisodePointDataset,
    PointDatasetCollator,
    create_dataloaders
)

from .transforms import (
    KeypointAwareTransform,
    ResizeTransform,
    HorizontalFlipTransform,
    ColorJitterTransform,
    NormalizeTransform,
    CompositeTransform,
    create_train_transforms,
    create_val_transforms,
)

from .utils import (
    normalize_coordinates,
    denormalize_coordinates
)

__all__ = [
    # Dataset and data loading
    'EpisodePointDataset',
    'PointDatasetCollator',
    'create_dataloaders',

    # Transform classes
    'KeypointAwareTransform',
    'ResizeTransform',
    'HorizontalFlipTransform',
    'ColorJitterTransform',
    'NormalizeTransform',
    'CompositeTransform',

    # Transform creation functions
    'create_train_transforms',
    'create_val_transforms',

    # Coordinate utilities
    'normalize_coordinates',
    'denormalize_coordinates',
]

# Package configuration
DATAIO_CONFIG = {
    'supported_image_formats': ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'],
    'supported_keypoint_formats': ['.npz', '.json', '.txt'],
    'required_sample_dir': 'samples_round2',
    'default_coordinate_range': [0, 1],
    'default_image_size': 384,
    'max_keypoints_per_sample': 1000,
    'default_val_split': 0.1,
    'default_random_seed': 42,
    'lazy_loading': True,
    'multi_dataset_support': True,
    'episode_level_split': True,
    'supported_transforms': [
        'resize', 'horizontal_flip',
        'color_jitter', 'normalize'
    ]
}

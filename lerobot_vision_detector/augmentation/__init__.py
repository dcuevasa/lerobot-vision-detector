"""Dataset augmentation and annotation package."""

from .annotator import DatasetVideoAugmentor, augment_dataset, match_camera_key

__all__ = [
    "DatasetVideoAugmentor",
    "augment_dataset",
    "match_camera_key",
]

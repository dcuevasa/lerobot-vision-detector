"""Dataset augmentation and annotation package."""

from .annotator import (
    DatasetVideoAugmentor,
    augment_dataset,
    extract_target_from_prompt,
    match_camera_key,
    parse_target_list,
)

__all__ = [
    "DatasetVideoAugmentor",
    "augment_dataset",
    "extract_target_from_prompt",
    "match_camera_key",
    "parse_target_list",
]

"""lerobot-vision-detector: Computer vision pipeline for LeRobot datasets and live camera streams.

Provides:
- Offline dataset augmentation with 2D Bounding Boxes or Instance Segmentation Masks.
- Real-time Multi-Camera Wrappers for online teleoperation, recording, and policy execution.
- Extensible vision detector backends (YOLO, YOLO-Seg, Mock).
"""

from .augmentation import DatasetVideoAugmentor, augment_dataset
from .detectors import (
    BaseDetector,
    Box,
    Detection,
    DetectionResult,
    Mask,
    MockDetector,
    YOLOBboxDetector,
    YOLODetector,
    YOLOSegDetector,
    make_detector,
)
from .wrappers import (
    DetectedCameraWrapper,
    MultiCameraVisionWrapper,
    wrap_cameras,
    wrap_robot_cameras,
)

__version__ = "0.1.0"

__all__ = [
    "BaseDetector",
    "Box",
    "Mask",
    "Detection",
    "DetectionResult",
    "YOLODetector",
    "YOLOBboxDetector",
    "YOLOSegDetector",
    "MockDetector",
    "make_detector",
    "DetectedCameraWrapper",
    "MultiCameraVisionWrapper",
    "wrap_cameras",
    "wrap_robot_cameras",
    "DatasetVideoAugmentor",
    "augment_dataset",
]

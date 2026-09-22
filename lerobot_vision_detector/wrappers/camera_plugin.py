"""LeRobot Camera Plugin integration for native CLI JSON configs."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any
import numpy as np

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig, ColorMode, Cv2Rotation
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from ..detectors import make_detector
from ..detectors.base import BaseDetector

logger = logging.getLogger("lerobot_vision_detector")


@CameraConfig.register_subclass("detected_opencv")
@dataclass(kw_only=True)
class DetectedOpenCVCameraConfig(OpenCVCameraConfig):
    """Configuration for an OpenCV camera with integrated real-time detection."""
    mode: str = "bbox"  # "bbox", "seg", or "both"
    target_objects: str | None = None  # e.g. "cup" or "cup,bottle"
    model_name: str | None = None  # e.g. "yolov8n.pt" or "yolov8n-seg.pt"
    conf_threshold: float = 0.25
    device: str | None = None


class DetectedOpenCVCamera(OpenCVCamera):
    """OpenCV Camera subclass that applies real-time YOLO detections to captured frames."""

    def __init__(self, config: DetectedOpenCVCameraConfig):
        super().__init__(config)
        self.detector_config = config
        self.mode = config.mode
        self.target_objects = config.target_objects
        self.conf_threshold = config.conf_threshold

        m_name = config.model_name
        if m_name is None:
            m_name = "yolov8n-seg.pt" if self.mode in ("seg", "mask") else "yolov8n.pt"

        self.detector: BaseDetector = make_detector(
            detector_type="yolo",
            model_name=m_name,
            mode=self.mode,
            target_objects=self.target_objects,
            conf_threshold=self.conf_threshold,
            device=config.device,
        )

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        return self.detector.annotate(
            frame,
            mode=self.mode,
            target_objects=self.target_objects,
            conf_threshold=self.conf_threshold,
        )

    def read(self) -> np.ndarray:
        raw = super().read()
        return self._process_frame(raw)

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        raw = super().async_read(timeout_ms=timeout_ms)
        return self._process_frame(raw)

    def read_latest(self, max_age_ms: int = 500) -> np.ndarray:
        raw = super().read_latest(max_age_ms=max_age_ms)
        return self._process_frame(raw)

"""Camera wrapper that performs real-time YOLO bounding box or segmentation mask rendering."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Sequence
import numpy as np

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig

from ..detectors.base import BaseDetector, DetectionResult
from ..detectors import make_detector

logger = logging.getLogger("lerobot_vision_detector")


class DetectedCameraWrapper(Camera):
    """Wraps an existing LeRobot Camera instance to augment frames with real-time detections.

    Delegates all camera hardware control to the underlying Camera object while
    intercepting frame capture (read, async_read, read_latest) to execute YOLO
    detection and visual rendering.
    """

    def __init__(
        self,
        base_camera: Camera,
        detector: BaseDetector | None = None,
        camera_name: str = "camera",
        mode: str = "bbox",  # "bbox", "seg", or "both"
        target_objects: str | Sequence[str] | None = None,
        conf_threshold: float = 0.25,
        enabled: bool = True,
        mask_alpha: float = 0.45,
        box_thickness: int = 2,
    ):
        # We initialize the parent Camera class with a dummy or cloned config
        dummy_config = getattr(base_camera, "config", None)
        if dummy_config is None:
            class _SimpleCamConfig(CameraConfig):
                pass
            dummy_config = _SimpleCamConfig(
                fps=getattr(base_camera, "fps", 30),
                width=getattr(base_camera, "width", 640),
                height=getattr(base_camera, "height", 480),
            )
        self.base_camera = base_camera
        self.camera_name = camera_name
        self.mode = mode
        self.target_objects = target_objects
        self.conf_threshold = conf_threshold
        self.enabled = enabled
        self.mask_alpha = mask_alpha
        self.box_thickness = box_thickness

        super().__init__(dummy_config)

        if detector is None:
            self.detector: BaseDetector = make_detector(
                detector_type="yolo",
                mode=mode,
                target_objects=target_objects,
                conf_threshold=conf_threshold,
            )
        else:
            self.detector = detector

        self._lock = threading.Lock()
        self.latest_detection_result: DetectionResult | None = None
        self.latest_annotated_frame: np.ndarray | None = None
        self.latest_raw_frame: np.ndarray | None = None

    @property
    def fps(self) -> int | None:
        return self.base_camera.fps

    @fps.setter
    def fps(self, value: int | None) -> None:
        self.base_camera.fps = value

    @property
    def width(self) -> int | None:
        return self.base_camera.width

    @width.setter
    def width(self, value: int | None) -> None:
        self.base_camera.width = value

    @property
    def height(self) -> int | None:
        return self.base_camera.height

    @height.setter
    def height(self, value: int | None) -> None:
        self.base_camera.height = value

    @property
    def is_connected(self) -> bool:
        return self.base_camera.is_connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        return Camera.find_cameras()

    def connect(self, warmup: bool = True) -> None:
        """Connect the underlying camera."""
        self.base_camera.connect(warmup=warmup)

    def disconnect(self) -> None:
        """Disconnect the underlying camera."""
        self.base_camera.disconnect()

    def _process_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        """Internal helper to run detection and annotate frame."""
        with self._lock:
            self.latest_raw_frame = raw_frame
            if not self.enabled:
                self.latest_annotated_frame = raw_frame
                self.latest_detection_result = None
                return raw_frame

            det_res = self.detector.predict(raw_frame)
            self.latest_detection_result = det_res

            annotated = self.detector.annotate(
                image=raw_frame,
                result=det_res,
                mode=self.mode,
                target_objects=self.target_objects,
                conf_threshold=self.conf_threshold,
                mask_alpha=self.mask_alpha,
                box_thickness=self.box_thickness,
            )
            self.latest_annotated_frame = annotated
            return annotated

    def read(self) -> np.ndarray:
        """Capture and return single frame synchronously with detections."""
        raw_frame = self.base_camera.read()
        return self._process_frame(raw_frame)

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        """Return the most recent new frame with detections."""
        raw_frame = self.base_camera.async_read(timeout_ms=timeout_ms)
        return self._process_frame(raw_frame)

    def read_latest(self, max_age_ms: int = 500) -> np.ndarray:
        """Return the most recent frame immediately with detections."""
        raw_frame = self.base_camera.read_latest(max_age_ms=max_age_ms)
        return self._process_frame(raw_frame)

    def get_latest_detections(self) -> DetectionResult | None:
        """Return the latest structured DetectionResult for external access."""
        with self._lock:
            return self.latest_detection_result

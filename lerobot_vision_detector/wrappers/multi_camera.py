"""Multi-Camera Vision Wrapper managing multiple cameras simultaneously with shared or dedicated detectors."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence
from lerobot.cameras.camera import Camera

from ..detectors.base import BaseDetector, merge_query_classes, parse_class_names
from ..detectors import make_detector
from .camera_wrapper import DetectedCameraWrapper

logger = logging.getLogger("lerobot_vision_detector")


class MultiCameraVisionWrapper:
    """Manages wrapping multiple cameras with real-time detection capabilities.

    Ensures that only SPECIFIED cameras are wrapped with detectors, while unspecified
    cameras continue to stream unmodified raw frames.
    """

    def __init__(
        self,
        cameras: dict[str, Camera],
        camera_keys: Sequence[str] | str | None = None,
        detector: BaseDetector | None = None,
        mode: str = "bbox",  # "bbox", "seg", "both", or "obscure"
        target_objects: str | Sequence[str] | None = None,
        conf_threshold: float = 0.25,
        model_name: str | None = None,
        device: str | None = None,
        mask_alpha: float = 0.45,
        box_thickness: int = 2,
        obscure_color: tuple[int, int, int] | Sequence[int] | str = (255, 255, 255),
        obscure_margin: int = 16,
        distractor_objects: str | Sequence[str] | None = None,
        detector_type: str = "yolo",
        **kwargs: Any,
    ):
        self.raw_cameras = cameras
        self.mode = mode
        self.target_objects = target_objects
        self.conf_threshold = conf_threshold
        self.mask_alpha = mask_alpha
        self.box_thickness = box_thickness
        self.obscure_color = obscure_color
        self.obscure_margin = obscure_margin
        self.distractor_objects = parse_class_names(distractor_objects)

        # Normalize specified camera keys
        if camera_keys is None or camera_keys == "all":
            self.specified_keys = set(cameras.keys())
        elif isinstance(camera_keys, str):
            self.specified_keys = {k.strip() for k in camera_keys.split(",") if k.strip()}
        else:
            self.specified_keys = {str(k).strip() for k in camera_keys if str(k).strip()}

        query_targets: str | Sequence[str] | None = target_objects
        if self.distractor_objects:
            merged = merge_query_classes(target_objects, self.distractor_objects)
            if merged:
                query_targets = merged

        # Instantiate shared detector if none provided
        if detector is None:
            self.detector: BaseDetector = make_detector(
                detector_type=detector_type,
                model_name=model_name,
                mode=mode,
                target_objects=query_targets,
                conf_threshold=conf_threshold,
                device=device,
                **kwargs,
            )
        else:
            self.detector = detector
            if (
                self.distractor_objects
                and isinstance(query_targets, list)
                and query_targets
                and hasattr(self.detector, "set_classes")
            ):
                try:
                    self.detector.set_classes(query_targets)
                except NotImplementedError:
                    pass

        self.wrapped_cameras: dict[str, Camera] = {}
        for cam_key, cam_obj in cameras.items():
            # Check if this camera is specified by exact match or normalized match
            is_specified = (
                cam_key in self.specified_keys
                or any(k in cam_key for k in self.specified_keys)
            )

            if is_specified:
                logger.info(f"Wrapping camera '{cam_key}' with {mode.upper()} detector (targets={target_objects})")
                self.wrapped_cameras[cam_key] = DetectedCameraWrapper(
                    base_camera=cam_obj,
                    detector=self.detector,
                    camera_name=cam_key,
                    mode=mode,
                    target_objects=target_objects,
                    conf_threshold=conf_threshold,
                    mask_alpha=mask_alpha,
                    box_thickness=box_thickness,
                    obscure_color=obscure_color,
                    obscure_margin=obscure_margin,
                    exclude_objects=self.distractor_objects or None,
                )
            else:
                logger.info(f"Camera '{cam_key}' is UNSPECIFIED. Keeping raw camera stream untouched.")
                self.wrapped_cameras[cam_key] = cam_obj

    def get_cameras(self) -> dict[str, Camera]:
        """Return the dictionary containing wrapped (and untouched) cameras."""
        return self.wrapped_cameras

    def set_target_objects(self, target_objects: str | Sequence[str] | None) -> None:
        """Update target objects dynamically across all wrapped cameras."""
        self.target_objects = target_objects
        for cam in self.wrapped_cameras.values():
            if isinstance(cam, DetectedCameraWrapper):
                cam.target_objects = target_objects
        if hasattr(self.detector, "set_classes"):
            query_classes = merge_query_classes(target_objects, self.distractor_objects)
            if query_classes:
                try:
                    self.detector.set_classes(query_classes)
                except NotImplementedError:
                    pass

    def set_mode(self, mode: str) -> None:
        """Update detection mode ('bbox', 'seg', 'both') dynamically across all wrapped cameras."""
        self.mode = mode
        for cam in self.wrapped_cameras.values():
            if isinstance(cam, DetectedCameraWrapper):
                cam.mode = mode

    def set_enabled(self, enabled: bool) -> None:
        """Enable or bypass detection overlays across all wrapped cameras."""
        for cam in self.wrapped_cameras.values():
            if isinstance(cam, DetectedCameraWrapper):
                cam.enabled = enabled


def wrap_cameras(
    cameras: dict[str, Camera],
    camera_keys: Sequence[str] | str | None = None,
    detector: BaseDetector | None = None,
    detector_type: str = "yolo",
    mode: str = "bbox",
    target_objects: str | Sequence[str] | None = None,
    conf_threshold: float = 0.25,
    model_name: str | None = None,
    device: str | None = None,
    **kwargs: Any,
) -> dict[str, Camera]:
    """Helper function to wrap a camera dictionary with detection capabilities.

    Args:
        cameras: Dictionary mapping camera name to LeRobot Camera instance.
        camera_keys: List or comma-separated string of camera names to process.
                     Cameras not specified will NOT be wrapped.
        detector: Pre-instantiated detector or None to auto-create YOLO detector.
        mode: 'bbox', 'seg', or 'both'.
        target_objects: Target object class name(s) (e.g. 'cup', 'bottle') or None for all.
        conf_threshold: Minimum detection confidence score.
        model_name: Model weights name or path.
        device: 'cuda', 'cpu', or None.

    Returns:
        Dictionary of cameras with specified cameras wrapped.
    """
    wrapper = MultiCameraVisionWrapper(
        cameras=cameras,
        camera_keys=camera_keys,
        detector=detector,
        detector_type=detector_type,
        mode=mode,
        target_objects=target_objects,
        conf_threshold=conf_threshold,
        model_name=model_name,
        device=device,
        **kwargs,
    )
    return wrapper.get_cameras()

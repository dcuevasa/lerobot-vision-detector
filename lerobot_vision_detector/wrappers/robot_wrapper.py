"""Robot wrapper utilities to inject real-time vision detection into any LeRobot robot."""

from __future__ import annotations

import logging
from typing import Any, Sequence
from lerobot.robots.robot import Robot

from ..detectors.base import BaseDetector
from .multi_camera import MultiCameraVisionWrapper, wrap_cameras

logger = logging.getLogger("lerobot_vision_detector")


def wrap_robot_cameras(
    robot: Robot,
    camera_keys: Sequence[str] | str | None = None,
    detector: BaseDetector | None = None,
    mode: str = "bbox",  # "bbox", "seg", or "both"
    target_objects: str | Sequence[str] | None = None,
    conf_threshold: float = 0.25,
    model_name: str | None = None,
    device: str | None = None,
    **kwargs: Any,
) -> Robot:
    """Wraps specified cameras of a LeRobot Robot instance with real-time detectors.

    This directly patches `robot.cameras` so that:
    1. During teleoperation, Rerun visualizer streams display the detection overlays.
    2. During dataset recording (lerobot-record, lerobot_record_physical.py), the recorded
       frames stored in MP4 videos already include the bounding boxes or segmentation masks.
    3. During policy evaluation (lerobot-eval), the policy receives live detected frames
       matching its training distribution.

    Args:
        robot: The instantiated LeRobot Robot.
        camera_keys: List of camera names to process (e.g. ['cam_high', 'cam_wrist']).
                     Cameras not specified will stream unmodified frames.
        detector: Pre-instantiated detector or None to auto-create YOLO detector.
        mode: 'bbox', 'seg', or 'both'.
        target_objects: Target object class name(s) (e.g. 'cup', 'bottle') or None for all.
        conf_threshold: Minimum confidence score.
        model_name: Model weights name or path.
        device: 'cuda', 'cpu', or None.

    Returns:
        The robot instance with wrapped cameras.
    """
    if not hasattr(robot, "cameras") or not robot.cameras:
        logger.warning(f"Robot '{robot}' has no cameras configured; skipping camera wrapping.")
        return robot

    logger.info(f"Wrapping robot cameras {list(robot.cameras.keys())} with vision detector...")
    wrapped_cams = wrap_cameras(
        cameras=robot.cameras,
        camera_keys=camera_keys,
        detector=detector,
        mode=mode,
        target_objects=target_objects,
        conf_threshold=conf_threshold,
        model_name=model_name,
        device=device,
        **kwargs,
    )
    robot.cameras = wrapped_cams
    return robot

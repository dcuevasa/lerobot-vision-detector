"""Wrappers package for real-time camera and robot vision detection."""

from .camera_wrapper import DetectedCameraWrapper
from .multi_camera import MultiCameraVisionWrapper, wrap_cameras
from .robot_wrapper import wrap_robot_cameras

__all__ = [
    "DetectedCameraWrapper",
    "MultiCameraVisionWrapper",
    "wrap_cameras",
    "wrap_robot_cameras",
]

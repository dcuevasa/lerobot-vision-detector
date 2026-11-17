"""Wrappers package for real-time camera and robot vision detection."""

from .camera_wrapper import DetectedCameraWrapper
from .multi_camera import MultiCameraVisionWrapper, reset_wrapped_cameras, wrap_cameras
from .robot_wrapper import reset_robot_cameras, wrap_robot_cameras

__all__ = [
    "DetectedCameraWrapper",
    "MultiCameraVisionWrapper",
    "reset_robot_cameras",
    "reset_wrapped_cameras",
    "wrap_cameras",
    "wrap_robot_cameras",
]

"""Detectors package and factory."""

from __future__ import annotations

from typing import Any, Sequence
from .base import BaseDetector, Box, Detection, DetectionResult, Mask
from .mock_detector import MockDetector
from .yolo_detector import YOLOBboxDetector, YOLODetector, YOLOSegDetector

_DETECTOR_REGISTRY: dict[str, type[BaseDetector]] = {
    "yolo": YOLODetector,
    "yolo_bbox": YOLOBboxDetector,
    "yolo_seg": YOLOSegDetector,
    "mock": MockDetector,
}


def register_detector(name: str, detector_cls: type[BaseDetector]) -> None:
    """Register a new custom detector class."""
    _DETECTOR_REGISTRY[name.lower()] = detector_cls


def make_detector(
    detector_type: str = "yolo",
    model_name: str | None = None,
    mode: str = "bbox",
    target_objects: str | Sequence[str] | None = None,
    conf_threshold: float = 0.25,
    device: str | None = None,
    half: bool = False,
    **kwargs: Any,
) -> BaseDetector:
    """Factory function to instantiate vision detectors.

    Args:
        detector_type: Type of detector ('yolo', 'yolo_bbox', 'yolo_seg', 'mock').
        model_name: Model weights name or file path (e.g. 'yolov8n.pt', 'yolov8n-seg.pt').
        mode: Visual detection mode ('bbox', 'seg', or 'both').
        target_objects: Target object class name(s) or 'all'.
        conf_threshold: Minimum confidence threshold.
        device: 'cuda', 'cpu', or None for auto-detect.
        half: Enable fp16 half precision.
        **kwargs: Extra arguments passed to detector constructor.

    Returns:
        Instance of BaseDetector.
    """
    dtype = detector_type.lower()
    if dtype not in _DETECTOR_REGISTRY:
        raise ValueError(
            f"Unknown detector type '{detector_type}'. Available: {list(_DETECTOR_REGISTRY.keys())}"
        )

    cls = _DETECTOR_REGISTRY[dtype]
    if dtype == "mock":
        return cls(
            mode=mode,
            target_objects=target_objects,
            conf_threshold=conf_threshold,
            **kwargs,
        )

    # For YOLO
    if model_name is None:
        model_name = "yolov8n-seg.pt" if mode in ("seg", "mask") else "yolov8n.pt"

    return cls(
        model_name_or_path=model_name,
        mode=mode,
        target_objects=target_objects,
        conf_threshold=conf_threshold,
        device=device,
        half=half,
        **kwargs,
    )


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
    "register_detector",
]

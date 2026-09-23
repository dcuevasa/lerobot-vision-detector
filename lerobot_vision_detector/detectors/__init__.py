"""Detectors package and factory."""

from __future__ import annotations

from typing import Any, Sequence
from ..utils.model_registry import is_open_vocab_model, resolve_model_path
from .base import BaseDetector, Box, Detection, DetectionResult, Mask
from .mock_detector import MockDetector
from .owl_detector import OWLv2Detector
from .yolo_detector import YOLOBboxDetector, YOLODetector, YOLOSegDetector, YOLOWorldDetector

_DETECTOR_REGISTRY: dict[str, type[BaseDetector]] = {
    "yolo": YOLODetector,
    "yolo_bbox": YOLOBboxDetector,
    "yolo_seg": YOLOSegDetector,
    "yolo_world": YOLOWorldDetector,
    "open_vocab": YOLOWorldDetector,
    "owlv2": OWLv2Detector,
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

    Supports:
    - Standard YOLO 2D Bounding Boxes ('yolo', 'yolo_bbox')
    - Instance Segmentation Masks ('yolo_seg')
    - Open-Vocabulary Detection via YOLO-World ('open_vocab', 'yolo_world')
    - Open-Vocabulary Transformer Detection via OWLv2 ('owlv2')
    - Synthetic Mock Detector ('mock')

    Args:
        detector_type: Type of detector ('yolo', 'yolo_bbox', 'yolo_seg', 'yolo_world', 'open_vocab', 'owlv2', 'mock').
        model_name: Model weights name or file path (e.g. 'yolov8n.pt', 'yolov8s-worldv2.pt', 'yolo11-detection-obj_s.pt').
        mode: Visual detection mode ('bbox', 'seg', or 'both').
        target_objects: Target object class name(s) or 'all' or open-vocab text queries.
        conf_threshold: Minimum confidence threshold.
        device: 'cuda', 'cpu', or None for auto-detect.
        half: Enable fp16 half precision.
        **kwargs: Extra arguments passed to detector constructor.

    Returns:
        Instance of BaseDetector.
    """
    dtype = detector_type.lower()

    # Automatically route open-vocabulary requests or models
    if dtype in ("open_vocab", "yolo_world"):
        if model_name is None:
            model_name = "yolov8s-worldv2.pt"
        resolved_path = resolve_model_path(model_name)
        return YOLOWorldDetector(
            model_name_or_path=resolved_path,
            mode=mode,
            target_objects=target_objects,
            conf_threshold=conf_threshold,
            device=device,
            half=half,
            **kwargs,
        )

    if dtype == "owlv2":
        model_id = model_name or "google/owlv2-base-patch16-ensemble"
        return OWLv2Detector(
            model_name_or_path=model_id,
            mode=mode,
            target_objects=target_objects,
            conf_threshold=conf_threshold,
            device=device,
            **kwargs,
        )

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

    # Set default YOLO model if none provided
    if model_name is None:
        model_name = "yolov8n-seg.pt" if mode in ("seg", "mask") else "yolov8n.pt"

    resolved_path = resolve_model_path(model_name)

    # If the model file itself is open-vocabulary, use YOLOWorldDetector
    if is_open_vocab_model(model_name):
        return YOLOWorldDetector(
            model_name_or_path=resolved_path,
            mode=mode,
            target_objects=target_objects,
            conf_threshold=conf_threshold,
            device=device,
            half=half,
            **kwargs,
        )

    return cls(
        model_name_or_path=resolved_path,
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
    "YOLOWorldDetector",
    "OWLv2Detector",
    "MockDetector",
    "make_detector",
    "register_detector",
]

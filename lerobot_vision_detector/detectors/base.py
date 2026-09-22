"""Abstract Base Detector and Data Structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np

from ..utils.drawing import (
    draw_bounding_box,
    draw_segmentation_mask,
    get_color_for_class,
)


@dataclass
class Box:
    """Bounding box representation in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def xywh(self) -> list[float]:
        return [self.x1, self.y1, self.width, self.height]


@dataclass
class Mask:
    """Instance segmentation mask representation."""
    mask: np.ndarray  # 2D boolean or uint8 mask (H, W)
    confidence: float
    class_id: int
    class_name: str
    polygon: list[list[float]] | None = None  # Optional polygon points [[x, y], ...]


@dataclass
class Detection:
    """A single detected instance containing box and/or mask."""
    class_name: str
    class_id: int
    confidence: float
    box: Box | None = None
    mask: Mask | None = None


@dataclass
class DetectionResult:
    """Full detection output for an image frame."""
    detections: list[Detection] = field(default_factory=list)
    image_shape: tuple[int, int, int] | None = None
    latency_ms: float = 0.0

    @property
    def is_empty(self) -> bool:
        return len(self.detections) == 0

    def filter_by_targets(
        self,
        target_objects: str | Sequence[str] | None = None,
        conf_threshold: float = 0.0,
    ) -> DetectionResult:
        """Filter detections by target class names and confidence threshold."""
        if not target_objects and conf_threshold <= 0.0:
            return self

        targets: set[str] | None = None
        if target_objects:
            if isinstance(target_objects, str):
                raw_targets = [t.strip().lower() for t in target_objects.split(",") if t.strip()]
            else:
                raw_targets = [str(t).strip().lower() for t in target_objects if str(t).strip()]
            if "all" in raw_targets or "*" in raw_targets:
                targets = None
            else:
                targets = set(raw_targets)

        filtered = []
        for det in self.detections:
            if det.confidence < conf_threshold:
                continue
            if targets is not None:
                c_name = det.class_name.lower()
                c_id_str = str(det.class_id)
                if c_name not in targets and c_id_str not in targets and not any(t == c_name or t == c_id_str for t in targets):
                    continue
            filtered.append(det)

        return DetectionResult(
            detections=filtered,
            image_shape=self.image_shape,
            latency_ms=self.latency_ms,
        )


class BaseDetector(ABC):
    """Abstract base detector interface for real-time and offline detection."""

    def __init__(
        self,
        mode: str = "bbox",  # "bbox", "seg", or "both"
        target_objects: str | Sequence[str] | None = None,
        conf_threshold: float = 0.25,
        device: str | None = None,
    ):
        self.mode = mode.lower()
        self.target_objects = target_objects
        self.conf_threshold = conf_threshold
        self.device = device

    @abstractmethod
    def predict(self, image: np.ndarray) -> DetectionResult:
        """Run detection on a single RGB frame (H, W, 3)."""
        pass

    def predict_batch(self, images: list[np.ndarray]) -> list[DetectionResult]:
        """Run detection on a batch of RGB frames. Default falls back to sequential predict."""
        return [self.predict(img) for img in images]

    def annotate(
        self,
        image: np.ndarray,
        result: DetectionResult | None = None,
        mode: str | None = None,
        target_objects: str | Sequence[str] | None = None,
        conf_threshold: float | None = None,
        color_palette: Sequence[tuple[int, int, int]] | None = None,
        draw_labels: bool = True,
        draw_scores: bool = True,
        mask_alpha: float = 0.45,
        box_thickness: int = 2,
    ) -> np.ndarray:
        """Render detections onto the input image array.

        Args:
            image: Input RGB image (H, W, 3).
            result: Precomputed DetectionResult (if None, calls predict(image)).
            mode: 'bbox', 'seg', or 'both'. Defaults to self.mode.
            target_objects: Object filter. Defaults to self.target_objects.
            conf_threshold: Confidence filter. Defaults to self.conf_threshold.
            color_palette: Custom RGB palette.
            draw_labels: Whether to render class labels.
            draw_scores: Whether to render confidence scores.
            mask_alpha: Opacity for segmentation masks.
            box_thickness: Bounding box line thickness.

        Returns:
            Annotated RGB numpy image array.
        """
        active_mode = (mode or self.mode).lower()
        active_targets = target_objects if target_objects is not None else self.target_objects
        active_conf = conf_threshold if conf_threshold is not None else self.conf_threshold

        if result is None:
            result = self.predict(image)

        filtered = result.filter_by_targets(active_targets, active_conf)
        annotated = image.copy()

        # Render segmentation masks first so boxes draw cleanly on top
        if active_mode in ("seg", "mask", "both"):
            for det in filtered.detections:
                if det.mask is not None and det.mask.mask is not None:
                    color = get_color_for_class(det.class_name, color_palette)
                    annotated = draw_segmentation_mask(
                        annotated,
                        det.mask.mask,
                        color=color,
                        alpha=mask_alpha,
                        contour_thickness=box_thickness,
                    )

        # Render bounding boxes
        if active_mode in ("bbox", "box", "both"):
            for det in filtered.detections:
                if det.box is not None:
                    color = get_color_for_class(det.class_name, color_palette)
                    annotated = draw_bounding_box(
                        annotated,
                        det.box.xyxy,
                        label=det.class_name if draw_labels else None,
                        score=det.confidence if draw_scores else None,
                        color=color,
                        thickness=box_thickness,
                        draw_label=draw_labels,
                        draw_score=draw_scores,
                    )

        return annotated

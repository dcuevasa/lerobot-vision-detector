"""Mock Detector for unit tests and synthetic testing without model weights."""

from __future__ import annotations

import time
from typing import Sequence
import numpy as np

from .base import BaseDetector, Box, Detection, DetectionResult, Mask


class MockDetector(BaseDetector):
    """Synthetic detector that returns predictable bounding boxes or masks."""

    def __init__(
        self,
        mode: str = "bbox",
        target_objects: str | Sequence[str] | None = None,
        conf_threshold: float = 0.25,
        default_class: str = "cup",
        mock_bbox: tuple[float, float, float, float] = (100.0, 100.0, 200.0, 200.0),
    ):
        super().__init__(
            mode=mode,
            target_objects=target_objects,
            conf_threshold=conf_threshold,
            device="cpu",
        )
        self.default_class = default_class
        self.mock_bbox = mock_bbox

    def predict(self, image: np.ndarray) -> DetectionResult:
        start_t = time.perf_counter()
        h, w = image.shape[:2]

        x1, y1, x2, y2 = self.mock_bbox
        x1 = min(x1, w - 10)
        y1 = min(y1, h - 10)
        x2 = min(x2, w - 5)
        y2 = min(y2, h - 5)

        box_obj = Box(
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
            confidence=0.95,
            class_id=0,
            class_name=self.default_class,
        )

        mask_np = np.zeros((h, w), dtype=bool)
        mask_np[int(y1):int(y2), int(x1):int(x2)] = True

        mask_obj = Mask(
            mask=mask_np,
            confidence=0.95,
            class_id=0,
            class_name=self.default_class,
        )

        det = Detection(
            class_name=self.default_class,
            class_id=0,
            confidence=0.95,
            box=box_obj,
            mask=mask_obj,
        )

        latency_ms = (time.perf_counter() - start_t) * 1e3
        return DetectionResult(
            detections=[det],
            image_shape=image.shape,
            latency_ms=latency_ms,
        )

    def predict_batch(self, images: list[np.ndarray]) -> list[DetectionResult]:
        return [self.predict(img) for img in images]

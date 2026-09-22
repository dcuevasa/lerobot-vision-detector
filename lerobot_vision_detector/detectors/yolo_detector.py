"""YOLO Detector implementation supporting both 2D Bounding Boxes and Instance Segmentation Masks."""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence
import numpy as np
import torch

from ..utils.env_check import find_and_import_ultralytics
from .base import BaseDetector, Box, Detection, DetectionResult, Mask

logger = logging.getLogger("lerobot_vision_detector")


class YOLODetector(BaseDetector):
    """High performance YOLO detector supporting Bounding Boxes and Instance Segmentation."""

    def __init__(
        self,
        model_name_or_path: str = "yolov8n.pt",
        mode: str = "bbox",  # "bbox", "seg", or "both"
        target_objects: str | Sequence[str] | None = None,
        conf_threshold: float = 0.25,
        device: str | None = None,
        half: bool = False,
    ):
        super().__init__(
            mode=mode,
            target_objects=target_objects,
            conf_threshold=conf_threshold,
            device=device,
        )
        self.model_name_or_path = model_name_or_path
        self.half = half

        # Auto-detect device if unspecified
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load YOLO model
        ultralytics = find_and_import_ultralytics()
        YOLO = ultralytics.YOLO
        logger.info(f"Loading YOLO model '{model_name_or_path}' on device '{self.device}'...")
        self.model = YOLO(model_name_or_path)

        # Warm up if using GPU
        if "cuda" in self.device:
            try:
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                self.model(dummy, verbose=False, device=self.device)
            except Exception as e:
                logger.warning(f"Warmup inference failed: {e}")

    def _parse_yolo_result(self, res: Any, start_t: float) -> DetectionResult:
        """Convert a single ultralytics Result object into DetectionResult."""
        detections: list[Detection] = []
        names = getattr(self.model, "names", {})

        boxes = getattr(res, "boxes", None)
        masks = getattr(res, "masks", None)

        has_boxes = boxes is not None and len(boxes) > 0
        has_masks = masks is not None and len(masks) > 0

        num_items = len(boxes) if has_boxes else (len(masks) if has_masks else 0)

        for i in range(num_items):
            box_obj: Box | None = None
            mask_obj: Mask | None = None

            conf = 0.0
            cls_id = -1
            cls_name = "object"

            if has_boxes:
                b = boxes[i]
                xyxy = b.xyxy[0].cpu().numpy()
                conf = float(b.conf[0].cpu().numpy())
                cls_id = int(b.cls[0].cpu().numpy())
                cls_name = str(names.get(cls_id, cls_id))
                box_obj = Box(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                )

            if has_masks and i < len(masks):
                m = masks[i]
                # mask data is tensor (1, H, W)
                mask_np = m.data[0].cpu().numpy().astype(bool)
                if not has_boxes:
                    conf = float(getattr(m, "conf", [1.0])[0]) if hasattr(m, "conf") else 1.0
                    cls_id = int(getattr(m, "cls", [0])[0]) if hasattr(m, "cls") else 0
                    cls_name = str(names.get(cls_id, cls_id))

                polygon_pts = None
                if hasattr(m, "xy") and m.xy is not None and len(m.xy) > 0:
                    polygon_pts = m.xy[0].tolist()

                mask_obj = Mask(
                    mask=mask_np,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    polygon=polygon_pts,
                )

            detections.append(
                Detection(
                    class_name=cls_name,
                    class_id=cls_id,
                    confidence=conf,
                    box=box_obj,
                    mask=mask_obj,
                )
            )

        latency_ms = (time.perf_counter() - start_t) * 1e3
        img_shape = getattr(res, "orig_shape", None)
        return DetectionResult(
            detections=detections,
            image_shape=img_shape,
            latency_ms=latency_ms,
        )

    def predict(self, image: np.ndarray) -> DetectionResult:
        """Run inference on a single RGB numpy image array (H, W, 3)."""
        start_t = time.perf_counter()
        results = self.model(
            image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
            half=self.half,
        )
        return self._parse_yolo_result(results[0], start_t)

    def predict_batch(self, images: list[np.ndarray]) -> list[DetectionResult]:
        """Run batched inference on a list of RGB numpy image arrays."""
        if not images:
            return []
        start_t = time.perf_counter()
        results = self.model(
            images,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
            half=self.half,
        )
        # Average latency per frame in batch
        batch_latency = ((time.perf_counter() - start_t) * 1e3) / max(1, len(images))
        parsed = []
        for res in results:
            dr = self._parse_yolo_result(res, start_t)
            dr.latency_ms = batch_latency
            parsed.append(dr)
        return parsed


class YOLOBboxDetector(YOLODetector):
    """Specialized YOLO 2D Bounding Box Detector."""

    def __init__(
        self,
        model_name_or_path: str = "yolov8n.pt",
        target_objects: str | Sequence[str] | None = None,
        conf_threshold: float = 0.25,
        device: str | None = None,
        half: bool = False,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            mode="bbox",
            target_objects=target_objects,
            conf_threshold=conf_threshold,
            device=device,
            half=half,
        )


class YOLOSegDetector(YOLODetector):
    """Specialized YOLO Instance Segmentation Mask Detector."""

    def __init__(
        self,
        model_name_or_path: str = "yolov8n-seg.pt",
        target_objects: str | Sequence[str] | None = None,
        conf_threshold: float = 0.25,
        device: str | None = None,
        half: bool = False,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            mode="seg",
            target_objects=target_objects,
            conf_threshold=conf_threshold,
            device=device,
            half=half,
        )

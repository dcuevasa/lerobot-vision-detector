"""OWL-ViT / OWLv2 Open-Vocabulary Detector implementation via Hugging Face Transformers."""

from __future__ import annotations

import logging
import time
from typing import Sequence
import numpy as np
from PIL import Image
import torch

from .base import BaseDetector, Box, Detection, DetectionResult

logger = logging.getLogger("lerobot_vision_detector")


class OWLv2Detector(BaseDetector):
    """Transformer-based Open-Vocabulary Object Detector using Google's OWLv2 architecture."""

    def __init__(
        self,
        model_name_or_path: str = "google/owlv2-base-patch16-ensemble",
        mode: str = "bbox",
        target_objects: str | Sequence[str] | None = None,
        conf_threshold: float = 0.20,
        device: str | None = None,
    ):
        super().__init__(
            mode=mode,
            target_objects=target_objects,
            conf_threshold=conf_threshold,
            device=device,
        )
        self.model_name = model_name_or_path

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading OWLv2 open-vocabulary model '{self.model_name}' on '{self.device}'...")
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.processor = Owlv2Processor.from_pretrained(self.model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def _get_query_texts(self) -> list[str]:
        if not self.target_objects:
            return ["object"]
        if isinstance(self.target_objects, str):
            return [t.strip() for t in self.target_objects.split(",") if t.strip()]
        return [str(t).strip() for t in self.target_objects if str(t).strip()]

    def predict(self, image: np.ndarray) -> DetectionResult:
        start_t = time.perf_counter()
        h, w = image.shape[:2]
        pil_image = Image.fromarray(image)

        texts = self._get_query_texts()
        # OWLv2 expects list of list of strings for batch of images: [[query1, query2, ...]]
        queries = [texts]

        inputs = self.processor(text=queries, images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process detections to image coordinates
        target_sizes = torch.tensor([[h, w]], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            threshold=self.conf_threshold,
            target_sizes=target_sizes,
        )

        res = results[0]
        boxes = res["boxes"].cpu().numpy()
        scores = res["scores"].cpu().numpy()
        labels = res["labels"].cpu().numpy()

        detections: list[Detection] = []
        for box, score, label_idx in zip(boxes, scores, labels):
            class_name = texts[label_idx] if label_idx < len(texts) else f"class_{label_idx}"
            b = Box(
                x1=float(box[0]),
                y1=float(box[1]),
                x2=float(box[2]),
                y2=float(box[3]),
                confidence=float(score),
                class_id=int(label_idx),
                class_name=class_name,
            )
            detections.append(
                Detection(
                    class_name=class_name,
                    class_id=int(label_idx),
                    confidence=float(score),
                    box=b,
                )
            )

        latency_ms = (time.perf_counter() - start_t) * 1e3
        return DetectionResult(
            detections=detections,
            image_shape=image.shape,
            latency_ms=latency_ms,
        )

    def predict_batch(self, images: list[np.ndarray]) -> list[DetectionResult]:
        return [self.predict(img) for img in images]

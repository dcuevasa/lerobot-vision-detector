"""Grounded-SAM Open-Vocabulary Object Detector and Segmenter.

Combines Grounding DINO (zero-shot text-to-box grounding) and Segment Anything (SAM)
into a unified BaseDetector compatible with the lerobot-vision-detector ecosystem.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
import time
from typing import Any, Sequence, Union

import numpy as np
from PIL import Image
import torch

from ..utils.env_check import find_and_import_grounded_sam
from .base import BaseDetector, Box, Detection, DetectionResult, Mask

logger = logging.getLogger("lerobot_vision_detector")


class GroundedSAMDetector(BaseDetector):
    """Open-Vocabulary Vision Detector and Segmenter using Grounded-Segment-Anything (Grounding DINO + SAM).

    Enables prompt-conditioned zero-shot detection and segmentation on overhead and wrist camera
    streams, eliminating perspective classification failures on manipulated objects.
    """

    def __init__(
        self,
        model_name_or_path: str = "vit_h",
        mode: str = "both",  # "bbox", "seg", or "both"
        target_objects: str | Sequence[str] | None = None,
        conf_threshold: float = 0.25,
        text_threshold: float = 0.25,
        device: str | None = None,
        sam_version: str | None = None,
        grounding_config: str | Path | None = None,
        grounding_checkpoint: str | Path | None = None,
        sam_checkpoint: str | Path | None = None,
        grounded_sam_instance: Any = None,
        **kwargs: Any,
    ):
        super().__init__(
            mode=mode,
            target_objects=target_objects,
            conf_threshold=conf_threshold,
            device=device,
        )
        self.text_threshold = text_threshold
        self.last_raw_result: Any = None

        # Resolve SAM backbone version (vit_h or vit_b)
        if sam_version:
            self.sam_version = sam_version.lower()
        elif any(k in str(model_name_or_path).lower() for k in ("vit_b", "vit-b", "base")):
            self.sam_version = "vit_b"
        else:
            self.sam_version = "vit_h"

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Format initial text prompt from target objects
        self.text_prompt = self._format_text_prompt(self.target_objects)

        if grounded_sam_instance is not None:
            self.grounded_sam = grounded_sam_instance
        else:
            GroundedSAM, _ = find_and_import_grounded_sam()
            logger.info(
                f"Initializing Grounded-SAM (SAM: {self.sam_version}, device: {self.device})..."
            )
            self.grounded_sam = GroundedSAM(
                sam_version=self.sam_version,
                grounding_config=grounding_config,
                grounding_checkpoint=grounding_checkpoint,
                sam_checkpoint=sam_checkpoint,
                device=self.device,
            )

        logger.info(f"GroundedSAMDetector ready with initial prompt: '{self.text_prompt}'")

    @staticmethod
    def _format_text_prompt(target_objects: str | Sequence[str] | None) -> str:
        """Format target objects into the dot-separated string expected by Grounding DINO.

        Examples:
            'cup, plate, robot' -> 'cup . plate . robot .'
            ['cup', 'bowl']     -> 'cup . bowl .'
            None                -> 'object .'
        """
        if not target_objects:
            return "object ."

        if isinstance(target_objects, str):
            parts = [p.strip().rstrip(".") for p in re.split(r"[,.]", target_objects) if p.strip()]
        else:
            parts = [str(p).strip().rstrip(".") for p in target_objects if str(p).strip()]

        if not parts:
            return "object ."

        # Grounding DINO requires phrases separated by ' . ' and ending with ' .'
        return " . ".join(parts) + " ."

    def set_classes(self, classes: str | Sequence[str]) -> None:
        """Update target object query classes dynamically."""
        self.target_objects = classes
        self.text_prompt = self._format_text_prompt(classes)
        logger.info(f"Updated GroundedSAM target classes: '{self.text_prompt}'")

    def predict(
        self,
        image: np.ndarray | Image.Image | str | Path,
        text_prompt: str | None = None,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> DetectionResult:
        """Run open-vocabulary detection and instance segmentation on an image.

        Args:
            image: RGB numpy array (H, W, 3), PIL Image, or file path.
            text_prompt: Optional override for object query text prompt.
            box_threshold: Optional override for confidence / box threshold.
            text_threshold: Optional override for text similarity threshold.

        Returns:
            DetectionResult populated with Box and Mask instances.
        """
        start_t = time.perf_counter()

        # Resolve image to PIL and shape
        if isinstance(image, (str, Path)):
            pil_image = Image.open(str(image)).convert("RGB")
            img_shape = (pil_image.height, pil_image.width, 3)
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
            img_shape = (pil_image.height, pil_image.width, 3)
        elif isinstance(image, np.ndarray):
            # LeRobot frames are RGB uint8 (H, W, 3)
            img_shape = image.shape
            pil_image = Image.fromarray(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        prompt = text_prompt if text_prompt is not None else self.text_prompt
        box_thresh = box_threshold if box_threshold is not None else self.conf_threshold
        txt_thresh = text_threshold if text_threshold is not None else self.text_threshold

        # Run Grounded-SAM
        raw_res = self.grounded_sam.predict(
            image=pil_image,
            text_prompt=prompt,
            box_threshold=box_thresh,
            text_threshold=txt_thresh,
        )
        self.last_raw_result = raw_res

        # Convert GroundedSAMResult to DetectionResult
        detections: list[Detection] = []
        num_items = len(raw_res)

        for i in range(num_items):
            label = str(raw_res.labels[i]).strip()
            score = float(raw_res.scores[i])

            # Extract bounding box [x1, y1, x2, y2]
            box_coords = raw_res.boxes[i].cpu().tolist()
            x1, y1, x2, y2 = [float(c) for c in box_coords]

            box_obj = Box(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=score,
                class_id=i,
                class_name=label,
            )

            # Extract segmentation mask (H, W) boolean numpy
            mask_np = raw_res.masks[i].cpu().numpy().astype(bool)
            mask_obj = Mask(
                mask=mask_np,
                confidence=score,
                class_id=i,
                class_name=label,
            )

            detections.append(
                Detection(
                    class_name=label,
                    class_id=i,
                    confidence=score,
                    box=box_obj,
                    mask=mask_obj,
                )
            )

        latency_ms = (time.perf_counter() - start_t) * 1e3
        return DetectionResult(
            detections=detections,
            image_shape=img_shape,
            latency_ms=latency_ms,
        )

    def predict_batch(self, images: list[np.ndarray]) -> list[DetectionResult]:
        """Run prediction on a list of images sequentially."""
        return [self.predict(img) for img in images]

    def predict_raw(
        self,
        image: np.ndarray | Image.Image | str | Path,
        text_prompt: str | None = None,
        box_threshold: float | None = None,
        text_threshold: float | None = None,
    ) -> Any:
        """Run prediction and return the native GroundedSAMResult object directly."""
        self.predict(
            image=image,
            text_prompt=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        return self.last_raw_result

    def render(self, random_color: bool = True) -> Image.Image:
        """Render the last prediction result using GroundedSAM's built-in visualizer."""
        if self.last_raw_result is None:
            raise RuntimeError("No prediction has been executed yet. Call predict() first.")
        return self.last_raw_result.render(random_color=random_color)

    def save(self, output_dir: str | Path) -> None:
        """Save the last prediction outputs (composite, masks, metadata) into a folder."""
        if self.last_raw_result is None:
            raise RuntimeError("No prediction has been executed yet. Call predict() first.")
        self.last_raw_result.save(output_dir)

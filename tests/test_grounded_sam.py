"""Unit and integration tests for GroundedSAMDetector."""

import unittest
from pathlib import Path
import sys
from unittest.mock import MagicMock

import numpy as np
from PIL import Image
import torch

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from lerobot_vision_detector.detectors import (
    Box,
    Detection,
    DetectionResult,
    GroundedSAMDetector,
    Mask,
    make_detector,
)


class TestGroundedSAMDetector(unittest.TestCase):
    def test_text_prompt_formatting(self):
        # Test comma-separated string
        prompt1 = GroundedSAMDetector._format_text_prompt("cup, plate, robot")
        self.assertEqual(prompt1, "cup . plate . robot .")

        # Test dot-separated string
        prompt2 = GroundedSAMDetector._format_text_prompt("cup . plate . robot .")
        self.assertEqual(prompt2, "cup . plate . robot .")

        # Test list of strings
        prompt3 = GroundedSAMDetector._format_text_prompt(["cup", "bottle", "bowl"])
        self.assertEqual(prompt3, "cup . bowl . bottle ." if False else "cup . bottle . bowl .")

        # Test empty / None
        self.assertEqual(GroundedSAMDetector._format_text_prompt(None), "object .")
        self.assertEqual(GroundedSAMDetector._format_text_prompt(""), "object .")

    def test_mock_grounded_sam_prediction(self):
        # Create dummy GroundedSAMResult
        mock_raw_result = MagicMock()
        mock_raw_result.raw_image = Image.new("RGB", (640, 480))
        mock_raw_result.boxes = torch.tensor([[100.0, 50.0, 200.0, 150.0], [300.0, 100.0, 450.0, 300.0]])
        mock_raw_result.masks = torch.zeros((2, 480, 640), dtype=torch.bool)
        mock_raw_result.masks[0, 50:150, 100:200] = True
        mock_raw_result.masks[1, 100:300, 300:450] = True
        mock_raw_result.labels = ["cup", "robot"]
        mock_raw_result.scores = [0.85, 0.72]
        mock_raw_result.__len__.return_value = 2

        mock_gsam = MagicMock()
        mock_gsam.predict.return_value = mock_raw_result

        detector = GroundedSAMDetector(
            mode="both",
            target_objects="cup, robot",
            grounded_sam_instance=mock_gsam,
        )
        self.assertEqual(detector.text_prompt, "cup . robot .")

        # Predict with numpy image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        res = detector.predict(img)

        self.assertIsInstance(res, DetectionResult)
        self.assertEqual(len(res.detections), 2)
        self.assertEqual(res.image_shape, (480, 640, 3))

        # Check detection 0 (cup)
        det0 = res.detections[0]
        self.assertEqual(det0.class_name, "cup")
        self.assertAlmostEqual(det0.confidence, 0.85, places=2)
        self.assertIsNotNone(det0.box)
        self.assertEqual(det0.box.xyxy, [100.0, 50.0, 200.0, 150.0])
        self.assertIsNotNone(det0.mask)
        self.assertEqual(det0.mask.mask.shape, (480, 640))
        self.assertTrue(det0.mask.mask[100, 150])

        # Check annotation rendering
        annotated = detector.annotate(img, result=res, mode="both")
        self.assertEqual(annotated.shape, img.shape)
        self.assertTrue(np.any(annotated > 0))

        # Test dynamic set_classes
        detector.set_classes(["bowl", "bottle"])
        self.assertEqual(detector.text_prompt, "bowl . bottle .")

    def test_factory_registration(self):
        # Verify that make_detector recognizes grounded_sam
        from unittest.mock import patch

        with patch("lerobot_vision_detector.detectors.grounded_sam_detector.find_and_import_grounded_sam") as mock_import:
            mock_class = MagicMock()
            mock_import.return_value = (mock_class, MagicMock())

            det = make_detector(
                detector_type="grounded_sam",
                mode="seg",
                target_objects="mug, plate",
                conf_threshold=0.30,
            )
            self.assertIsInstance(det, GroundedSAMDetector)
            self.assertEqual(det.mode, "seg")
            self.assertEqual(det.text_prompt, "mug . plate .")
            self.assertEqual(det.conf_threshold, 0.30)


if __name__ == "__main__":
    unittest.main()

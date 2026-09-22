"""Unit tests for drawing utilities."""

import unittest
import numpy as np

import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from lerobot_vision_detector.utils.drawing import (
    draw_bounding_box,
    draw_segmentation_mask,
    get_color_for_class,
)


class TestDrawing(unittest.TestCase):
    def test_draw_bounding_box(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        box = [10, 10, 50, 50]
        annotated = draw_bounding_box(img, box, label="cup", score=0.92, color=(0, 255, 0))
        self.assertEqual(annotated.shape, (100, 100, 3))
        self.assertTrue(np.any(annotated > 0))

    def test_draw_segmentation_mask(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:60, 20:60] = True
        blended = draw_segmentation_mask(img, mask, color=(255, 0, 0), alpha=0.5)
        self.assertEqual(blended.shape, (100, 100, 3))
        self.assertTrue(np.any(blended > 0))

    def test_get_color_for_class(self):
        c1 = get_color_for_class("cup")
        c2 = get_color_for_class("cup")
        self.assertEqual(c1, c2)
        self.assertEqual(len(c1), 3)


if __name__ == "__main__":
    unittest.main()

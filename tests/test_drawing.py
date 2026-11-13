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
    draw_obscuring_mask,
    draw_segmentation_mask,
    get_color_for_class,
    parse_rgb_color,
    restrict_palette_color,
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

    def test_parse_rgb_color(self):
        self.assertEqual(parse_rgb_color("white"), (255, 255, 255))
        self.assertEqual(parse_rgb_color("255, 0, 128"), (255, 0, 128))
        self.assertEqual(parse_rgb_color("#00ff80"), (0, 255, 128))

    def test_restrict_palette_and_obscuring_mask_white(self):
        # Image with pure white background highlights, near-white, and a blue region
        img = np.full((64, 64, 3), 255, dtype=np.uint8)
        img[0:10, 0:10] = (248, 252, 255)
        img[50:64, 50:64] = (20, 90, 220)

        restricted = restrict_palette_color(img, color=(255, 255, 255), margin=16)
        self.assertFalse(np.any(np.all(restricted == 255, axis=-1)))
        self.assertTrue(np.all(restricted[0:10, 0:10] == (239, 239, 239)))
        self.assertTrue(np.all(restricted[50:64, 50:64] == (20, 90, 220)))

        mask = np.zeros((64, 64), dtype=bool)
        mask[16:40, 16:40] = True
        obscured = draw_obscuring_mask(img, mask, color=(255, 255, 255), margin=16)

        # Inside mask: strictly perfectly white (255, 255, 255)
        self.assertTrue(np.all(obscured[mask] == (255, 255, 255)))
        # Outside mask: never white or within margin=16 of white
        bg = obscured[~mask]
        self.assertFalse(np.any(np.all(bg == 255, axis=-1)))
        self.assertTrue(np.all(np.max(255 - bg.astype(np.int16), axis=-1) >= 16))


if __name__ == "__main__":
    unittest.main()

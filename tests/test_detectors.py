"""Unit tests for vision detectors."""

import unittest
import numpy as np

import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from lerobot_vision_detector.detectors import (
    Box,
    Detection,
    DetectionResult,
    Mask,
    MockDetector,
    YOLOBboxDetector,
    YOLOSegDetector,
    make_detector,
)


class TestDetectors(unittest.TestCase):
    def test_box_properties(self):
        b = Box(x1=10, y1=20, x2=50, y2=80, confidence=0.9, class_id=1, class_name="cup")
        self.assertEqual(b.width, 40)
        self.assertEqual(b.height, 60)
        self.assertEqual(b.xyxy, [10, 20, 50, 80])
        self.assertEqual(b.xywh, [10, 20, 40, 60])

    def test_detection_result_filter(self):
        det1 = Detection(
            class_name="cup",
            class_id=0,
            confidence=0.85,
            box=Box(0, 0, 10, 10, 0.85, 0, "cup"),
        )
        det2 = Detection(
            class_name="bottle",
            class_id=1,
            confidence=0.40,
            box=Box(20, 20, 30, 30, 0.40, 1, "bottle"),
        )
        res = DetectionResult(detections=[det1, det2], image_shape=(100, 100, 3))

        cup_only = res.filter_by_targets(target_objects="cup")
        self.assertEqual(len(cup_only.detections), 1)
        self.assertEqual(cup_only.detections[0].class_name, "cup")

        high_conf = res.filter_by_targets(conf_threshold=0.50)
        self.assertEqual(len(high_conf.detections), 1)
        self.assertEqual(high_conf.detections[0].class_name, "cup")

        all_res = res.filter_by_targets(target_objects="all")
        self.assertEqual(len(all_res.detections), 2)

    def test_mock_detector_bbox_and_seg(self):
        img = np.zeros((240, 320, 3), dtype=np.uint8)

        det_bbox = MockDetector(mode="bbox", default_class="cup")
        ann_bbox = det_bbox.annotate(img)
        self.assertEqual(ann_bbox.shape, img.shape)
        self.assertTrue(np.any(ann_bbox > 0))

        det_seg = MockDetector(mode="seg", default_class="cup")
        ann_seg = det_seg.annotate(img)
        self.assertEqual(ann_seg.shape, img.shape)
        self.assertTrue(np.any(ann_seg > 0))

    def test_make_detector_factory(self):
        mock = make_detector("mock", mode="bbox", default_class="tape")
        self.assertIsInstance(mock, MockDetector)
        self.assertEqual(mock.default_class, "tape")

    def test_yolo_detector_inference(self):
        img = np.zeros((320, 320, 3), dtype=np.uint8)
        yolo_bbox = make_detector("yolo", model_name="yolov8n.pt", mode="bbox")
        res = yolo_bbox.predict(img)
        self.assertIsInstance(res, DetectionResult)
        ann = yolo_bbox.annotate(img, result=res)
        self.assertEqual(ann.shape, img.shape)

        yolo_seg = make_detector("yolo", model_name="yolov8n-seg.pt", mode="seg")
        res_seg = yolo_seg.predict(img)
        self.assertIsInstance(res_seg, DetectionResult)
        ann_seg = yolo_seg.annotate(img, result=res_seg)
        self.assertEqual(ann_seg.shape, img.shape)


if __name__ == "__main__":
    unittest.main()

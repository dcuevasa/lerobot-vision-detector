"""Unit tests for standardized model registry and open-vocabulary detection."""

import unittest
from pathlib import Path
import numpy as np

import sys
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from lerobot_vision_detector.utils.model_registry import (
    get_models_dir,
    is_open_vocab_model,
    resolve_model_path,
)
from lerobot_vision_detector.detectors import (
    YOLOWorldDetector,
    make_detector,
)


class TestModelRegistryAndOpenVocab(unittest.TestCase):
    def test_models_dir_exists(self):
        models_dir = get_models_dir()
        self.assertTrue(models_dir.exists())
        self.assertTrue(models_dir.is_dir())

    def test_resolve_model_path(self):
        models_dir = get_models_dir()
        dummy_model = models_dir / "yolo11-detection-obj_s.pt"
        created_dummy = False
        if not dummy_model.exists():
            dummy_model.touch()
            created_dummy = True

        try:
            # Should resolve yolo11-detection-obj_s.pt inside models/
            resolved = resolve_model_path("yolo11-detection-obj_s.pt")
            self.assertTrue(Path(resolved).is_file())
            self.assertTrue(resolved.endswith("yolo11-detection-obj_s.pt"))

            # Fallback when file doesn't exist
            fallback = resolve_model_path("non_existent_model.pt")
            self.assertEqual(fallback, "non_existent_model.pt")
        finally:
            if created_dummy and dummy_model.exists():
                dummy_model.unlink()

    def test_is_open_vocab_model(self):
        self.assertTrue(is_open_vocab_model("yolov8s-worldv2.pt"))
        self.assertTrue(is_open_vocab_model("yolo11s-world.pt"))
        self.assertTrue(is_open_vocab_model("google/owlv2-base-patch16-ensemble"))
        self.assertFalse(is_open_vocab_model("yolov8n.pt"))
        self.assertFalse(is_open_vocab_model("yolo11-detection-obj_s.pt"))

    def test_open_vocab_detector_inference(self):
        det = make_detector(
            detector_type="open_vocab",
            target_objects=["mug", "tape", "robot gripper"],
        )
        self.assertIsInstance(det, YOLOWorldDetector)
        self.assertEqual(det.target_objects, ["mug", "tape", "robot gripper"])

        img = np.zeros((320, 320, 3), dtype=np.uint8)
        res = det.predict(img)
        self.assertIsNotNone(res)
        ann = det.annotate(img, result=res)
        self.assertEqual(ann.shape, img.shape)

    def test_open_vocab_dynamic_class_update(self):
        det = make_detector(
            detector_type="yolo_world",
            target_objects=["red cup"],
        )
        self.assertEqual(det.target_objects, ["red cup"])
        det.set_classes(["blue tape", "screwdriver"])
        self.assertEqual(det.target_objects, ["blue tape", "screwdriver"])


if __name__ == "__main__":
    unittest.main()

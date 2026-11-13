"""Unit tests for offline dataset augmentation."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import sys
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from lerobot_vision_detector.augmentation import (
    DatasetVideoAugmentor,
    extract_target_from_prompt,
    match_camera_key,
    parse_target_list,
)
from lerobot_vision_detector.detectors import make_detector


class TestDatasetAugment(unittest.TestCase):
    def test_match_camera_key(self):
        # Exact match
        self.assertTrue(match_camera_key("cam_high", {"cam_high"}))
        # Short name match from full LeRobot video key
        self.assertTrue(match_camera_key("observation.images.cam_high", {"cam_high"}))
        self.assertTrue(match_camera_key("observation.images.realsense", {"realsense"}))
        # Negative match
        self.assertFalse(match_camera_key("observation.images.cam_wrist", {"cam_high"}))
        # All match
        self.assertTrue(match_camera_key("observation.images.wrist_cam", {"all"}))

    def test_extract_target_from_prompt(self):
        candidates = "mug, bowl, remote, bottle, cellphone"
        self.assertEqual(
            parse_target_list(" mug , bowl, Mug , cellphone "),
            ["mug", "bowl", "cellphone"],
        )
        self.assertEqual(
            extract_target_from_prompt(
                "Move the mug from the pink surface to the yellow surface",
                candidates,
            ),
            "mug",
        )
        self.assertEqual(
            extract_target_from_prompt(
                "Move the cellphone from the blue surface to the pink surface",
                candidates,
            ),
            "cellphone",
        )
        # Whole-word check: 'cupboard' should not match 'cup'
        self.assertIsNone(
            extract_target_from_prompt("Put the object in the cupboard", "cup, bowl")
        )
        # Earliest mention wins; longer phrase wins tie at same start position
        self.assertEqual(
            extract_target_from_prompt(
                "Pick the measuring cup next to the bowl",
                "cup, measuring cup, bowl",
            ),
            "measuring cup",
        )
        self.assertIsNone(
            extract_target_from_prompt("Move the plate to the tray", candidates)
        )

    def test_offline_augment_single_episode_with_prompt_target(self):
        source_repo = "bendca61/mujoco-so101-cube_on_tray-mouse-v1"
        output_repo = "local/test_unittest_cube"

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            out_dir = tmp_dir / "augmented_dataset"
            mock_det = make_detector("mock", mode="bbox", default_class="cube")
            recorded_classes: list[list[str]] = []

            def _mock_set_classes(classes):
                recorded_classes.append(list(classes))
                mock_det.default_class = str(classes[0])

            mock_det.set_classes = _mock_set_classes  # type: ignore[attr-defined]
            augmentor = DatasetVideoAugmentor(
                detector=mock_det,
                mode="bbox",
                target_objects="tray, cube, bottle",
            )

            out_path = augmentor.augment(
                input_repo=source_repo,
                output_repo=output_repo,
                cameras="realsense",
                output_dir=out_dir,
                overwrite=True,
                max_episodes=1,
                target_from_prompt=True,
            )

            self.assertTrue(out_path.exists())
            self.assertTrue((out_path / "meta" / "info.json").exists())
            ann_path = out_path / "meta" / "lerobot_vision_detections.json"
            self.assertTrue(ann_path.exists())

            ann_data = json.loads(ann_path.read_text(encoding="utf-8"))
            self.assertTrue(ann_data["target_from_prompt"])
            ep0_targets = ann_data["episodes"]["0"]["target_objects"]
            self.assertEqual(len(ep0_targets), 1)
            self.assertEqual(recorded_classes, [ep0_targets])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_offline_augment_obscure_with_prompt_and_distractor(self):
        source_repo = "bendca61/mujoco-so101-cube_on_tray-mouse-v1"
        output_repo = "local/test_unittest_obscure_robot"

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            out_dir = tmp_dir / "augmented_dataset_obscure"
            mock_det = make_detector("mock", mode="obscure", default_class="cube")
            recorded_classes: list[list[str]] = []

            def _mock_set_classes(classes):
                recorded_classes.append(list(classes))
                mock_det.default_class = str(classes[0])

            mock_det.set_classes = _mock_set_classes  # type: ignore[attr-defined]
            augmentor = DatasetVideoAugmentor(
                detector=mock_det,
                mode="obscure",
                target_objects="tray, cube, bottle",
                distractor_objects="robot",
            )

            out_path = augmentor.augment(
                input_repo=source_repo,
                output_repo=output_repo,
                cameras="realsense",
                output_dir=out_dir,
                overwrite=True,
                max_episodes=1,
                target_from_prompt=True,
            )

            ann_path = out_path / "meta" / "lerobot_vision_detections.json"
            ann_data = json.loads(ann_path.read_text(encoding="utf-8"))
            self.assertTrue(ann_data["target_from_prompt"])
            self.assertEqual(ann_data["distractor_objects"], ["robot"])
            self.assertEqual(ann_data["obscure_color"], [255, 255, 255])
            ep0_targets = ann_data["episodes"]["0"]["target_objects"]
            self.assertEqual(len(ep0_targets), 1)
            # Detector queried BOTH target + 'robot' distractor on ep0, while only target is logged
            self.assertEqual(recorded_classes[-1], [*ep0_targets, "robot"])
            logged_classes = {d["class_name"] for d in ann_data["episodes"]["0"]["detections"]}
            self.assertNotIn("robot", logged_classes)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

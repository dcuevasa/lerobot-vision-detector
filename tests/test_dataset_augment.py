"""Unit tests for offline dataset augmentation."""

import shutil
import tempfile
import unittest
from pathlib import Path

import sys
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from lerobot_vision_detector.augmentation import DatasetVideoAugmentor, match_camera_key
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

    def test_offline_augment_single_episode(self):
        source_repo = "bendca61/mujoco-so101-cube_on_tray-mouse-v1"
        output_repo = "local/test_unittest_cube"

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            out_dir = tmp_dir / "augmented_dataset"
            mock_det = make_detector("mock", mode="bbox", default_class="cube")
            augmentor = DatasetVideoAugmentor(detector=mock_det, mode="bbox", target_objects="cube")

            out_path = augmentor.augment(
                input_repo=source_repo,
                output_repo=output_repo,
                cameras="realsense",
                output_dir=out_dir,
                overwrite=True,
                max_episodes=1,
            )

            self.assertTrue(out_path.exists())
            self.assertTrue((out_path / "meta" / "info.json").exists())
            self.assertTrue((out_path / "meta" / "lerobot_vision_detections.json").exists())
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

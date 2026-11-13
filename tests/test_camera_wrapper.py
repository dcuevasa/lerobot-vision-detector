"""Unit tests for camera wrappers and robot wrapping."""

import unittest
import numpy as np

import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig

from lerobot_vision_detector.detectors import make_detector
from lerobot_vision_detector.wrappers import (
    DetectedCameraWrapper,
    MultiCameraVisionWrapper,
    wrap_cameras,
    wrap_robot_cameras,
)


class DummyCamera(Camera):
    def __init__(self, fps=30, width=640, height=480):
        class DummyConfig(CameraConfig):
            pass
        super().__init__(DummyConfig(fps=fps, width=width, height=height))
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @staticmethod
    def find_cameras():
        return []

    def connect(self, warmup: bool = True):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def read(self):
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def async_read(self, timeout_ms: float = 200):
        return self.read()

    def read_latest(self, max_age_ms: int = 500):
        return self.read()


class TestCameraWrapper(unittest.TestCase):
    def test_detected_camera_wrapper(self):
        base_cam = DummyCamera(fps=30, width=640, height=480)
        mock_det = make_detector("mock", mode="bbox", default_class="cup")

        wrapper = DetectedCameraWrapper(
            base_camera=base_cam,
            detector=mock_det,
            mode="bbox",
            target_objects="cup",
        )

        self.assertEqual(wrapper.fps, 30)
        self.assertEqual(wrapper.width, 640)
        self.assertEqual(wrapper.height, 480)
        self.assertFalse(wrapper.is_connected)

        wrapper.connect()
        self.assertTrue(wrapper.is_connected)

        frame = wrapper.read()
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertTrue(np.any(frame > 0))

        latest_dets = wrapper.get_latest_detections()
        self.assertIsNotNone(latest_dets)
        self.assertEqual(len(latest_dets.detections), 1)
        self.assertEqual(latest_dets.detections[0].class_name, "cup")

        wrapper.disconnect()
        self.assertFalse(wrapper.is_connected)

    def test_multi_camera_vision_wrapper_only_specified(self):
        cams = {
            "cam_high": DummyCamera(),
            "cam_wrist": DummyCamera(),
            "realsense": DummyCamera(),
        }
        mock_det = make_detector("mock", mode="seg", default_class="bottle")

        # Only specify cam_high and realsense; cam_wrist MUST remain untouched!
        wrapped_dict = wrap_cameras(
            cameras=cams,
            camera_keys=["cam_high", "realsense"],
            detector=mock_det,
            mode="seg",
        )

        self.assertIsInstance(wrapped_dict["cam_high"], DetectedCameraWrapper)
        self.assertIsInstance(wrapped_dict["realsense"], DetectedCameraWrapper)
        self.assertNotIsInstance(wrapped_dict["cam_wrist"], DetectedCameraWrapper)
        self.assertIs(wrapped_dict["cam_wrist"], cams["cam_wrist"])

    def test_wrap_robot_cameras(self):
        class DummyRobot:
            def __init__(self):
                self.cameras = {
                    "cam_high": DummyCamera(),
                    "cam_wrist": DummyCamera(),
                }

        robot = DummyRobot()
        mock_det = make_detector("mock", mode="bbox", default_class="tape")
        wrap_robot_cameras(robot, camera_keys=["cam_high"], detector=mock_det)

        self.assertIsInstance(robot.cameras["cam_high"], DetectedCameraWrapper)
        self.assertNotIsInstance(robot.cameras["cam_wrist"], DetectedCameraWrapper)

    def test_multi_camera_wrapper_obscure_with_distractor(self):
        from lerobot_vision_detector.detectors.base import Box, Detection, DetectionResult, Mask

        class DualMockDetector:
            def __init__(self):
                self.classes = []
                self.mode = "obscure"
                self.target_objects = None
                self.conf_threshold = 0.25

            def set_classes(self, classes):
                self.classes = list(classes)

            def predict(self, image):
                h, w = image.shape[:2]
                m_target = np.zeros((h, w), dtype=bool)
                m_target[20:60, 20:60] = True
                m_robot = np.zeros((h, w), dtype=bool)
                m_robot[100:150, 100:150] = True
                return DetectionResult(
                    detections=[
                        Detection("cup", 0, 0.9, Box(20, 20, 60, 60, 0.9, 0, "cup"), Mask(m_target, 0.9, 0, "cup")),
                        Detection("robot", 1, 0.95, Box(100, 100, 150, 150, 0.95, 1, "robot"), Mask(m_robot, 0.95, 1, "robot")),
                    ],
                    image_shape=image.shape,
                )

            def annotate(self, *args, **kwargs):
                # Delegate to BaseDetector.annotate implementation via MockDetector helper
                helper = make_detector("mock", mode="obscure")
                return helper.annotate(*args, **kwargs)

        det = DualMockDetector()
        cams = {"cam_high": DummyCamera()}
        wrapper = MultiCameraVisionWrapper(
            cameras=cams,
            camera_keys=["cam_high"],
            detector=det,  # type: ignore[arg-type]
            mode="obscure",
            target_objects="cup",
            distractor_objects="robot",
        )
        self.assertEqual(det.classes, ["cup", "robot"])
        frame = wrapper.get_cameras()["cam_high"].read()
        # Cup masked in solid pure white; robot region stays dark background (untouched by mask)
        self.assertTrue(np.all(frame[20:60, 20:60] == (255, 255, 255)))
        self.assertTrue(np.all(frame[100:150, 100:150] == (0, 0, 0)))


if __name__ == "__main__":
    unittest.main()

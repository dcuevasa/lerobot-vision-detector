"""Real-time Teleoperation with Multi-Camera YOLO Vision Wrapping.

Extends standard LeRobot teleoperation to display live bounding boxes or segmentation masks
in Rerun without recording datasets.
"""

from dataclasses import dataclass, field, asdict
import logging
from pprint import pformat
import time

from lerobot.configs import parser
from lerobot.processor import make_default_processors
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleop_loop
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun

from ..wrappers.robot_wrapper import wrap_robot_cameras
from .record import VisionDetectorConfig

logger = logging.getLogger("lerobot_vision_teleoperate")


@dataclass
class DetectedTeleoperateConfig(TeleoperateConfig):
    """Teleoperation configuration with vision detector options."""
    detector: VisionDetectorConfig = field(default_factory=VisionDetectorConfig)


@parser.wrap()
def teleoperate(cfg: DetectedTeleoperateConfig) -> None:
    """Main teleoperation loop with vision detection wrapper."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="teleoperation-detected", ip=cfg.display_ip, port=cfg.display_port)

    # 1. Instantiate Robot and Teleoperator
    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    # 2. Inject Vision Detector Wrappers into Specified Cameras
    if cfg.detector.enabled and hasattr(robot, "cameras") and robot.cameras:
        target_cams = None
        if cfg.detector.cameras and cfg.detector.cameras != "all":
            target_cams = [c.strip() for c in cfg.detector.cameras.split(",") if c.strip()]

        target_objs = None
        if cfg.detector.target_objects:
            target_objs = [t.strip() for t in cfg.detector.target_objects.split(",") if t.strip()]

        logger.info(
            f"Wrapping cameras with live {cfg.detector.mode.upper()} detector "
            f"(targets={target_objs}, cameras={target_cams or list(robot.cameras.keys())})"
        )
        wrap_robot_cameras(
            robot=robot,
            camera_keys=target_cams,
            mode=cfg.detector.mode,
            target_objects=target_objs,
            conf_threshold=cfg.detector.conf_threshold,
            model_name=cfg.detector.model_name,
            device=cfg.detector.device,
        )

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    robot.connect()
    teleop.connect()

    print("\n=================================================")
    print("Teleoperation active with live YOLO detections!")
    print("Visualizing in Rerun. Press Ctrl+C to exit.")
    print("=================================================\n")

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            display_compressed_images=cfg.display_compressed_images,
        )
    finally:
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()


if __name__ == "__main__":
    teleoperate()

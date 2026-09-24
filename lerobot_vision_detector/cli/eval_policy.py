"""Real-time Policy Evaluation with Multi-Camera YOLO Vision Wrapping.

Enables evaluating trained policies (ACT, Diffusion, etc.) on robots where the policy
depends on bounding boxes or segmentation masks on specific cameras.
"""

from dataclasses import dataclass, field, asdict
import logging
from pprint import pformat
import time

from lerobot.utils.import_utils import register_third_party_plugins

# Discover and register all third-party plugins (robots, teleoperators, policies)
try:
    register_third_party_plugins()
except Exception:
    pass

try:
    import lerobot_teleoperator_so101_ik.so101physicalwrapper  # noqa: F401
except ImportError:
    pass

from lerobot.configs import parser
from lerobot.policies.pretrained import PreTrainedConfig
from lerobot.robots.config import RobotConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.robots import make_robot_from_config
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun

from ..wrappers.robot_wrapper import wrap_robot_cameras

logger = logging.getLogger("lerobot_vision_eval")


@dataclass
class VisionEvalDetectorConfig:
    enabled: bool = True
    mode: str = "bbox"  # "bbox", "seg", or "both"
    cameras: str = "all"  # Comma-separated cameras to process, or 'all'
    target_objects: str | None = None  # Comma-separated target objects
    model_name: str | None = None  # e.g. 'yolov8n.pt' or 'yolov8n-seg.pt'
    conf_threshold: float = 0.25
    device: str | None = None


@dataclass
class DetectedEvalConfig:
    robot: RobotConfig
    policy: PreTrainedConfig
    detector: VisionEvalDetectorConfig = field(default_factory=VisionEvalDetectorConfig)
    display_data: bool = False
    display_ip: str = "127.0.0.1"
    display_port: int | None = None
    eval_fps: int = 30


@parser.wrap()
def eval_policy(cfg: DetectedEvalConfig) -> None:
    """Run real-time policy evaluation with YOLO-wrapped cameras."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="eval-detected-policy", ip=cfg.display_ip, port=cfg.display_port)

    # 1. Instantiate Robot
    robot = make_robot_from_config(cfg.robot)

    # 2. Inject Vision Detector Wrappers into Specified Cameras
    if cfg.detector.enabled and hasattr(robot, "cameras") and robot.cameras:
        target_cams = None
        if cfg.detector.cameras and cfg.detector.cameras != "all":
            target_cams = [c.strip() for c in cfg.detector.cameras.split(",") if c.strip()]

        target_objs = None
        if cfg.detector.target_objects:
            target_objs = [t.strip() for t in cfg.detector.target_objects.split(",") if t.strip()]

        logger.info(f"Wrapping cameras with real-time {cfg.detector.mode.upper()} detector for policy execution...")
        wrap_robot_cameras(
            robot=robot,
            camera_keys=target_cams,
            mode=cfg.detector.mode,
            target_objects=target_objs,
            conf_threshold=cfg.detector.conf_threshold,
            model_name=cfg.detector.model_name,
            device=cfg.detector.device,
        )

    # 3. Load Policy and Processors
    policy = make_policy(cfg.policy)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": cfg.policy.device}},
    )

    robot.connect()
    policy.eval()

    logger.info("Starting policy evaluation loop with live vision detections...")
    try:
        dt = 1.0 / cfg.eval_fps if hasattr(cfg, "eval_fps") and cfg.eval_fps else 1.0 / 30.0
        while True:
            t0 = time.perf_counter()
            obs = robot.get_observation()
            processed_obs = preprocessor(obs) if preprocessor else obs
            action = policy.select_action(processed_obs)
            robot_action = postprocessor(action) if postprocessor else action
            robot.send_action(robot_action)
            elapsed = time.perf_counter() - t0
            sleep_t = max(0.0, dt - elapsed)
            time.sleep(sleep_t)
    except KeyboardInterrupt:
        logger.info("Evaluation stopped by user.")
    finally:
        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    eval_policy()

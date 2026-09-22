"""Real-time Dataset Recording with Multi-Camera YOLO Vision Wrapping.

Inspired by FLAG-Embodied-data/lerobot_record_physical.py and scripts/coffee.
Wraps specified physical or simulated robot cameras with live YOLO detections,
allowing episode recording directly into LeRobot datasets with detections embedded.
"""

from dataclasses import dataclass, field, asdict
import logging
from pathlib import Path
from pprint import pformat
import sys
import time
from typing import Any, Sequence

from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_record import RecordConfig, record_loop
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.control_utils import (
    is_headless,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun

from ..wrappers.robot_wrapper import wrap_robot_cameras

logger = logging.getLogger("lerobot_vision_record")


@dataclass
class VisionDetectorConfig:
    """Configuration for real-time vision detection on camera feeds."""
    enabled: bool = True
    mode: str = "bbox"  # "bbox", "seg", or "both"
    cameras: str = "all"  # Comma-separated camera names, or 'all'
    target_objects: str | None = None  # Comma-separated target classes, or None for all
    model_name: str | None = None  # e.g. 'yolov8n.pt' or 'yolov8n-seg.pt'
    conf_threshold: float = 0.25
    device: str | None = None


@dataclass
class DetectedRecordConfig(RecordConfig):
    """Extends LeRobot RecordConfig with Vision Detector options."""
    detector: VisionDetectorConfig = field(default_factory=VisionDetectorConfig)


def init_manual_episode_listener() -> tuple[object | None, dict[str, bool]]:
    """Terminal keyboard listener for manual episode acceptance/discard controls:
    - Enter: start episode
    - Right: accept/save episode
    - Left: discard current buffered episode
    - Space: skip/retry without saving
    - Esc: stop recording session
    """
    events = {
        "start_episode": False,
        "accept_episode": False,
        "discard_episode": False,
        "skip_attempt": False,
        "stop_recording": False,
        "exit_early": False,
        "rerecord_episode": False,
    }

    if is_headless():
        logging.warning("Headless environment detected. Auto-starting and auto-accepting episodes.")
        events["start_episode"] = True
        events["accept_episode"] = True
        return None, events

    import threading

    class TerminalKeyListener(threading.Thread):
        def __init__(self):
            super().__init__(daemon=True)
            self.running = True

        def run(self):
            try:
                import msvcrt
                self._run_windows()
            except ImportError:
                self._run_unix()

        def _handle_key(self, key_name: str):
            if key_name == "enter":
                print("\r[KEY] Enter -> start episode")
                events["start_episode"] = True
            elif key_name == "right":
                print("\r[KEY] Right -> accept episode")
                events["accept_episode"] = True
                events["exit_early"] = True
            elif key_name == "left":
                print("\r[KEY] Left -> discard episode")
                events["discard_episode"] = True
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key_name == "space":
                print("\r[KEY] Space -> skip/redo attempt")
                events["skip_attempt"] = True
                events["exit_early"] = True
            elif key_name == "esc":
                print("\r[KEY] Esc -> stop recording")
                events["stop_recording"] = True
                events["exit_early"] = True

        def _run_windows(self):
            import msvcrt
            while self.running:
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch in (b'\x00', b'\xe0'):
                        ch2 = msvcrt.getch()
                        if ch2 == b'K': self._handle_key("left")
                        elif ch2 == b'M': self._handle_key("right")
                    elif ch == b'\r': self._handle_key("enter")
                    elif ch == b' ': self._handle_key("space")
                    elif ch == b'\x1b': self._handle_key("esc")
                else:
                    time.sleep(0.01)

        def _run_unix(self):
            import select, termios, tty
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while self.running:
                    [r, _, _] = select.select([fd], [], [], 0.05)
                    if r:
                        ch = os.read(fd, 1)
                        if ch == b'\x1b':
                            [r2, _, _] = select.select([fd], [], [], 0.1)
                            if r2:
                                seq = os.read(fd, 2)
                                if seq in (b'[D', b'OD'): self._handle_key("left")
                                elif seq in (b'[C', b'OC'): self._handle_key("right")
                            else:
                                self._handle_key("esc")
                        elif ch in (b'\n', b'\r'): self._handle_key("enter")
                        elif ch == b' ': self._handle_key("space")
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        def stop(self):
            self.running = False

    listener = TerminalKeyListener()
    listener.start()
    return listener, events


@parser.wrap()
def record(cfg: DetectedRecordConfig) -> LeRobotDataset:
    """Main recording loop with vision detection wrapper."""
    init_logging()
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="recording-detected-episodes", ip=cfg.display_ip, port=cfg.display_port)

    # 1. Instantiate Robot and Teleoperator
    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # 2. Inject Vision Detector Wrappers into Robot Cameras
    if cfg.detector.enabled and hasattr(robot, "cameras") and robot.cameras:
        target_cams = None
        if cfg.detector.cameras and cfg.detector.cameras != "all":
            target_cams = [c.strip() for c in cfg.detector.cameras.split(",") if c.strip()]

        target_objs = None
        if cfg.detector.target_objects:
            target_objs = [t.strip() for t in cfg.detector.target_objects.split(",") if t.strip()]

        logger.info(
            f"Wrapping cameras with {cfg.detector.mode.upper()} detector "
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

    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    dataset = None
    listener, events = init_manual_episode_listener()

    try:
        if cfg.resume:
            dataset = LeRobotDataset(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )
            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_processes,
                    num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )

        policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
        preprocessor = None
        postprocessor = None
        if cfg.policy is not None:
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=cfg.policy,
                pretrained_path=cfg.policy.pretrained_path,
                dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
                preprocessor_overrides={
                    "device_processor": {"device": cfg.policy.device},
                    "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
                },
            )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        captured_episodes = 0
        total_episodes = cfg.dataset.num_episodes

        print("\n=================================================")
        print("Ready to record with real-time Vision Detections!")
        print("Controls: Enter=Start | Right=Save | Left=Discard | Space=Retry | Esc=Exit")
        print("=================================================\n")

        while captured_episodes < total_episodes and not events["stop_recording"]:
            print(f"Waiting to start episode {captured_episodes + 1}/{total_episodes} (Press Enter)...")
            while not events["start_episode"] and not events["stop_recording"]:
                time.sleep(0.05)

            if events["stop_recording"]:
                break

            events["start_episode"] = False
            events["exit_early"] = False
            events["accept_episode"] = False
            events["discard_episode"] = False

            print(f"Recording episode {captured_episodes + 1}/{total_episodes}...")
            start_time = time.perf_counter()
            max_duration = cfg.dataset.episode_time_s

            while True:
                elapsed = time.perf_counter() - start_time
                if events["stop_recording"] or events["exit_early"] or (max_duration and elapsed >= max_duration):
                    break

                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=1.0 / cfg.dataset.fps,
                    display_data=cfg.display_data,
                    single_task=cfg.dataset.single_task,
                )

            if events["discard_episode"] or events["skip_attempt"]:
                print(f"Episode {captured_episodes + 1} discarded.")
                dataset.clear_episode_buffer()
            else:
                print(f"Episode {captured_episodes + 1} saved successfully!")
                dataset.save_episode()
                captured_episodes += 1

        print(f"\nRecording completed! Total episodes captured: {captured_episodes}")
        dataset.finalize()

        if cfg.dataset.push_to_hub:
            print(f"Pushing dataset to Hugging Face Hub: {cfg.dataset.repo_id}...")
            dataset.push_to_hub()
            print("Upload complete!")

    finally:
        if listener is not None:
            listener.stop()
        if robot.is_connected:
            robot.disconnect()
        if teleop is not None and teleop.is_connected:
            teleop.disconnect()

    return dataset


if __name__ == "__main__":
    record()

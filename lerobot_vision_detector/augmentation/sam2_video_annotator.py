"""SAM 2 Video Annotation Engine for LeRobot Datasets.

This module provides spatio-temporal segmentation mask and bounding box propagation
using Meta's SAM 2 (Segment Anything Model 2) video predictor. It resolves single-frame
detection failures (such as top-down perspective errors and gripper occlusions during grasps).

Supports dual-mode interaction:
1. Headless CLI / Remote SSH terminal execution.
2. Interactive Web UI (Gradio) for visual point-prompt placement and inspection.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Sequence

import cv2
from huggingface_hub import HfApi, snapshot_download
import numpy as np
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

from sam2.build_sam import build_sam2_video_predictor

logger = logging.getLogger("lerobot_vision_detector.sam2")

DEFAULT_IGNORE_FEATURES = {
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
    "subtask_index",
    "subtask",
    "next.reward",
    "next.done",
    "next.truncated",
}


def match_camera_key(video_key: str, specified_cameras: set[str]) -> bool:
    """Check if a video feature key matches any of the specified camera names."""
    if not specified_cameras:
        return False
    if "all" in specified_cameras or "*" in specified_cameras:
        return True
    if video_key in specified_cameras:
        return True
    short_key = video_key.split(".")[-1]
    if short_key in specified_cameras:
        return True
    for spec in specified_cameras:
        if spec in video_key or video_key in spec:
            return True
    return False


class SAM2VideoAnnotator:
    """Propagates visual prompts (bounding box / instance mask) across LeRobot video episodes."""

    def __init__(
        self,
        model_cfg: str = "configs/sam2.1/sam2.1_hiera_b+.yaml",
        checkpoint_path: str = "/workspace/sam2.1_b.pt",
        device: str = "cuda",
        mode: str = "both",  # "bbox", "mask", or "both"
        label: str = "cup",
        color: tuple[int, int, int] = (0, 255, 0),  # Green in RGB
        mask_alpha: float = 0.40,
        box_thickness: int = 2,
    ):
        self.model_cfg = model_cfg
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.mode = mode.lower()
        self.label = label
        self.color = color
        self.mask_alpha = mask_alpha
        self.box_thickness = box_thickness

        logger.info(f"Loading SAM 2 Video Predictor ({model_cfg}) from {checkpoint_path}...")
        self.predictor = build_sam2_video_predictor(
            config_file=self.model_cfg,
            ckpt_path=self.checkpoint_path,
            device=self.device,
        )
        logger.info("SAM 2 Video Predictor initialized successfully.")

    def detect_auto_seed(self, frame_rgb: np.ndarray) -> tuple[int, int]:
        """Detect initial seed point on Frame 0 using foreground contrast heuristics.

        In the surface transfer task, the mug is placed on a colored surface plate.
        If YOLO proposal or plate detection is unavailable, defaults to empirical mug center (435, 122).
        """
        try:
            hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
            # Find colored plates (saturated regions)
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            plate_mask = (sat > 60) & (val > 50)
            
            # Find contours of colored plates
            contours, _ = cv2.findContours(plate_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Find the plate contour in the upper half of the table
                valid_contours = [c for c in contours if cv2.contourArea(c) > 1500]
                if valid_contours:
                    # Sort by area descending
                    valid_contours.sort(key=cv2.contourArea, reverse=True)
                    # Pick largest contour
                    largest = valid_contours[0]
                    M = cv2.moments(largest)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        logger.info(f"Auto-detected seed point from surface centroid: ({cx}, {cy})")
                        return (cx, cy)
        except Exception as e:
            logger.warning(f"Auto-seed heuristic encountered error: {e}. Falling back to default.")

        # Default fallback calibrated for overhead SO-101 workspace
        fallback = (435, 122)
        logger.info(f"Using default calibrated seed point: {fallback}")
        return fallback

    def annotate_episode_frames(
        self,
        frames_rgb: np.ndarray,  # Shape (N, H, W, 3), uint8
        points: list[tuple[int, int]] | None = None,
        labels: list[int] | None = None,
        box: list[int] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Runs SAM 2 tracking on a sequence of RGB frames and renders visual prompts.

        Args:
            frames_rgb: Array of shape (N, H, W, 3) in RGB format.
            points: List of (x, y) seed points for Frame 0.
            labels: List of labels (1 for positive/foreground, 0 for negative).
            box: Optional [xmin, ymin, xmax, ymax] bounding box prompt.

        Returns:
            annotated_frames: Shape (N, H, W, 3) uint8 numpy array with prompts rendered.
            frame_detections: List of metadata dicts per frame with bounding box info.
        """
        num_frames, H, W, _ = frames_rgb.shape
        if num_frames == 0:
            return frames_rgb, []

        # If no prompt provided, run auto-seed detection on Frame 0
        if (points is None or len(points) == 0) and box is None:
            seed_x, seed_y = self.detect_auto_seed(frames_rgb[0])
            points = [(seed_x, seed_y)]
            labels = [1]
        elif points is not None and labels is None:
            labels = [1] * len(points)

        # Dump frames to a temporary directory for SAM 2 video predictor
        with tempfile.TemporaryDirectory() as tmp_dir:
            for i in range(num_frames):
                # SAM 2 expects JPEG files
                cv2.imwrite(
                    os.path.join(tmp_dir, f"{i:05d}.jpg"),
                    cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR),
                )

            # Initialize SAM 2 video state
            inference_state = self.predictor.init_state(video_path=tmp_dir)

            # Add prompts to Frame 0
            pts_arr = np.array(points, dtype=np.float32) if points else None
            lbls_arr = np.array(labels, dtype=np.int32) if labels else None
            box_arr = np.array(box, dtype=np.float32) if box else None

            self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=pts_arr,
                labels=lbls_arr,
                box=box_arr,
            )

            # Propagate masks across entire episode
            annotated_frames = frames_rgb.copy()
            frame_detections: list[dict[str, Any]] = []

            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                # out_mask_logits: (num_objects, 1, H, W)
                mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy()  # (H, W) boolean

                y_indices, x_indices = np.where(mask)
                has_mask = len(x_indices) > 0

                bbox = None
                if has_mask:
                    x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
                    y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
                    bbox = [x_min, y_min, x_max, y_max]

                frame_detections.append({
                    "frame": int(out_frame_idx),
                    "label": self.label,
                    "has_mask": bool(has_mask),
                    "box_xyxy": bbox,
                })

                if not has_mask:
                    continue

                curr_frame = annotated_frames[out_frame_idx]

                # Render Instance Mask overlay
                if self.mode in ("mask", "both"):
                    colored_mask = np.zeros_like(curr_frame)
                    colored_mask[mask] = self.color
                    # Alpha blend
                    blended = np.where(
                        mask[:, :, None],
                        (curr_frame * (1.0 - self.mask_alpha) + colored_mask * self.mask_alpha).astype(np.uint8),
                        curr_frame,
                    )
                    curr_frame[:] = blended

                # Render Bounding Box overlay
                if self.mode in ("bbox", "both") and bbox is not None:
                    x1, y1, x2, y2 = bbox
                    # Convert to BGR for cv2 drawing then back or draw directly on RGB
                    # cv2 works in place on RGB with color tuple
                    cv2.rectangle(
                        curr_frame,
                        (x1, y1),
                        (x2, y2),
                        color=self.color,
                        thickness=self.box_thickness,
                    )
                    # Text label header
                    label_text = f"{self.label}"
                    (text_w, text_h), baseline = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    text_y1 = max(y1 - 4, text_h + 2)
                    cv2.rectangle(
                        curr_frame,
                        (x1, text_y1 - text_h - 2),
                        (x1 + text_w + 4, text_y1 + 2),
                        self.color,
                        -1,
                    )
                    cv2.putText(
                        curr_frame,
                        label_text,
                        (x1 + 2, text_y1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

        return annotated_frames, frame_detections

    def augment(
        self,
        input_repo: str,
        output_repo: str,
        cameras: str | Sequence[str] = "cam_high",
        prompt_points: dict[int, list[tuple[int, int]]] | None = None,
        default_point: tuple[int, int] | None = None,
        auto_prompt: bool = True,
        output_dir: str | Path | None = None,
        local_dir: str | Path | None = None,
        push_to_hub: bool = False,
        hf_token: str | None = None,
        overwrite: bool = True,
        vcodec: str = "libsvtav1",
        max_episodes: int | None = None,
    ) -> Path:
        """Runs the SAM 2 augmentation across a complete LeRobot dataset.

        Args:
            input_repo: Source LeRobot dataset identifier or local path.
            output_repo: Target repo ID for the augmented dataset.
            cameras: Cameras to annotate (unspecified cameras are preserved untouched).
            prompt_points: Optional dict mapping episode index -> list of (x, y) seed points.
            default_point: Fallback seed point for all episodes if not in prompt_points.
            auto_prompt: If True and no prompt is provided, auto-detects seed on Frame 0.
            output_dir: Destination path for the new dataset.
            local_dir: Directory containing cached source dataset.
            push_to_hub: If True, uploads dataset to Hugging Face Hub.
            hf_token: Hugging Face API token.
            overwrite: If True, deletes existing destination directory.
            vcodec: Video codec for encoding (default libsvtav1).
            max_episodes: Limit number of episodes to process (for quick testing).

        Returns:
            Path to the saved augmented dataset.
        """
        # Parse camera keys
        if isinstance(cameras, str):
            specified_cams = {c.strip() for c in cameras.split(",") if c.strip()}
        else:
            specified_cams = {str(c).strip() for c in cameras if str(c).strip()}

        logger.info(f"Starting SAM 2 Video Augmentation Pipeline")
        logger.info(f"Source: {input_repo} -> Destination: {output_repo}")
        logger.info(f"Target Camera(s): {sorted(specified_cams)}, Mode: {self.mode.upper()}")

        # 1. Load source dataset
        if local_dir:
            src_root = Path(local_dir)
        else:
            src_root = Path.home() / ".cache" / "huggingface" / "lerobot" / input_repo

        if not (src_root / "meta" / "info.json").exists() and not Path(input_repo).exists():
            logger.info(f"Downloading source dataset '{input_repo}'...")
            src_root.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=input_repo, repo_type="dataset", local_dir=str(src_root), token=hf_token)

        src_ds = LeRobotDataset(input_repo, root=src_root)
        video_keys = list(src_ds.meta.video_keys)
        process_video_keys = [vk for vk in video_keys if match_camera_key(vk, specified_cams)]
        passthrough_keys = [vk for vk in video_keys if not match_camera_key(vk, specified_cams)]

        if not process_video_keys:
            raise ValueError(f"No cameras in {video_keys} matched specified {specified_cams}")

        logger.info(f"Cameras to augment with SAM 2: {process_video_keys}")
        logger.info(f"Cameras to preserve unmodified: {passthrough_keys}")

        # 2. Output path setup
        if output_dir is None:
            safe_slug = output_repo.replace("/", "_")
            out_path = Path.cwd() / "outputs" / safe_slug
        else:
            out_path = Path(output_dir)

        if out_path.exists():
            if overwrite:
                logger.info(f"Removing existing output directory: {out_path}")
                shutil.rmtree(out_path)
            else:
                raise FileExistsError(f"Output directory {out_path} already exists. Pass overwrite=True.")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 3. Schema & LeRobotDataset creation
        data_features = {k: dict(v) for k, v in src_ds.meta.features.items() if k not in DEFAULT_IGNORE_FEATURES}
        fps = float(src_ds.meta.fps)

        create_kwargs: dict[str, Any] = {
            "repo_id": output_repo,
            "fps": src_ds.meta.fps,
            "root": out_path,
            "robot_type": src_ds.meta.robot_type,
            "features": data_features,
            "use_videos": len(video_keys) > 0,
            "image_writer_processes": 0,
            "image_writer_threads": 4,
        }

        import inspect
        create_sig_params = inspect.signature(LeRobotDataset.create).parameters
        if "rgb_encoder" in create_sig_params:
            try:
                from lerobot.configs.video import RGBEncoderConfig
                create_kwargs["rgb_encoder"] = RGBEncoderConfig(vcodec=vcodec, video_backend="pyav")
            except Exception:
                pass
            if "video_backend" in create_sig_params:
                create_kwargs["video_backend"] = "pyav"
        elif "vcodec" in create_sig_params:
            create_kwargs["vcodec"] = vcodec

        new_ds = LeRobotDataset.create(**create_kwargs)

        total_episodes = src_ds.meta.total_episodes
        if max_episodes is not None and max_episodes > 0:
            total_episodes = min(total_episodes, max_episodes)

        all_detections_meta: dict[str, Any] = {}

        # 4. Process each episode
        for ep_idx in tqdm(range(total_episodes), desc=f"SAM 2 Augmenting {output_repo}"):
            ep_meta = src_ds.meta.episodes[ep_idx]
            ep_from_idx = int(ep_meta["dataset_from_index"])
            ep_to_idx = int(ep_meta["dataset_to_index"])
            num_frames = int(ep_meta["length"])

            if num_frames <= 0:
                continue

            hf_slice = src_ds.hf_dataset.select(range(ep_from_idx, ep_to_idx))
            decoded_videos: dict[str, np.ndarray] = {}

            # Decode all cameras
            backend = getattr(src_ds, "video_backend", getattr(src_ds, "_video_backend", "pyav"))
            tol = getattr(src_ds, "tolerance_s", getattr(src_ds, "_tolerance_s", 0.0001))

            for vk in video_keys:
                from_ts = float(ep_meta[f"videos/{vk}/from_timestamp"])
                shifted_ts = [from_ts + i * (1.0 / fps) for i in range(num_frames)]
                vid_file_rel = src_ds.meta.get_video_file_path(ep_idx, vk)
                vid_path = src_ds.root / vid_file_rel

                frames_tensor = decode_video_frames(vid_path, shifted_ts, tol, backend)
                frames_np = (
                    (frames_tensor * 255.0)
                    .clamp(0, 255)
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                decoded_videos[vk] = frames_np

            # Determine prompt for this episode
            ep_points = None
            if prompt_points and ep_idx in prompt_points:
                ep_points = prompt_points[ep_idx]
            elif default_point is not None:
                ep_points = [default_point]

            # Annotate specified cameras with SAM 2
            ep_det_records = []
            for vk in process_video_keys:
                raw_cam_frames = decoded_videos[vk]
                annotated_cam, det_records = self.annotate_episode_frames(
                    frames_rgb=raw_cam_frames,
                    points=ep_points,
                )
                decoded_videos[vk] = annotated_cam
                ep_det_records.extend(det_records)

            all_detections_meta[str(ep_idx)] = {"frames": ep_det_records}

            # Resolve task string
            task_str = "default_task"
            if hasattr(src_ds.meta, "tasks") and src_ds.meta.tasks is not None:
                if hasattr(src_ds.meta.tasks, "index") and len(src_ds.meta.tasks) > 0:
                    task_str = str(src_ds.meta.tasks.index[0])
                elif isinstance(src_ds.meta.tasks, dict) and src_ds.meta.tasks:
                    task_str = str(list(src_ds.meta.tasks.keys())[0])

            # Write frames into new dataset
            for i in range(num_frames):
                frame_dict = {}
                for k in data_features:
                    if k in decoded_videos:
                        frame_dict[k] = decoded_videos[k][i]
                    else:
                        frame_dict[k] = hf_slice[i][k]

                current_task = task_str
                if "task" in hf_slice.column_names and hf_slice[i]["task"]:
                    current_task = str(hf_slice[i]["task"])
                elif "task_index" in hf_slice.column_names and hasattr(src_ds.meta, "tasks") and hasattr(src_ds.meta.tasks, "index"):
                    t_idx = int(hf_slice[i]["task_index"])
                    if 0 <= t_idx < len(src_ds.meta.tasks.index):
                        current_task = str(src_ds.meta.tasks.index[t_idx])

                frame_dict["task"] = current_task
                new_ds.add_frame(frame_dict)

            new_ds.save_episode()

        # 5. Finalize dataset
        logger.info("Finalizing new LeRobotDataset...")
        new_ds.finalize()

        # 6. Save annotation metadata
        meta_dir = out_path / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        with open(meta_dir / "sam2_annotations.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_dataset": input_repo,
                    "target_cameras": list(process_video_keys),
                    "mode": self.mode,
                    "model_cfg": self.model_cfg,
                    "checkpoint": os.path.basename(self.checkpoint_path),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "episodes": all_detections_meta,
                },
                f,
                indent=2,
            )

        # 7. Push to Hub if requested
        if push_to_hub:
            logger.info(f"Pushing augmented dataset to Hugging Face Hub: {output_repo}...")
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=output_repo, repo_type="dataset", exist_ok=True)
            api.upload_folder(
                repo_id=output_repo,
                repo_type="dataset",
                folder_path=str(out_path),
                commit_message=f"Upload SAM 2 visual-prompt augmented dataset ({self.mode})",
            )
            # Create codebase_version tag
            try:
                api.create_tag(repo_id=output_repo, repo_type="dataset", tag="v3.0")
                logger.info(f"Tagged Hub dataset {output_repo} with 'v3.0'")
            except Exception as tag_err:
                logger.warning(f"Could not tag repository (may already exist): {tag_err}")

        logger.info(f"Dataset augmentation completed successfully: {out_path}")
        return out_path

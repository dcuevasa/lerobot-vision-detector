"""Offline Dataset Vision Augmentation Pipeline.

Augments existing LeRobot datasets with 2D Bounding Boxes or Instance Segmentation Masks
for specified target objects across user-specified cameras, leaving unspecified cameras untouched,
and publishes the augmented dataset locally and/or to Hugging Face Hub.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Sequence
from huggingface_hub import HfApi, snapshot_download
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

from ..detectors.base import BaseDetector, DetectionResult
from ..detectors import make_detector

logger = logging.getLogger("lerobot_vision_detector")

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

    # Exact match
    if video_key in specified_cameras:
        return True

    # Short name match: "observation.images.cam_high" -> "cam_high"
    short_key = video_key.split(".")[-1]
    if short_key in specified_cameras:
        return True

    # Substring match
    for spec in specified_cameras:
        if spec in video_key or video_key in spec:
            return True

    return False


class DatasetVideoAugmentor:
    """Processes existing LeRobot datasets to render spatial detections onto video frames."""

    def __init__(
        self,
        detector: BaseDetector | None = None,
        mode: str = "bbox",  # "bbox", "seg", or "both"
        target_objects: str | Sequence[str] | None = None,
        conf_threshold: float = 0.25,
        model_name: str | None = None,
        device: str | None = None,
        half: bool = False,
        mask_alpha: float = 0.45,
        box_thickness: int = 2,
    ):
        self.mode = mode
        self.target_objects = target_objects
        self.conf_threshold = conf_threshold
        self.mask_alpha = mask_alpha
        self.box_thickness = box_thickness

        if detector is None:
            self.detector: BaseDetector = make_detector(
                detector_type="yolo",
                model_name=model_name,
                mode=mode,
                target_objects=target_objects,
                conf_threshold=conf_threshold,
                device=device,
                half=half,
            )
        else:
            self.detector = detector

    def augment(
        self,
        input_repo: str,
        output_repo: str,
        cameras: str | Sequence[str],
        output_dir: str | Path | None = None,
        local_dir: str | Path | None = None,
        push_to_hub: bool = False,
        hf_token: str | None = None,
        overwrite: bool = False,
        vcodec: str = "libsvtav1",
        batch_size: int = 16,
        max_episodes: int | None = None,
    ) -> Path:
        """Run detection pipeline on the specified cameras of an input dataset.

        Args:
            input_repo: Source LeRobot dataset repo ID or local path.
            output_repo: Target repo ID for the augmented dataset.
            cameras: List or comma-separated string of camera keys to process.
                     Cameras NOT specified will NOT be processed.
            output_dir: Local destination directory for new dataset.
            local_dir: Directory used to cache/download source dataset.
            push_to_hub: If True, upload resulting dataset to Hugging Face Hub.
            hf_token: Optional Hugging Face authorization token.
            overwrite: If True, overwrite existing output directory.
            vcodec: Video codec for encoding (default libsvtav1).
            batch_size: Frame batch size for GPU detection.
            max_episodes: Optional limit on number of episodes to process (for debugging).

        Returns:
            Path to the newly created local dataset directory.
        """
        # Normalize specified cameras
        if isinstance(cameras, str):
            specified_cams = {c.strip() for c in cameras.split(",") if c.strip()}
        else:
            specified_cams = {str(c).strip() for c in cameras if str(c).strip()}

        if not specified_cams:
            raise ValueError("No camera specified. Please specify at least one camera to process via 'cameras'.")

        logger.info(f"Target object(s): {self.target_objects}")
        logger.info(f"Detection mode: {self.mode.upper()}")
        logger.info(f"Specified camera(s) to process: {sorted(specified_cams)}")

        # 1. Load source dataset
        logger.info(f"Loading source dataset: {input_repo}...")
        src_root = Path(local_dir) if local_dir else None
        src_ds = LeRobotDataset(input_repo, root=src_root)

        # 2. Identify video keys to process vs pass through
        video_keys = list(src_ds.meta.video_keys)
        process_video_keys: list[str] = []
        passthrough_video_keys: list[str] = []

        for vk in video_keys:
            if match_camera_key(vk, specified_cams):
                process_video_keys.append(vk)
            else:
                passthrough_video_keys.append(vk)

        if not process_video_keys:
            raise ValueError(
                f"None of the specified cameras {specified_cams} matched the dataset video keys: {video_keys}."
            )

        logger.info(f"==> Cameras to be PROCESSED with YOLO detections: {process_video_keys}")
        logger.info(f"==> Cameras to be KEPT UNTOUCHED (unspecified): {passthrough_video_keys}")

        # 3. Setup output directory
        if output_dir is None:
            safe_slug = output_repo.replace("/", "_")
            out_path = Path.cwd() / "outputs" / safe_slug
        else:
            out_path = Path(output_dir)

        if out_path.exists():
            if overwrite:
                logger.info(f"Overwriting existing output directory: {out_path}")
                shutil.rmtree(out_path)
            else:
                raise FileExistsError(
                    f"Destination '{out_path}' already exists. Pass overwrite=True to replace it."
                )

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # 4. Prepare feature schema
        data_features = {}
        for k, v in src_ds.meta.features.items():
            if k in DEFAULT_IGNORE_FEATURES:
                continue
            data_features[k] = dict(v)

        use_videos = len(video_keys) > 0
        fps = float(src_ds.meta.fps)

        # 5. Create new LeRobotDataset
        logger.info(f"Creating augmented LeRobotDataset '{output_repo}' at {out_path}...")
        new_ds = LeRobotDataset.create(
            repo_id=output_repo,
            fps=src_ds.meta.fps,
            root=out_path,
            robot_type=src_ds.meta.robot_type,
            features=data_features,
            use_videos=use_videos,
            image_writer_processes=0,
            image_writer_threads=4 if use_videos else 0,
            vcodec=vcodec,
        )

        total_episodes = src_ds.meta.total_episodes
        if max_episodes is not None and max_episodes > 0:
            total_episodes = min(total_episodes, max_episodes)

        # Metadata dictionary for visualizer annotations
        annotated_events: dict[str, dict[str, Any]] = {}

        # 6. Process each episode
        for ep_idx in tqdm(range(total_episodes), desc=f"Augmenting {output_repo}"):
            ep_meta = src_ds.meta.episodes[ep_idx]
            ep_from_idx = int(ep_meta["dataset_from_index"])
            ep_to_idx = int(ep_meta["dataset_to_index"])
            num_frames = int(ep_meta["length"])

            if num_frames <= 0:
                continue

            # Select tabular data
            hf_slice = src_ds.hf_dataset.select(range(ep_from_idx, ep_to_idx))

            # Decode video frames for each video key
            decoded_videos: dict[str, np.ndarray] = {}
            if use_videos:
                for vk in video_keys:
                    from_ts = float(ep_meta[f"videos/{vk}/from_timestamp"])
                    shifted_ts = [from_ts + i * (1.0 / fps) for i in range(num_frames)]
                    vid_file_rel = src_ds.meta.get_video_file_path(ep_idx, vk)
                    vid_path = src_ds.root / vid_file_rel

                    # Decode returns (N, C, H, W) in [0, 1]
                    frames_tensor = decode_video_frames(
                        vid_path, shifted_ts, src_ds.tolerance_s, src_ds.video_backend
                    )
                    # Convert to (N, H, W, C) uint8 numpy
                    frames_np = (
                        (frames_tensor * 255.0)
                        .clamp(0, 255)
                        .to(torch.uint8)
                        .permute(0, 2, 3, 1)
                        .cpu()
                        .numpy()
                    )
                    decoded_videos[vk] = frames_np

            # Process SPECIFIED cameras with detector
            ep_box_atoms: list[dict[str, Any]] = []

            for vk in process_video_keys:
                raw_frames = decoded_videos[vk]
                annotated_frames = np.empty_like(raw_frames)

                # Process in batches for maximum GPU throughput
                for b_start in range(0, num_frames, batch_size):
                    b_end = min(b_start + batch_size, num_frames)
                    batch_raw = [raw_frames[j] for j in range(b_start, b_end)]

                    # Run batched detector inference
                    batch_results = self.detector.predict_batch(batch_raw)

                    # Annotate frames
                    for j, (frame, det_res) in enumerate(zip(batch_raw, batch_results)):
                        idx = b_start + j
                        annotated = self.detector.annotate(
                            image=frame,
                            result=det_res,
                            mode=self.mode,
                            target_objects=self.target_objects,
                            conf_threshold=self.conf_threshold,
                            mask_alpha=self.mask_alpha,
                            box_thickness=self.box_thickness,
                        )
                        annotated_frames[idx] = annotated

                        # Log detection atom for visualizer if matches target
                        filtered = det_res.filter_by_targets(self.target_objects, self.conf_threshold)
                        if not filtered.is_empty:
                            for det in filtered.detections:
                                ep_box_atoms.append({
                                    "frame": idx,
                                    "timestamp": idx / fps,
                                    "camera": vk,
                                    "class_name": det.class_name,
                                    "confidence": det.confidence,
                                    "box_xyxy": det.box.xyxy if det.box else None,
                                })

                # Replace raw video with annotated video
                decoded_videos[vk] = annotated_frames

            # Unspecified cameras in passthrough_video_keys remain unmodified!

            # Build task string lookup
            task_str = "default_task"
            if hasattr(src_ds.meta, "tasks") and src_ds.meta.tasks is not None:
                if hasattr(src_ds.meta.tasks, "index") and len(src_ds.meta.tasks) > 0:
                    task_str = str(src_ds.meta.tasks.index[0])
                elif isinstance(src_ds.meta.tasks, dict) and src_ds.meta.tasks:
                    task_str = str(list(src_ds.meta.tasks.keys())[0])
            elif hasattr(src_ds.meta, "task") and src_ds.meta.task:
                task_str = str(src_ds.meta.task)
            elif "tasks" in ep_meta:
                ep_tasks = ep_meta["tasks"]
                task_str = str(ep_tasks[0] if isinstance(ep_tasks, list) else ep_tasks)

            # Write frames into new dataset
            for i in range(num_frames):
                frame_dict = {}
                for k in data_features:
                    if k in decoded_videos:
                        frame_dict[k] = decoded_videos[k][i]
                    else:
                        frame_dict[k] = hf_slice[i][k]

                # Ensure task string is present as required by LeRobot
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
            annotated_events[str(ep_idx)] = {"detections": ep_box_atoms}

        # 7. Finalize dataset
        logger.info("Finalizing new LeRobotDataset...")
        new_ds.finalize()

        # 8. Save detection annotations metadata for visualizer
        meta_dir = out_path / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        ann_meta_path = meta_dir / "lerobot_vision_detections.json"
        with open(ann_meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_dataset": input_repo,
                    "target_objects": self.target_objects,
                    "mode": self.mode,
                    "processed_cameras": process_video_keys,
                    "unprocessed_cameras": passthrough_video_keys,
                    "episodes": annotated_events,
                },
                f,
                indent=2,
            )

        logger.info(f"Dataset successfully created and finalized at: {out_path}")

        # 9. Push to Hugging Face Hub if requested
        if push_to_hub:
            logger.info(f"Pushing augmented dataset to Hugging Face Hub: {output_repo}...")
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=output_repo, repo_type="dataset", exist_ok=True)
            api.upload_folder(
                folder_path=str(out_path),
                repo_id=output_repo,
                repo_type="dataset",
                commit_message=(
                    f"Augment dataset with {self.mode.upper()} detections for "
                    f"targets={self.target_objects} on cameras={process_video_keys}"
                ),
            )
            logger.info(f"Done! Dataset published to https://huggingface.co/datasets/{output_repo}")

        return out_path


def augment_dataset(
    input_repo: str,
    output_repo: str,
    cameras: str | Sequence[str],
    mode: str = "bbox",
    target_objects: str | Sequence[str] | None = None,
    conf_threshold: float = 0.25,
    model_name: str | None = None,
    device: str | None = None,
    output_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    push_to_hub: bool = False,
    hf_token: str | None = None,
    overwrite: bool = False,
    batch_size: int = 16,
    max_episodes: int | None = None,
    **kwargs: Any,
) -> Path:
    """Convenience functional API for augmenting a LeRobot dataset."""
    augmentor = DatasetVideoAugmentor(
        mode=mode,
        target_objects=target_objects,
        conf_threshold=conf_threshold,
        model_name=model_name,
        device=device,
        **kwargs,
    )
    return augmentor.augment(
        input_repo=input_repo,
        output_repo=output_repo,
        cameras=cameras,
        output_dir=output_dir,
        local_dir=local_dir,
        push_to_hub=push_to_hub,
        hf_token=hf_token,
        overwrite=overwrite,
        batch_size=batch_size,
        max_episodes=max_episodes,
    )

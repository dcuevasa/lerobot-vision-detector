#!/usr/bin/env python3
"""Fast hybrid dataset builder: keep visual prompt on initial N frames per episode and raw frames thereafter.

Reuses already-generated obscured (or detected) frames for frames `0 .. N-1` of each
episode on the target camera (e.g. `observation.images.cam_high`) and splices them
with the unmodified raw frames from the source dataset for frames `N .. length-1`.
All tabular data, task prompts, metadata, and unspecified cameras (e.g. `cam_wrist`)
are preserved directly from `--obscured_dir`.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import av
from huggingface_hub import HfApi
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mix_initial_frames")


def _get_video_encoder_config(info_meta: dict[str, Any], camera_key: str) -> tuple[str, str, int, int, int, dict[str, str]]:
    """Build PyAV stream configuration matching `info.json` / LeRobot v3.0 defaults."""
    feat_info = info_meta["features"][camera_key].get("info", {})
    width = int(feat_info.get("video.width", 640))
    height = int(feat_info.get("video.height", 480))
    fps = int(feat_info.get("video.fps", info_meta.get("fps", 30)))
    pix_fmt = str(feat_info.get("video.pix_fmt", "yuv420p"))
    codec_name = str(feat_info.get("video.codec", "av1"))
    vcodec = "libsvtav1" if codec_name in ("av1", "libsvtav1") else codec_name

    options: dict[str, str] = {
        "g": str(feat_info.get("video.g", 2)),
        "crf": str(feat_info.get("video.crf", 30)),
    }
    if vcodec == "libsvtav1":
        options["preset"] = str(feat_info.get("video.preset", 12))
    elif "video.preset" in feat_info:
        options["preset"] = str(feat_info["video.preset"])

    return vcodec, pix_fmt, width, height, fps, options


def mix_initial_frames(
    obscured_dir: Path,
    raw_repo: str,
    output_dir: Path,
    output_repo: str,
    camera: str = "observation.images.cam_high",
    initial_frames: int = 5,
    overwrite: bool = False,
    push_to_hub: bool = False,
    hf_token: str | None = None,
) -> Path:
    if initial_frames <= 0:
        raise ValueError(f"initial_frames must be positive, got {initial_frames}")
    if not obscured_dir.exists():
        raise FileNotFoundError(f"Obscured dataset directory not found: {obscured_dir}")

    if output_dir.exists():
        if overwrite:
            logger.info(f"Removing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"Output directory exists: {output_dir}. Pass --overwrite to replace it.")

    logger.info(f"Copying base structure from {obscured_dir} -> {output_dir} (skipping target camera video '{camera}')...")
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in obscured_dir.iterdir():
        if item.name == "videos":
            dst_videos = output_dir / "videos"
            dst_videos.mkdir(parents=True, exist_ok=True)
            for vid_cam_dir in item.iterdir():
                if vid_cam_dir.name == camera:
                    continue
                shutil.copytree(vid_cam_dir, dst_videos / vid_cam_dir.name)
        elif item.is_dir():
            shutil.copytree(item, output_dir / item.name)
        else:
            shutil.copy2(item, output_dir / item.name)

    info_meta = json.loads((obscured_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    vcodec, pix_fmt, width, height, fps, codec_options = _get_video_encoder_config(info_meta, camera)
    logger.info(f"Output video config for {camera}: {vcodec} {width}x{height}@{fps} {pix_fmt} opts={codec_options}")

    ep_files = sorted((obscured_dir / "meta" / "episodes").glob("**/*.parquet"))
    if len(ep_files) != 1:
        raise RuntimeError(f"Expected 1 episode metadata parquet in {obscured_dir}, found {len(ep_files)}")
    ep_df = pd.read_parquet(ep_files[0])

    logger.info(f"Loading raw dataset '{raw_repo}'...")
    raw_ds = LeRobotDataset(raw_repo, video_backend="pyav")
    if len(raw_ds.meta.episodes) != len(ep_df):
        raise RuntimeError(
            f"Episode count mismatch: obscured={len(ep_df)} vs raw={len(raw_ds.meta.episodes)}"
        )

    obscured_vid_files = sorted((obscured_dir / "videos" / camera).glob("**/*.mp4"))
    if len(obscured_vid_files) != 1:
        raise RuntimeError(f"Expected 1 obscured video file for {camera}, found {obscured_vid_files}")
    obscured_vid_path = obscured_vid_files[0]
    rel_vid_path = obscured_vid_path.relative_to(obscured_dir)
    out_vid_path = output_dir / rel_vid_path
    out_vid_path.parent.mkdir(parents=True, exist_ok=True)

    backend = getattr(raw_ds, "video_backend", getattr(raw_ds, "_video_backend", "pyav"))
    tol = getattr(raw_ds, "tolerance_s", getattr(raw_ds, "_tolerance_s", 0.0001))

    logging.getLogger("libav").setLevel(av.logging.WARNING)
    total_written = 0

    with av.open(str(obscured_vid_path), "r") as in_container, av.open(str(out_vid_path), "w") as out_container:
        out_stream = out_container.add_stream(vcodec, fps, options=codec_options)
        out_stream.pix_fmt = pix_fmt
        out_stream.width = width
        out_stream.height = height

        obscured_frame_iter = (f.to_ndarray(format="rgb24") for f in in_container.decode(video=0))

        for ep_idx in tqdm(range(len(ep_df)), desc=f"Splicing init-{initial_frames} frames"):
            num_frames = int(ep_df.loc[ep_idx, "length"])
            raw_ep_meta = raw_ds.meta.episodes[ep_idx]
            if int(raw_ep_meta["length"]) != num_frames:
                raise RuntimeError(
                    f"Episode {ep_idx} length mismatch: obscured={num_frames} vs raw={raw_ep_meta['length']}"
                )

            # Pull obscured frames for this episode sequentially
            obs_ep_frames = [next(obscured_frame_iter) for _ in range(num_frames)]

            # Decode raw frames for this episode from raw_ds
            from_ts = float(raw_ep_meta[f"videos/{camera}/from_timestamp"])
            shifted_ts = [from_ts + i * (1.0 / fps) for i in range(num_frames)]
            raw_vid_rel = raw_ds.meta.get_video_file_path(ep_idx, camera)
            raw_vid_path = raw_ds.root / raw_vid_rel
            frames_tensor = decode_video_frames(raw_vid_path, shifted_ts, tol, backend)
            raw_ep_frames = (
                (frames_tensor * 255.0)
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )

            split_idx = min(initial_frames, num_frames)
            for f_idx in range(num_frames):
                frame_rgb = obs_ep_frames[f_idx] if f_idx < split_idx else raw_ep_frames[f_idx]
                av_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
                for packet in out_stream.encode(av_frame):
                    out_container.mux(packet)
                total_written += 1

        for packet in out_stream.encode():
            out_container.mux(packet)

    av.logging.restore_default_callback()
    logger.info(f"Wrote {total_written} spliced frames into {out_vid_path}")

    # Update visual detections annotation file if present
    ann_meta_path = output_dir / "meta" / "lerobot_vision_detections.json"
    if ann_meta_path.exists():
        ann_data = json.loads(ann_meta_path.read_text(encoding="utf-8"))
        ann_data["initial_frames"] = int(initial_frames)
        ann_data["raw_source_dataset"] = raw_repo
        for ep_data in ann_data.get("episodes", {}).values():
            if "detections" in ep_data:
                ep_data["detections"] = [
                    d for d in ep_data["detections"] if int(d.get("frame", 0)) < initial_frames
                ]
        ann_meta_path.write_text(json.dumps(ann_data, indent=2), encoding="utf-8")

    # Update image stats for target camera from raw dataset (since >98% of frames are raw)
    stats_path = output_dir / "meta" / "stats.json"
    if stats_path.exists() and raw_ds.meta.stats and camera in raw_ds.meta.stats:
        stats_data = json.loads(stats_path.read_text(encoding="utf-8"))
        stats_data[camera] = {
            k: v.tolist() if hasattr(v, "tolist") else v for k, v in raw_ds.meta.stats[camera].items()
        }
        stats_path.write_text(json.dumps(stats_data, indent=4), encoding="utf-8")

    # Verify with LeRobotDataset across first, second, and final episodes
    logger.info("Verifying output dataset with LeRobotDataset...")
    out_ds = LeRobotDataset(output_repo, root=output_dir, video_backend="pyav")
    assert out_ds.meta.total_episodes == len(ep_df), f"Unexpected episodes: {out_ds.meta.total_episodes}"
    assert out_ds.meta.total_frames == total_written, f"Unexpected frames: {out_ds.meta.total_frames} != {total_written}"

    sample0 = out_ds[0][camera]
    sample_post = out_ds[min(initial_frames + 5, out_ds.meta.total_frames - 1)][camera]
    raw_sample_post = raw_ds[min(initial_frames + 5, raw_ds.meta.total_frames - 1)][camera]
    sample_last = out_ds[out_ds.meta.total_frames - 1][camera]
    raw_sample_last = raw_ds[raw_ds.meta.total_frames - 1][camera]
    white_frac_0 = float(((sample0[0] > 0.98) & (sample0[1] > 0.98) & (sample0[2] > 0.98)).float().mean())
    raw_diff_post = float((sample_post - raw_sample_post).abs().mean())
    raw_diff_last = float((sample_last - raw_sample_last).abs().mean())
    logger.info(
        f"Sanity check: ep0 frame 0 near-white fraction={white_frac_0:.4f}, "
        f"frame {initial_frames + 5} diff vs raw={raw_diff_post:.4f}, "
        f"last frame ({out_ds.meta.total_frames - 1}) diff vs raw={raw_diff_last:.4f}"
    )

    if push_to_hub:
        logger.info(f"Uploading {output_dir} to Hugging Face Hub dataset '{output_repo}'...")
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=output_repo, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=output_repo,
            repo_type="dataset",
            commit_message=(
                f"Create initial-{initial_frames}-frame obscured hybrid dataset from "
                f"{obscured_dir.name} + {raw_repo}"
            ),
        )
        try:
            api.delete_tag(repo_id=output_repo, tag="v3.0", repo_type="dataset")
        except Exception:
            pass
        api.create_tag(repo_id=output_repo, tag="v3.0", repo_type="dataset")
        logger.info(f"Published and tagged v3.0: https://huggingface.co/datasets/{output_repo}")

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Splice initial N obscured frames per episode with raw subsequent frames.")
    parser.add_argument(
        "--obscured_dir",
        type=Path,
        default=Path("/workspace/outputs/bendca61_so101-surface-transfer-v1-gsam-obscure"),
        help="Path to the full obscured local LeRobot dataset.",
    )
    parser.add_argument(
        "--raw_repo",
        type=str,
        default="bendca61/so101-surface-transfer-v1-pruned",
        help="Raw source LeRobot dataset repo ID or local path.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/workspace/outputs/bendca61_so101-surface-transfer-v1-gsam-obscure-init5"),
        help="Destination directory for hybrid dataset.",
    )
    parser.add_argument(
        "--output_repo",
        type=str,
        default="bendca61/so101-surface-transfer-v1-gsam-obscure-init5",
        help="Target Hugging Face dataset repository ID.",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="observation.images.cam_high",
        help="Camera key to splice (default: observation.images.cam_high).",
    )
    parser.add_argument(
        "--initial_frames",
        type=int,
        default=5,
        help="Number of initial frames per episode to keep obscured (default: 5).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output_dir if it exists.")
    parser.add_argument("--push_to_hub", action="store_true", help="Upload dataset to Hugging Face Hub and tag v3.0.")
    parser.add_argument("--hf_token", type=str, default=None, help="Optional Hugging Face API token.")
    args = parser.parse_args()

    mix_initial_frames(
        obscured_dir=args.obscured_dir,
        raw_repo=args.raw_repo,
        output_dir=args.output_dir,
        output_repo=args.output_repo,
        camera=args.camera,
        initial_frames=args.initial_frames,
        overwrite=args.overwrite,
        push_to_hub=args.push_to_hub,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Interactive Web UI for SAM 2 Visual Prompting & Dataset Augmentation.

Runs inside the Docker container and exposes a web interface on port 7860.
Accessible locally or remotely over SSH port forwarding:
    ssh -L 7860:localhost:7860 user@pgx-ip
    Browser: http://localhost:7860
"""

import logging
import os
from pathlib import Path
import sys
import tempfile
import time

import cv2
import gradio as gr
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import decode_video_frames

from lerobot_vision_detector.augmentation.sam2_video_annotator import SAM2VideoAnnotator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("sam2_webui")

# Global state
annotator_instance = None
cached_dataset = None
cached_repo_id = None
current_episode_frames = None
current_prompt_points = []


def get_annotator(checkpoint_path="/workspace/sam2.1_b.pt", mode="both"):
    global annotator_instance
    if annotator_instance is None:
        logger.info("Initializing SAM 2 Video Predictor in Web UI...")
        annotator_instance = SAM2VideoAnnotator(
            checkpoint_path=checkpoint_path,
            mode=mode,
        )
    annotator_instance.mode = mode
    return annotator_instance


def load_dataset(repo_id_or_path: str):
    global cached_dataset, cached_repo_id
    if cached_dataset is None or cached_repo_id != repo_id_or_path:
        logger.info(f"Loading LeRobot dataset: {repo_id_or_path}...")
        local_dir = None
        if Path(repo_id_or_path).exists():
            root = Path(repo_id_or_path)
        else:
            root = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id_or_path
            if not root.exists() and os.path.exists("/cache/temp_prune_surface_transfer"):
                root = Path("/cache/temp_prune_surface_transfer")
        cached_dataset = LeRobotDataset(repo_id_or_path, root=root)
        cached_repo_id = repo_id_or_path
    return cached_dataset


def load_episode_frame0(repo_id: str, episode_idx: int, camera_key: str):
    global current_episode_frames, current_prompt_points
    current_prompt_points = []
    try:
        ds = load_dataset(repo_id)
        total_eps = ds.meta.total_episodes
        if episode_idx >= total_eps:
            return None, f"Error: Episode {episode_idx} exceeds dataset max ({total_eps - 1})"

        ep_meta = ds.meta.episodes[episode_idx]
        num_frames = int(ep_meta["length"])
        fps = float(ds.meta.fps)

        # Match camera key
        actual_vk = None
        for vk in ds.meta.video_keys:
            if camera_key in vk:
                actual_vk = vk
                break
        if actual_vk is None:
            actual_vk = list(ds.meta.video_keys)[0]

        from_ts = float(ep_meta[f"videos/{actual_vk}/from_timestamp"])
        shifted_ts = [from_ts + i * (1.0 / fps) for i in range(num_frames)]
        vid_file_rel = ds.meta.get_video_file_path(episode_idx, actual_vk)
        vid_path = ds.root / vid_file_rel

        backend = getattr(ds, "video_backend", getattr(ds, "_video_backend", "pyav"))
        tol = getattr(ds, "tolerance_s", getattr(ds, "_tolerance_s", 0.0001))

        frames_tensor = decode_video_frames(vid_path, shifted_ts, tol, backend)
        current_episode_frames = (
            (frames_tensor * 255.0)
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )

        frame0 = current_episode_frames[0].copy()
        # Auto seed suggestion
        annotator = get_annotator()
        auto_x, auto_y = annotator.detect_auto_seed(frame0)

        # Draw a small crosshair for candidate seed
        preview = frame0.copy()
        cv2.drawMarker(preview, (auto_x, auto_y), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(preview, f"Suggested ({auto_x},{auto_y})", (auto_x + 10, auto_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        info_msg = (
            f"Loaded Episode {episode_idx} ({num_frames} frames, {actual_vk}).\n"
            f"Auto-suggested mug center: ({auto_x}, {auto_y}). Click on the image to set or override prompt point."
        )
        return preview, info_msg
    except Exception as e:
        logger.exception("Error loading episode frame 0")
        return None, f"Error: {e}"


def on_image_click(evt: gr.SelectData, image_in):
    global current_prompt_points
    if image_in is None:
        return None, "No image loaded"
    x, y = evt.index[0], evt.index[1]
    current_prompt_points.append((x, y))
    
    # Render clicked points on image
    annotated = image_in.copy()
    for pt in current_prompt_points:
        cv2.circle(annotated, pt, 5, (0, 255, 0), -1)
        cv2.circle(annotated, pt, 8, (255, 255, 255), 1)
        cv2.putText(annotated, f"({pt[0]},{pt[1]})", (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    return annotated, f"Prompt point added: ({x}, {y}). Total points: {len(current_prompt_points)}"


def preview_frame0_segmentation(mode: str):
    global current_episode_frames, current_prompt_points
    if current_episode_frames is None:
        return None, "Please load an episode first."

    annotator = get_annotator(mode=mode)
    frame0 = current_episode_frames[0:1].copy()

    pts = current_prompt_points if current_prompt_points else None
    annotated_seq, det_records = annotator.annotate_episode_frames(frame0, points=pts)
    
    status = f"Frame 0 segmented successfully using mode '{mode}'."
    if det_records and det_records[0]["box_xyxy"]:
        bbox = det_records[0]["box_xyxy"]
        status += f" Bounding Box: {bbox}"

    return annotated_seq[0], status


def test_propagation_clip(mode: str, num_frames_to_track: int = 40):
    global current_episode_frames, current_prompt_points
    if current_episode_frames is None:
        return None, "Please load an episode first."

    annotator = get_annotator(mode=mode)
    clip_frames = current_episode_frames[:num_frames_to_track].copy()
    pts = current_prompt_points if current_prompt_points else None

    logger.info(f"Propagating clip across {len(clip_frames)} frames...")
    annotated_seq, det_records = annotator.annotate_episode_frames(clip_frames, points=pts)

    # Save temporary video for video player preview
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
        out_vid_path = tmp_vid.name

    h, w, _ = clip_frames.shape[1:]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(out_vid_path, fourcc, 30.0, (w, h))
    for f in annotated_seq:
        out_writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out_writer.release()

    return out_vid_path, f"Successfully tracked object across {len(clip_frames)} frames."


def run_full_dataset_augmentation(
    input_repo: str,
    output_repo: str,
    cameras: str,
    mode: str,
    push_to_hub: bool,
    max_episodes: int,
):
    try:
        annotator = get_annotator(mode=mode)
        max_eps = int(max_episodes) if max_episodes > 0 else None
        out_path = annotator.augment(
            input_repo=input_repo,
            output_repo=output_repo,
            cameras=cameras,
            auto_prompt=True,
            push_to_hub=push_to_hub,
            max_episodes=max_eps,
        )
        return f"Dataset augmentation complete!\nSaved to: {out_path}\nHub upload: {push_to_hub}"
    except Exception as e:
        logger.exception("Full dataset augmentation error")
        return f"Error: {e}"


def build_app():
    with gr.Blocks(title="SAM 2 Visual Prompt Annotator") as demo:
        gr.Markdown(
            """
            # 🎯 SO-101 Visual Prompt Annotator (SAM 2.1 Video Predictor)
            **Interactive Video Mask & Bounding Box Generation for Robot Zero-Shot Generalization**
            * Uses SAM 2 spatio-temporal memory tracking to persist target object through physical gripper occlusion.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Dataset & Episode Selection")
                repo_input = gr.Textbox(
                    value="bendca61/so101-surface-transfer-v1-pruned",
                    label="Source LeRobot Dataset",
                )
                camera_input = gr.Dropdown(
                    choices=["cam_high", "cam_wrist", "all"],
                    value="cam_high",
                    label="Target Camera Stream",
                )
                episode_slider = gr.Slider(
                    minimum=0,
                    maximum=14,
                    step=1,
                    value=0,
                    label="Episode Index",
                )
                load_btn = gr.Button("📥 Load Episode Frame 0", variant="primary")

                gr.Markdown("### 2. Prompting & Overlay Mode")
                overlay_mode = gr.Radio(
                    choices=["both", "bbox", "mask"],
                    value="both",
                    label="Visual Overlay Mode",
                )
                preview_btn = gr.Button("🔍 Preview Frame 0 Segmentation", variant="secondary")
                track_clip_btn = gr.Button("▶️ Test Video Propagation (40 Frames)")

                gr.Markdown("### 3. Full Dataset Batch Augmentation")
                out_repo_input = gr.Textbox(
                    value="bendca61/so101-surface-transfer-sam2-v1",
                    label="Output LeRobot Dataset Repo ID",
                )
                max_eps_input = gr.Number(value=0, label="Max Episodes (0 = all 15 episodes)")
                push_hub_checkbox = gr.Checkbox(value=True, label="Push directly to Hugging Face Hub")
                batch_run_btn = gr.Button("🚀 Run Full Augmentation Pipeline", variant="stop")

            with gr.Column(scale=2):
                gr.Markdown("### 4. Interactive Frame Canvas & Video Player")
                status_box = gr.Textbox(label="Status / Output Log", lines=2)
                image_display = gr.Image(
                    label="Frame 0 (Click anywhere on the object to set prompt seed point)",
                    interactive=True,
                )
                video_display = gr.Video(label="Tracked Video Propagation Preview")

        # Event wiring
        load_btn.click(
            fn=load_episode_frame0,
            inputs=[repo_input, episode_slider, camera_input],
            outputs=[image_display, status_box],
        )

        image_display.select(
            fn=on_image_click,
            inputs=[image_display],
            outputs=[image_display, status_box],
        )

        preview_btn.click(
            fn=preview_frame0_segmentation,
            inputs=[overlay_mode],
            outputs=[image_display, status_box],
        )

        track_clip_btn.click(
            fn=test_propagation_clip,
            inputs=[overlay_mode],
            outputs=[video_display, status_box],
        )

        batch_run_btn.click(
            fn=run_full_dataset_augmentation,
            inputs=[repo_input, out_repo_input, camera_input, overlay_mode, push_hub_checkbox, max_eps_input],
            outputs=[status_box],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    port = int(os.getenv("PORT", "7860"))
    logger.info(f"Starting SAM 2 Web UI on http://0.0.0.0:{port}...")
    app.launch(server_name="0.0.0.0", server_port=port, share=False)

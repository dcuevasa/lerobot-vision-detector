"""CLI tool for offline LeRobot dataset augmentation with YOLO spatial detections."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from ..augmentation.annotator import DatasetVideoAugmentor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("lerobot_vision_augment")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Augment an existing LeRobot dataset with YOLO bounding boxes or segmentation masks "
            "for specific target objects across specified cameras, and publish to a new repo."
        )
    )
    parser.add_argument(
        "--input_repo",
        type=str,
        required=True,
        help="Input LeRobot dataset identifier (e.g. 'bendca61/so101-test-leader-v1' or local path).",
    )
    parser.add_argument(
        "--output_repo",
        type=str,
        required=True,
        help="Target repository identifier for the augmented dataset (e.g. 'user/so101-test-detected').",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        required=True,
        help="One or more camera names to process (e.g. --cameras cam_high or --cameras cam_high cam_wrist). Unspecified cameras are NOT processed.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["bbox", "seg", "both"],
        default="bbox",
        help="Detection modality: 'bbox' (2D bounding box), 'seg' (instance mask), or 'both'. Default: bbox.",
    )
    parser.add_argument(
        "--target_objects",
        type=str,
        default=None,
        help="Target object class name(s) comma-separated (e.g. 'cup' or 'cup,bottle'). Defaults to all detected objects.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model weights name or path (defaults to yolov8n.pt for bbox, yolov8n-seg.pt for seg).",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device ('cuda', 'cpu', or auto-detect).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Local filesystem directory to write new dataset to (default: outputs/<output_repo>).",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Local directory where source dataset is stored or cached.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Upload the augmented dataset to Hugging Face Hub.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face API token (can also be set via HF_TOKEN environment variable).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if it exists.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Frame batch size for GPU inference (default: 16).",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Limit number of episodes to process (useful for quick verification).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Flatten camera list if comma-separated strings were provided
    cams: list[str] = []
    for c in args.cameras:
        cams.extend([x.strip() for x in c.split(",") if x.strip()])

    targets = None
    if args.target_objects:
        targets = [t.strip() for t in args.target_objects.split(",") if t.strip()]

    augmentor = DatasetVideoAugmentor(
        mode=args.mode,
        target_objects=targets,
        conf_threshold=args.conf_threshold,
        model_name=args.model_name,
        device=args.device,
    )

    out_path = augmentor.augment(
        input_repo=args.input_repo,
        output_repo=args.output_repo,
        cameras=cams,
        output_dir=args.output_dir,
        local_dir=args.local_dir,
        push_to_hub=args.push_to_hub,
        hf_token=args.hf_token,
        overwrite=args.overwrite,
        batch_size=args.batch_size,
        max_episodes=args.max_episodes,
    )

    logger.info(f"Augmentation pipeline finished successfully! Dataset saved at: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CLI Entry point for SAM 2 Video Dataset Augmentation.

Runs offline visual prompt propagation over LeRobot datasets in headless / SSH server environments.
Zero UI dependencies required; completely automated.

Example:
    python scripts/sam2_annotate.py \\
        --input_repo bendca61/so101-surface-transfer-v1-pruned \\
        --output_repo bendca61/so101-surface-transfer-sam2-v1 \\
        --cameras cam_high \\
        --mode both \\
        --auto_prompt \\
        --push_to_hub
"""

import argparse
import logging
import os
import sys

from lerobot_vision_detector.augmentation.sam2_video_annotator import SAM2VideoAnnotator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("sam2_annotate")


def parse_args():
    parser = argparse.ArgumentParser(description="Augment LeRobot dataset with SAM 2 visual prompts (CLI mode)")
    parser.add_argument("--input_repo", type=str, required=True, help="Source LeRobot dataset ID or path")
    parser.add_argument("--output_repo", type=str, required=True, help="Destination LeRobot dataset repo ID")
    parser.add_argument("--cameras", type=str, default="cam_high", help="Comma-separated camera keys to annotate")
    parser.add_argument("--mode", type=str, default="both", choices=["bbox", "mask", "both"], help="Prompt visual overlay mode")
    parser.add_argument("--label", type=str, default="cup", help="Target object label text")
    parser.add_argument("--model_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_b+.yaml", help="SAM 2 model config")
    parser.add_argument("--checkpoint", type=str, default="/workspace/sam2.1_b.pt", help="Path to SAM 2 checkpoint")
    parser.add_argument("--point", type=int, nargs=2, default=None, metavar=("X", "Y"), help="Default seed point on Frame 0")
    parser.add_argument("--auto_prompt", action="store_true", default=True, help="Automatically detect seed point on Frame 0")
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit number of episodes to process (for debugging)")
    parser.add_argument("--output_dir", type=str, default=None, help="Local destination directory")
    parser.add_argument("--local_dir", type=str, default=None, help="Local source dataset directory")
    parser.add_argument("--push_to_hub", action="store_true", help="Upload resulting dataset to Hugging Face Hub")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face API token")
    parser.add_argument("--vcodec", type=str, default="libsvtav1", help="Video codec for LeRobot output (default libsvtav1)")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Initializing SAM 2 Video Annotator...")
    annotator = SAM2VideoAnnotator(
        model_cfg=args.model_cfg,
        checkpoint_path=args.checkpoint,
        mode=args.mode,
        label=args.label,
    )

    default_point = tuple(args.point) if args.point else None

    logger.info(f"Augmenting {args.input_repo} -> {args.output_repo}...")
    annotator.augment(
        input_repo=args.input_repo,
        output_repo=args.output_repo,
        cameras=args.cameras,
        default_point=default_point,
        auto_prompt=args.auto_prompt,
        output_dir=args.output_dir,
        local_dir=args.local_dir,
        push_to_hub=args.push_to_hub,
        hf_token=args.hf_token,
        vcodec=args.vcodec,
        max_episodes=args.max_episodes,
    )
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()

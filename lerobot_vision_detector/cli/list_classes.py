"""CLI tool to list and search available detectable object classes for a given vision/YOLO model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..utils.env_check import find_and_import_ultralytics
from ..utils.model_registry import is_open_vocab_model, resolve_model_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print all available detectable object classes for a given vision/YOLO model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Model weights name or path (e.g. 'yolov8n.pt', 'yolo11-detection-obj_s.pt', 'yolov8s-worldv2.pt'). Default: yolov8n.pt",
    )
    parser.add_argument(
        "--search",
        "-s",
        type=str,
        default=None,
        help="Optional search filter to find classes matching a keyword (case-insensitive).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON format.",
    )
    return parser


def get_model_classes(model_path: str) -> dict[int, str]:
    """Load model and retrieve its class names mapping."""
    ultralytics = find_and_import_ultralytics()
    YOLO = ultralytics.YOLO
    model = YOLO(model_path)
    names = getattr(model, "names", {})
    return {int(k): str(v) for k, v in names.items()}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    resolved_path = resolve_model_path(args.model)
    is_open_vocab = is_open_vocab_model(args.model)

    try:
        class_map = get_model_classes(resolved_path)
    except Exception as e:
        print(f"Error loading model '{args.model}' (resolved to '{resolved_path}'): {e}", file=sys.stderr)
        sys.exit(1)

    # Filter if search query is provided
    if args.search:
        query = args.search.strip().lower()
        filtered = {k: v for k, v in class_map.items() if query in v.lower()}
    else:
        filtered = class_map

    if args.json:
        payload = {
            "model": args.model,
            "resolved_path": resolved_path,
            "is_open_vocabulary": is_open_vocab,
            "classes": filtered,
        }
        print(json.dumps(payload, indent=2))
        return

    print("=" * 60)
    print(f"Model:         {args.model}")
    print(f"Resolved Path: {resolved_path}")
    if is_open_vocab:
        print("Type:          OPEN-VOCABULARY (YOLO-World)")
        print("               Accepts ANY custom text queries via --target_objects")
        print("               e.g. --target_objects 'red mug, blue tape, robot gripper'")
        print("-" * 60)
        print(f"Default Base Classes ({len(class_map)}):")
    else:
        print(f"Total Available Classes: {len(class_map)}")

    if args.search:
        print(f"Matching '{args.search}': {len(filtered)}")
    print("=" * 60)

    if not filtered:
        print(f"No classes found matching '{args.search}'.")
        return

    # Print in clean 3-column format
    sorted_items = sorted(filtered.items())
    num_cols = 3

    for i in range(0, len(sorted_items), num_cols):
        row = sorted_items[i : i + num_cols]
        line = "  ".join(f"[{k:2d}] {v:<18}" for k, v in row)
        print(line)

    print("=" * 60)


if __name__ == "__main__":
    main()

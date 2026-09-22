"""CLI tool to list and search available detectable object classes for a given YOLO model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..utils.env_check import find_and_import_ultralytics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print all available detectable object classes for a given vision/YOLO model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Model weights name or path (e.g. 'yolov8n.pt', 'yolov8n-seg.pt', 'yolo11n.pt', or custom .pt path). Default: yolov8n.pt",
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


def get_model_classes(model_name: str) -> dict[int, str]:
    """Load model and retrieve its class names mapping."""
    ultralytics = find_and_import_ultralytics()
    YOLO = ultralytics.YOLO
    model = YOLO(model_name)
    names = getattr(model, "names", {})
    return {int(k): str(v) for k, v in names.items()}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        class_map = get_model_classes(args.model)
    except Exception as e:
        print(f"Error loading model '{args.model}': {e}", file=sys.stderr)
        sys.exit(1)

    # Filter if search query is provided
    if args.search:
        query = args.search.strip().lower()
        filtered = {k: v for k, v in class_map.items() if query in v.lower()}
    else:
        filtered = class_map

    if args.json:
        print(json.dumps(filtered, indent=2))
        return

    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Total Available Classes: {len(class_map)}")
    if args.search:
        print(f"Matching '{args.search}': {len(filtered)}")
    print("=" * 60)

    if not filtered:
        print(f"No classes found matching '{args.search}'.")
        return

    # Print in clean 3-column format
    sorted_items = sorted(filtered.items())
    col_width = 25
    num_cols = 3

    for i in range(0, len(sorted_items), num_cols):
        row = sorted_items[i : i + num_cols]
        line = "  ".join(f"[{k:2d}] {v:<18}" for k, v in row)
        print(line)

    print("=" * 60)


if __name__ == "__main__":
    main()

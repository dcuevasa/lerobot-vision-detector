"""Standardized model path resolver and registry utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

logger = logging.getLogger("lerobot_vision_detector")

KNOWN_OPEN_VOCAB_MODELS = {
    "yolov8s-world.pt",
    "yolov8s-worldv2.pt",
    "yolov8m-world.pt",
    "yolov8m-worldv2.pt",
    "yolov8l-world.pt",
    "yolov8l-worldv2.pt",
    "yolov8x-world.pt",
    "yolov8x-worldv2.pt",
    "yolo11s-world.pt",
    "yolo11m-world.pt",
    "yolo11l-world.pt",
    "yolo11x-world.pt",
    "google/owlv2-base-patch16-ensemble",
    "google/owlvit-base-patch32",
}


def get_models_dir() -> Path:
    """Return the standardized models directory of the repository."""
    # This file is in lerobot_vision_detector/utils/model_registry.py
    # Repo root is 2 levels up
    repo_root = Path(__file__).resolve().parent.parent.parent
    models_dir = repo_root / "models"
    if not models_dir.exists():
        # Fallback to CWD / models
        cwd_models = Path.cwd() / "models"
        if cwd_models.exists():
            return cwd_models
    return models_dir


def resolve_model_path(model_name_or_path: str) -> str:
    """Resolve a model name to its standardized file path if present in models/.

    Checks:
    1. Exact existing path on filesystem
    2. Under <repo_root>/models/<model_name>
    3. Under <current_working_dir>/models/<model_name>
    4. Falls back to original string for hub download / built-in resolution

    Args:
        model_name_or_path: String model name or path.

    Returns:
        Resolved path string.
    """
    p = Path(model_name_or_path)
    if p.is_file():
        return str(p.resolve())

    # Check standardized models/ directory
    models_dir = get_models_dir()
    candidate = models_dir / model_name_or_path
    if candidate.is_file():
        return str(candidate.resolve())

    # Check without extension or with .pt
    if not model_name_or_path.endswith(".pt"):
        cand_pt = models_dir / f"{model_name_or_path}.pt"
        if cand_pt.is_file():
            return str(cand_pt.resolve())

    return model_name_or_path


def is_open_vocab_model(model_name: str) -> bool:
    """Check if model name implies an open-vocabulary vision detector."""
    m = model_name.lower()
    if any(k in m for k in ("world", "owl", "grounding", "glip", "clip")):
        return True
    return m in KNOWN_OPEN_VOCAB_MODELS

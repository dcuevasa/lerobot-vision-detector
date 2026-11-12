"""Environment check and dependency resolver for lerobot-vision-detector."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any

logger = logging.getLogger("lerobot_vision_detector")


def find_and_import_ultralytics() -> Any:
    """Import ultralytics, checking active environment and sister pyenv versions if needed.

    Returns:
        The ultralytics module.

    Raises:
        ImportError: If ultralytics cannot be found in the current environment or pyenv paths.
    """
    try:
        import ultralytics
        return ultralytics
    except ImportError:
        pass

    # Search for ultralytics in pyenv environments (e.g. sister pyenv 'yolo' or similar)
    home = Path.home()
    pyenv_root = home / ".pyenv" / "versions"
    candidates: list[Path] = []
    if pyenv_root.exists():
        candidates.extend(pyenv_root.glob("**/envs/yolo/lib/python*/site-packages"))
        candidates.extend(pyenv_root.glob("yolo/lib/python*/site-packages"))
        candidates.extend(pyenv_root.glob("*/envs/*/lib/python*/site-packages"))

    for cand in candidates:
        if cand.is_dir() and str(cand) not in sys.path:
            sys.path.append(str(cand))
            try:
                import ultralytics
                logger.info(f"Loaded ultralytics from pyenv path: {cand}")
                return ultralytics
            except ImportError:
                continue

    raise ImportError(
        "Could not import 'ultralytics'. Please install it using `pip install ultralytics` "
        "or ensure the 'yolo' pyenv virtual environment is present."
    )


def find_and_import_grounded_sam() -> tuple[Any, Any]:
    """Import GroundedSAM and GroundedSAMResult, checking sys.path, workspace, and sister paths.

    Returns:
        tuple (GroundedSAM, GroundedSAMResult) classes.

    Raises:
        ImportError: If grounded_sam cannot be imported or found.
    """
    try:
        from grounded_sam import GroundedSAM, GroundedSAMResult
        return GroundedSAM, GroundedSAMResult
    except (ImportError, ModuleNotFoundError):
        pass

    import os
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidates: list[Path] = []

    # 1. Explicit environment variable
    if os.environ.get("GROUNDED_SAM_PATH"):
        candidates.append(Path(os.environ["GROUNDED_SAM_PATH"]))

    # 2. Sister repository in workspace
    candidates.append(repo_root.parent / "Grounded-Segment-Anything")
    candidates.append(Path("/workspace/Grounded-Segment-Anything"))
    candidates.append(Path.cwd() / "Grounded-Segment-Anything")
    candidates.append(Path.cwd().parent / "Grounded-Segment-Anything")
    candidates.append(Path.home() / "Tesis" / "Grounded-Segment-Anything")

    for cand in candidates:
        if cand.is_dir() and (cand / "grounded_sam").is_dir():
            for p in [cand, cand / "GroundingDINO", cand / "segment_anything"]:
                if p.is_dir() and str(p) not in sys.path:
                    sys.path.insert(0, str(p))
            try:
                from grounded_sam import GroundedSAM, GroundedSAMResult
                logger.info(f"Loaded grounded_sam from path: {cand}")
                return GroundedSAM, GroundedSAMResult
            except (ImportError, ModuleNotFoundError) as err:
                logger.debug(f"Attempted to import from {cand} but failed: {err}")
                continue

    raise ImportError(
        "Could not import 'grounded_sam'. Ensure 'Grounded-Segment-Anything' is present in your "
        "workspace, in PYTHONPATH, or set the GROUNDED_SAM_PATH environment variable."
    )


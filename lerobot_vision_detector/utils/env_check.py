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

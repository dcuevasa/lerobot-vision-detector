"""CLI interfaces for lerobot-vision-detector."""

from .augment import main as augment_main
from .record import record as record_main
from .teleoperate import teleoperate as teleoperate_main
from .eval_policy import eval_policy as eval_main
from .list_classes import main as list_classes_main

__all__ = [
    "augment_main",
    "record_main",
    "teleoperate_main",
    "eval_main",
    "list_classes_main",
]

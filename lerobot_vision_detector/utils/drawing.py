"""Drawing and visualization utilities for bounding boxes and segmentation masks."""

from __future__ import annotations

from typing import Any, Sequence
import cv2
import numpy as np


# Standard distinct colors (RGB)
DEFAULT_PALETTE = [
    (239, 68, 68),    # Red
    (34, 197, 94),    # Green
    (59, 130, 246),   # Blue
    (245, 158, 11),   # Amber / Orange
    (168, 85, 247),   # Purple
    (236, 72, 153),   # Pink
    (20, 184, 166),   # Teal
    (234, 179, 8),    # Yellow
    (99, 102, 241),   # Indigo
    (6, 182, 212),    # Cyan
]


def get_color_for_class(class_name_or_id: str | int, palette: Sequence[tuple[int, int, int]] | None = None) -> tuple[int, int, int]:
    """Return a deterministic RGB color for a given class name or id."""
    palette = palette or DEFAULT_PALETTE
    if isinstance(class_name_or_id, int):
        idx = class_name_or_id % len(palette)
    else:
        idx = sum(ord(c) for c in str(class_name_or_id)) % len(palette)
    return palette[idx]


def draw_bounding_box(
    image: np.ndarray,
    box: tuple[float, float, float, float] | list[float] | np.ndarray,
    label: str | None = None,
    score: float | None = None,
    color: tuple[int, int, int] = (34, 197, 94),
    thickness: int = 2,
    draw_label: bool = True,
    draw_score: bool = True,
) -> np.ndarray:
    """Draw a 2D bounding box with optional text badge on an RGB image.

    Args:
        image: RGB uint8 image array (H, W, 3).
        box: (x1, y1, x2, y2) in pixel coordinates.
        label: Class name or label text.
        score: Confidence score in [0, 1].
        color: RGB tuple for box outline and badge.
        thickness: Line thickness in pixels.
        draw_label: Whether to draw class label.
        draw_score: Whether to draw confidence score.

    Returns:
        Annotated RGB image copy or modified array.
    """
    img = image.copy()
    h, w = img.shape[:2]

    x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1, min(x2, w - 1))
    y2 = max(y1, min(y2, h - 1))

    # In OpenCV, drawing expects BGR or RGB depending on convention;
    # since img is RGB, we use color directly as (R, G, B)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Format text banner
    text_parts = []
    if draw_label and label:
        text_parts.append(str(label))
    if draw_score and score is not None:
        text_parts.append(f"{score:.2f}")

    if text_parts:
        text = " ".join(text_parts)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thick = 1
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thick)

        # Draw filled background for label
        badge_y1 = max(0, y1 - th - 6)
        badge_y2 = y1
        badge_x2 = min(w, x1 + tw + 6)
        cv2.rectangle(img, (x1, badge_y1), (badge_x2, badge_y2), color, -1)

        # Draw white text
        text_y = badge_y2 - 3
        cv2.putText(
            img,
            text,
            (x1 + 3, text_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thick,
            cv2.LINE_AA,
        )

    return img


def draw_segmentation_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (34, 197, 94),
    alpha: float = 0.45,
    contour_thickness: int = 2,
) -> np.ndarray:
    """Draw a semi-transparent instance segmentation mask on an RGB image.

    Args:
        image: RGB uint8 image array (H, W, 3).
        mask: 2D binary mask (H, W) or polygon, boolean or uint8.
        color: RGB tuple for the mask overlay.
        alpha: Transparency coefficient in [0, 1].
        contour_thickness: Line thickness for the mask boundary contour.

    Returns:
        RGB image array with blended mask.
    """
    img = image.copy()
    h, w = img.shape[:2]

    # Normalize mask to boolean (H, W)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    elif mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]

    # Resize mask if shape does not match image
    if mask.shape[:2] != (h, w):
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        mask_uint8 = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)
        mask_bool = mask_uint8 > 0
    else:
        mask_bool = mask > 0

    if not np.any(mask_bool):
        return img

    # Alpha blending
    colored_layer = np.zeros_like(img, dtype=np.uint8)
    colored_layer[mask_bool] = color

    # Blend masked region
    img[mask_bool] = (
        (1.0 - alpha) * img[mask_bool].astype(np.float32)
        + alpha * np.array(color, dtype=np.float32)
    ).astype(np.uint8)

    # Optional boundary contour for crisp visual boundary
    if contour_thickness > 0:
        mask_uint8 = mask_bool.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(img, contours, -1, color, contour_thickness, cv2.LINE_AA)

    return img


_NAMED_RGB_COLORS: dict[str, tuple[int, int, int]] = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
}


def parse_rgb_color(
    color: tuple[int, int, int] | Sequence[int] | str | None,
    default: tuple[int, int, int] = (255, 255, 255),
) -> tuple[int, int, int]:
    """Parse an RGB color specification (tuple, sequence, name, hex, or 'R,G,B') cleanly."""
    if color is None:
        return default
    if isinstance(color, str):
        s = color.strip().lower()
        if s in _NAMED_RGB_COLORS:
            return _NAMED_RGB_COLORS[s]
        if s.startswith("#") and len(s) == 7:
            return (int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16))
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) == 3:
            return tuple(int(max(0, min(255, int(p)))) for p in parts)  # type: ignore[return-value]
        raise ValueError(f"Invalid RGB color string '{color}'. Expected 'R,G,B', '#RRGGBB', or color name.")
    vals = list(color)
    if len(vals) != 3:
        raise ValueError(f"Expected 3 RGB channels, got {len(vals)}: {color}")
    return tuple(int(max(0, min(255, int(v)))) for v in vals)  # type: ignore[return-value]


def _normalize_mask_bool(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Normalize a 2D/3D binary mask to a boolean (H, W) array matching frame shape."""
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    elif mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]

    if mask.shape[:2] != (height, width):
        mask_uint8 = (mask > 0).astype(np.uint8) * 255
        mask_uint8 = cv2.resize(mask_uint8, (width, height), interpolation=cv2.INTER_NEAREST)
        return mask_uint8 > 0
    return mask > 0


def restrict_palette_color(
    image: np.ndarray,
    color: tuple[int, int, int] | Sequence[int] | str = (255, 255, 255),
    margin: int = 16,
) -> np.ndarray:
    """Restrict an RGB image so no pixel equals or falls within `margin` (L_inf) of `color`.

    For example, with ``color=(255, 255, 255)`` and ``margin=16``, every nearly-white
    pixel (all channels >= 240) is clamped down to ``(239, 239, 239)`` so that pure white
    is strictly reserved for obscuring object masks.
    """
    target = parse_rgb_color(color)
    safe_margin = max(1, int(margin))
    img = image.copy()

    target_arr = np.array(target, dtype=np.int16)
    diff = np.abs(img.astype(np.int16) - target_arr)
    near_mask = np.all(diff < safe_margin, axis=-1)
    if not np.any(near_mask):
        return img

    img_i16 = img[near_mask].astype(np.int16)
    for ch in range(3):
        c_val = int(target_arr[ch])
        hi = c_val + safe_margin
        lo = c_val - safe_margin
        if hi > 255:
            img_i16[:, ch] = lo
        elif lo < 0:
            img_i16[:, ch] = hi
        else:
            img_i16[:, ch] = np.where(img_i16[:, ch] >= c_val, hi, lo)

    img[near_mask] = np.clip(img_i16, 0, 255).astype(np.uint8)
    return img


def draw_obscuring_mask(
    image: np.ndarray,
    masks: Sequence[np.ndarray] | np.ndarray,
    color: tuple[int, int, int] | Sequence[int] | str = (255, 255, 255),
    margin: int = 16,
) -> np.ndarray:
    """Anonymize detected objects with an opaque solid mask in a reserved color.

    Restricts all non-masked pixels away from `color` by at least `margin` (L_inf),
    then paints all masked object regions with the exact solid `color` (no blending,
    contours, boxes, or text).
    """
    target = parse_rgb_color(color)
    restricted = restrict_palette_color(image, color=target, margin=margin)
    h, w = restricted.shape[:2]

    if isinstance(masks, np.ndarray):
        mask_list = [masks] if masks.ndim <= 2 or (masks.ndim == 3 and (masks.shape[0] == 1 or masks.shape[-1] == 1)) else list(masks)
    else:
        mask_list = list(masks)

    if not mask_list:
        return restricted

    combined = np.zeros((h, w), dtype=bool)
    for m in mask_list:
        if m is not None:
            combined |= _normalize_mask_bool(m, h, w)

    if np.any(combined):
        restricted[combined] = target
    return restricted

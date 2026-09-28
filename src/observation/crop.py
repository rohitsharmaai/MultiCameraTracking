"""Person-crop extraction. Segmentation runs on this crop, never the full frame."""

from __future__ import annotations

import numpy as np

from src.types import BBox


def xyxy_to_int(bbox: BBox, width: int, height: int) -> tuple[int, int, int, int]:
    clipped = bbox.clip(width, height)
    x1 = int(np.floor(clipped.x1))
    y1 = int(np.floor(clipped.y1))
    x2 = int(np.ceil(clipped.x2))
    y2 = int(np.ceil(clipped.y2))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def extract_crop(
    frame: np.ndarray,
    bbox: BBox,
    pad_ratio: float = 0.05,
) -> tuple[np.ndarray, BBox]:
    h, w = frame.shape[:2]
    pw = bbox.width() * pad_ratio
    ph = bbox.height() * pad_ratio
    padded = BBox(bbox.x1 - pw, bbox.y1 - ph, bbox.x2 + pw, bbox.y2 + ph)
    x1, y1, x2, y2 = xyxy_to_int(padded, w, h)
    crop = frame[y1:y2, x1:x2].copy()
    return crop, BBox(float(x1), float(y1), float(x2), float(y2))

"""Align a 112x112 face crop from YuNet five-point landmarks for AuraFace."""

from __future__ import annotations

import cv2
import numpy as np

# ArcFace / InsightFace reference landmarks for 112x112.
_REF = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_face(image: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    src = landmarks.astype(np.float32)
    if src.shape[0] < 5:
        return cv2.resize(image, (size, size))
    dst = _REF * (size / 112.0)
    m, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if m is None:
        return cv2.resize(image, (size, size))
    return cv2.warpAffine(image, m, (size, size), borderValue=0.0)

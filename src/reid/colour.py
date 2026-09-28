"""Part-based clothing colour descriptor, fused with the OSNet body embedding.

On low-resolution footage (the EPFL clips are 360x288, people ~70 px wide)
OSNet alone cannot reliably separate people in similar outfits: different
people often score 0.7+. What does separate them is the colour of the upper
and lower clothing. Measured on labelled 4p-c0 crops:

    OSNet cosine                      AUC 0.915
    upper/lower HSV histograms        AUC 0.971
    OSNet x weaker part colour match  AUC 0.978

A fused body vector is ``[body embedding (D) | sqrt(hist upper) | sqrt(hist
lower)]`` where D is the body model's size (OSNet 512, Youtu ReID 768). Each
part is a square-rooted, L1-normalised HSV histogram, so the dot product of
two parts is their Bhattacharyya coefficient. Everything stays a plain list
of floats (no schema change).

Vectors of different lengths come from different models (or with/without
colour) and are never compared: the similarity is 0.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

OSNET_DIM = 512
H_BINS, S_BINS, V_BINS = 8, 4, 4
PART_DIM = H_BINS * S_BINS * V_BINS
COLOUR_DIM = 2 * PART_DIM
FUSED_DIM = OSNET_DIM + COLOUR_DIM  # OSNet + colour
# Body model sizes that can carry a colour suffix.
BODY_DIMS = (512, 768)
# Vertical extent of each part as a fraction of the crop height (the crop is
# the detection box plus 5% padding, so it spans the whole person even when
# the legs are hidden): skip the head, split torso and legs, skip the feet.
# Bands are NOT relative to the visible mask - with occluded legs that would
# slide the "lower" band up onto the torso.
PARTS = ((0.18, 0.50), (0.55, 0.92))
_MIN_PART_PIXELS = 20


def part_colour(crop: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """``2 * PART_DIM`` vector: sqrt-histogram of the upper then lower body.

    A part with too few person pixels is all zeros (and is ignored when
    comparing).
    """
    h, w = crop.shape[:2]
    person = (
        mask > 0
        if mask is not None and mask.shape[:2] == (h, w)
        else np.ones((h, w), dtype=bool)
    )
    out = np.zeros(2 * PART_DIM, dtype=np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    for i, (a, b) in enumerate(PARTS):
        band = np.zeros_like(person)
        band[int(a * h) : int(b * h)] = True
        part = (person & band).astype(np.uint8)
        if int(part.sum()) < _MIN_PART_PIXELS:
            continue
        hist = cv2.calcHist(
            [hsv], [0, 1, 2], part, [H_BINS, S_BINS, V_BINS], [0, 180, 0, 256, 0, 256]
        ).ravel()
        total = float(hist.sum())
        if total > 0:
            out[i * PART_DIM : (i + 1) * PART_DIM] = np.sqrt(hist / total)
    return out


def fuse(body: Sequence[float], colour: np.ndarray) -> list[float]:
    return [float(x) for x in body] + [float(x) for x in colour]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def _body_dim(size: int) -> int | None:
    """Body-embedding length of a fused vector of ``size`` values, else None."""
    d = size - COLOUR_DIM
    return d if d in BODY_DIMS else None


def appearance_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Similarity of two embeddings of the same model.

    Fused body vectors: body cosine (clipped at 0) times the weaker of the
    two part colour matches, so a mismatch in either the top or the bottom
    clothing blocks a match. Plain vectors (faces, body without colour):
    cosine. Different lengths - different models - are not comparable: 0.
    """
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    if va.size != vb.size:
        return 0.0
    d = _body_dim(va.size)
    if d is None:
        return _cos(va, vb)
    body = max(0.0, _cos(va[:d], vb[:d]))
    colour = []
    for i in range(2):
        pa = va[d + i * PART_DIM : d + (i + 1) * PART_DIM]
        pb = vb[d + i * PART_DIM : d + (i + 1) * PART_DIM]
        if float(pa @ pa) > 0.5 and float(pb @ pb) > 0.5:
            colour.append(float(np.clip(pa @ pb, 0.0, 1.0)))
    return body * min(colour) if colour else body

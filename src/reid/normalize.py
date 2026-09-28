"""Embedding normalization and cosine similarity."""

from __future__ import annotations

import numpy as np


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm < eps:
        return arr
    return arr / norm


def cosine_similarity(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> float:
    va = l2_normalize(np.asarray(a, dtype=np.float32))
    vb = l2_normalize(np.asarray(b, dtype=np.float32))
    return float(np.clip(np.dot(va, vb), -1.0, 1.0))


def cosine_to_unit(sim: float) -> float:
    """Map cosine [-1, 1] to [0, 1]. Kept for tests and non-ReID terms."""
    return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))


def reid_cosine_term(sim: float) -> float:
    """ReID cosine as a [0, 1] score. Negative similarity is evidence against a match."""
    return float(np.clip(sim, 0.0, 1.0))

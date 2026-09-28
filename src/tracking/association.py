"""IoU association. No appearance / FastReID weights.

Greedy highest-IoU matching. SciPy Hungarian is optional and unused by
default so the tracker does not depend on a SciPy/NumPy ABI match.
"""

from __future__ import annotations

import numpy as np

from src.types import BBox


def iou_matrix(tracks: list[BBox], detections: list[BBox]) -> np.ndarray:
    n, m = len(tracks), len(detections)
    mat = np.zeros((n, m), dtype=np.float32)
    for i, tb in enumerate(tracks):
        for j, db in enumerate(detections):
            mat[i, j] = tb.iou(db)
    return mat


def associate(
    track_boxes: list[BBox],
    det_boxes: list[BBox],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    if not track_boxes or not det_boxes:
        return [], list(range(len(track_boxes))), list(range(len(det_boxes)))
    iou = iou_matrix(track_boxes, det_boxes)
    pairs = [
        (float(iou[i, j]), i, j)
        for i in range(iou.shape[0])
        for j in range(iou.shape[1])
        if iou[i, j] >= iou_threshold
    ]
    pairs.sort(reverse=True)
    used_t: set[int] = set()
    used_d: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, i, j in pairs:
        if i in used_t or j in used_d:
            continue
        matches.append((i, j))
        used_t.add(i)
        used_d.add(j)
    unmatched_tracks = [i for i in range(len(track_boxes)) if i not in used_t]
    unmatched_dets = [j for j in range(len(det_boxes)) if j not in used_d]
    return matches, unmatched_tracks, unmatched_dets

"""Body and face quality scoring. Quality is a gate, never an additive match bonus."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.types import Observation, Viewpoint


def sharpness(image: np.ndarray) -> float:
    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def body_quality_score(
    observation: Observation,
    weights: dict[str, Any] | None = None,
) -> float:
    weights = weights or {}
    w_blur = float(weights.get("laplacian_blur_weight", 0.25))
    w_area = float(weights.get("area_weight", 0.15))
    w_aspect = float(weights.get("aspect_weight", 0.10))
    w_occ = float(weights.get("occlusion_weight", 0.20))
    w_pose = float(weights.get("pose_weight", 0.20))
    w_det = float(weights.get("detector_weight", 0.10))

    blur_term = _clamp01(observation.sharpness / 200.0)
    area_term = _clamp01(observation.bbox_area / 40000.0)
    # Standing person typical aspect ~0.35–0.55 (w/h). Penalize extreme boxes.
    aspect = observation.bbox_aspect_ratio
    aspect_term = _clamp01(1.0 - abs(aspect - 0.42) / 0.6)
    occ_term = _clamp01(1.0 - observation.occlusion_score)
    pose_term = _clamp01(observation.pose_score)
    det_term = _clamp01(observation.detector_confidence)
    vis = _clamp01(observation.body_visibility_score)

    score = (
        w_blur * blur_term
        + w_area * area_term
        + w_aspect * aspect_term
        + w_occ * occ_term
        + w_pose * pose_term
        + w_det * det_term
    )
    return _clamp01(score * vis)


def face_quality_score(
    *,
    bbox_size: float,
    detector_confidence: float,
    sharpness_value: float,
    yaw: float,
    pitch: float,
    roll: float,
    landmark_score: float,
    occlusion: float,
    exposure: float,
    completeness: float,
) -> float:
    size_term = _clamp01(bbox_size / 80.0)
    blur_term = _clamp01(sharpness_value / 150.0)
    pose_term = _clamp01(1.0 - (abs(yaw) + abs(pitch) + abs(roll)) / 135.0)
    occ_term = _clamp01(1.0 - occlusion)
    exp_term = 1.0 - abs(_clamp01(exposure) - 0.5) * 2.0
    return _clamp01(
        0.18 * size_term
        + 0.16 * _clamp01(detector_confidence)
        + 0.16 * blur_term
        + 0.14 * pose_term
        + 0.12 * _clamp01(landmark_score)
        + 0.10 * occ_term
        + 0.08 * exp_term
        + 0.06 * _clamp01(completeness)
    )


def estimate_viewpoint(keypoints: np.ndarray | None, scores: np.ndarray | None) -> Viewpoint:
    """Approximate front/side/back from shoulders and hips when available.

    RTMPose-Body2d uses COCO-WholeBody indexing. We use:
    left_shoulder=5, right_shoulder=6, left_hip=11, right_hip=12 when present.
    """
    if keypoints is None or scores is None or len(keypoints) < 13:
        return Viewpoint.UNKNOWN
    ls, rs = keypoints[5], keypoints[6]
    if scores[5] < 0.3 or scores[6] < 0.3:
        return Viewpoint.UNKNOWN
    shoulder_dx = float(rs[0] - ls[0])
    if abs(shoulder_dx) < 8:
        return Viewpoint.SIDE
    # Image x increases to the right. If left shoulder is to the right of right
    # shoulder, the person is likely facing away (back).
    if shoulder_dx < 0:
        return Viewpoint.BACK
    return Viewpoint.FRONT

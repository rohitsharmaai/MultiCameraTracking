"""Global identity score.

S = w_body * body + w_face * face + w_time * time + w_camera * camera + w_motion * motion

Every term is scaled to [0, 1] before weights. Quality is a gate: it may
scale w_body / w_face to zero. It must not add a bonus that creates a match
when visual similarity is weak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.reid.normalize import reid_cosine_term


@dataclass
class ScoreBreakdown:
    S: float
    body_similarity: float
    face_similarity: float
    temporal_plausibility: float
    camera_plausibility: float
    motion_plausibility: float
    used_body: bool
    used_face: bool
    both_quality_failed: bool


def temporal_plausibility(dt_seconds: float, max_gap: float = 180.0) -> float:
    if dt_seconds < 0:
        return 0.0
    if dt_seconds == 0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - dt_seconds / max_gap))


def motion_plausibility(
    expected_direction: tuple[float, float] | None,
    observed_direction: tuple[float, float] | None,
) -> float:
    if expected_direction is None or observed_direction is None:
        return 0.5
    a = expected_direction
    b = observed_direction
    na = (a[0] ** 2 + a[1] ** 2) ** 0.5
    nb = (b[0] ** 2 + b[1] ** 2) ** 0.5
    if na < 1e-6 or nb < 1e-6:
        return 0.5
    cos = (a[0] * b[0] + a[1] * b[1]) / (na * nb)
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))


def compute_score(
    *,
    body_cosine: float | None,
    face_cosine: float | None,
    body_quality: float,
    face_quality: float,
    temporal: float,
    camera: float,
    motion: float,
    cfg: dict[str, Any],
    face_weight_scale: float = 1.0,
) -> ScoreBreakdown:
    w_body = float(cfg.get("w_body", 0.45))
    w_face = float(cfg.get("w_face", 0.30))
    w_time = float(cfg.get("w_time", 0.10))
    w_camera = float(cfg.get("w_camera", 0.10))
    w_motion = float(cfg.get("w_motion", 0.05))
    body_gate = float(cfg.get("body_quality_gate", 0.35))
    face_gate = float(cfg.get("face_quality_gate", 0.45))

    # Cross-view face matches (e.g. query front vs gallery side) are much less
    # reliable than same-view ones.  Scale the face weight down instead of
    # letting an unreliable face cosine dilute a strong body match.  A scale
    # of 0.0 removes the face term from the weighted average entirely.
    w_face *= max(0.0, min(1.0, float(face_weight_scale)))

    used_body = body_cosine is not None and body_quality >= body_gate
    used_face = face_cosine is not None and face_quality >= face_gate
    both_failed = (not used_body) and (not used_face)

    body_term = reid_cosine_term(body_cosine) if used_body and body_cosine is not None else 0.0
    face_term = reid_cosine_term(face_cosine) if used_face and face_cosine is not None else 0.0
    if not used_body:
        w_body = 0.0
    if not used_face:
        w_face = 0.0

    weight_sum = w_body + w_face + w_time + w_camera + w_motion
    if weight_sum <= 0 or both_failed:
        return ScoreBreakdown(
            S=0.0,
            body_similarity=body_term,
            face_similarity=face_term,
            temporal_plausibility=temporal,
            camera_plausibility=camera,
            motion_plausibility=motion,
            used_body=used_body,
            used_face=used_face,
            both_quality_failed=both_failed,
        )

    S = (
        w_body * body_term
        + w_face * face_term
        + w_time * temporal
        + w_camera * camera
        + w_motion * motion
    ) / weight_sum
    return ScoreBreakdown(
        S=float(S),
        body_similarity=body_term,
        face_similarity=face_term,
        temporal_plausibility=temporal,
        camera_plausibility=camera,
        motion_plausibility=motion,
        used_body=used_body,
        used_face=used_face,
        both_quality_failed=both_failed,
    )

"""YuNet face detector on the person crop. Exact artifact: face_detection_yunet_2023mar.onnx."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.observation.quality import face_quality_score, sharpness
from src.registry import ModelRegistry


@dataclass
class FaceDetection:
    bbox: tuple[float, float, float, float]
    landmarks: np.ndarray
    confidence: float
    quality: float
    yaw: float
    pitch: float
    roll: float


class YuNetDetector:
    name = "YuNet"
    version = "2023mar"

    def __init__(self, registry: Optional[ModelRegistry] = None, model=None) -> None:
        self._model = model
        self._path = None
        if registry is not None:
            entry = registry.require_loadable("YuNet")
            self.version = entry.version
            self._path = registry.resolve_path(entry)

    def load(self, width: int = 320, height: int = 320) -> None:
        if self._model is not None:
            return
        if self._path is None or not self._path.exists():
            raise FileNotFoundError("YuNet ONNX missing")
        self._model = cv2.FaceDetectorYN.create(
            str(self._path),
            "",
            (width, height),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5,
        )

    def detect(self, crop: np.ndarray) -> Optional[FaceDetection]:
        if crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if w < 16 or h < 16:
            return None
        if self._model is None:
            self.load(width=w, height=h)
        assert self._model is not None
        try:
            self._model.setInputSize((w, h))
            _, faces = self._model.detect(crop)
        except cv2.error:
            return None
        if faces is None or len(faces) == 0:
            return None
        # Take the largest high-confidence face inside the person crop.
        faces = sorted(faces, key=lambda f: float(f[2] * f[3]), reverse=True)
        face = faces[0]
        x, y, bw, bh = (float(v) for v in face[:4])
        conf = float(face[-1])
        lms = np.array(face[4:14], dtype=np.float32).reshape(5, 2)
        yaw, pitch, roll = _head_pose(lms)
        y1c, y2c = int(max(0, y)), int(min(h, y + bh))
        x1c, x2c = int(max(0, x)), int(min(w, x + bw))
        face_crop = crop[y1c:y2c, x1c:x2c]
        sharp = sharpness(face_crop) if face_crop.size else 0.0
        exposure = float(np.mean(crop) / 255.0) if crop.size else 0.5
        completeness = 1.0 if (x >= 0 and y >= 0 and x + bw <= w and y + bh <= h) else 0.6
        quality = face_quality_score(
            bbox_size=min(bw, bh),
            detector_confidence=conf,
            sharpness_value=sharp,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            landmark_score=_landmark_geometry(lms),
            occlusion=0.0,
            exposure=exposure,
            completeness=completeness,
        )
        return FaceDetection(
            bbox=(x, y, x + bw, y + bh),
            landmarks=lms,
            confidence=conf,
            quality=quality,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
        )


def _head_pose(lms: np.ndarray) -> tuple[float, float, float]:
    """Approximate yaw/pitch/roll in degrees from 5 YuNet landmarks."""
    if lms.shape[0] < 5:
        return 0.0, 0.0, 0.0
    left_eye, right_eye, nose, left_mouth, right_mouth = lms
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    roll = float(np.degrees(np.arctan2(dy, dx + 1e-6)))
    mid_eyes = (left_eye + right_eye) / 2.0
    eye_dist = np.linalg.norm(right_eye - left_eye) + 1e-6
    yaw = float(np.degrees(np.arctan2(nose[0] - mid_eyes[0], eye_dist)))
    pitch = float(np.degrees(np.arctan2(nose[1] - mid_eyes[1], eye_dist)))
    return yaw, pitch, roll


def _landmark_geometry(lms: np.ndarray) -> float:
    if lms.shape[0] < 5:
        return 0.0
    left_eye, right_eye, nose, left_mouth, right_mouth = lms
    eye_dist = np.linalg.norm(right_eye - left_eye)
    mouth_dist = np.linalg.norm(right_mouth - left_mouth)
    if eye_dist < 1:
        return 0.0
    ratio = mouth_dist / eye_dist
    return float(np.clip(1.0 - abs(ratio - 0.85) / 0.85, 0.0, 1.0))

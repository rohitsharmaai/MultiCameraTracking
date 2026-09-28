"""Constant-velocity Kalman filter for axis-aligned boxes (x, y, a, h)."""

from __future__ import annotations

import numpy as np

from src.types import BBox


class KalmanBoxTracker:
    """8-D state: [x, y, a, h, vx, vy, va, vh]."""

    def __init__(self, bbox: BBox) -> None:
        self._F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self._F[i, i + 4] = 1.0
        self._H = np.eye(4, 8, dtype=np.float32)
        self._P = np.eye(8, dtype=np.float32) * 10.0
        self._P[4:, 4:] *= 100.0
        self._Q = np.eye(8, dtype=np.float32)
        self._Q[4:, 4:] *= 0.01
        self._R = np.eye(4, dtype=np.float32) * 1.0
        z = self._bbox_to_z(bbox)
        self._x = np.zeros((8, 1), dtype=np.float32)
        self._x[:4, 0] = z

    @staticmethod
    def _bbox_to_z(bbox: BBox) -> np.ndarray:
        w = max(bbox.width(), 1e-3)
        h = max(bbox.height(), 1e-3)
        x, y = bbox.center()
        a = w / h
        return np.array([x, y, a, h], dtype=np.float32)

    @staticmethod
    def _z_to_bbox(z: np.ndarray) -> BBox:
        x, y, a, h = (float(v) for v in z.reshape(-1)[:4])
        h = max(h, 1e-3)
        w = max(a * h, 1e-3)
        return BBox(x - w / 2.0, y - h / 2.0, x + w / 2.0, y + h / 2.0)

    def predict(self) -> BBox:
        if self._x[7, 0] + self._x[3, 0] <= 0:
            self._x[7, 0] = 0.0
        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return self._z_to_bbox(self._x[:4, 0])

    def update(self, bbox: BBox) -> BBox:
        z = self._bbox_to_z(bbox).reshape(4, 1)
        y = z - self._H @ self._x
        s = self._H @ self._P @ self._H.T + self._R
        try:
            k = self._P @ self._H.T @ np.linalg.inv(s)
        except np.linalg.LinAlgError:
            k = self._P @ self._H.T @ np.linalg.pinv(s)
        self._x = self._x + k @ y
        self._P = (np.eye(8, dtype=np.float32) - k @ self._H) @ self._P
        return self._z_to_bbox(self._x[:4, 0])

    def velocity_xy(self) -> tuple[float, float]:
        return (float(self._x[4, 0]), float(self._x[5, 0]))

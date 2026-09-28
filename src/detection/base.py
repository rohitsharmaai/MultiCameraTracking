"""Detector interface. Detectors find people; they do not assign Global IDs."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src.types import Detection


class PersonDetector(Protocol):
    name: str
    version: str

    def detect(
        self,
        frame: np.ndarray,
        *,
        camera_id: str,
        frame_id: int,
        timestamp: float,
    ) -> list[Detection]:
        ...

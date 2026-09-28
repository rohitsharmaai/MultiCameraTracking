"""PPHumanSeg on the person crop only. Never the full frame.

Required before pose and OSNet. Quality model, not an identity model.
Exact artifact: human_segmentation_pphumanseg_2023mar.onnx
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.registry import ModelRegistry


@dataclass
class SegmentationResult:
    mask: np.ndarray
    coverage: float
    occlusion_score: float


class PPHumanSeg:
    name = "PPHumanSeg"
    version = "2023mar"
    input_size = (192, 192)

    def __init__(self, registry: Optional[ModelRegistry] = None, session=None) -> None:
        self._session = session
        self._path = None
        if registry is not None:
            entry = registry.require_loadable("PPHumanSeg")
            self.version = entry.version
            self._path = registry.resolve_path(entry)

    def load(self) -> None:
        if self._session is not None:
            return
        if self._path is None:
            raise FileNotFoundError("PPHumanSeg local_path is not configured")
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(self._path),
            providers=self._providers(),
        )

    @staticmethod
    def _providers() -> list[str]:
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            preferred = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
            return preferred or ["CPUExecutionProvider"]
        except Exception:
            return ["CPUExecutionProvider"]

    def run(self, crop: np.ndarray) -> SegmentationResult:
        if crop.size == 0:
            return SegmentationResult(
                mask=np.zeros((1, 1), dtype=np.uint8),
                coverage=0.0,
                occlusion_score=1.0,
            )
        if self._session is None:
            self.load()
        assert self._session is not None
        h, w = crop.shape[:2]
        inp = cv2.resize(crop, self.input_size)
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # The OpenCV Zoo PPHumanSeg export expects ImageNet-style
        # normalization to [-1, 1] (mean=0.5, std=0.5).
        inp = (inp - 0.5) / 0.5
        inp = np.transpose(inp, (2, 0, 1))[None, ...]
        input_name = self._session.get_inputs()[0].name
        out = self._session.run(None, {input_name: inp})[0]
        if out.ndim == 4:
            # (1, 2, H, W) or (1, 1, H, W)
            logits = out[0]
            if logits.shape[0] == 2:
                prob = _softmax(logits, axis=0)[1]
            else:
                prob = 1.0 / (1.0 + np.exp(-logits[0]))
        else:
            prob = out.squeeze()
        mask = cv2.resize(prob.astype(np.float32), (w, h))
        binary = (mask >= 0.5).astype(np.uint8)
        coverage = float(binary.mean()) if binary.size else 0.0
        occlusion = float(1.0 - coverage)
        return SegmentationResult(mask=binary, coverage=coverage, occlusion_score=occlusion)


def _softmax(x: np.ndarray, axis: int = 0) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-9, None)

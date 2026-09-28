"""RTMPose-Body2d (Qualcomm ONNX float, 256x192). Quality/geometry only.

Exact artifact: rtmpose_body2d-onnx-float. Not an identity model.
Returns 133 joints with confidences. Use body keypoints for quality/viewpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.observation.quality import estimate_viewpoint
from src.registry import ModelRegistry
from src.types import Viewpoint

# COCO body subset used for visibility (first 17 of whole-body).
_TORSO = (5, 6, 11, 12)
_LEFT = (5, 7, 9, 11, 13, 15)
_RIGHT = (6, 8, 10, 12, 14, 16)


@dataclass
class PoseResult:
    keypoints: np.ndarray
    scores: np.ndarray
    pose_score: float
    body_visibility_score: float
    viewpoint: Viewpoint
    visible_count: int


class RTMPoseBody2d:
    name = "RTMPose-Body2d"
    version = "onnx-float-256x192"
    input_size = (192, 256)  # (W, H) of the Qualcomm export

    def __init__(self, registry: Optional[ModelRegistry] = None, session=None) -> None:
        self._session = session
        self._path = None
        if registry is not None:
            entry = registry.require_loadable("RTMPose-Body2d")
            self.version = entry.version
            self._path = registry.resolve_path(entry)

    def load(self) -> None:
        if self._session is not None:
            return
        if self._path is None:
            raise FileNotFoundError("RTMPose-Body2d local_path is not configured")
        import onnxruntime as ort

        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self._session = ort.InferenceSession(str(self._path), providers=providers or ["CPUExecutionProvider"])

    def run(self, crop: np.ndarray) -> PoseResult:
        if crop.size == 0:
            empty = np.zeros((133, 2), dtype=np.float32)
            scores = np.zeros((133,), dtype=np.float32)
            return PoseResult(empty, scores, 0.0, 0.0, Viewpoint.UNKNOWN, 0)
        if self._session is None:
            self.load()
        assert self._session is not None
        h, w = crop.shape[:2]
        inp_w, inp_h = self.input_size
        letter, scale, pad_x, pad_y = _letterbox(crop, inp_w, inp_h)
        inp = cv2.cvtColor(letter, cv2.COLOR_BGR2RGB).astype(np.float32)
        # The Qualcomm end-to-end export contains its own BGR->RGB and
        # ImageNet mean/std preprocessing nodes.  Pass raw 0-255 RGB here;
        # normalizing outside the graph double-normalizes the input.
        inp = np.transpose(inp, (2, 0, 1))[None, ...]
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: inp})
        kpts, scores = _parse_keypoints(
            outputs, src_w=inp_w, src_h=inp_h, inp_w=inp_w, inp_h=inp_h
        )
        kpts[:, 0] = (kpts[:, 0] - pad_x) / max(scale, 1e-6)
        kpts[:, 1] = (kpts[:, 1] - pad_y) / max(scale, 1e-6)
        kpts[:, 0] = np.clip(kpts[:, 0], 0, w - 1)
        kpts[:, 1] = np.clip(kpts[:, 1], 0, h - 1)
        visible = int((scores[:17] >= 0.3).sum()) if len(scores) else 0
        torso = _mean_score(scores, _TORSO)
        left = _mean_score(scores, _LEFT)
        right = _mean_score(scores, _RIGHT)
        visibility = float(np.clip((torso * 0.5 + (left + right) * 0.25), 0.0, 1.0))
        pose_score = float(np.clip(scores[:17].mean() if len(scores) else 0.0, 0.0, 1.0))
        viewpoint = estimate_viewpoint(kpts, scores)
        return PoseResult(
            keypoints=kpts,
            scores=scores,
            pose_score=pose_score,
            body_visibility_score=visibility,
            viewpoint=viewpoint,
            visible_count=visible,
        )


def _letterbox(crop: np.ndarray, dst_w: int, dst_h: int) -> tuple[np.ndarray, float, int, int]:
    h, w = crop.shape[:2]
    scale = min(dst_w / max(w, 1), dst_h / max(h, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(crop, (nw, nh))
    canvas = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    x0 = (dst_w - nw) // 2
    y0 = (dst_h - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas, scale, x0, y0


def _mean_score(scores: np.ndarray, idxs: tuple[int, ...]) -> float:
    valid = [scores[i] for i in idxs if i < len(scores)]
    return float(np.mean(valid)) if valid else 0.0


def _parse_keypoints(
    outputs: list[np.ndarray],
    src_w: int,
    src_h: int,
    inp_w: int,
    inp_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Accept Qualcomm SimCC (pred_x, pred_y), (1, N, 3), or heatmaps."""
    if (
        len(outputs) >= 2
        and outputs[0].ndim == 3
        and outputs[1].ndim == 3
        and outputs[0].shape[1] == outputs[1].shape[1]
        and outputs[0].shape[-1] > 4
        and outputs[1].shape[-1] > 4
    ):
        px = outputs[0][0]
        py = outputs[1][0]
        xs = px.argmax(axis=-1).astype(np.float32)
        ys = py.argmax(axis=-1).astype(np.float32)
        # SimCC split_ratio=2: bin index / bins * input_size == index / 2
        kpts = np.stack(
            [xs * src_w / float(px.shape[-1]), ys * src_h / float(py.shape[-1])],
            axis=1,
        )
        # rtmlib get_simcc_maximum uses the smaller raw maximum. This float
        # export's peaks sit near 0.05-0.35; scale so the quality gate is usable.
        raw = np.minimum(px.max(axis=-1), py.max(axis=-1)).astype(np.float32)
        scores = np.clip(raw / 0.25, 0.0, 1.0)
        return kpts, scores
    primary = outputs[0]
    if primary.ndim == 3 and primary.shape[-1] >= 2 and primary.shape[-1] <= 4:
        # (1, 133, 2/3)
        arr = primary[0]
        kpts = arr[:, :2].astype(np.float32)
        scores = arr[:, 2] if arr.shape[1] > 2 else np.ones((arr.shape[0],), dtype=np.float32)
        kpts[:, 0] *= src_w / float(inp_w)
        kpts[:, 1] *= src_h / float(inp_h)
        return kpts, scores.astype(np.float32)
    if primary.ndim == 4:
        # heatmap (1, K, H, W)
        heat = primary[0]
        k, hh, ww = heat.shape
        kpts = np.zeros((k, 2), dtype=np.float32)
        scores = np.zeros((k,), dtype=np.float32)
        for i in range(k):
            flat = heat[i].reshape(-1)
            idx = int(flat.argmax())
            y, x = divmod(idx, ww)
            scores[i] = float(flat[idx])
            kpts[i, 0] = x * src_w / float(ww)
            kpts[i, 1] = y * src_h / float(hh)
        return kpts, scores
    # Fallback
    flat = primary.reshape(-1)
    n = min(133, flat.size // 3)
    arr = flat[: n * 3].reshape(n, 3)
    return arr[:, :2].astype(np.float32), arr[:, 2].astype(np.float32)

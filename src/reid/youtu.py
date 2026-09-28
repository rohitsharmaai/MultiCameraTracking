"""Youtu person ReID (OpenCV Model Zoo, Apache-2.0) body encoder.

Chosen over OSNet x1.0 after benchmarking on labelled EPFL Lab crops
(AUC of same-person vs different-person similarity):

    camera A vs camera B pairs      alone    x clothing colour
    OSNet x1.0 msmt17 (masked)      0.802
    OSNet x1.0 msmt17               0.951    0.994
    Youtu ReID                      0.989    0.998

The full crop is used (no background masking): zeroing the background with
PPHumanSeg on ~70 px wide crops also removes parts of the person and cost
every model tested 0.05-0.15 AUC.

Exact artifact: person_reid_youtu_2021nov.onnx, input 1x3x256x128 RGB with
ImageNet mean/std, output 768-d.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.registry import ModelRegistry
from src.reid.normalize import l2_normalize

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class YoutuReIDEncoder:
    name = "YoutuReID"
    version = "2021nov"
    input_size = (128, 256)  # (w, h)
    dim = 768

    def __init__(self, registry: Optional[ModelRegistry] = None, session=None) -> None:
        self._session = session
        self._path = None
        if registry is not None:
            entry = registry.require_loadable(self.name)
            self.version = entry.version
            self._path = registry.resolve_path(entry)

    def load(self) -> None:
        if self._session is not None:
            return
        if self._path is None or not self._path.exists():
            raise FileNotFoundError(
                "Youtu ReID ONNX missing. Run scripts/download_models.py "
                "(opencv/person_reid_youtureid, person_reid_youtu_2021nov.onnx)"
            )
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = [
            p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available
        ] or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(str(self._path), providers=providers)

    def embed(self, crop: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """768-d L2-normalised embedding of the full crop.

        ``mask`` is accepted for interface compatibility with OSNetEncoder
        and deliberately ignored (see module docstring).
        """
        if self._session is None:
            self.load()
        assert self._session is not None
        x = cv2.resize(crop, self.input_size)[:, :, ::-1].astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        name = self._session.get_inputs()[0].name
        out = self._session.run(None, {name: x})[0]
        return l2_normalize(np.asarray(out).reshape(-1))

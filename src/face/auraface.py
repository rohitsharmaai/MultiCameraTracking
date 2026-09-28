"""AuraFace-v1 face embeddings. Exact artifact: glintr100.onnx.

Source: https://huggingface.co/fal/AuraFace-v1 (Apache-2.0).
Do not download scrfd_*, genderage, or InsightFace buffalo/antelope packs.
Face detection stays YuNet. Input is a 112x112 aligned face.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from src.registry import ModelRegistry
from src.reid.normalize import l2_normalize


class AuraFaceEncoder:
    name = "AuraFace-v1"
    version = "glintr100"
    input_size = 112

    def __init__(self, registry: Optional[ModelRegistry] = None, session=None) -> None:
        self._session = session
        self._path = None
        if registry is not None:
            entry = registry.require_loadable("AuraFace-v1")
            self.version = entry.version
            self._path = registry.resolve_path(entry)

    def load(self) -> None:
        if self._session is not None:
            return
        if self._path is None or not self._path.exists():
            raise FileNotFoundError(
                "AuraFace-v1 glintr100.onnx missing. Download only that file from "
                "https://huggingface.co/fal/AuraFace-v1"
            )
        import onnxruntime as ort

        providers = [
            p
            for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
            if p in ort.get_available_providers()
        ]
        self._session = ort.InferenceSession(str(self._path), providers=providers or ["CPUExecutionProvider"])

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        if self._session is None:
            self.load()
        assert self._session is not None
        face = cv2.resize(aligned_face, (self.input_size, self.input_size))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        face = (face - 127.5) / 127.5
        tensor = np.transpose(face, (2, 0, 1))[None, ...]
        input_name = self._session.get_inputs()[0].name
        out = self._session.run(None, {input_name: tensor})[0].reshape(-1)
        return l2_normalize(out)

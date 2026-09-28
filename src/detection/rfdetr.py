"""RF-DETR-L person detector. Exact artifact: RFDETRLarge / rfdetr-large.

Do not use XL/2XL. Person class only. Detector does not own identity.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np

from src.exceptions import ModelNotApprovedError
from src.registry import ModelRegistry
from src.types import BBox, Detection

logger = logging.getLogger(__name__)

# rf-detr-large-2026: patch_size=16, num_windows=2, official resolution=704.
# 560 is the old Large default (divisible by 14) and is rejected by this export.
_RESOLUTION_BLOCK = 32
_OFFICIAL_LARGE_RESOLUTION = 704


def _align_resolution(value: int, block: int = _RESOLUTION_BLOCK) -> int:
    if value in (0, 560):
        return _OFFICIAL_LARGE_RESOLUTION
    if value < 0:
        return _OFFICIAL_LARGE_RESOLUTION
    if value % block == 0:
        return value
    return ((value + block - 1) // block) * block


def _ensure_nvidia_lib_path() -> None:
    """Make pip-installed NVIDIA wheel libs visible to cuDNN."""
    extras: list[str] = []
    try:
        import nvidia  # type: ignore

        root = Path(nvidia.__file__).resolve().parent
        for name in ("cudnn", "cublas", "cuda_runtime", "cuda_nvrtc", "cufft", "curand", "cusolver", "cusparse"):
            lib = root / name / "lib"
            if lib.is_dir():
                extras.append(str(lib))
    except Exception:
        return
    if not extras:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    merged = ":".join(extras + ([current] if current else []))
    os.environ["LD_LIBRARY_PATH"] = merged


def configure_cuda_runtime() -> None:
    """Keep CUDA usable when the pip cuDNN wheel cannot load its sublibraries."""
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available() or not torch.backends.cudnn.enabled:
        return
    try:
        x = torch.zeros(1, 3, 8, 8, device="cuda")
        w = torch.zeros(3, 3, 1, 1, device="cuda")
        torch.nn.functional.conv2d(x, w)
    except RuntimeError as exc:
        if "cudnn" not in str(exc).lower():
            raise
        torch.backends.cudnn.enabled = False
        logger.warning("cuDNN is unusable on this GPU; using CUDA without cuDNN")


def _pick_device(preferred: Optional[str], allow_cpu_fallback: bool) -> str:
    if preferred:
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    if not allow_cpu_fallback:
        raise RuntimeError("CUDA is unavailable and ALLOW_CPU_FALLBACK is false")
    return "cpu"


class RFDETRPersonDetector:
    name = "RF-DETR-L"
    version = "rfdetr-large"

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        *,
        confidence_threshold: float = 0.4,
        person_class_id: int = 1,
        input_resolution: int = _OFFICIAL_LARGE_RESOLUTION,
        device: Optional[str] = None,
        allow_cpu_fallback: bool = True,
    ) -> None:
        if registry is not None:
            entry = registry.require_loadable("RF-DETR-L")
            self.version = entry.version or self.version
        self.confidence_threshold = confidence_threshold
        self.person_class_id = person_class_id
        self.input_resolution = _align_resolution(input_resolution)
        self.device = device
        self.allow_cpu_fallback = allow_cpu_fallback
        self._model = None

    def load(self) -> None:
        try:
            from rfdetr import RFDETRLarge  # type: ignore
        except ImportError as exc:
            raise ModelNotApprovedError(
                "rfdetr package is not installed. Install the approved package "
                "`pip install rfdetr` and use RFDETRLarge only. Do not substitute "
                "an Ultralytics detector."
            ) from exc
        _ensure_nvidia_lib_path()
        configure_cuda_runtime()
        self.device = _pick_device(self.device, self.allow_cpu_fallback)
        self._model = RFDETRLarge(resolution=self.input_resolution, device=self.device)
        logger.info(
            "Loaded RF-DETR-L artifact=rfdetr-large resolution=%s device=%s",
            self.input_resolution,
            self.device,
        )

    def _predict(self, frame: np.ndarray):
        res = self.input_resolution
        return self._model.predict(frame, threshold=self.confidence_threshold, shape=(res, res))

    def detect(
        self,
        frame: np.ndarray,
        *,
        camera_id: str,
        frame_id: int,
        timestamp: float,
    ) -> list[Detection]:
        if self._model is None:
            self.load()
        assert self._model is not None
        try:
            preds = self._predict(frame)
        except RuntimeError as exc:
            message = str(exc).lower()
            gpu_failed = any(token in message for token in ("cudnn", "cuda", "cublas"))
            if not (gpu_failed and self.allow_cpu_fallback and self.device != "cpu"):
                raise
            logger.warning("RF-DETR GPU inference failed (%s); retrying on CPU", exc)
            from rfdetr import RFDETRLarge  # type: ignore

            self.device = "cpu"
            self._model = RFDETRLarge(resolution=self.input_resolution, device="cpu")
            preds = self._predict(frame)
        detections: list[Detection] = []
        xyxy = getattr(preds, "xyxy", None)
        conf = getattr(preds, "confidence", None)
        cls = getattr(preds, "class_id", None)
        if xyxy is None:
            return detections
        for i, box in enumerate(xyxy):
            class_id = int(cls[i]) if cls is not None else self.person_class_id
            if class_id != self.person_class_id:
                continue
            score = float(conf[i]) if conf is not None else 0.0
            if score < self.confidence_threshold:
                continue
            x1, y1, x2, y2 = (float(v) for v in box)
            detections.append(
                Detection(
                    bbox=BBox(x1, y1, x2, y2),
                    confidence=score,
                    class_id=class_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    camera_id=camera_id,
                )
            )
        return detections


class FakePersonDetector:
    """Test double. Never used in production."""

    name = "FakeDetector"
    version = "test"

    def __init__(self, scripted: Optional[dict[int, list[Detection]]] = None) -> None:
        self.scripted = scripted or {}

    def detect(
        self,
        frame: np.ndarray,
        *,
        camera_id: str,
        frame_id: int,
        timestamp: float,
    ) -> list[Detection]:
        return list(self.scripted.get(frame_id, []))

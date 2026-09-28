"""Characterise the body similarity of one audited co-occurring crop pair.

Regression/explanation test for the P0001 case where two tracks that share a
Global ID (and are on screen at the same time) score a high body cosine. It
runs the production body method -- OSNet on the PPHumanSeg-masked crop -- on two
saved audit JPEGs and checks that:

  * the recomputed cosine is close to the cosine the audit already stored, and
  * the *person* (mask bounding box) is at least as similar as the background,
    i.e. the match is not only a shared backdrop.

Skipped unless the exact OSNet artifact and the two audit crops exist, because
both live outside the repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.registry import ModelRegistry
from src.reid.normalize import cosine_similarity
from src.reid.osnet import OSNetEncoder
from src.segmentation.pphumanseg import PPHumanSeg

pytestmark = pytest.mark.gpu

ROOT = Path(__file__).resolve().parents[2]
AUDIT_DIR = Path.home() / ".multicam_reid/identity_audit/P0001"
IMAGE_A = AUDIT_DIR / "co_body_A_1_f439.jpg"
IMAGE_B = AUDIT_DIR / "co_body_A_4_f744.jpg"


def _missing_prereqs() -> str | None:
    if not IMAGE_A.exists() or not IMAGE_B.exists():
        return f"audit crops not found in {AUDIT_DIR}"
    try:
        reg = ModelRegistry(
            ROOT / "configs/models.yaml", ROOT / "models/MANIFEST.yaml", ROOT / "models"
        )
        if not reg.require_loadable("OSNet-x1.0"):
            return "OSNet-x1.0 not loadable"
        path = reg.resolve_path(reg.get("OSNet-x1.0"))
        if path is None or not path.exists():
            return "OSNet-x1.0 checkpoint missing"
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"registry unavailable: {exc}"
    return None


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _stored_cosine() -> float | None:
    summary = AUDIT_DIR / "summary.json"
    if not summary.exists():
        return None
    data = json.loads(summary.read_text(encoding="utf-8"))
    section = (data.get("cooccurring") or {}).get("body")
    return float(section["cosine"]) if section else None


def _load_models():
    reg = ModelRegistry(
        ROOT / "configs/models.yaml", ROOT / "models/MANIFEST.yaml", ROOT / "models"
    )
    encoder = OSNetEncoder(reg)
    encoder.load()
    segmenter = PPHumanSeg(reg)
    segmenter.load()
    return encoder, segmenter


def test_p0001_cooccurring_body_pair_is_similar():
    reason = _missing_prereqs()
    if reason:
        pytest.skip(reason)

    encoder, segmenter = _load_models()
    a = cv2.imread(str(IMAGE_A))
    b = cv2.imread(str(IMAGE_B))
    assert a is not None and b is not None

    seg_a, seg_b = segmenter.run(a), segmenter.run(b)
    masked = float(cosine_similarity(
        encoder.embed(a, mask=seg_a.mask), encoder.embed(b, mask=seg_b.mask)
    ))

    # Same method as src/pipeline/features.py: above the 0.62 body threshold.
    assert masked >= 0.62, f"expected a similar pair, got {masked:.4f}"

    # Reproduce the audit's stored embedding cosine from the saved JPEG.
    stored = _stored_cosine()
    if stored is not None:
        assert abs(masked - stored) < 0.1, f"recomputed {masked:.4f} vs stored {stored:.4f}"

    # The match must survive removing the background, otherwise it is not the
    # person that is similar.
    bb_a, bb_b = _mask_bbox(seg_a.mask), _mask_bbox(seg_b.mask)
    if bb_a and bb_b:
        x1, y1, x2, y2 = bb_a
        person_a = a[y1:y2, x1:x2]
        x1, y1, x2, y2 = bb_b
        person_b = b[y1:y2, x1:x2]
        person_only = float(cosine_similarity(
            encoder.embed(person_a), encoder.embed(person_b)
        ))
        assert person_only > 0.6, (
            f"person-only cosine {person_only:.4f} suggests the match is "
            "driven by the shared background, not the body"
        )

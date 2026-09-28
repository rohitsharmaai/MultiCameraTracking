"""A body crop shared with another person must not become a body embedding."""

from __future__ import annotations

import numpy as np

from src.pipeline.features import FeatureExtractor, overlap_fraction
from src.types import BBox, LocalTrack, TrackState


class _FakeEncoder:
    _model = object()

    def embed(self, crop, mask=None):
        return np.ones(8, dtype=np.float32)


def _track(box: BBox) -> LocalTrack:
    return LocalTrack(
        track_id="A_4", camera_id="A", state=TrackState.CONFIRMED,
        bbox=box, confidence=0.9, frame_id=0, timestamp=0.0,
    )


FRAME = np.full((300, 400, 3), 120, dtype=np.uint8)
TARGET = BBox(100, 50, 180, 250)


def test_overlap_fraction():
    assert overlap_fraction(TARGET, []) == 0.0
    assert overlap_fraction(TARGET, [BBox(300, 0, 350, 100)]) == 0.0
    assert abs(overlap_fraction(TARGET, [BBox(140, 50, 180, 250)]) - 0.5) < 0.02
    # Union, not sum: two identical boxes still cover half.
    half = BBox(140, 50, 180, 250)
    assert abs(overlap_fraction(TARGET, [half, half]) - 0.5) < 0.02


def test_crowded_crop_has_no_body_embedding():
    ext = FeatureExtractor(body_encoder=_FakeEncoder(), max_body_overlap=0.35)
    feats = ext.extract(FRAME, _track(TARGET), other_bboxes=[BBox(130, 40, 200, 260)])
    assert feats.body_embedding is None
    assert ext.crowded_bodies_skipped == 1


def test_small_overlap_keeps_body_embedding():
    ext = FeatureExtractor(body_encoder=_FakeEncoder(), max_body_overlap=0.35)
    feats = ext.extract(FRAME, _track(TARGET), other_bboxes=[BBox(170, 40, 260, 260)])
    assert feats.body_embedding is not None


def test_check_disabled_by_default():
    ext = FeatureExtractor(body_encoder=_FakeEncoder())
    feats = ext.extract(FRAME, _track(TARGET), other_bboxes=[BBox(100, 50, 180, 250)])
    assert feats.body_embedding is not None


class _FakeFaceDet:
    _model = object()

    def __init__(self, size):
        self.size = size

    def detect(self, crop):
        from src.face.yunet import FaceDetection

        lm = np.array([[10, 10], [20, 10], [15, 15], [11, 20], [19, 20]], dtype=np.float32)
        return FaceDetection((5, 5, 5 + self.size, 5 + self.size), lm, 0.9, 0.9, 0, 0, 0)


class _FakeFaceEnc:
    _model = object()

    def embed(self, aligned):
        return np.ones(8, dtype=np.float32)


def test_tiny_face_gets_no_embedding():
    ext = FeatureExtractor(face_detector=_FakeFaceDet(15), face_encoder=_FakeFaceEnc(), min_face_px=32)
    feats = ext.extract(FRAME, _track(TARGET))
    assert feats.face_embedding is None and feats.face_quality == 0.0
    assert feats.face_crop is not None  # still shown in the GUI
    big = FeatureExtractor(face_detector=_FakeFaceDet(40), face_encoder=_FakeFaceEnc(), min_face_px=32)
    assert big.extract(FRAME, _track(TARGET)).face_embedding is not None

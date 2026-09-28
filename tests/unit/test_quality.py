import numpy as np

from src.observation.quality import body_quality_score, estimate_viewpoint, face_quality_score, sharpness
from src.types import Observation, Viewpoint
from src.types import BBox


def test_sharpness_blank_is_low():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    assert sharpness(img) < 1.0


def test_sharpness_edges_is_higher():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:, 16] = 255
    assert sharpness(img) > sharpness(np.zeros((32, 32, 3), dtype=np.uint8))


def test_body_quality_in_unit_interval():
    obs = Observation(
        frame_id=1,
        camera_id="A",
        timestamp=0.0,
        bbox=BBox(0, 0, 40, 80),
        track_id="A_1",
        sharpness=80,
        bbox_area=3200,
        bbox_aspect_ratio=0.5,
        occlusion_score=0.2,
        pose_score=0.8,
        body_visibility_score=0.9,
        detector_confidence=0.7,
    )
    q = body_quality_score(obs)
    assert 0.0 <= q <= 1.0
    assert q > 0.3


def test_zero_body_visibility_is_not_treated_as_perfect():
    base = dict(
        frame_id=1,
        camera_id="A",
        timestamp=0.0,
        bbox=BBox(0, 0, 40, 80),
        track_id="A_1",
        sharpness=80,
        bbox_area=3200,
        bbox_aspect_ratio=0.5,
        occlusion_score=0.2,
        pose_score=0.8,
        detector_confidence=0.7,
    )
    visible = body_quality_score(Observation(**base, body_visibility_score=0.9))
    hidden = body_quality_score(Observation(**base, body_visibility_score=0.0))
    assert hidden < visible


def test_face_quality_penalizes_profile():
    frontal = face_quality_score(
        bbox_size=80,
        detector_confidence=0.9,
        sharpness_value=120,
        yaw=5,
        pitch=2,
        roll=1,
        landmark_score=0.9,
        occlusion=0.0,
        exposure=0.5,
        completeness=1.0,
    )
    profile = face_quality_score(
        bbox_size=80,
        detector_confidence=0.9,
        sharpness_value=120,
        yaw=80,
        pitch=40,
        roll=30,
        landmark_score=0.4,
        occlusion=0.4,
        exposure=0.2,
        completeness=0.5,
    )
    assert frontal > profile


def test_viewpoint_from_shoulders():
    kpts = np.zeros((17, 2), dtype=np.float32)
    scores = np.ones(17, dtype=np.float32)
    kpts[5] = [10, 20]  # left
    kpts[6] = [40, 20]  # right
    assert estimate_viewpoint(kpts, scores) == Viewpoint.FRONT
    kpts[5] = [40, 20]
    kpts[6] = [10, 20]
    assert estimate_viewpoint(kpts, scores) == Viewpoint.BACK

import numpy as np

from src.observation.manager import ObservationManager
from src.types import BBox, LocalTrack, TrackState, Viewpoint


def test_observation_selection_top_k_and_diversity():
    mgr = ObservationManager(top_k_body=3, min_body_quality=0.01, min_sharpness=0.0, min_bbox_area=1)
    frame = np.zeros((120, 80, 3), dtype=np.uint8)
    frame[:, 20] = 255
    track = LocalTrack(
        track_id="A_1",
        camera_id="A",
        state=TrackState.CONFIRMED,
        bbox=BBox(10, 10, 50, 90),
        confidence=0.9,
        frame_id=0,
        timestamp=0.0,
    )
    for i, vp in enumerate([Viewpoint.FRONT, Viewpoint.SIDE, Viewpoint.BACK, Viewpoint.FRONT]):
        track.frame_id = i
        mgr.consider(
            frame,
            track,
            detector_confidence=0.9,
            pose_score=0.8,
            viewpoint=vp,
        )
    selected = mgr.select_body("A_1")
    assert 1 <= len(selected) <= 3
    views = {b.observation.viewpoint for b in selected}
    assert len(views) >= 2


def test_tentative_track_not_observed():
    mgr = ObservationManager(min_sharpness=0.0, min_bbox_area=1)
    frame = np.ones((80, 40, 3), dtype=np.uint8) * 40
    track = LocalTrack(
        track_id="A_1",
        camera_id="A",
        state=TrackState.TENTATIVE,
        bbox=BBox(5, 5, 25, 55),
        confidence=0.8,
        frame_id=0,
        timestamp=0.0,
    )
    assert mgr.consider(frame, track, detector_confidence=0.8) is None

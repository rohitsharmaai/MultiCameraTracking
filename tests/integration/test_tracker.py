from src.tracking.botsort import BoTSORTTracker
from src.types import BBox, Detection, TrackState


def _det(cam, fid, ts, box, conf=0.9) -> Detection:
    return Detection(
        bbox=BBox(*box),
        confidence=conf,
        class_id=1,
        frame_id=fid,
        timestamp=ts,
        camera_id=cam,
    )


def test_track_lifecycle_and_namespace():
    tracker = BoTSORTTracker(camera_id="A", min_confirmed_frames=3, max_lost_frames=2)
    box = (10, 10, 40, 80)
    for i in range(5):
        tracks = tracker.update([_det("A", i, i / 25, box)], i, i / 25)
    confirmed = [t for t in tracks if t.state == TrackState.CONFIRMED]
    assert confirmed
    assert confirmed[0].track_id.startswith("A_")

    # disappear — confirmed tracks are lost, then terminated and dropped
    last = []
    for i in range(5, 9):
        last = tracker.update([], i, i / 25)
    assert all(t.state != TrackState.CONFIRMED for t in last)
    assert tracker.confirmed() == []


def test_rejects_foreign_camera_detections():
    tracker = BoTSORTTracker(camera_id="A")
    try:
        tracker.update([_det("B", 0, 0, (0, 0, 10, 10))], 0, 0)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_detector_to_tracker_ids_are_local_only():
    tracker = BoTSORTTracker(camera_id="B", min_confirmed_frames=2)
    tracks = []
    for i in range(3):
        tracks = tracker.update([_det("B", i, i / 25, (5, 5, 25, 55))], i, i / 25)
    assert all(t.track_id.startswith("B_") for t in tracks)
    assert all(t.track_id != "P0001" for t in tracks)

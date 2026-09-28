"""Identity audit: tracks are (video, local track); crops come from the right frame."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.gui.duplicates import audit_identity, cooccurring_pairs_from_review
from src.identity.gallery import PersonGallery
from src.types import EmbeddingRecord, Viewpoint
from tests.conftest import near, unit_vec


def _rec(track, video, frame, emb):
    return EmbeddingRecord(
        global_person_id="P0001", track_id=track, camera_id="A", timestamp=float(frame),
        embedding=emb, quality_score=0.8, viewpoint=Viewpoint.FRONT, source_frame_id=frame,
        source_video_id=video, source_video_name=f"{video}.avi", source_frame_number=frame,
        source_timestamp_ms=frame * 40, model_name="OSNet-x1.0", model_version="", modality="body",
    )


def _snap(root: Path, video: str, frame: int, track: str, value: int) -> None:
    d = root / "P0001"
    d.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / f"body_{video}_{frame}_{track}.jpg"), np.full((20, 10, 3), value, np.uint8))


def test_same_local_id_in_two_videos_is_two_tracks(tmp_path):
    base = unit_vec(1)
    gal = PersonGallery("P0001", body=[
        _rec("A_1", "v1", 10, base),
        _rec("A_1", "v2", 10, near(base, seed=2)),
    ])
    _snap(tmp_path / "snaps", "v1", 10, "A_1", 50)
    _snap(tmp_path / "snaps", "v2", 10, "A_1", 200)
    out = audit_identity(gal, tmp_path / "snaps", tmp_path / "audit")
    assert out["body"] is not None
    assert {out["body"]["a"]["video"], out["body"]["b"]["video"]} == {"v1.avi", "v2.avi"}
    # Frame 10 of v1 and frame 10 of v2 are not "on screen together".
    assert out["cooccurring"]["pair"] is None


def test_crop_uses_nearest_frame_of_same_track_and_video(tmp_path):
    base = unit_vec(3)
    gal = PersonGallery("P0001", body=[
        _rec("A_1", "v1", 439, base),
        _rec("A_4", "v1", 744, near(base, seed=4)),
    ])
    snaps = tmp_path / "snaps"
    _snap(snaps, "v1", 180, "A_1", 10)
    _snap(snaps, "v1", 430, "A_1", 20)
    _snap(snaps, "v2", 439, "A_1", 30)  # other video, same local ID: never used
    _snap(snaps, "v1", 744, "A_4", 40)
    out = audit_identity(gal, snaps, tmp_path / "audit")
    sides = {out["body"]["a"]["track"]: out["body"]["a"], out["body"]["b"]["track"]: out["body"]["b"]}
    assert sides["A_1"]["frame"] == 439 and sides["A_1"]["crop_frame"] == 430
    assert sides["A_4"]["crop_frame"] == 744


def test_review_pairs_are_scoped_per_video(tmp_path):
    job = tmp_path / "job1"
    job.mkdir()
    rows = [
        {"frame": 5, "people": [
            {"global_id": "P0001", "local_track": "A_1", "video_name": "v1.avi"},
            {"global_id": "P0001", "local_track": "A_4", "video_name": "v1.avi"},
        ]},
        {"frame": 9, "people": [
            {"global_id": "P0001", "local_track": "A_1", "video_name": "v1.avi"},
            {"global_id": "P0001", "local_track": "A_4", "video_name": "v1.avi"},
        ]},
    ]
    (job / "timeline.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert cooccurring_pairs_from_review(tmp_path, "P0001") == {("v1", "A_1", "A_4"): (5, 9)}

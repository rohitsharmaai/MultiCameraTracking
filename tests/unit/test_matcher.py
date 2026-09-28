from __future__ import annotations

import numpy as np

from src.identity.candidates import Candidate
from src.identity.gallery import PersonGallery
from src.identity.matcher import IdentityMatcher, MatcherConfig
from src.identity.topology import CameraTopology
from src.types import EmbeddingRecord, Viewpoint


def _record(gid: str, camera: str, embedding: list[float]) -> EmbeddingRecord:
    return EmbeddingRecord(
        global_person_id=gid,
        track_id=f"{camera}_1",
        camera_id=camera,
        timestamp=1.0,
        embedding=embedding,
        quality_score=0.9,
        viewpoint=Viewpoint.FRONT,
        source_frame_id=0,
        source_video_id="v1",
        source_video_name="v1.mp4",
        source_frame_number=0,
        source_timestamp_ms=0,
        model_name="OSNet-x1.0",
        model_version="v",
        modality="body",
    )


def test_impossible_same_run_transition_is_not_accepted():
    topology = CameraTopology(
        {"edges": [{"from_camera": "A", "to_camera": "B", "min_transition_seconds": 1, "max_transition_seconds": 10}]}
    )
    config = MatcherConfig(
        accept_threshold=0.55,
        minimum_score_margin=0.08,
        scoring={
            "w_body": 0.5,
            "w_face": 0.0,
            "w_time": 0.1,
            "w_camera": 0.1,
            "w_motion": 0.0,
            "body_quality_gate": 0.3,
            "face_quality_gate": 0.4,
        },
        topology_enabled=True,
        temporal_enabled=True,
        unknown_edge_penalty=1.0,
    )
    matcher = IdentityMatcher(config, topology)
    gallery = PersonGallery("P0001", last_camera_id="A", last_seen_at=1.0, first_camera_id="A")
    query = np.eye(4, dtype=np.float32)[0].tolist()
    gallery.body.append(_record("P0001", "A", query))
    candidate = Candidate("P0001", 1.0, -1.0, "A", 1.0, ["same_camera"])
    breakdown = matcher.score_candidate(
        candidate,
        gallery,
        camera_id="Z",
        timestamp=1.0,
        body_embedding=query,
        face_embedding=None,
        body_quality=0.9,
        face_quality=0.0,
    )
    decision = matcher.decide("A_1", "Z", [(candidate, breakdown)], both_quality_failed=False)
    assert decision.state.value == "UNCERTAIN"
    assert decision.global_person_id is None

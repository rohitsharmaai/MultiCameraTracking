from src.identity.candidates import generate_candidates
from src.identity.gallery import PersonGallery
from src.identity.topology import CameraTopology
from src.types import EmbeddingRecord, Viewpoint
from tests.conftest import unit_vec


def _gal(gid, cam, ts, emb):
    g = PersonGallery(gid, last_camera_id=cam, last_seen_at=ts, first_camera_id=cam)
    g.body.append(
        EmbeddingRecord(
            global_person_id=gid,
            track_id=f"{cam}_1",
            camera_id=cam,
            timestamp=ts,
            embedding=emb,
            quality_score=0.9,
            viewpoint=Viewpoint.FRONT,
            source_frame_id=0,
            source_video_id="v",
            source_video_name="v.mp4",
            source_frame_number=0,
            source_timestamp_ms=0,
            model_name="OSNet-x1.0",
            model_version="t",
            modality="body",
        )
    )
    return g


def test_candidate_filtering_prefers_recent_and_same_camera():
    topo = CameraTopology(
        {
            "edges": [
                {
                    "from_camera": "A",
                    "to_camera": "B",
                    "min_transition_seconds": 1,
                    "max_transition_seconds": 90,
                    "expected_direction": "e",
                    "confidence": 0.9,
                }
            ]
        }
    )
    q = unit_vec(1)
    galleries = {
        "P0001": _gal("P0001", "A", 10.0, q),
        "P0002": _gal("P0002", "C", 10.0, unit_vec(2)),
    }
    cands = generate_candidates(
        galleries,
        camera_id="A",
        timestamp=12.0,
        body_embedding=q,
        face_embedding=None,
        topology=topo,
        candidate_k=8,
    )
    ids = [c.global_person_id for c in cands]
    assert "P0001" in ids
    assert any("same_camera" in c.reasons for c in cands if c.global_person_id == "P0001")

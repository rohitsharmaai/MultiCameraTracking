from pathlib import Path

from src.database.repository import IdentityRepository
from src.database.models import EmbeddingRow
from src.database.session import create_engine_from_url, init_db
from src.identity.gallery import PersonGallery
from src.identity.manager import GlobalIdentityManager
from src.identity.matcher import IdentityMatcher, MatcherConfig
from src.types import EmbeddingRecord, Viewpoint


def test_sqlite_survives_new_engine(tmp_path: Path):
    url = f"sqlite:///{tmp_path / 'gallery.db'}"
    engine = create_engine_from_url(url, environment="test")
    init_db(engine)
    store = IdentityRepository(engine)
    gal = PersonGallery(global_person_id="P0001", last_camera_id="A", last_seen_at=1.0, first_camera_id="A")
    store.upsert_person(gal)
    store.add_embedding(
        EmbeddingRecord(
            global_person_id="P0001",
            track_id="A_1",
            camera_id="A",
            timestamp=1.0,
            embedding=[0.1, 0.2, 0.3],
            quality_score=0.9,
            viewpoint=Viewpoint.FRONT,
            source_frame_id=3,
            source_video_id="abc",
            source_video_name="video_01.mp4",
            source_frame_number=3,
            source_timestamp_ms=120,
            model_name="OSNet-x1.0",
            model_version="t",
            modality="body",
        )
    )

    engine2 = create_engine_from_url(url, environment="test")
    store2 = IdentityRepository(engine2)
    loaded = store2.load_galleries()
    assert "P0001" in loaded
    rec = loaded["P0001"].body[0]
    assert rec.source_video_name == "video_01.mp4"
    assert rec.source_frame_number == 3
    assert rec.model_name == "OSNet-x1.0"
    assert rec.embedding[0] == 0.1


def test_repeated_same_embedding_is_not_persisted_twice(tmp_path: Path, thresholds, topology):
    url = f"sqlite:///{tmp_path / 'dedupe.db'}"
    engine = create_engine_from_url(url, environment="test")
    init_db(engine)
    store = IdentityRepository(engine)
    matcher = IdentityMatcher(
        MatcherConfig(
            accept_threshold=0.55,
            minimum_score_margin=0.08,
            scoring=thresholds["scoring"],
            topology_enabled=True,
            temporal_enabled=True,
            unknown_edge_penalty=1.0,
        ),
        topology,
    )
    manager = GlobalIdentityManager(store, matcher, topology, thresholds)
    embedding = [0.1] * 512
    for frame in (1, 2):
        manager.assign(
            track_id="A_1",
            camera_id="A",
            timestamp=float(frame),
            body_embedding=embedding,
            face_embedding=None,
            body_quality=0.8,
            face_quality=0.0,
            source_video_id="video-1",
            source_video_name="video_1.mp4",
            force_decide=True,
        )
    with store._session() as session:
        count = session.query(EmbeddingRow).filter_by(global_person_id="P0001").count()
    assert count == 1


def test_production_rejects_sqlite():
    from src.exceptions import ProductionDatabaseError
    import pytest

    with pytest.raises(ProductionDatabaseError):
        create_engine_from_url("sqlite:///./x.db", environment="production")


def _track(store, video_id, gid, status="MATCHED"):
    store.add_track(
        track_id="A_1", camera_id="A", global_person_id=gid, start_time=0.0,
        end_time=None, status=status, source_video_id=video_id,
        source_video_name=f"{video_id}.mp4", confidence=0.9,
    )


def test_track_rows_are_kept_per_source_video(tmp_path: Path):
    """A_1 is reused by every video; a later video must not overwrite it."""
    engine = create_engine_from_url(f"sqlite:///{tmp_path / 'g.db'}", environment="test")
    init_db(engine)
    store = IdentityRepository(engine)
    _track(store, "vid1", "P0001")
    _track(store, "vid1", "P0001")  # same video: update, not a new row
    _track(store, "vid2", "P0002")
    rows = store.list_tracks("A")
    assert sorted((r.source_video_id, r.global_person_id) for r in rows) == [
        ("vid1", "P0001"),
        ("vid2", "P0002"),
    ]
    assert [r.source_video_id for r in store.person_timeline("P0001")] == ["vid1"]


def test_track_rows_fall_back_on_old_sqlite_schema(tmp_path: Path):
    """A dev DB created with the old UNIQUE(track_id, camera_id) still works."""
    import sqlite3

    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE camera_tracks (id INTEGER PRIMARY KEY, track_id VARCHAR(64) NOT NULL,"
        " camera_id VARCHAR(64) NOT NULL, global_person_id VARCHAR(32), start_time FLOAT,"
        " end_time FLOAT, first_bbox TEXT, last_bbox TEXT, confidence FLOAT, status VARCHAR(32),"
        " source_video_id VARCHAR(64), source_video_name VARCHAR(256),"
        " CONSTRAINT uq_track_cam UNIQUE (track_id, camera_id))"
    )
    con.commit()
    con.close()
    engine = create_engine_from_url(f"sqlite:///{db}", environment="test")
    init_db(engine)
    store = IdentityRepository(engine)
    _track(store, "vid1", "P0001")
    _track(store, "vid2", "P0002")
    rows = store.list_tracks("A")
    assert [(r.source_video_id, r.global_person_id) for r in rows] == [("vid2", "P0002")]

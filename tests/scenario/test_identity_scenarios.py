"""Scenario tests A–M from the development specification."""

from __future__ import annotations

from src.database.repository import InMemoryStore
from src.identity.manager import GlobalIdentityManager
from src.types import IdentityAssignmentState, MatchKind
from tests.conftest import near, unit_vec


def _assign(manager, track, cam, ts, body, face=None, bq=0.8, fq=0.0, **kw):
    return manager.assign(
        track_id=track,
        camera_id=cam,
        timestamp=ts,
        body_embedding=body,
        face_embedding=face,
        body_quality=bq,
        face_quality=fq,
        source_video_name=kw.get("video", "video_01.mp4"),
        source_frame_number=kw.get("frame", 0),
        force_decide=True,
    )


def test_a_single_camera_one_person(manager):
    body = unit_vec(1)
    d1 = _assign(manager, "A_1", "A", 1.0, body, video="video_01.mp4")
    d2 = _assign(manager, "A_1", "A", 2.0, near(body, seed=2), video="video_01.mp4")
    assert d1.global_person_id == "P0001"
    assert d2.global_person_id == "P0001"
    assert len(manager.list_persons()) == 1


def test_b_tracker_fragmentation_same_id(manager):
    body = unit_vec(3)
    a = _assign(manager, "A_1", "A", 1.0, body)
    b = _assign(manager, "A_2", "A", 8.0, near(body, seed=4))
    assert a.global_person_id == b.global_person_id == "P0001"


def test_c_cross_camera_transition(manager):
    body = unit_vec(5)
    a = _assign(manager, "A_1", "A", 1.0, body)
    b = _assign(manager, "B_3", "B", 6.0, near(body, seed=6))
    assert a.global_person_id == b.global_person_id
    assert manager.store.transitions


def test_d_similar_people_not_blind_merge(manager):
    a = unit_vec(10)
    b = unit_vec(11)
    d1 = _assign(manager, "A_1", "A", 1.0, a)
    d2 = _assign(manager, "A_2", "A", 1.2, b)
    assert d1.global_person_id != d2.global_person_id


def test_e_face_unavailable_body_works(manager):
    body = unit_vec(12)
    d = _assign(manager, "A_1", "A", 1.0, body, face=None, fq=0.0)
    assert d.global_person_id == "P0001"
    assert d.match_kind in {MatchKind.NEW, MatchKind.EXISTING}


def test_same_track_upgrades_gallery_when_better_pose_arrives(manager):
    entry = unit_vec(50)
    first = _assign(manager, "A_1", "A", 1.0, entry, bq=0.40, video="enter.mp4", frame=3)
    assert first.global_person_id == "P0001"
    later = _assign(
        manager,
        "A_1",
        "A",
        2.4,
        unit_vec(51),
        face=unit_vec(52),
        bq=0.85,
        fq=0.9,
        video="enter.mp4",
        frame=40,
    )
    assert later.global_person_id == "P0001"
    assert later.match_kind == MatchKind.EXISTING
    gal = manager.show_person("P0001")
    assert gal is not None
    assert max(e.quality_score for e in gal.body) >= 0.85
    assert gal.face
    assert max(e.quality_score for e in gal.face) >= 0.9


def test_require_face_blocks_until_face_arrives(manager):
    manager.thresholds.setdefault("identity", {})["require_face_for_id"] = True
    body = unit_vec(40)
    blocked = _assign(manager, "A_1", "A", 1.0, body, face=None, fq=0.0)
    assert blocked.global_person_id is None
    assert blocked.match_kind == MatchKind.UNRESOLVED
    assigned = _assign(manager, "A_1", "A", 2.0, body, face=unit_vec(41), fq=0.9)
    assert assigned.global_person_id == "P0001"


def test_require_body_blocks_face_only(manager):
    manager.thresholds.setdefault("identity", {})["require_body_for_id"] = True
    blocked = manager.assign(
        track_id="A_1",
        camera_id="A",
        timestamp=1.0,
        body_embedding=None,
        face_embedding=unit_vec(42),
        body_quality=0.0,
        face_quality=0.9,
        force_decide=True,
        source_video_name="video_01.mp4",
    )
    assert blocked.global_person_id is None
    assert blocked.match_kind == MatchKind.UNRESOLVED


def test_f_body_poor_face_good(manager):
    face = unit_vec(20)
    first = _assign(manager, "A_1", "A", 1.0, body=unit_vec(21), face=face, bq=0.8, fq=0.9)
    later = manager.assign(
        track_id="B_1",
        camera_id="B",
        timestamp=5.0,
        body_embedding=unit_vec(99),
        face_embedding=near(face, seed=22),
        body_quality=0.1,
        face_quality=0.95,
        force_decide=True,
        source_video_name="video_02.mp4",
    )
    assert later.global_person_id == first.global_person_id


def test_g_both_poor_unresolved(manager):
    d = manager.assign(
        track_id="A_9",
        camera_id="A",
        timestamp=1.0,
        body_embedding=None,
        face_embedding=None,
        body_quality=0.05,
        face_quality=0.05,
        force_decide=True,
    )
    assert d.state == IdentityAssignmentState.UNRESOLVED
    assert d.global_person_id is None


def test_h_same_camera_return(manager):
    body = unit_vec(30)
    first = _assign(manager, "A_1", "A", 1.0, body)
    later = _assign(manager, "A_42", "A", 40.0, near(body, seed=31))
    assert later.global_person_id == first.global_person_id


def test_i_impossible_topology_reduces(manager):
    body = unit_vec(40)
    _assign(manager, "A_1", "A", 1.0, body)
    # Camera Z has no edge from A.
    later = _assign(manager, "Z_1", "Z", 3.0, near(body, seed=41))
    # Strong body but impossible transition: either new ID or unresolved, not silent A-match.
    assert later.global_person_id != "P0001" or later.camera_plausibility == 0.0
    if later.match_kind == MatchKind.EXISTING:
        assert later.camera_plausibility == 0.0


def test_j_video_clock_reset_still_reidentifies(manager):
    body = unit_vec(120)
    first = manager.assign(
        track_id="A_1",
        camera_id="A",
        timestamp=120.0,
        body_embedding=body,
        face_embedding=None,
        body_quality=0.9,
        face_quality=0.0,
        source_video_id="video-01",
        source_video_name="video_01.mp4",
        force_decide=True,
    )
    later = manager.assign(
        track_id="B_1",
        camera_id="B",
        timestamp=0.0,  # the second video-file clock restarts
        body_embedding=near(body, noise=0.01, seed=121),
        face_embedding=None,
        body_quality=0.9,
        face_quality=0.0,
        source_video_id="video-02",
        source_video_name="video_02.mp4",
        force_decide=True,
    )
    assert first.global_person_id == "P0001"
    assert later.global_person_id == "P0001"
    assert later.temporal_plausibility == 1.0
    assert {e.source_video_name for e in manager.show_person("P0001").body} == {
        "video_01.mp4",
        "video_02.mp4",
    }


def test_provisional_new_id_can_switch_after_better_evidence(manager):
    manager.thresholds.setdefault("identity", {})["new_id_confirmation_frames"] = 1
    known = unit_vec(149)
    first = _assign(manager, "A_1", "A", 1.0, known, video="video_01.mp4")
    provisional = _assign(
        manager,
        "B_1",
        "B",
        2.0,
        unit_vec(150),
        bq=0.9,
        video="video_02.mp4",
    )
    assert first.global_person_id == "P0001"
    assert provisional.global_person_id == "P0002"

    # This makes the public state MATCHED, but the assignment is still
    # provisional internally; hysteresis must not block the later recovery.
    _assign(
        manager,
        "B_1",
        "B",
        2.5,
        unit_vec(151),
        bq=0.9,
        video="video_02.mp4",
    )
    # Decisions use the median over the track's recent crops, so the better
    # evidence has to hold for more than one crop before the ID switches.
    for i, ts in enumerate((3.0, 3.5)):
        later = _assign(
            manager,
            "B_1",
            "B",
            ts,
            near(known, noise=0.01, seed=152 + i),
            bq=0.9,
            video="video_02.mp4",
        )
    assert later.global_person_id == "P0001"
    # The spurious provisional id is archived, not deleted, so the operator
    # can still inspect it.
    archived = manager.show_person("P0002")
    assert archived is not None
    assert archived.status == "inactive"
    assert "P0002" in {p.global_person_id for p in manager.list_persons()}
    # ...but it must not be offered as a match candidate any more.
    from src.identity.candidates import generate_candidates

    cands = generate_candidates(
        manager.galleries,
        camera_id="B",
        timestamp=4.0,
        body_embedding=near(known, noise=0.01, seed=153),
        face_embedding=None,
        topology=manager.topology,
    )
    assert "P0002" not in {c.global_person_id for c in cands}


def test_provisional_id_can_recover_after_max_window(manager):
    manager.thresholds.setdefault("identity", {}).update(
        {"new_id_confirmation_frames": 1, "new_id_max_frames": 1}
    )
    known = unit_vec(160)
    first = _assign(manager, "A_1", "A", 1.0, known, video="video_01.mp4")
    provisional = _assign(
        manager,
        "B_1",
        "B",
        2.0,
        unit_vec(161),
        bq=0.9,
        video="video_02.mp4",
    )
    later = _assign(
        manager,
        "B_1",
        "B",
        3.0,
        near(known, noise=0.01, seed=162),
        bq=0.9,
        video="video_02.mp4",
    )
    assert first.global_person_id == "P0001"
    assert provisional.global_person_id == "P0002"
    assert later.global_person_id == "P0001"
    # Archived, not deleted.
    assert manager.show_person("P0002") is not None
    assert manager.show_person("P0002").status == "inactive"


def test_unrelated_candidate_without_visual_gallery_does_not_block_match(manager):
    from src.identity.gallery import PersonGallery

    body = unit_vec(140)
    first = _assign(manager, "A_1", "A", 1.0, body, video="video_01.mp4")
    manager.galleries["P0002"] = PersonGallery(
        "P0002", last_camera_id="A", last_seen_at=1.0, first_camera_id="A"
    )
    later = _assign(
        manager, "A_2", "A", 2.0, near(body, seed=141), video="video_01.mp4"
    )
    assert first.global_person_id == "P0001"
    assert later.global_person_id == "P0001"


def test_new_run_does_not_reuse_a_stale_local_track_id(manager):
    first_body = unit_vec(130)
    second_body = unit_vec(131)
    first = manager.assign(
        track_id="A_1",
        camera_id="A",
        timestamp=0.0,
        body_embedding=first_body,
        face_embedding=None,
        body_quality=0.9,
        face_quality=0.0,
        source_video_id="video-01",
        force_decide=True,
    )
    manager.begin_run()
    second = manager.assign(
        track_id="A_1",  # the new local tracker starts at A_1 again
        camera_id="A",
        timestamp=0.0,
        body_embedding=second_body,
        face_embedding=None,
        body_quality=0.9,
        face_quality=0.0,
        source_video_id="video-02",
        force_decide=True,
    )
    assert first.global_person_id == "P0001"
    assert second.global_person_id != first.global_person_id
    assert len(manager.list_persons()) == 2


def test_j_and_m_persistent_video_to_video(thresholds, topology):
    store = InMemoryStore()
    from src.identity.matcher import IdentityMatcher, MatcherConfig

    def make():
        return GlobalIdentityManager(
            store,
            IdentityMatcher(
                MatcherConfig(0.55, 0.08, thresholds["scoring"], True, True, 1.0),
                topology,
            ),
            topology,
            thresholds,
        )

    body = unit_vec(50)
    m1 = make()
    d1 = _assign(m1, "A_1", "A", 1.0, body, video="video_01.mp4")
    # process restart
    m2 = make()
    m2.reload()
    d2 = _assign(m2, "B_1", "B", 8.0, near(body, seed=51), video="video_02.mp4")
    assert d2.global_person_id == d1.global_person_id
    gal = m2.show_person(d1.global_person_id)
    videos = {e.source_video_name for e in gal.body}
    assert "video_01.mp4" in videos
    assert "video_02.mp4" in videos


def test_k_new_identity_added(manager):
    known = unit_vec(60)
    unseen = unit_vec(61)
    d1 = _assign(manager, "A_1", "A", 1.0, known, video="video_01.mp4")
    d2 = _assign(manager, "A_2", "A", 20.0, near(known, seed=62), video="video_02.mp4")
    d3 = _assign(manager, "A_3", "A", 21.0, unseen, video="video_02.mp4")
    assert d2.global_person_id == d1.global_person_id
    assert d3.global_person_id != d1.global_person_id
    assert len(manager.list_persons()) == 2


def test_l_source_provenance(manager):
    body = unit_vec(70)
    _assign(manager, "A_1", "A", 1.5, body, video="video_01.mp4", frame=12)
    gal = manager.show_person("P0001")
    rec = gal.body[0]
    assert rec.source_video_name == "video_01.mp4"
    assert rec.source_frame_number == 12
    assert rec.track_id == "A_1"
    assert rec.model_name
    assert rec.quality_score > 0


def test_already_matched_track_does_not_mint_second_id(manager):
    body = unit_vec(90)
    first = _assign(manager, "A_1", "A", 1.0, body)
    # Later evidence is poor; must keep P0001 rather than mint P0002.
    later = manager.assign(
        track_id="A_1",
        camera_id="A",
        timestamp=4.0,
        body_embedding=unit_vec(91),
        face_embedding=None,
        body_quality=0.9,
        face_quality=0.0,
        force_decide=True,
    )
    assert first.global_person_id == "P0001"
    assert later.global_person_id == "P0001"
    assert len(manager.list_persons()) == 1


def test_uncertain_when_two_close_matches(manager):
    a = unit_vec(80)
    # Construct a second identity whose embedding is almost the same.
    _assign(manager, "A_1", "A", 1.0, a)
    # Manually insert a second person with a very similar embedding.
    from src.identity.gallery import PersonGallery
    from src.types import EmbeddingRecord, Viewpoint

    gal = PersonGallery("P0002", last_camera_id="A", last_seen_at=1.0, first_camera_id="A")
    gal.body.append(
        EmbeddingRecord(
            global_person_id="P0002",
            track_id="A_9",
            camera_id="A",
            timestamp=1.0,
            embedding=near(a, noise=0.01, seed=1),
            quality_score=0.9,
            viewpoint=Viewpoint.FRONT,
            source_frame_id=0,
            source_video_id="x",
            source_video_name="x.mp4",
            source_frame_number=0,
            source_timestamp_ms=0,
            model_name="OSNet-x1.0",
            model_version="t",
            modality="body",
        )
    )
    manager.galleries["P0002"] = gal
    d = _assign(manager, "A_7", "A", 3.0, near(a, noise=0.01, seed=2))
    assert d.state in {IdentityAssignmentState.UNCERTAIN, IdentityAssignmentState.MATCHED}
    if d.state == IdentityAssignmentState.UNCERTAIN:
        assert d.global_person_id is None

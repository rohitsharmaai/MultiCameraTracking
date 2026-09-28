from src.identity.gallery import evict_then_insert, upgrade_or_insert
from src.types import EmbeddingRecord, Viewpoint


def _rec(qid: str, quality: float, ts: float, vp: Viewpoint) -> EmbeddingRecord:
    return EmbeddingRecord(
        global_person_id="P0001",
        track_id="A_1",
        camera_id="A",
        timestamp=ts,
        embedding=[1.0, 0.0],
        quality_score=quality,
        viewpoint=vp,
        source_frame_id=0,
        source_video_id="v",
        source_video_name="v.mp4",
        source_frame_number=0,
        source_timestamp_ms=0,
        model_name="OSNet-x1.0",
        model_version="t",
        modality="body",
    )


def test_insert_under_cap():
    items = [_rec("a", 0.9, 1, Viewpoint.FRONT)]
    out = evict_then_insert(items, _rec("b", 0.8, 2, Viewpoint.SIDE), gallery_size=3)
    assert len(out) == 2


def test_evict_lowest_quality():
    items = [
        _rec("a", 0.9, 1, Viewpoint.FRONT),
        _rec("b", 0.2, 2, Viewpoint.FRONT),
        _rec("c", 0.8, 3, Viewpoint.FRONT),
    ]
    new = _rec("d", 0.7, 4, Viewpoint.FRONT)
    out = evict_then_insert(items, new, gallery_size=3)
    assert len(out) == 3
    qualities = sorted(x.quality_score for x in out)
    assert 0.2 not in qualities
    assert 0.7 in qualities


def test_never_drop_last_of_viewpoint_if_extras_exist():
    items = [
        _rec("front1", 0.1, 1, Viewpoint.FRONT),
        _rec("front2", 0.9, 2, Viewpoint.FRONT),
        _rec("side", 0.05, 3, Viewpoint.SIDE),
    ]
    new = _rec("n", 0.8, 4, Viewpoint.FRONT)
    out = evict_then_insert(items, new, gallery_size=3)
    viewpoints = {x.viewpoint for x in out}
    assert Viewpoint.SIDE in viewpoints


def test_tie_quality_evicts_oldest():
    items = [
        _rec("old", 0.5, 1, Viewpoint.FRONT),
        _rec("new", 0.5, 9, Viewpoint.FRONT),
    ]
    out = evict_then_insert(items, _rec("x", 0.9, 10, Viewpoint.FRONT), gallery_size=2)
    stamps = [x.timestamp for x in out]
    assert 1 not in stamps


def test_upgrade_replaces_weaker_same_viewpoint():
    items = [_rec("entry", 0.25, 1, Viewpoint.FRONT)]
    out, victim = upgrade_or_insert(items, _rec("full", 0.8, 8, Viewpoint.FRONT), gallery_size=4)
    assert victim is not None and victim.quality_score == 0.25
    assert len(out) == 1
    assert out[0].quality_score == 0.8


def test_upgrade_does_not_replace_same_local_id_from_another_video():
    old = _rec("old", 0.2, 1, Viewpoint.FRONT)
    new = _rec("new", 0.9, 2, Viewpoint.FRONT)
    new.track_id = old.track_id
    new.source_video_id = "video-02"
    new.source_video_name = "video_02.mp4"
    out, victim = upgrade_or_insert([old], new, gallery_size=4)
    assert victim is None
    assert len(out) == 2


def test_upgrade_keeps_different_viewpoint():
    items = [_rec("front", 0.8, 1, Viewpoint.FRONT)]
    out, victim = upgrade_or_insert(items, _rec("side", 0.7, 2, Viewpoint.SIDE), gallery_size=4)
    assert victim is None
    assert {x.viewpoint for x in out} == {Viewpoint.FRONT, Viewpoint.SIDE}


def test_upgrade_does_not_replace_another_track_or_video():
    items = [_rec("cam1", 0.8, 1, Viewpoint.FRONT)]
    other = _rec("cam2", 0.7, 8, Viewpoint.FRONT)
    other.track_id = "B_1"
    other.source_video_name = "video_02.mp4"
    out, victim = upgrade_or_insert(items, other, gallery_size=4)
    assert victim is None
    assert len(out) == 2


def test_upgrade_replaces_unknown_entrance_with_real_pose():
    items = [_rec("partial", 0.2, 1, Viewpoint.UNKNOWN)]
    out, victim = upgrade_or_insert(items, _rec("posed", 0.7, 6, Viewpoint.FRONT), gallery_size=4)
    assert victim is not None and victim.viewpoint == Viewpoint.UNKNOWN
    assert len(out) == 1
    assert out[0].viewpoint == Viewpoint.FRONT


def test_supported_similarity_needs_several_matching_crops():
    from src.identity.gallery import supported_similarity
    from tests.conftest import near, unit_vec

    x, y = unit_vec(200), unit_vec(201)
    rec = lambda v, i: EmbeddingRecord(
        global_person_id="P0001", track_id=f"A_{i}", camera_id="A", timestamp=float(i),
        embedding=v, quality_score=0.8, viewpoint=Viewpoint.FRONT, source_frame_id=i,
        source_video_id="v", source_video_name="v.avi", source_frame_number=i,
        source_timestamp_ms=i, model_name="m", model_version="", modality="body",
    )
    # One contaminated crop (looks like y) in a gallery of person x.
    gallery = [rec(near(x, seed=i), i) for i in range(5)] + [rec(near(y, seed=9), 9)]
    assert supported_similarity(near(y, seed=10), gallery, k=1) > 0.8
    assert supported_similarity(near(y, seed=10), gallery, k=3) < 0.2
    assert supported_similarity(near(x, seed=11), gallery, k=3) > 0.8
    assert supported_similarity(near(x, seed=11), gallery[:2], k=3) > 0.8  # small gallery

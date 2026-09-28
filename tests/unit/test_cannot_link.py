"""Tracks on screen together are different people (run-level cannot-link).

Regression for the P0001 case: A_1 and A_4 were visible in the same frame,
their bodies look alike (cosine ~0.71 with OSNet), and A_4 took A_1's Global
ID. Blocking only the shared frame is not enough - A_4 must also not take
that ID in later frames where A_1 is gone.
"""

from __future__ import annotations

from src.types import BBox, MatchKind
from tests.conftest import near, unit_vec

LEFT = BBox(0, 0, 50, 150)
RIGHT = BBox(200, 0, 250, 150)


def _assign(manager, track, ts, body, frame=0):
    return manager.assign(
        track_id=track,
        camera_id="A",
        timestamp=ts,
        body_embedding=body,
        face_embedding=None,
        body_quality=0.8,
        face_quality=0.0,
        source_video_id="vid",
        source_video_name="4p-c0.avi",
        source_frame_number=frame,
        force_decide=True,
    )


def test_covisible_track_cannot_take_the_other_tracks_id(manager):
    body = unit_vec(1)
    lookalike = near(body, noise=0.03, seed=7)  # would match P0001 on its own
    manager.observe_frame("A", [("A_1", LEFT)], 1.0)
    a1 = _assign(manager, "A_1", 1.0, body)
    assert a1.global_person_id == "P0001"

    manager.observe_frame("A", [("A_1", LEFT), ("A_4", RIGHT)], 2.0)
    a4 = _assign(manager, "A_4", 2.0, lookalike)
    assert a4.global_person_id not in {None, "P0001"}


def test_constraint_holds_after_the_other_track_leaves(manager):
    body = unit_vec(2)
    manager.observe_frame("A", [("A_1", LEFT)], 1.0)
    _assign(manager, "A_1", 1.0, body)
    manager.observe_frame("A", [("A_1", LEFT), ("A_4", RIGHT)], 2.0)
    first = _assign(manager, "A_4", 2.0, near(body, noise=0.03, seed=8))
    # A_1 is gone; A_4 is still provisional and keeps searching the gallery.
    for i in range(5):
        manager.observe_frame("A", [("A_4", RIGHT)], 3.0 + i)
        later = _assign(manager, "A_4", 3.0 + i, near(body, noise=0.03, seed=20 + i))
        assert later.global_person_id == first.global_person_id != "P0001"


def test_unrelated_track_can_still_match_the_id(manager):
    body = unit_vec(3)
    manager.observe_frame("A", [("A_1", LEFT), ("A_4", RIGHT)], 1.0)
    _assign(manager, "A_1", 1.0, body)
    # A_9 never shared a frame with A_1: tracker fragmentation, same person.
    manager.observe_frame("A", [("A_9", LEFT)], 5.0)
    d = _assign(manager, "A_9", 5.0, near(body, seed=9))
    assert d.global_person_id == "P0001"


def test_heavily_overlapping_boxes_are_not_linked(manager):
    body = unit_vec(4)
    manager.observe_frame("A", [("A_1", LEFT), ("A_2", BBox(2, 0, 52, 150))], 1.0)
    _assign(manager, "A_1", 1.0, body)
    d = _assign(manager, "A_2", 1.0, near(body, seed=10))
    assert d.global_person_id == "P0001"


def test_shared_id_is_split_when_both_tracks_appear_together(manager):
    body = unit_vec(5)
    manager.observe_frame("A", [("A_1", LEFT)], 1.0)
    _assign(manager, "A_1", 1.0, body)
    # A_4 matches while A_1 is briefly not visible.
    manager.observe_frame("A", [("A_4", RIGHT)], 2.0)
    a4 = _assign(manager, "A_4", 2.0, near(body, noise=0.03, seed=11))
    assert a4.global_person_id == "P0001"
    before = len(manager.galleries["P0001"].body)

    splits = manager.observe_frame("A", [("A_1", LEFT), ("A_4", RIGHT)], 3.0)
    assert [s.track_id for s in splits] == ["A_4"]
    split = splits[0]
    assert split.split_from == "P0001"
    assert split.match_kind == MatchKind.NEW
    new_gid = split.global_person_id
    assert new_gid != "P0001"
    # A_4's evidence moved with it; A_1's stayed.
    assert all(r.track_id == "A_1" for r in manager.galleries["P0001"].body)
    assert len(manager.galleries["P0001"].body) < before
    assert manager.galleries[new_gid].body
    assert all(r.global_person_id == new_gid for r in manager.galleries[new_gid].body)
    assert frozenset(("P0001", new_gid)) in manager.distinct_pairs

    # And A_4 does not drift back to P0001 afterwards.
    manager.observe_frame("A", [("A_4", RIGHT)], 4.0)
    again = _assign(manager, "A_4", 4.0, near(body, noise=0.03, seed=12))
    assert again.global_person_id == new_gid


def test_constraints_reset_between_runs(manager):
    body = unit_vec(6)
    manager.begin_run("r1")
    manager.observe_frame("A", [("A_1", LEFT), ("A_4", RIGHT)], 1.0)
    _assign(manager, "A_1", 1.0, body)
    manager.begin_run("r2")
    # Local IDs restart per run: this A_4 is someone new and may match P0001.
    d = _assign(manager, "A_4", 1.0, near(body, seed=13))
    assert d.global_person_id == "P0001"


def test_can_be_disabled(manager):
    manager.thresholds.setdefault("identity", {})["exclusive_per_frame"] = False
    body = unit_vec(7)
    manager.observe_frame("A", [("A_1", LEFT), ("A_4", RIGHT)], 1.0)
    _assign(manager, "A_1", 1.0, body)
    d = _assign(manager, "A_4", 1.0, near(body, seed=14))
    assert d.global_person_id == "P0001"


def _mix(x, y, wx, wy):
    import numpy as np

    v = wx * np.asarray(x) + wy * np.asarray(y)
    return (v / np.linalg.norm(v)).tolist()


def test_lookalike_of_a_blocked_id_does_not_fall_through_to_the_next_id(manager):
    """A_34 case: best match is an ID it cannot be; a weaker ID must not win.

    The query looks most like P0001 (held by A_1, on screen together) and
    only moderately like P0002. Taking P0002 just because P0001 is blocked
    would merge two different people; a new ID is the safe answer.
    """
    x, y = unit_vec(40), unit_vec(41)
    manager.observe_frame("A", [("A_1", LEFT)], 1.0)
    assert _assign(manager, "A_1", 1.0, x).global_person_id == "P0001"
    manager.observe_frame("A", [("A_2", LEFT)], 2.0)
    manager.begin_run("second")  # P0002 from another video: no link to A_1
    manager.observe_frame("A", [("A_2", LEFT)], 1.0)
    assert _assign(manager, "A_2", 1.0, y).global_person_id == "P0002"

    manager.begin_run("third")
    manager.observe_frame("A", [("A_1", LEFT)], 1.0)
    assert _assign(manager, "A_1", 1.0, near(x, seed=42)).global_person_id == "P0001"
    manager.observe_frame("A", [("A_1", LEFT), ("A_34", RIGHT)], 2.0)
    query = _mix(x, y, 0.85, 0.7)  # cos ~0.77 to P0001, ~0.63 to P0002
    d = _assign(manager, "A_34", 2.0, query)
    assert d.global_person_id not in {"P0001", "P0002"}


MID = BBox(20, 0, 70, 150)


def _cross(manager, fresh_1, fresh_2, partner_leaves=False):
    manager.thresholds["identity"]["exclusive_per_frame"] = False
    x, y = unit_vec(60), unit_vec(61)
    manager.observe_frame("A", [("A_1", LEFT), ("A_2", RIGHT)], 1.0)
    assert _assign(manager, "A_1", 1.0, x).global_person_id == "P0001"
    assert _assign(manager, "A_2", 1.0, y).global_person_id == "P0002"
    _establish(manager, "A_1", "A_2")
    # Boxes overlap heavily: the tracker may swap them here.
    out = manager.observe_frame("A", [("A_1", LEFT), ("A_2", MID)], 2.0)
    assert {d.track_id for d in out} == {"A_1", "A_2"}
    assert all(d.global_person_id is None for d in out)  # ID withdrawn
    n_before = len(manager.galleries["P0001"].body)
    held = _assign(manager, "A_1", 2.0, near(y, seed=64))
    assert held.global_person_id is None  # not verified while overlapping
    assert len(manager.galleries["P0001"].body) == n_before
    # Clear of each other again; fresh crops decide.
    visible = [("A_1", LEFT)] if partner_leaves else [("A_1", LEFT), ("A_2", RIGHT)]
    manager.observe_frame("A", visible, 3.0)
    d1 = _assign(manager, "A_1", 3.1, fresh_1(x, y))
    d2 = None if partner_leaves else _assign(manager, "A_2", 3.1, fresh_2(x, y))
    return d1, d2


def test_crossing_swap_is_detected_and_undone(manager):
    d1, d2 = _cross(manager, lambda x, y: near(y, seed=65), lambda x, y: near(x, seed=66))
    assert d1.global_person_id == "P0002"
    assert d2.global_person_id == "P0001"


def test_crossing_without_swap_keeps_ids(manager):
    d1, d2 = _cross(manager, lambda x, y: near(x, seed=67), lambda x, y: near(y, seed=68))
    assert d1.global_person_id == "P0001" and "re-verified" in d1.reason
    assert d2.global_person_id == "P0002"
    assert not manager._verify


def test_swap_detected_even_when_the_partner_track_leaves(manager):
    """A_13 case: the other track ended during the crossing."""
    d1, _ = _cross(manager, lambda x, y: near(y, seed=69), None, partner_leaves=True)
    assert d1.global_person_id == "P0002"


def _establish(manager, *tracks):
    # New IDs stay provisional (re-matched on every crop) for the
    # confirmation window; afterwards only the drift check can move them.
    for t in tracks:
        manager.assignments[t].provisional = False


def test_hijacked_track_moves_to_the_person_it_now_shows(manager):
    """Tracker handed A_7's box to another person: A_7 must follow by appearance."""
    manager.thresholds["identity"]["exclusive_per_frame"] = False
    x, y = unit_vec(80), unit_vec(81)
    for ts in (1.0, 1.5):
        _assign(manager, "A_1", ts, near(x, seed=int(ts * 10)))
        _assign(manager, "A_2", ts, near(y, seed=int(ts * 10) + 1))
    assert manager.assignments["A_1"].global_person_id == "P0001"
    assert manager.assignments["A_2"].global_person_id == "P0002"
    _establish(manager, "A_1", "A_2")
    # Well after its own recent crops, A_1 starts showing person y.
    out = [_assign(manager, "A_1", 10.0 + i, near(y, seed=90 + i)) for i in range(3)]
    assert [d.global_person_id for d in out[:2]] == ["P0001", "P0001"]
    assert out[2].global_person_id == "P0002"
    assert "drifted" in out[2].reason


def test_single_odd_crop_does_not_move_a_track(manager):
    x, y = unit_vec(82), unit_vec(83)
    for ts in (1.0, 1.5):
        _assign(manager, "A_1", ts, near(x, seed=int(ts * 10)))
        _assign(manager, "A_2", ts, near(y, seed=int(ts * 10) + 1))
    _establish(manager, "A_1", "A_2")
    _assign(manager, "A_1", 10.0, near(y, seed=95))
    d = _assign(manager, "A_1", 11.0, near(x, seed=96))
    assert d.global_person_id == "P0001"
    assert "A_1" not in manager._drift


def test_one_lucky_crop_does_not_join_an_existing_id(manager):
    """Brown-sweater case: a single borderline crop must not decide the match."""
    manager.thresholds["identity"]["exclusive_per_frame"] = False
    x, y = unit_vec(100), unit_vec(101)
    _assign(manager, "A_1", 1.0, x)
    # A_4 is a different person; one of its crops happens to look like x.
    _assign(manager, "A_4", 5.0, near(y, seed=102))
    _assign(manager, "A_4", 5.2, near(y, seed=103))
    d = _assign(manager, "A_4", 5.4, near(x, seed=104))
    assert d.global_person_id != "P0001"


def test_borderline_match_uses_the_id_but_does_not_widen_its_gallery(manager):
    """Snowball guard: only confident matches add crops to an ID."""
    manager.thresholds["identity"]["exclusive_per_frame"] = False
    manager.thresholds["identity"]["enroll_threshold"] = 0.95
    x = unit_vec(110)
    _assign(manager, "A_1", 1.0, x)
    founder = len(manager.galleries["P0001"].body)
    # Similar enough to be assigned (fixture threshold 0.55), not to enroll.
    for i in range(3):
        d = _assign(manager, "A_2", 5.0 + i, near(x, noise=0.03, seed=111 + i))
    assert d.global_person_id == "P0001"
    assert len(manager.galleries["P0001"].body) == founder
    assert all(r.track_id == "A_1" for r in manager.galleries["P0001"].body)


def test_confident_match_is_enrolled(manager):
    manager.thresholds["identity"]["enroll_threshold"] = 0.5
    x = unit_vec(112)
    _assign(manager, "A_1", 1.0, x)
    for i in range(3):
        _assign(manager, "A_2", 5.0 + i, near(x, noise=0.01, seed=113 + i))
    assert any(r.track_id == "A_2" for r in manager.galleries["P0001"].body)


def test_slightly_low_crops_after_crossing_keep_the_id(manager):
    """A pose change may put genuine crops just under the threshold: no new ID."""
    manager.thresholds["identity"]["exclusive_per_frame"] = False
    manager.thresholds["reid"]["body_similarity_threshold"] = 0.9
    x, y = unit_vec(120), unit_vec(121)
    manager.observe_frame("A", [("A_1", LEFT), ("A_2", RIGHT)], 1.0)
    _assign(manager, "A_1", 1.0, x)
    _assign(manager, "A_2", 1.0, y)
    _establish(manager, "A_1", "A_2")
    manager.observe_frame("A", [("A_1", LEFT), ("A_2", MID)], 2.0)
    manager.observe_frame("A", [("A_1", LEFT), ("A_2", RIGHT)], 3.0)
    # cos ~0.85 to x: under the (raised) 0.9 threshold, far above anyone else.
    for i in range(4):
        d = _assign(manager, "A_1", 3.1 + i, near(x, noise=0.03, seed=122 + i))
    assert d.global_person_id == "P0001"
    assert len(manager.galleries) == 2


def _raise_threshold(manager, thr):
    # The matcher keeps its own copy of the threshold; keep both in step.
    manager.thresholds["reid"]["body_similarity_threshold"] = thr
    manager.matcher.config.body_similarity_threshold = thr
    manager.matcher.config.accept_threshold = thr


def test_unclear_new_track_waits_instead_of_taking_a_new_id(manager):
    """Fragment of a known person: below threshold at first, clear a bit later."""
    _raise_threshold(manager, 0.9)
    manager.thresholds["identity"].update(unclear_band=0.3, unclear_wait_seconds=3.0)
    x = unit_vec(130)
    _assign(manager, "A_1", 1.0, x)
    first = _assign(manager, "A_9", 5.0, near(x, noise=0.03, seed=131))  # ~0.85: unclear
    assert first.global_person_id is None and "unclear" in first.reason
    for i in range(3):
        d = _assign(manager, "A_9", 5.5 + i * 0.2, near(x, noise=0.005, seed=132 + i))
    assert d.global_person_id == "P0001"
    assert len(manager.galleries) == 1


def test_unclear_track_gets_a_new_id_after_waiting(manager):
    _raise_threshold(manager, 0.9)
    manager.thresholds["identity"].update(unclear_band=0.3, unclear_wait_seconds=3.0)
    x = unit_vec(140)
    _assign(manager, "A_1", 1.0, x)
    _assign(manager, "A_9", 5.0, near(x, noise=0.03, seed=141))
    d = _assign(manager, "A_9", 8.5, near(x, noise=0.03, seed=142))
    assert d.global_person_id not in (None, "P0001")


def test_new_track_is_observed_before_it_gets_a_new_id(manager):
    """A_23 case: poor first crops must not create an ID at once."""
    manager.thresholds["identity"]["new_id_min_seconds"] = 1.0
    x = unit_vec(150)
    # Every new person, the first one too, is observed >= 1 s first.
    assert _assign(manager, "A_1", 0.0, x).global_person_id is None
    assert _assign(manager, "A_1", 1.1, near(x, seed=153)).global_person_id == "P0001"
    first = _assign(manager, "A_9", 5.0, unit_vec(151))  # looks like nobody yet
    assert first.global_person_id is None
    d = _assign(manager, "A_9", 5.3, near(x, seed=152))  # now clearly A_1's person
    assert d.global_person_id == "P0001"
    assert len(manager.galleries) == 1


def test_one_dip_does_not_end_the_unclear_wait(manager):
    """A_16 case: 0.49 (unclear), one dip, then clear -> old ID, no new ID."""
    _raise_threshold(manager, 0.9)
    manager.thresholds["identity"].update(unclear_band=0.2, unclear_wait_seconds=3.0)
    x = unit_vec(160)
    _assign(manager, "A_1", 0.0, x)
    assert _assign(manager, "A_9", 5.0, near(x, noise=0.03, seed=161)).global_person_id is None
    for i, v in enumerate((unit_vec(162), unit_vec(163), unit_vec(164))):  # dip
        assert _assign(manager, "A_9", 5.1 + 0.1 * i, v).global_person_id is None
    for i in range(3):
        d = _assign(manager, "A_9", 6.0 + 0.1 * i, near(x, noise=0.005, seed=165 + i))
    assert d.global_person_id == "P0001"
    assert len(manager.galleries) == 1


def test_drift_to_nobody_withdraws_the_id_instead_of_a_new_one(manager):
    x, y = unit_vec(170), unit_vec(171)
    _assign(manager, "A_1", 0.0, x)
    _assign(manager, "A_2", 0.0, y)
    _establish(manager, "A_1", "A_2")
    for i in range(3):  # three crops like nobody known
        d = _assign(manager, "A_1", 10.0 + i * 0.1, unit_vec(172 + i))
    assert d.global_person_id is None and "unverified" in d.reason
    assert len(manager.galleries) == 2
    back = _assign(manager, "A_1", 11.0, near(x, seed=180))  # its own crops again
    assert back.global_person_id == "P0001"

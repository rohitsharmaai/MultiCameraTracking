from src.identity.state import TrackAssignment, apply_hysteresis
from src.types import IdentityAssignmentState


def test_hysteresis_blocks_weak_switch():
    current = TrackAssignment(
        track_id="A_1",
        camera_id="A",
        state=IdentityAssignmentState.MATCHED,
        global_person_id="P0001",
        score=0.80,
    )
    assert not apply_hysteresis(
        current, "P0002", 0.82, 0.70, enabled=True, switch_margin=0.15
    )


def test_hysteresis_allows_large_advantage():
    current = TrackAssignment(
        track_id="A_1",
        camera_id="A",
        state=IdentityAssignmentState.MATCHED,
        global_person_id="P0001",
        score=0.60,
    )
    assert apply_hysteresis(
        current, "P0002", 0.90, 0.50, enabled=True, switch_margin=0.15
    )


def test_hysteresis_disabled():
    current = TrackAssignment(
        track_id="A_1",
        camera_id="A",
        state=IdentityAssignmentState.MATCHED,
        global_person_id="P0001",
        score=0.90,
    )
    assert apply_hysteresis(current, "P0002", 0.50, 0.1, enabled=False, switch_margin=0.15)

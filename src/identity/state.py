"""Identity assignment state machine + hysteresis (Sections 19, 35).

These states belong to a local track's identity assignment, not the person record.

UNRESOLVED -> MATCHED | NEW | UNCERTAIN
MATCHED stays MATCHED unless another ID has a large score advantage
UNCERTAIN stays UNCERTAIN unless more evidence arrives (do not force)
NEW -> MATCHED to the ID just created
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.types import IdentityAssignmentState


@dataclass
class TrackAssignment:
    track_id: str
    camera_id: str
    state: IdentityAssignmentState = IdentityAssignmentState.UNRESOLVED
    global_person_id: Optional[str] = None
    score: float = 0.0
    second_best: float = 0.0
    frames_in_state: int = 0
    history: list[str] = field(default_factory=list)
    source_video_id: str | None = None
    source_video_name: str | None = None
    provisional: bool = False
    # Order in which this track took its current Global ID. When two tracks on
    # screen together hold the same ID, the one that took it first keeps it.
    bound_seq: int = 0


def apply_hysteresis(
    current: TrackAssignment,
    proposed_id: Optional[str],
    proposed_score: float,
    second_best: float,
    *,
    enabled: bool,
    switch_margin: float,
) -> bool:
    """Return True if an established MATCHED assignment may switch IDs."""
    if not enabled:
        return True
    if current.state != IdentityAssignmentState.MATCHED:
        return True
    if current.global_person_id is None:
        return True
    if proposed_id == current.global_person_id:
        return False
    return (proposed_score - current.score) >= switch_margin and (
        proposed_score - second_best
    ) >= switch_margin

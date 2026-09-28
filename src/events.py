"""Internal event model (Section 25). No embeddings, no raw frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Event:
    name: str
    timestamp: str
    camera_id: Optional[str] = None
    track_id: Optional[str] = None
    global_person_id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[Callable[[Event], None]] = []
        self.history: list[Event] = []

    def subscribe(self, fn: Callable[[Event], None]) -> None:
        self._subscribers.append(fn)

    def publish(self, name: str, **kwargs: Any) -> Event:
        event = Event(
            name=name,
            timestamp=kwargs.pop("timestamp", None) or _now(),
            camera_id=kwargs.pop("camera_id", None),
            track_id=kwargs.pop("track_id", None),
            global_person_id=kwargs.pop("global_person_id", None),
            payload=kwargs,
        )
        self.history.append(event)
        for fn in self._subscribers:
            fn(event)
        return event


DETECTION_CREATED = "DetectionCreated"
TRACK_STARTED = "TrackStarted"
TRACK_CONFIRMED = "TrackConfirmed"
TRACK_LOST = "TrackLost"
TRACK_TERMINATED = "TrackTerminated"
OBSERVATION_ACCEPTED = "ObservationAccepted"
BODY_EMBEDDING_CREATED = "BodyEmbeddingCreated"
FACE_EMBEDDING_CREATED = "FaceEmbeddingCreated"
IDENTITY_MATCHED = "IdentityMatched"
IDENTITY_CREATED = "IdentityCreated"
IDENTITY_UNRESOLVED = "IdentityUnresolved"
CAMERA_TRANSITION = "CameraTransitionDetected"

"""BoT-SORT local tracker (tracker-only).

MIT algorithm. Appearance / FastReID is OFF. Global identity is NOT owned here.
Each camera has its own namespace: A_1, A_2, ... never assume A_17 == B_17.

Lifecycle: NEW -> TENTATIVE -> CONFIRMED -> LOST -> TERMINATED
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.events import (
    TRACK_CONFIRMED,
    TRACK_LOST,
    TRACK_STARTED,
    TRACK_TERMINATED,
    EventBus,
)
from src.tracking.association import associate
from src.tracking.kalman import KalmanBoxTracker
from src.types import BBox, Detection, LocalTrack, TrackState


@dataclass
class _InternalTrack:
    local_id: int
    camera_id: str
    kalman: KalmanBoxTracker
    bbox: BBox
    confidence: float
    state: TrackState
    hits: int = 0
    age: int = 0
    time_since_update: int = 0
    frame_id: int = 0
    timestamp: float = 0.0
    start_time: float = 0.0
    first_bbox: Optional[BBox] = None
    last_bbox: Optional[BBox] = None

    @property
    def track_id(self) -> str:
        return f"{self.camera_id}_{self.local_id}"

    def to_public(self) -> LocalTrack:
        return LocalTrack(
            track_id=self.track_id,
            camera_id=self.camera_id,
            state=self.state,
            bbox=self.bbox,
            confidence=self.confidence,
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            hits=self.hits,
            age=self.age,
            time_since_update=self.time_since_update,
            first_bbox=self.first_bbox,
            last_bbox=self.last_bbox,
            start_time=self.start_time,
            end_time=self.timestamp if self.state == TrackState.TERMINATED else None,
            velocity=self.kalman.velocity_xy(),
        )


@dataclass
class BoTSORTTracker:
    camera_id: str
    min_confirmed_frames: int = 3
    max_lost_frames: int = 30
    iou_threshold: float = 0.3
    high_score_threshold: float = 0.6
    low_score_threshold: float = 0.1
    events: Optional[EventBus] = None
    _next_id: int = 1
    _tracks: list[_InternalTrack] = field(default_factory=list)

    def update(self, detections: list[Detection], frame_id: int, timestamp: float) -> list[LocalTrack]:
        self._tracks = [t for t in self._tracks if t.state != TrackState.TERMINATED]
        for det in detections:
            if det.camera_id != self.camera_id:
                raise ValueError(
                    f"Tracker {self.camera_id} received detection from {det.camera_id}"
                )

        for track in self._tracks:
            track.bbox = track.kalman.predict()
            track.age += 1
            track.time_since_update += 1

        high = [d for d in detections if d.confidence >= self.high_score_threshold]
        low = [
            d
            for d in detections
            if self.low_score_threshold <= d.confidence < self.high_score_threshold
        ]

        active = [
            t
            for t in self._tracks
            if t.state in {TrackState.NEW, TrackState.TENTATIVE, TrackState.CONFIRMED, TrackState.LOST}
        ]
        used_high = self._associate_and_update(active, high, frame_id, timestamp)

        unmatched_tracks = [
            t for t in active if t.time_since_update > 0 and t.state != TrackState.TERMINATED
        ]
        self._associate_and_update(unmatched_tracks, low, frame_id, timestamp)

        for i, det in enumerate(high):
            if i not in used_high:
                self._start_track(det, frame_id, timestamp)

        self._advance_lifecycle(timestamp)
        return [t.to_public() for t in self._tracks if t.state != TrackState.TERMINATED]

    def _associate_and_update(
        self,
        tracks: list[_InternalTrack],
        detections: list[Detection],
        frame_id: int,
        timestamp: float,
    ) -> set[int]:
        used: set[int] = set()
        if not tracks or not detections:
            return used
        matches, _, _ = associate(
            [t.bbox for t in tracks],
            [d.bbox for d in detections],
            self.iou_threshold,
        )
        for ti, di in matches:
            track = tracks[ti]
            det = detections[di]
            track.bbox = track.kalman.update(det.bbox)
            track.confidence = det.confidence
            track.hits += 1
            track.time_since_update = 0
            track.frame_id = frame_id
            track.timestamp = timestamp
            track.last_bbox = det.bbox
            used.add(di)
            if track.state == TrackState.LOST:
                track.state = TrackState.CONFIRMED if track.hits >= self.min_confirmed_frames else TrackState.TENTATIVE
            elif track.state == TrackState.NEW:
                track.state = TrackState.TENTATIVE
            if (
                track.state == TrackState.TENTATIVE
                and track.hits >= self.min_confirmed_frames
            ):
                track.state = TrackState.CONFIRMED
                self._emit(TRACK_CONFIRMED, track)
        return used

    def _start_track(self, det: Detection, frame_id: int, timestamp: float) -> None:
        track = _InternalTrack(
            local_id=self._next_id,
            camera_id=self.camera_id,
            kalman=KalmanBoxTracker(det.bbox),
            bbox=det.bbox,
            confidence=det.confidence,
            state=TrackState.NEW,
            hits=1,
            age=1,
            time_since_update=0,
            frame_id=frame_id,
            timestamp=timestamp,
            start_time=timestamp,
            first_bbox=det.bbox,
            last_bbox=det.bbox,
        )
        self._next_id += 1
        self._tracks.append(track)
        self._emit(TRACK_STARTED, track)

    def _advance_lifecycle(self, timestamp: float) -> None:
        for track in self._tracks:
            if track.state == TrackState.TERMINATED:
                continue
            if track.time_since_update == 1 and track.state in {
                TrackState.CONFIRMED,
                TrackState.TENTATIVE,
            }:
                track.state = TrackState.LOST
                self._emit(TRACK_LOST, track)
            if track.time_since_update > self.max_lost_frames:
                track.state = TrackState.TERMINATED
                track.timestamp = timestamp
                self._emit(TRACK_TERMINATED, track)
            elif track.state == TrackState.NEW and track.hits >= 1 and track.age >= 1:
                track.state = TrackState.TENTATIVE

    def _emit(self, name: str, track: _InternalTrack) -> None:
        if self.events is None:
            return
        self.events.publish(
            name,
            camera_id=track.camera_id,
            track_id=track.track_id,
            state=track.state.value,
        )

    def confirmed(self) -> list[LocalTrack]:
        return [
            t.to_public()
            for t in self._tracks
            if t.state == TrackState.CONFIRMED
        ]

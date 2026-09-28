"""Per-camera isolated pipeline. Must not mutate another camera's tracker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.events import DETECTION_CREATED, EventBus
from src.tracking.botsort import BoTSORTTracker
from src.types import Detection, LocalTrack


@dataclass
class CameraWorker:
    camera_id: str
    detector: object
    tracker: BoTSORTTracker
    events: Optional[EventBus] = None

    def process_frame(self, frame, frame_id: int, timestamp: float) -> tuple[list[Detection], list[LocalTrack]]:
        detections = self.detector.detect(
            frame,
            camera_id=self.camera_id,
            frame_id=frame_id,
            timestamp=timestamp,
        )
        if self.events:
            for det in detections:
                self.events.publish(
                    DETECTION_CREATED,
                    camera_id=self.camera_id,
                    frame_id=frame_id,
                    confidence=det.confidence,
                )
        tracks = self.tracker.update(detections, frame_id, timestamp)
        return detections, tracks

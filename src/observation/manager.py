"""Collect and select high-quality observations per local track.

Do not generate embeddings from every frame. Best body and best face
observations do not have to be the same frame.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.events import OBSERVATION_ACCEPTED, EventBus
from src.observation.crop import extract_crop
from src.observation.quality import body_quality_score, sharpness
from src.types import LocalTrack, Observation, TrackState, Viewpoint


@dataclass
class BufferedObservation:
    observation: Observation
    crop: np.ndarray
    mask: Optional[np.ndarray] = None


@dataclass
class ObservationManager:
    top_k_body: int = 8
    top_k_face: int = 3
    min_body_quality: float = 0.35
    min_face_quality: float = 0.45
    min_bbox_area: float = 1600.0
    min_sharpness: float = 20.0
    quality_weights: dict[str, Any] = field(default_factory=dict)
    events: Optional[EventBus] = None
    max_buffer: int = 48
    _buffers: dict[str, list[BufferedObservation]] = field(default_factory=lambda: defaultdict(list))

    def consider(
        self,
        frame: np.ndarray,
        track: LocalTrack,
        *,
        detector_confidence: float,
        source_video_id: Optional[str] = None,
        source_video_name: Optional[str] = None,
        mask_coverage: float = 1.0,
        occlusion_score: float = 0.0,
        pose_score: float = 0.0,
        body_visibility_score: float = 1.0,
        face_quality: float = 0.0,
        viewpoint: Viewpoint = Viewpoint.UNKNOWN,
        crop: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
    ) -> Optional[Observation]:
        if track.state != TrackState.CONFIRMED:
            return None
        if crop is None:
            crop, _ = extract_crop(frame, track.bbox)
        if crop.size == 0:
            return None
        sharp = sharpness(crop)
        if sharp < self.min_sharpness or track.bbox.area() < self.min_bbox_area:
            return None
        obs = Observation(
            frame_id=track.frame_id,
            camera_id=track.camera_id,
            timestamp=track.timestamp,
            bbox=track.bbox,
            track_id=track.track_id,
            sharpness=sharp,
            bbox_area=track.bbox.area(),
            bbox_aspect_ratio=track.bbox.aspect_ratio(),
            occlusion_score=occlusion_score,
            pose_score=pose_score,
            body_visibility_score=body_visibility_score,
            detector_confidence=detector_confidence,
            face_quality=face_quality,
            viewpoint=viewpoint,
            mask_coverage=mask_coverage,
            source_video_id=source_video_id,
            source_video_name=source_video_name,
            source_frame_number=track.frame_id,
            source_timestamp_ms=int(track.timestamp * 1000),
        )
        obs.body_quality = body_quality_score(obs, self.quality_weights)
        if obs.body_quality < self.min_body_quality and obs.face_quality < self.min_face_quality:
            return None
        buf = self._buffers[track.track_id]
        buf.append(BufferedObservation(observation=obs, crop=crop, mask=mask))
        if len(buf) > self.max_buffer:
            buf.sort(key=lambda b: b.observation.body_quality, reverse=True)
            del buf[self.max_buffer :]
        if self.events:
            self.events.publish(
                OBSERVATION_ACCEPTED,
                camera_id=track.camera_id,
                track_id=track.track_id,
                frame_id=track.frame_id,
                body_quality=obs.body_quality,
            )
        return obs

    def select_body(self, track_id: str) -> list[BufferedObservation]:
        return self._select(track_id, key=lambda b: b.observation.body_quality, k=self.top_k_body)

    def select_face(self, track_id: str) -> list[BufferedObservation]:
        items = [
            b
            for b in self._buffers.get(track_id, [])
            if b.observation.face_quality >= self.min_face_quality
        ]
        items.sort(key=lambda b: b.observation.face_quality, reverse=True)
        return self._diverse(items, self.top_k_face)

    def _select(
        self,
        track_id: str,
        key,
        k: int,
    ) -> list[BufferedObservation]:
        items = [b for b in self._buffers.get(track_id, []) if key(b) >= self.min_body_quality]
        items.sort(key=key, reverse=True)
        return self._diverse(items, k)

    def _diverse(self, items: list[BufferedObservation], k: int) -> list[BufferedObservation]:
        """Keep Top-K while preferring viewpoint diversity."""
        if len(items) <= k:
            return items
        chosen: list[BufferedObservation] = []
        seen: set[Viewpoint] = set()
        for item in items:
            vp = item.observation.viewpoint
            if vp not in seen or vp == Viewpoint.UNKNOWN:
                chosen.append(item)
                seen.add(vp)
            if len(chosen) >= k:
                return chosen
        for item in items:
            if item not in chosen:
                chosen.append(item)
            if len(chosen) >= k:
                break
        return chosen

    def clear(self) -> None:
        """Drop all run-local observation buffers."""
        self._buffers.clear()

    def clear_track(self, track_id: str) -> None:
        self._buffers.pop(track_id, None)

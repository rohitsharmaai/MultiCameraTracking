"""Shared dataclasses. Local Track IDs are never Global Person IDs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TrackState(str, Enum):
    NEW = "NEW"
    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    LOST = "LOST"
    TERMINATED = "TERMINATED"


class IdentityAssignmentState(str, Enum):
    """Assignment of a *local track*, not the person record lifecycle."""

    UNRESOLVED = "UNRESOLVED"
    MATCHED = "MATCHED"
    NEW = "NEW"
    UNCERTAIN = "UNCERTAIN"
    CONFLICT = "CONFLICT"


class PersonStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"


class MatchKind(str, Enum):
    EXISTING = "existing"
    NEW = "new"
    UNRESOLVED = "unresolved"
    CONFLICT = "conflict"


class Viewpoint(str, Enum):
    FRONT = "front"
    SIDE = "side"
    BACK = "back"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def area(self) -> float:
        return self.width() * self.height()

    def aspect_ratio(self) -> float:
        h = self.height()
        return self.width() / h if h > 0 else 0.0

    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def clip(self, width: int, height: int) -> "BBox":
        return BBox(
            x1=max(0.0, min(float(width - 1), self.x1)),
            y1=max(0.0, min(float(height - 1), self.y1)),
            x2=max(0.0, min(float(width), self.x2)),
            y2=max(0.0, min(float(height), self.y2)),
        )

    def iou(self, other: "BBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area() + other.area() - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Detection:
    bbox: BBox
    confidence: float
    class_id: int
    frame_id: int
    timestamp: float
    camera_id: str


@dataclass
class LocalTrack:
    track_id: str
    camera_id: str
    state: TrackState
    bbox: BBox
    confidence: float
    frame_id: int
    timestamp: float
    hits: int = 0
    age: int = 0
    time_since_update: int = 0
    first_bbox: Optional[BBox] = None
    last_bbox: Optional[BBox] = None
    start_time: float = 0.0
    end_time: Optional[float] = None
    velocity: tuple[float, float] = (0.0, 0.0)


@dataclass
class Observation:
    frame_id: int
    camera_id: str
    timestamp: float
    bbox: BBox
    track_id: str
    sharpness: float = 0.0
    bbox_area: float = 0.0
    bbox_aspect_ratio: float = 0.0
    occlusion_score: float = 0.0
    pose_score: float = 0.0
    body_visibility_score: float = 0.0
    detector_confidence: float = 0.0
    face_quality: float = 0.0
    viewpoint: Viewpoint = Viewpoint.UNKNOWN
    body_quality: float = 0.0
    mask_coverage: float = 0.0
    source_video_id: Optional[str] = None
    source_video_name: Optional[str] = None
    source_frame_number: Optional[int] = None
    source_timestamp_ms: Optional[int] = None


@dataclass
class EmbeddingRecord:
    global_person_id: str
    track_id: str
    camera_id: str
    timestamp: float
    embedding: list[float]
    quality_score: float
    viewpoint: Viewpoint
    source_frame_id: int
    source_video_id: Optional[str]
    source_video_name: Optional[str]
    source_frame_number: Optional[int]
    source_timestamp_ms: Optional[int]
    model_name: str
    model_version: str
    modality: str  # body | face


@dataclass
class IdentityDecision:
    track_id: str
    camera_id: str
    state: IdentityAssignmentState
    global_person_id: Optional[str]
    match_kind: MatchKind
    score: float
    second_best_score: float
    margin: float
    body_similarity: float = 0.0
    face_similarity: float = 0.0
    temporal_plausibility: float = 0.0
    camera_plausibility: float = 0.0
    motion_plausibility: float = 0.0
    reason: str = ""
    candidate_count: int = 0
    # Transient: set when this decision abandoned a provisional Global ID and
    # rebound the track to `global_person_id`. Used to keep the evidence that
    # caused the reassignment with the old identity.
    archived_gid: Optional[str] = None
    # Transient: set when _maybe_snapshot wrote a better body/face crop this
    # frame so the operator GUI can refresh its reference images.
    snapshot_updated: bool = False
    # Transient: set when this track was split off ``split_from`` because it
    # was on screen together with another track holding that ID.
    split_from: Optional[str] = None


@dataclass
class OverlayInfo:
    global_id: Optional[str]
    local_track: str
    source: str
    match: MatchKind
    confidence: float


@dataclass
class VideoRunStats:
    run_id: str
    source_video_name: str
    source_video_path_or_uri: str
    video_sha256: str
    started_at: str
    completed_at: Optional[str] = None
    frames_processed: int = 0
    tracks_detected: int = 0
    global_ids_seen: list[str] = field(default_factory=list)
    new_global_ids_created: int = 0
    existing_global_ids_reused: int = 0
    camera_files: dict[str, str] = field(default_factory=dict)
    output_videos: dict[str, str] = field(default_factory=dict)

"""SQLAlchemy models matching Section 20."""

from __future__ import annotations

from datetime import datetime, timezone


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class CameraRow(Base):
    __tablename__ = "cameras"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), default="")
    source_type: Mapped[str] = mapped_column(String(32), default="video_file")
    source_uri: Mapped[str] = mapped_column(Text, default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    declared_for_video_testing: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class PersonRow(Base):
    __tablename__ = "persons"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    global_person_id: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    last_seen_at: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(16), default="active")
    first_camera_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    last_camera_id: Mapped[str | None] = mapped_column(String(64), nullable=True)


class CameraTrackRow(Base):
    __tablename__ = "camera_tracks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    track_id: Mapped[str] = mapped_column(String(64), nullable=False)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False)
    global_person_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    start_time: Mapped[float] = mapped_column(Float, default=0.0)
    end_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    first_bbox: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_bbox: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(32), default="")
    source_video_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_video_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    # Local track IDs (A_1, A_2, ...) restart for every video, so a track row
    # is scoped to its source video as well.
    __table_args__ = (
        UniqueConstraint(
            "track_id", "camera_id", "source_video_id", name="uq_track_cam_video"
        ),
    )


class EmbeddingRow(Base):
    __tablename__ = "person_embeddings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    global_person_id: Mapped[str] = mapped_column(String(32), index=True)
    track_id: Mapped[str] = mapped_column(String(64))
    camera_id: Mapped[str] = mapped_column(String(64))
    timestamp: Mapped[float] = mapped_column(Float)
    embedding: Mapped[str] = mapped_column(Text)  # JSON; pgvector column added when available
    quality_score: Mapped[float] = mapped_column(Float)
    viewpoint: Mapped[str] = mapped_column(String(16))
    source_frame_id: Mapped[int] = mapped_column(Integer, default=0)
    source_video_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_video_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    source_frame_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_timestamp_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(64), default="")


class FaceEmbeddingRow(Base):
    __tablename__ = "face_embeddings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    global_person_id: Mapped[str] = mapped_column(String(32), index=True)
    track_id: Mapped[str] = mapped_column(String(64))
    camera_id: Mapped[str] = mapped_column(String(64))
    timestamp: Mapped[float] = mapped_column(Float)
    embedding: Mapped[str] = mapped_column(Text)
    quality_score: Mapped[float] = mapped_column(Float)
    viewpoint: Mapped[str] = mapped_column(String(16))
    source_frame_id: Mapped[int] = mapped_column(Integer, default=0)
    source_video_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_video_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    source_frame_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_timestamp_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_name: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(64), default="")


class VideoRunRow(Base):
    __tablename__ = "video_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True)
    source_video_name: Mapped[str] = mapped_column(Text)
    source_video_path_or_uri: Mapped[str] = mapped_column(Text)
    video_sha256: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    frames_processed: Mapped[int] = mapped_column(Integer, default=0)
    tracks_detected: Mapped[int] = mapped_column(Integer, default=0)
    global_ids_seen: Mapped[str] = mapped_column(Text, default="[]")
    new_global_ids_created: Mapped[int] = mapped_column(Integer, default=0)
    existing_global_ids_reused: Mapped[int] = mapped_column(Integer, default=0)
    camera_files: Mapped[str] = mapped_column(Text, default="{}")


class SourceVideoRow(Base):
    __tablename__ = "source_videos"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_video_id: Mapped[str] = mapped_column(String(64), unique=True)
    source_video_name: Mapped[str] = mapped_column(String(256))
    source_video_path_or_uri: Mapped[str] = mapped_column(Text)
    video_sha256: Mapped[str] = mapped_column(String(64), default="")
    first_processed_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    last_processed_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class TransitionRow(Base):
    __tablename__ = "transitions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    global_person_id: Mapped[str] = mapped_column(String(32), index=True)
    from_camera_id: Mapped[str] = mapped_column(String(64))
    to_camera_id: Mapped[str] = mapped_column(String(64))
    from_time: Mapped[float] = mapped_column(Float)
    to_time: Mapped[float] = mapped_column(Float)
    match_score: Mapped[float] = mapped_column(Float)
    match_method: Mapped[str] = mapped_column(String(64))
    confidence: Mapped[float] = mapped_column(Float)


class ModelRegistryRow(Base):
    __tablename__ = "model_registry"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(64))
    artifact: Mapped[str] = mapped_column(String(256))
    version: Mapped[str] = mapped_column(String(64))
    sha256: Mapped[str] = mapped_column(String(64), default="")
    license: Mapped[str] = mapped_column(String(64))
    commercial_use_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    verification_date: Mapped[str] = mapped_column(String(32), default="")
    status: Mapped[str] = mapped_column(String(16))


class AccessLogRow(Base):
    __tablename__ = "access_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    actor: Mapped[str] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64))
    resource: Mapped[str] = mapped_column(String(128))
    detail: Mapped[str] = mapped_column(Text, default="")

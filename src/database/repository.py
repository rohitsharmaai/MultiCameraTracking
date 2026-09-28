"""Persistent identity store. PostgreSQL/pgvector is the source of truth.

A process restart must reconstruct the gallery from this store.
SQLite is allowed only for development/tests (JSON embeddings).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import delete, select, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.database.models import (
    CameraTrackRow,
    EmbeddingRow,
    FaceEmbeddingRow,
    PersonRow,
    SourceVideoRow,
    TransitionRow,
    VideoRunRow,
)
from src.identity.gallery import PersonGallery
from src.types import EmbeddingRecord, VideoRunStats, Viewpoint


def _viewpoint(value: str) -> Viewpoint:
    try:
        return Viewpoint(value)
    except ValueError:
        return Viewpoint.UNKNOWN


def _emb_to_json(vec: list[float]) -> str:
    return json.dumps([float(x) for x in vec])


def _json_to_emb(raw: str) -> list[float]:
    return [float(x) for x in json.loads(raw)]


def _row_to_record(row, modality: str) -> EmbeddingRecord:
    return EmbeddingRecord(
        global_person_id=row.global_person_id,
        track_id=row.track_id,
        camera_id=row.camera_id,
        timestamp=row.timestamp,
        embedding=_json_to_emb(row.embedding),
        quality_score=row.quality_score,
        viewpoint=_viewpoint(row.viewpoint),
        source_frame_id=row.source_frame_id,
        source_video_id=row.source_video_id,
        source_video_name=row.source_video_name,
        source_frame_number=row.source_frame_number,
        source_timestamp_ms=row.source_timestamp_ms,
        model_name=row.model_name,
        model_version=row.model_version,
        modality=modality,
    )


class IdentityRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    def _session(self) -> Session:
        return Session(self.engine, expire_on_commit=False)

    def load_galleries(self) -> dict[str, PersonGallery]:
        with self._session() as s:
            persons = s.scalars(select(PersonRow)).all()
            bodies = s.scalars(select(EmbeddingRow)).all()
            faces = s.scalars(select(FaceEmbeddingRow)).all()
        galleries: dict[str, PersonGallery] = {}
        for p in persons:
            galleries[p.global_person_id] = PersonGallery(
                global_person_id=p.global_person_id,
                last_camera_id=p.last_camera_id,
                last_seen_at=p.last_seen_at,
                first_camera_id=p.first_camera_id,
                status=p.status,
            )
        for row in bodies:
            gal = galleries.setdefault(row.global_person_id, PersonGallery(row.global_person_id))
            gal.body.append(_row_to_record(row, "body"))
        for row in faces:
            gal = galleries.setdefault(row.global_person_id, PersonGallery(row.global_person_id))
            gal.face.append(_row_to_record(row, "face"))
        return galleries

    def next_person_id(self, prefix: str, width: int) -> str:
        with self._session() as s:
            ids = s.scalars(select(PersonRow.global_person_id)).all()
        n = 0
        for gid in ids:
            if gid.startswith(prefix):
                try:
                    n = max(n, int(gid[len(prefix) :]))
                except ValueError:
                    continue
        return f"{prefix}{n + 1:0{width}d}"

    def upsert_person(self, gallery: PersonGallery) -> None:
        with self._session() as s:
            row = s.scalar(
                select(PersonRow).where(PersonRow.global_person_id == gallery.global_person_id)
            )
            if row is None:
                row = PersonRow(global_person_id=gallery.global_person_id)
                s.add(row)
            row.last_seen_at = gallery.last_seen_at
            row.status = gallery.status
            row.first_camera_id = gallery.first_camera_id
            row.last_camera_id = gallery.last_camera_id
            s.commit()

    def add_embedding(self, rec: EmbeddingRecord) -> None:
        table = EmbeddingRow if rec.modality == "body" else FaceEmbeddingRow
        with self._session() as s:
            s.add(
                table(
                    global_person_id=rec.global_person_id,
                    track_id=rec.track_id,
                    camera_id=rec.camera_id,
                    timestamp=rec.timestamp,
                    embedding=_emb_to_json(rec.embedding),
                    quality_score=rec.quality_score,
                    viewpoint=rec.viewpoint.value,
                    source_frame_id=rec.source_frame_id,
                    source_video_id=rec.source_video_id,
                    source_video_name=rec.source_video_name,
                    source_frame_number=rec.source_frame_number,
                    source_timestamp_ms=rec.source_timestamp_ms,
                    model_name=rec.model_name,
                    model_version=rec.model_version,
                )
            )
            s.commit()

    def remove_embedding(self, rec: EmbeddingRecord) -> None:
        table = EmbeddingRow if rec.modality == "body" else FaceEmbeddingRow
        with self._session() as s:
            rows = s.scalars(
                select(table).where(
                    table.global_person_id == rec.global_person_id,
                    table.track_id == rec.track_id,
                    table.timestamp == rec.timestamp,
                    table.viewpoint == rec.viewpoint.value,
                    table.quality_score == rec.quality_score,
                )
            ).all()
            for row in rows[:1]:
                s.delete(row)
            s.commit()

    def add_track(
        self,
        track_id: str,
        camera_id: str,
        global_person_id: Optional[str],
        start_time: float,
        end_time: Optional[float],
        status: str,
        source_video_id: Optional[str],
        source_video_name: Optional[str],
        confidence: float,
    ) -> None:
        # Local track IDs restart per video: scope the row to its source video
        # so a later video's A_1 does not overwrite this one's history.
        same_video = (
            CameraTrackRow.source_video_id == source_video_id
            if source_video_id is not None
            else CameraTrackRow.source_video_id.is_(None)
        )
        for scoped in (True, False):
            with self._session() as s:
                conditions = [
                    CameraTrackRow.track_id == track_id,
                    CameraTrackRow.camera_id == camera_id,
                ]
                if scoped:
                    conditions.append(same_video)
                row = s.scalar(select(CameraTrackRow).where(*conditions))
                if row is None:
                    row = CameraTrackRow(
                        track_id=track_id, camera_id=camera_id, start_time=start_time
                    )
                    s.add(row)
                row.global_person_id = global_person_id
                row.end_time = end_time
                row.status = status
                row.source_video_id = source_video_id
                row.source_video_name = source_video_name
                row.confidence = confidence
                try:
                    s.commit()
                    return
                except IntegrityError:
                    # A development SQLite file created before the per-video
                    # key still has UNIQUE(track_id, camera_id); fall back to
                    # updating that row (the old behaviour).
                    s.rollback()
                    if not scoped:
                        raise

    def add_transition(
        self,
        global_person_id: str,
        from_camera_id: str,
        to_camera_id: str,
        from_time: float,
        to_time: float,
        match_score: float,
        match_method: str,
        confidence: float,
    ) -> None:
        with self._session() as s:
            s.add(
                TransitionRow(
                    global_person_id=global_person_id,
                    from_camera_id=from_camera_id,
                    to_camera_id=to_camera_id,
                    from_time=from_time,
                    to_time=to_time,
                    match_score=match_score,
                    match_method=match_method,
                    confidence=confidence,
                )
            )
            s.commit()

    def reset_gallery(self) -> None:
        with self._session() as s:
            for model in (
                EmbeddingRow,
                FaceEmbeddingRow,
                CameraTrackRow,
                TransitionRow,
                PersonRow,
                VideoRunRow,
                SourceVideoRow,
            ):
                s.execute(delete(model))
            s.commit()

    def upsert_source_video(self, name: str, path: str, digest: str) -> str:
        vid = digest[:16] if digest else uuid.uuid4().hex[:16]
        with self._session() as s:
            row = s.scalar(select(SourceVideoRow).where(SourceVideoRow.video_sha256 == digest)) if digest else None
            if row is None:
                row = s.scalar(select(SourceVideoRow).where(SourceVideoRow.source_video_name == name))
            if row is None:
                row = SourceVideoRow(
                    source_video_id=vid,
                    source_video_name=name,
                    source_video_path_or_uri=path,
                    video_sha256=digest,
                )
                s.add(row)
            else:
                row.last_processed_at = datetime.now(timezone.utc)
                row.source_video_path_or_uri = path
                vid = row.source_video_id
            s.commit()
        return vid

    def record_run(self, stats: VideoRunStats) -> None:
        with self._session() as s:
            row = s.scalar(select(VideoRunRow).where(VideoRunRow.run_id == stats.run_id))
            if row is None:
                row = VideoRunRow(run_id=stats.run_id)
                s.add(row)
            row.source_video_name = stats.source_video_name
            row.source_video_path_or_uri = stats.source_video_path_or_uri
            row.video_sha256 = stats.video_sha256
            row.completed_at = datetime.fromisoformat(stats.completed_at) if stats.completed_at else None
            row.frames_processed = stats.frames_processed
            row.tracks_detected = stats.tracks_detected
            row.global_ids_seen = json.dumps(stats.global_ids_seen)
            row.new_global_ids_created = stats.new_global_ids_created
            row.existing_global_ids_reused = stats.existing_global_ids_reused
            row.camera_files = json.dumps(stats.camera_files)
            s.commit()

    def list_tracks(self, camera_id: Optional[str] = None) -> list[CameraTrackRow]:
        with self._session() as s:
            stmt = select(CameraTrackRow)
            if camera_id:
                stmt = stmt.where(CameraTrackRow.camera_id == camera_id)
            return list(s.scalars(stmt).all())

    def list_transitions(self) -> list[TransitionRow]:
        with self._session() as s:
            return list(s.scalars(select(TransitionRow)).all())

    def person_timeline(self, gid: str) -> list[CameraTrackRow]:
        with self._session() as s:
            return list(
                s.scalars(
                    select(CameraTrackRow).where(CameraTrackRow.global_person_id == gid)
                ).all()
            )

    def delete_person(self, gid: str) -> None:
        """Retention / deletion workflow."""
        with self._session() as s:
            s.execute(delete(EmbeddingRow).where(EmbeddingRow.global_person_id == gid))
            s.execute(delete(FaceEmbeddingRow).where(FaceEmbeddingRow.global_person_id == gid))
            s.execute(delete(CameraTrackRow).where(CameraTrackRow.global_person_id == gid))
            s.execute(delete(TransitionRow).where(TransitionRow.global_person_id == gid))
            s.execute(delete(PersonRow).where(PersonRow.global_person_id == gid))
            s.commit()

    def merge_persons(self, source: str, target: str) -> None:
        """Move every row of ``source`` onto ``target`` and drop the source.

        Explicit user action only (GUI merge button). Nothing calls this
        automatically.
        """
        if source == target:
            return
        with self._session() as s:
            for table in (EmbeddingRow, FaceEmbeddingRow, CameraTrackRow, TransitionRow):
                s.execute(
                    update(table)
                    .where(table.global_person_id == source)
                    .values(global_person_id=target)
                )
            target_row = s.scalar(
                select(PersonRow).where(PersonRow.global_person_id == target)
            )
            src_row = s.scalar(select(PersonRow).where(PersonRow.global_person_id == source))
            if target_row is not None and src_row is not None:
                target_row.last_seen_at = max(
                    target_row.last_seen_at or 0.0, src_row.last_seen_at or 0.0
                )
            s.execute(delete(PersonRow).where(PersonRow.global_person_id == source))
            s.commit()


class InMemoryStore:
    """Test double. Production must use IdentityRepository."""

    def __init__(self) -> None:
        self.galleries: dict[str, PersonGallery] = {}
        self.tracks: list[dict] = []
        self.transitions: list[dict] = []
        self._next = 1

    def load_galleries(self) -> dict[str, PersonGallery]:
        return self.galleries

    def next_person_id(self, prefix: str, width: int) -> str:
        n = self._next - 1
        for gid in self.galleries:
            if gid.startswith(prefix):
                try:
                    n = max(n, int(gid[len(prefix) :]))
                except ValueError:
                    continue
        self._next = n + 2
        return f"{prefix}{n + 1:0{width}d}"

    def upsert_person(self, gallery: PersonGallery) -> None:
        self.galleries[gallery.global_person_id] = gallery

    def add_embedding(self, rec: EmbeddingRecord) -> None:
        gal = self.galleries.setdefault(rec.global_person_id, PersonGallery(rec.global_person_id))
        target = gal.body if rec.modality == "body" else gal.face
        if rec not in target:
            target.append(rec)

    def remove_embedding(self, rec: EmbeddingRecord) -> None:
        gal = self.galleries.get(rec.global_person_id)
        if gal is None:
            return
        target = gal.body if rec.modality == "body" else gal.face
        if rec in target:
            target.remove(rec)
            return
        for i, it in enumerate(target):
            if (
                it.timestamp == rec.timestamp
                and it.quality_score == rec.quality_score
                and it.viewpoint == rec.viewpoint
                and it.track_id == rec.track_id
            ):
                target.pop(i)
                return

    def add_track(self, **kwargs) -> None:
        self.tracks.append(kwargs)

    def add_transition(self, **kwargs) -> None:
        self.transitions.append(kwargs)

    def delete_person(self, global_person_id: str) -> None:
        self.galleries.pop(global_person_id, None)
        self.tracks[:] = [
            row
            for row in self.tracks
            if row.get("global_person_id") != global_person_id
        ]
        self.transitions[:] = [
            row
            for row in self.transitions
            if row.get("global_person_id") != global_person_id
        ]

    def merge_persons(self, source: str, target: str) -> None:
        if source == target:
            return
        src = self.galleries.pop(source, None)
        dst = self.galleries.get(target)
        if src is not None and dst is not None:
            for rec in src.body + src.face:
                rec.global_person_id = target
            dst.body.extend(src.body)
            dst.face.extend(src.face)
        for row in self.tracks + self.transitions:
            if row.get("global_person_id") == source:
                row["global_person_id"] = target

    def reset_gallery(self) -> None:
        self.galleries.clear()
        self.tracks.clear()
        self.transitions.clear()
        self._next = 1


def create_store(engine: Optional[Engine] = None):
    if engine is None:
        return InMemoryStore()
    return IdentityRepository(engine)

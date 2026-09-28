"""Engine and schema bootstrap. Production requires PostgreSQL + pgvector."""

from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.database.models import Base
from src.exceptions import ProductionDatabaseError

logger = logging.getLogger(__name__)


def create_engine_from_url(url: str, *, environment: str = "development") -> Engine:
    is_postgres = url.startswith("postgresql")
    if environment.lower() in {"prod", "production"} and not is_postgres:
        raise ProductionDatabaseError(
            "Production requires PostgreSQL + pgvector. SQLite is not allowed."
        )
    if url.startswith("sqlite"):
        # Ensure parent directory exists for file-backed sqlite.
        if ":///" in url:
            raw = url.split(":///", 1)[1]
            if raw and raw != ":memory:" and not raw.startswith(":memory:"):
                Path(raw).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(url, future=True)
        if environment.lower() not in {"test", "testing"}:
            logger.warning(
                "Using SQLite persistence. Identities survive restart, but this is "
                "not the production PostgreSQL/pgvector store."
            )
        return engine
    return create_engine(url, future=True, pool_pre_ping=True)


def init_db(engine: Engine) -> None:
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    if engine.dialect.name == "postgresql":
        with engine.begin() as conn:
            # Older development databases used VARCHAR(64), which cannot
            # hold the semicolon-joined hashes of a multi-camera run.
            conn.execute(text("ALTER TABLE video_runs ALTER COLUMN video_sha256 TYPE TEXT"))
            # Older databases keyed camera_tracks on (track_id, camera_id)
            # only, so every new video overwrote the previous video's A_1 row.
            # Only the constraint changes; no rows are touched.
            conn.execute(
                text("ALTER TABLE camera_tracks DROP CONSTRAINT IF EXISTS uq_track_cam")
            )
            conn.execute(
                text(
                    """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM pg_constraint WHERE conname = 'uq_track_cam_video'
                        ) THEN
                            ALTER TABLE camera_tracks ADD CONSTRAINT uq_track_cam_video
                                UNIQUE (track_id, camera_id, source_video_id);
                        END IF;
                    END $$;
                    """
                )
            )

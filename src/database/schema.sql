-- Reference schema (SQLAlchemy also creates these tables).
-- Production: PostgreSQL + pgvector.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS cameras (
    id SERIAL PRIMARY KEY,
    camera_id VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(128),
    source_type VARCHAR(32),
    source_uri TEXT,
    enabled BOOLEAN DEFAULT TRUE,
    declared_for_video_testing BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS persons (
    id SERIAL PRIMARY KEY,
    global_person_id VARCHAR(32) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    last_seen_at DOUBLE PRECISION,
    status VARCHAR(16),
    first_camera_id VARCHAR(64),
    last_camera_id VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS camera_tracks (
    id SERIAL PRIMARY KEY,
    track_id VARCHAR(64) NOT NULL,
    camera_id VARCHAR(64) NOT NULL,
    global_person_id VARCHAR(32),
    start_time DOUBLE PRECISION,
    end_time DOUBLE PRECISION,
    first_bbox TEXT,
    last_bbox TEXT,
    confidence DOUBLE PRECISION,
    status VARCHAR(32),
    source_video_id VARCHAR(64),
    source_video_name VARCHAR(256),
    CONSTRAINT uq_track_cam_video UNIQUE (track_id, camera_id, source_video_id)
);

CREATE TABLE IF NOT EXISTS person_embeddings (
    id SERIAL PRIMARY KEY,
    global_person_id VARCHAR(32),
    track_id VARCHAR(64),
    camera_id VARCHAR(64),
    timestamp DOUBLE PRECISION,
    embedding TEXT,
    quality_score DOUBLE PRECISION,
    viewpoint VARCHAR(16),
    source_frame_id INTEGER,
    source_video_id VARCHAR(64),
    source_video_name VARCHAR(256),
    source_frame_number INTEGER,
    source_timestamp_ms INTEGER,
    model_name VARCHAR(64),
    model_version VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id SERIAL PRIMARY KEY,
    global_person_id VARCHAR(32),
    track_id VARCHAR(64),
    camera_id VARCHAR(64),
    timestamp DOUBLE PRECISION,
    embedding TEXT,
    quality_score DOUBLE PRECISION,
    viewpoint VARCHAR(16),
    source_frame_id INTEGER,
    source_video_id VARCHAR(64),
    source_video_name VARCHAR(256),
    source_frame_number INTEGER,
    source_timestamp_ms INTEGER,
    model_name VARCHAR(64),
    model_version VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS video_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(64) UNIQUE,
    source_video_name TEXT,
    source_video_path_or_uri TEXT,
    video_sha256 TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    frames_processed INTEGER,
    tracks_detected INTEGER,
    global_ids_seen TEXT,
    new_global_ids_created INTEGER,
    existing_global_ids_reused INTEGER,
    camera_files TEXT
);

CREATE TABLE IF NOT EXISTS source_videos (
    id SERIAL PRIMARY KEY,
    source_video_id VARCHAR(64) UNIQUE,
    source_video_name VARCHAR(256),
    source_video_path_or_uri TEXT,
    video_sha256 VARCHAR(64),
    first_processed_at TIMESTAMP,
    last_processed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS transitions (
    id SERIAL PRIMARY KEY,
    global_person_id VARCHAR(32),
    from_camera_id VARCHAR(64),
    to_camera_id VARCHAR(64),
    from_time DOUBLE PRECISION,
    to_time DOUBLE PRECISION,
    match_score DOUBLE PRECISION,
    match_method VARCHAR(64),
    confidence DOUBLE PRECISION
);

CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(64),
    artifact VARCHAR(256),
    version VARCHAR(64),
    sha256 VARCHAR(64),
    license VARCHAR(64),
    commercial_use_verified BOOLEAN,
    verification_date VARCHAR(32),
    status VARCHAR(16)
);

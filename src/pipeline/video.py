"""Video-file ingestion: single file and synchronized multi-camera.

Identities persist in the database. A process restart must not erase them.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from src.camera.source import VideoFileSource
from src.camera.worker import CameraWorker
from src.events import EventBus
from src.identity.manager import GlobalIdentityManager
from src.observation.manager import ObservationManager
from src.pipeline.features import FeatureExtractor
from src.rendering.overlay import annotate_frame
from src.tracking.botsort import BoTSORTTracker
from src.types import MatchKind, OverlayInfo, TrackState, VideoRunStats

logger = logging.getLogger(__name__)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VideoProcessor:
    def __init__(
        self,
        manager: GlobalIdentityManager,
        detector,
        feature_extractor: FeatureExtractor,
        observation_manager: ObservationManager,
        store,
        events: Optional[EventBus] = None,
        tracking_cfg: Optional[dict] = None,
        output_dir: Path | str = "outputs",
        snapshot_store=None,
        on_frame: Optional[Callable[..., None]] = None,
    ) -> None:
        self.manager = manager
        self.detector = detector
        self.features = feature_extractor
        self.observations = observation_manager
        self.store = store
        self.events = events or EventBus()
        self.tracking_cfg = tracking_cfg or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_store = snapshot_store
        self.on_frame = on_frame
        self.playback_gate: Optional[Callable[[], str]] = None
        # Global IDs whose bounding boxes should be hidden in the live frame.
        self.hidden_ids: set[str] = set()

    def _maybe_snapshot(
        self,
        decision,
        crop,
        face_crop,
        source_name,
        camera_id,
        frame_idx,
        body_quality: float,
        face_quality: float,
        viewpoint: str,
    ) -> bool:
        if self.snapshot_store is None or not decision.global_person_id:
            return False
        if decision.match_kind not in {MatchKind.NEW, MatchKind.EXISTING}:
            return False
        qualities = {"body": body_quality, "face": face_quality}
        saved = False
        for modality, image in (("body", crop), ("face", face_crop)):
            if image is None:
                continue
            if self.snapshot_store.save_track_best(
                global_id=decision.global_person_id,
                crop=image,
                modality=modality,
                video_name=source_name,
                camera_id=camera_id,
                frame_number=frame_idx,
                match_kind=decision.match_kind.value,
                quality=qualities[modality],
                viewpoint=viewpoint,
                local_track=decision.track_id,
            ):
                saved = True
        # If this decision abandoned a provisional identity (e.g. P0001) and
        # rebound the track to another one (e.g. P0002), keep a copy of the
        # crops that caused it with the *old* identity, so an operator can see
        # exactly which face/body justified the reassignment.
        archived = getattr(decision, "archived_gid", None)
        if archived and archived != decision.global_person_id:
            label = f"reason_reassign_to_{decision.global_person_id}"
            for modality, image in (("body", crop), ("face", face_crop)):
                if image is None:
                    continue
                if self.snapshot_store.save(
                    global_id=archived,
                    crop=image,
                    modality=modality,
                    video_name=source_name,
                    camera_id=camera_id,
                    frame_number=frame_idx,
                    match_kind=label,
                    quality=qualities[modality],
                    viewpoint=viewpoint,
                    local_track=decision.track_id,
                ):
                    saved = True
        return saved

    def _move_split_snapshots(self, decision, source_name: str) -> None:
        """Move a split track's saved crops to the ID it was split onto."""
        old = getattr(decision, "split_from", None)
        if not old or self.snapshot_store is None or not decision.global_person_id:
            return
        move = getattr(self.snapshot_store, "move_track", None)
        if callable(move):
            move(old, decision.global_person_id, decision.track_id, source_name)
        decision.snapshot_updated = True

    def process_single(
        self,
        input_path: Path | str,
        camera_id: str = "A",
        fps: Optional[float] = None,
        run_name: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> VideoRunStats:
        return self.process_synchronized(
            cameras={camera_id: str(input_path)},
            fps=fps,
            run_name=run_name,
            max_frames=max_frames,
        )

    def process_synchronized(
        self,
        cameras: dict[str, str],
        fps: Optional[float] = None,
        run_name: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> VideoRunStats:
        sources = {
            cam: VideoFileSource(path, cam, fps=fps) for cam, path in cameras.items()
        }
        shared_fps = fps or next(iter(sources.values())).fps
        run_id = run_name or uuid.uuid4().hex[:12]
        # Local tracker IDs restart for every video run.  Clear only the
        # manager's run-local assignments; the persistent gallery is retained
        # so this run can perform ReID against earlier videos.
        begin_run = getattr(self.manager, "begin_run", None)
        if callable(begin_run):
            begin_run(run_id=run_id)
        clear_observations = getattr(self.observations, "clear", None)
        if callable(clear_observations):
            clear_observations()
        names = ",".join(Path(p).name for p in cameras.values())
        digests = {cam: file_sha256(Path(p)) for cam, p in cameras.items()}
        source_ids = {}
        for cam, path in cameras.items():
            if hasattr(self.store, "upsert_source_video"):
                source_ids[cam] = self.store.upsert_source_video(
                    Path(path).name, str(path), digests[cam]
                )
            else:
                source_ids[cam] = digests[cam][:16]

        workers = {
            cam: CameraWorker(
                camera_id=cam,
                detector=self.detector,
                tracker=BoTSORTTracker(
                    camera_id=cam,
                    min_confirmed_frames=int(self.tracking_cfg.get("min_confirmed_frames", 3)),
                    max_lost_frames=int(self.tracking_cfg.get("max_lost_frames", 30)),
                    iou_threshold=float(self.tracking_cfg.get("iou_threshold", 0.3)),
                    high_score_threshold=float(self.tracking_cfg.get("high_score_threshold", 0.6)),
                    low_score_threshold=float(self.tracking_cfg.get("low_score_threshold", 0.1)),
                    events=self.events,
                ),
                events=self.events,
            )
            for cam in cameras
        }

        import cv2

        writers = {}
        output_videos: dict[str, str] = {}
        for cam, src in sources.items():
            if src.width < 2 or src.height < 2:
                raise RuntimeError(f"Invalid frame size for {cam}: {src.width}x{src.height}")
            out_path = self.output_dir / f"{run_id}_{cam}_{src.source_name}.mp4"
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                shared_fps,
                (src.width, src.height),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open output video writer: {out_path}")
            writers[cam] = writer
            output_videos[cam] = str(out_path)

        started = datetime.now(timezone.utc).isoformat()
        frame_idx = 0
        seen_tracks: set[str] = set()
        seen_ids: set[str] = set()
        created_before = {g.global_person_id for g in self.manager.list_persons()}
        last_decision: dict = {}
        extract_stride = max(1, int(self.tracking_cfg.get("extract_stride", 4)))

        try:
            while True:
                if self.playback_gate is not None and self.playback_gate() == "stop":
                    break
                if max_frames is not None and frame_idx >= max_frames:
                    break
                timestamp = frame_idx / shared_fps
                batch = {}
                any_ok = False
                for cam, src in sources.items():
                    frame = src.read_next(frame_idx)
                    if frame is None:
                        continue
                    any_ok = True
                    dets, tracks = workers[cam].process_frame(frame.image, frame_idx, timestamp)
                    batch[cam] = (frame, dets, tracks)
                if not any_ok:
                    break

                proposals = []
                pair_embeddings: dict[str, list[float]] = {}
                overlays: dict[str, list] = {cam: [] for cam in batch}
                observe_frame = getattr(self.manager, "observe_frame", None)
                for cam, (frame, dets, tracks) in batch.items():
                    if callable(observe_frame):
                        # Tracks on screen together are different people:
                        # record that before any of them is (re)assigned.
                        on_screen = [
                            (t.track_id, t.bbox)
                            for t in tracks
                            if t.state == TrackState.CONFIRMED
                        ]
                        for split in observe_frame(cam, on_screen, timestamp):
                            last_decision[split.track_id] = split
                            self._move_split_snapshots(split, frame.source_name)
                    for track in tracks:
                        seen_tracks.add(track.track_id)
                        if track.state != TrackState.CONFIRMED:
                            info = OverlayInfo(
                                global_id=None,
                                local_track=track.track_id,
                                source=frame.source_name,
                                match=MatchKind.UNRESOLVED,
                                confidence=0.0,
                            )
                            overlays[cam].append((track, info))
                            continue
                        should_extract = (
                            track.track_id not in last_decision
                            or track.hits % extract_stride == 0
                            or self.manager.track_needs_dense_extract(track.track_id)
                        )
                        if should_extract:
                            other_bboxes = [
                                o.bbox
                                for o in tracks
                                if o.track_id != track.track_id
                                and o.state == TrackState.CONFIRMED
                            ]
                            try:
                                feats = self.features.extract(
                                    frame.image, track, other_bboxes=other_bboxes
                                )
                            except Exception:
                                logger.exception(
                                    "feature extraction failed camera=%s track=%s frame=%s",
                                    cam,
                                    track.track_id,
                                    frame_idx,
                                )
                                feats = None
                            if feats is not None:
                                observation = self.observations.consider(
                                    frame.image,
                                    track,
                                    detector_confidence=track.confidence,
                                    source_video_id=source_ids[cam],
                                    source_video_name=frame.source_name,
                                    mask_coverage=feats.mask_coverage,
                                    occlusion_score=feats.occlusion_score,
                                    pose_score=feats.pose_score,
                                    body_visibility_score=feats.body_visibility,
                                    face_quality=feats.face_quality,
                                    viewpoint=feats.viewpoint,
                                    crop=feats.crop,
                                )
                                if observation is not None:
                                    decision = self.manager.assign(
                                        track_id=track.track_id,
                                        camera_id=cam,
                                        timestamp=timestamp,
                                        body_embedding=feats.body_embedding,
                                        face_embedding=feats.face_embedding,
                                        body_quality=feats.body_quality,
                                        face_quality=feats.face_quality,
                                        viewpoint=feats.viewpoint,
                                        source_video_id=source_ids[cam],
                                        source_video_name=frame.source_name,
                                        source_frame_number=frame_idx,
                                        source_timestamp_ms=int(timestamp * 1000),
                                        body_model=getattr(self.features, "body_model", "OSNet-x1.0"),
                                        body_model_version=getattr(self.features, "body_model_version", ""),
                                        velocity=track.velocity,
                                    )
                                    last_decision[track.track_id] = decision
                                    if feats.body_embedding:
                                        pair_embeddings[track.track_id] = feats.body_embedding
                                    snapshot_saved = self._maybe_snapshot(
                                        decision,
                                        feats.crop,
                                        feats.face_crop,
                                        frame.source_name,
                                        cam,
                                        frame_idx,
                                        observation.body_quality,
                                        observation.face_quality,
                                        feats.viewpoint.value,
                                    )
                                    # Refresh the operator UI when a better
                                    # reference crop was written, even if the
                                    # gallery embeddings did not change.
                                    if snapshot_saved:
                                        decision.snapshot_updated = True
                        decision = last_decision.get(track.track_id)
                        if decision is None:
                            overlays[cam].append(
                                (
                                    track,
                                    OverlayInfo(
                                        global_id=None,
                                        local_track=track.track_id,
                                        source=frame.source_name,
                                        match=MatchKind.UNRESOLVED,
                                        confidence=0.0,
                                    ),
                                )
                            )
                            continue
                        proposals.append(decision)
                        if decision.global_person_id:
                            seen_ids.add(decision.global_person_id)
                        overlays[cam].append(
                            (
                                track,
                                OverlayInfo(
                                    global_id=decision.global_person_id,
                                    local_track=track.track_id,
                                    source=frame.source_name,
                                    match=decision.match_kind,
                                    confidence=decision.score,
                                ),
                            )
                        )
                if len(cameras) > 1:
                    self.manager.resolve_overlap(
                        proposals, simultaneous=True, embeddings=pair_embeddings
                    )
                    overlay_by_track = {p.track_id: p for p in proposals}
                    for cam in overlays:
                        updated = []
                        for track, info in overlays[cam]:
                            dec = overlay_by_track.get(track.track_id)
                            if dec is not None:
                                info = OverlayInfo(
                                    global_id=dec.global_person_id,
                                    local_track=track.track_id,
                                    source=info.source,
                                    match=dec.match_kind,
                                    confidence=dec.score,
                                )
                            updated.append((track, info))
                        overlays[cam] = updated
                for cam, (frame, _, _) in batch.items():
                    annotated = annotate_frame(frame.image, overlays[cam])
                    writers[cam].write(annotated)
                    if self.on_frame is not None:
                        live = annotated
                        if self.hidden_ids:
                            # Drop the boxes the operator unticked, for the
                            # live stream only (the saved video keeps them all).
                            visible = [
                                (t, info)
                                for (t, info) in overlays[cam]
                                if info.global_id not in self.hidden_ids
                            ]
                            live = annotate_frame(frame.image, visible)
                        self.on_frame(
                            frame_idx=frame_idx,
                            camera_id=cam,
                            source_name=frame.source_name,
                            annotated=live,
                            overlays=overlays[cam],
                            raw=frame.image,
                            decisions=[
                                last_decision[t.track_id]
                                for t, _ in overlays[cam]
                                if t.track_id in last_decision
                            ],
                        )
                frame_idx += 1
                if frame_idx == 1 or frame_idx % 100 == 0:
                    logger.info(
                        "processed frame=%s tracks=%s ids=%s",
                        frame_idx,
                        len(seen_tracks),
                        ",".join(sorted(seen_ids)) or "-",
                    )
        finally:
            for src in sources.values():
                src.close()
            for w in writers.values():
                w.release()

        # A provisional ID can be discarded (archived as inactive) when a
        # later, better embedding recovers an existing person.  Do not report
        # that transient ID as a final identity for the run.
        current_ids = {
            g.global_person_id
            for g in self.manager.list_persons()
            if getattr(g, "status", "active") == "active"
        }
        final_new_ids = current_ids - created_before
        final_seen_ids = (seen_ids | final_new_ids) & current_ids
        reused_ids = final_seen_ids & created_before
        stats = VideoRunStats(
            run_id=run_id,
            source_video_name=names,
            source_video_path_or_uri=";".join(f"{c}={p}" for c, p in cameras.items()),
            video_sha256=";".join(digests.values()),
            started_at=started,
            completed_at=datetime.now(timezone.utc).isoformat(),
            frames_processed=frame_idx,
            tracks_detected=len(seen_tracks),
            global_ids_seen=sorted(final_seen_ids),
            new_global_ids_created=len(final_new_ids),
            existing_global_ids_reused=len(reused_ids),
            camera_files=dict(cameras),
            output_videos=output_videos,
        )
        if hasattr(self.store, "record_run"):
            self.store.record_run(stats)
        logger.info(
            "run %s frames=%s tracks=%s new=%s reused=%s",
            run_id,
            stats.frames_processed,
            stats.tracks_detected,
            stats.new_global_ids_created,
            stats.existing_global_ids_reused,
        )
        return stats

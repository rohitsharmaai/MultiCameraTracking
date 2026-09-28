"""Background video jobs for the local operator GUI."""

from __future__ import annotations

import base64
import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from src.camera.ids import camera_from_filename
from src.gui.review import ReviewRecorder
from src.rendering.overlay import annotate_frame
from src.types import MatchKind

logger = logging.getLogger(__name__)


@dataclass
class GuiJob:
    job_id: str
    video_name: str
    camera_id: str
    path: Path
    status: str = "queued"
    frame: int = 0
    total_frames: int = 0
    error: Optional[str] = None
    stats: dict[str, Any] = field(default_factory=dict)
    events: deque = field(default_factory=lambda: deque(maxlen=200))
    paused: bool = False
    stop_requested: bool = False
    output_videos: list[dict[str, str]] = field(default_factory=list)
    id_events: list[dict[str, Any]] = field(default_factory=list)
    reviewed_frames: int = 0
    track_ids: dict[str, Optional[str]] = field(default_factory=dict)
    last: Optional[dict[str, Any]] = None
    _seq: int = 0
    _cv: threading.Condition = field(default_factory=threading.Condition)

    def push(self, payload: dict[str, Any]) -> None:
        with self._cv:
            self._seq += 1
            payload["seq"] = self._seq
            self.events.append(payload)
            self._cv.notify_all()

    def wait_after(self, seq: int, timeout: float = 2.0) -> list[dict[str, Any]]:
        with self._cv:
            if not any(ev["seq"] > seq for ev in self.events):
                self._cv.wait(timeout=timeout)
            return [ev for ev in list(self.events) if ev["seq"] > seq]

    def playback_gate(self) -> str:
        with self._cv:
            while self.paused and not self.stop_requested:
                self._cv.wait(timeout=0.25)
            return "stop" if self.stop_requested else "run"


class GuiRuntime:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.upload_dir = project_root / "data" / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = project_root / "data" / "gui_snapshots"
        self.outputs_dir = project_root / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.review = ReviewRecorder(project_root / "data" / "gui_reviews")
        self.processor = None
        self.job: Optional[GuiJob] = None
        self.hidden_ids: set[str] = set()
        self._lock = threading.Lock()

    def ensure_processor(self, cfg, manager, store, snapshots) -> None:
        if self.processor is not None:
            self.processor.manager = manager
            self.processor.store = store
            self.processor.snapshot_store = snapshots
            self.processor.hidden_ids = set(self.hidden_ids)
            return
        from src.bootstrap import build_processor

        self.processor = build_processor(cfg, store=store, snapshot_store=snapshots)
        self.processor.manager = manager
        self.processor.snapshot_store = snapshots
        self.processor.hidden_ids = set(self.hidden_ids)

    def start(self, path: Path, *, max_frames: Optional[int] = None) -> GuiJob:
        with self._lock:
            if self.job and self.job.status in {"queued", "running", "paused", "stopping"}:
                raise RuntimeError("A video is already running")
            camera_id, video_name = camera_from_filename(path)
            cap = cv2.VideoCapture(str(path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            job = GuiJob(
                job_id=uuid.uuid4().hex[:10],
                video_name=video_name,
                camera_id=camera_id,
                path=path,
                total_frames=total,
            )
            self.job = job
        thread = threading.Thread(target=self._run, args=(job, max_frames), daemon=True)
        thread.start()
        return job

    def _require_job(self, *ok: str) -> GuiJob:
        job = self.job
        if job is None or job.status not in ok:
            raise RuntimeError("No active video to control")
        return job

    def pause(self) -> GuiJob:
        job = self._require_job("running", "paused")
        with job._cv:
            job.paused = True
            if job.status == "running":
                job.status = "paused"
                job.push({"type": "paused", "frame": job.frame})
            job._cv.notify_all()
        return job

    def resume(self) -> GuiJob:
        job = self._require_job("paused", "running")
        with job._cv:
            job.paused = False
            if job.status == "paused":
                job.status = "running"
                job.push({"type": "resumed", "frame": job.frame})
            job._cv.notify_all()
        return job

    def stop(self) -> GuiJob:
        job = self._require_job("queued", "running", "paused", "stopping")
        with job._cv:
            job.stop_requested = True
            job.paused = False
            job.status = "stopping"
            job.push({"type": "stopping"})
            job._cv.notify_all()
        return job

    def refresh_live_frame(self) -> None:
        """Re-annotate the current frame so a paused view reflects tick changes."""
        job = self.job
        if job is None or not job.last or job.last.get("raw") is None:
            return
        overlays = job.last["overlays"]
        visible = (
            [(t, i) for (t, i) in overlays if i.global_id not in self.hidden_ids]
            if self.hidden_ids
            else overlays
        )
        live = annotate_frame(job.last["raw"], visible)
        job.push(
            {
                "type": "frame",
                "frame": job.last["frame"],
                "total": job.total_frames,
                "video_name": job.last["source"],
                "camera_id": job.last["camera"],
                "jpeg": _jpeg_b64(live),
                "people": _people_payload(
                    overlays, job.last["source"], job.last["camera"]
                ),
                "id_events": [],
            }
        )

    def _run(self, job: GuiJob, max_frames: Optional[int]) -> None:
        job.status = "running"
        job.push({"type": "started", "video_name": job.video_name, "camera_id": job.camera_id})
        seen: dict[str, dict[str, Any]] = {}
        emitted: set[tuple] = set()

        def on_frame(*, frame_idx, camera_id, source_name, annotated, overlays, decisions, raw=None):
            job.frame = frame_idx + 1
            people = _people_payload(overlays, source_name, camera_id)
            if raw is not None:
                job.last = {
                    "frame": frame_idx,
                    "source": source_name,
                    "camera": camera_id,
                    "overlays": list(overlays),
                    "raw": raw.copy(),
                }
            id_events = _id_changes(job.track_ids, people, frame_idx)
            job.track_ids = {p["local_track"]: p.get("global_id") for p in people}
            self.review.save_frame(job.job_id, frame_idx, annotated, people, id_events)
            job.reviewed_frames = frame_idx + 1
            for ev in id_events:
                job.id_events.append(ev)
                job.push({**ev, "type": ev["type"], "video_name": source_name, "camera_id": camera_id})
            for dec in decisions:
                if not dec.global_person_id:
                    continue
                kind = dec.match_kind.value
                gal = self.processor.manager.show_person(dec.global_person_id)
                prior = []
                if gal is not None:
                    for rec in gal.body + gal.face:
                        name = rec.source_video_name
                        if name and name != source_name and name not in prior:
                            prior.append(name)
                if kind == MatchKind.NEW.value:
                    event_type = "new_id"
                elif prior:
                    event_type = "reid_cross_video"
                elif kind == MatchKind.EXISTING.value:
                    event_type = "reid_same_video"
                else:
                    continue
                if "upgraded gallery" in (dec.reason or ""):
                    job.push(
                        {
                            "type": "gallery_updated",
                            "global_id": dec.global_person_id,
                            "local_track": dec.track_id,
                            "reason": dec.reason,
                        }
                    )
                if getattr(dec, "snapshot_updated", False):
                    # A better body/face reference crop was written this frame.
                    # Tell the dashboard to refresh so the new image shows.
                    dec.snapshot_updated = False
                    job.push(
                        {
                            "type": "gallery_updated",
                            "global_id": dec.global_person_id,
                            "local_track": dec.track_id,
                            "reason": "snapshot improved",
                        }
                    )
                ev_key = (dec.global_person_id, dec.track_id, event_type)
                if ev_key in emitted:
                    continue
                emitted.add(ev_key)
                seen.setdefault(
                    dec.global_person_id,
                    {"video_name": source_name, "camera_id": camera_id},
                )
                job.push(
                    {
                        "type": event_type,
                        "global_id": dec.global_person_id,
                        "local_track": dec.track_id,
                        "video_name": source_name,
                        "camera_id": camera_id,
                        "score": dec.score,
                        "reason": dec.reason,
                        "match_kind": kind,
                        "frame": frame_idx,
                        "first_video": prior[0] if prior else source_name,
                        "prior_videos": prior,
                    }
                )
            if frame_idx % 3 != 0 and frame_idx > 0:
                return
            jpeg = _jpeg_b64(annotated)
            job.push(
                {
                    "type": "frame",
                    "frame": frame_idx,
                    "total": job.total_frames,
                    "video_name": source_name,
                    "camera_id": camera_id,
                    "jpeg": jpeg,
                    "people": people,
                    "id_events": id_events,
                }
            )

        assert self.processor is not None
        self.processor.on_frame = on_frame
        self.processor.playback_gate = job.playback_gate
        self.processor.hidden_ids = set(self.hidden_ids)
        try:
            stats = self.processor.process_single(
                job.path,
                camera_id=job.camera_id,
                run_name=f"gui-{job.job_id}",
                max_frames=max_frames,
            )
            job.output_videos = [
                {
                    "camera_id": cam,
                    "path": path,
                    "url": f"/gui/outputs/{Path(path).name}",
                    "name": Path(path).name,
                }
                for cam, path in (stats.output_videos or {}).items()
                if path
            ]
            job.stats = {
                "run_id": stats.run_id,
                "frames": stats.frames_processed,
                "tracks": stats.tracks_detected,
                "global_ids": stats.global_ids_seen,
                "new": stats.new_global_ids_created,
                "reused": stats.existing_global_ids_reused,
                "output_videos": job.output_videos,
            }
            # Keep evidence of identities that look like the same person by
            # body, and per-identity track comparisons. This only writes
            # crops/summary - it never deletes/merges.
            try:
                from src.gui.duplicates import (
                    audit_identity,
                    cooccurring_pairs_from_review,
                    find_similar_pairs,
                    save_pairs,
                )
                from src.gui.snapshots import default_snapshot_dir

                reid_cfg = self.processor.manager.thresholds.get("reid", {})
                body_thr = float(reid_cfg.get("body_similarity_threshold", 0.62))
                face_thr = float(reid_cfg.get("face_similarity_threshold", 0.45))
                base = default_snapshot_dir(self.project_root).parent
                snap_root = self.processor.snapshot_store.root

                pairs = find_similar_pairs(
                    self.processor.manager.galleries, body_thr, face_thr,
                    distinct_pairs=getattr(self.processor.manager, "distinct_pairs", None),
                )
                saved = save_pairs(pairs, snap_root, base / "similar_identities")
                if saved:
                    job.push({"type": "duplicates", "pairs": saved})

                for gal in self.processor.manager.list_persons():
                    if len({r.track_id for r in gal.body + gal.face}) >= 2:
                        audit_identity(
                            gal, snap_root, base / "identity_audit", body_thr, face_thr,
                            cooccurring_pairs_from_review(self.review.root, gal.global_person_id),
                        )
            except Exception:
                logger.exception("similar-body/identity audit failed")
            if job.stop_requested:
                job.status = "stopped"
                job.push({"type": "stopped", **job.stats})
            else:
                job.status = "done"
                job.push({"type": "done", **job.stats})
        except Exception as exc:
            logger.exception("GUI video job failed")
            job.status = "error"
            job.error = str(exc)
            job.push({"type": "error", "message": str(exc)})
        finally:
            job.paused = False
            self.processor.on_frame = None
            self.processor.playback_gate = None


def _people_payload(overlays, source_name: str, camera_id: str) -> list[dict[str, Any]]:
    people = []
    for track, info in overlays:
        people.append(
            {
                "global_id": info.global_id,
                "local_track": info.local_track,
                "match": info.match.value,
                "confidence": info.confidence,
                "video_name": source_name,
                "camera_id": camera_id,
                "bbox": [round(v, 1) for v in track.bbox.as_xyxy()],
            }
        )
    return people


def _id_changes(
    previous: dict[str, Optional[str]],
    people: list[dict[str, Any]],
    frame_idx: int,
) -> list[dict[str, Any]]:
    current = {p["local_track"]: p.get("global_id") for p in people}
    events = []
    for track_id, gid in current.items():
        old = previous.get(track_id)
        if gid and old != gid:
            events.append(
                {
                    "type": "id_changed" if old else "id_assigned",
                    "frame": frame_idx,
                    "local_track": track_id,
                    "global_id": gid,
                    "previous_id": old,
                    "match": next((p.get("match") for p in people if p["local_track"] == track_id), None),
                }
            )
        elif old and not gid:
            events.append(
                {
                    "type": "id_cleared",
                    "frame": frame_idx,
                    "local_track": track_id,
                    "global_id": None,
                    "previous_id": old,
                }
            )
    return events


def _jpeg_b64(frame: np.ndarray) -> str:
    h, w = frame.shape[:2]
    if w > 720:
        scale = 720 / w
        frame = cv2.resize(frame, (720, max(1, int(h * scale))))
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")

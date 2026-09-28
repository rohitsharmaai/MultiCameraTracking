"""Local operator GUI routes (reset, upload, live process, identity panel)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.gui.jobs import GuiRuntime
from src.gui.snapshots import SnapshotStore, default_snapshot_dir


class IdGateSettings(BaseModel):
    require_body: bool = False
    require_face: bool = False


class VisibilitySettings(BaseModel):
    hidden: list[str] = []


class MergeRequest(BaseModel):
    source: str
    target: str
    # Required to merge two IDs that were on screen together in one frame.
    force: bool = False


def attach_gui(app, *, project_root: Path, manager, store, cfg) -> None:
    web_dir = project_root / "web"
    snapshots = SnapshotStore(default_snapshot_dir(project_root))
    runtime = GuiRuntime(project_root)
    app.state.gui_runtime = runtime
    app.state.snapshots = snapshots

    router = APIRouter(prefix="/gui", tags=["gui"])

    def _apply_id_gates(require_body: bool, require_face: bool) -> dict:
        identity = manager.thresholds.setdefault("identity", {})
        identity["require_body_for_id"] = bool(require_body)
        identity["require_face_for_id"] = bool(require_face)
        return {
            "require_body": bool(identity.get("require_body_for_id", False)),
            "require_face": bool(identity.get("require_face_for_id", False)),
        }

    def _id_gates() -> dict:
        identity = manager.thresholds.get("identity", {})
        return {
            "require_body": bool(identity.get("require_body_for_id", False)),
            "require_face": bool(identity.get("require_face_for_id", False)),
        }

    def _image_payload(row: dict | None) -> dict | None:
        if not row:
            return None
        return {
            "url": f"/gui/snapshots/{row['path']}",
            "video_name": row.get("video_name"),
            "camera_id": row.get("camera_id"),
            "frame": row.get("frame_number"),
            "quality": row.get("quality"),
        }

    def _person_card(gal) -> dict:
        videos = sorted(
            {e.source_video_name for e in gal.body + gal.face if e.source_video_name}
        )
        snaps = snapshots.for_person(gal.global_person_id)
        body_image = _image_payload(snapshots.latest(gal.global_person_id, "body"))
        face_image = _image_payload(snapshots.latest(gal.global_person_id, "face"))
        snapshot_dir = (snapshots.root / gal.global_person_id).resolve()
        # Local track IDs restart per video: count per (video, track).
        track_counts: dict[tuple[str, str], int] = {}
        for e in gal.body + gal.face:
            key = (e.source_video_name or "", e.track_id)
            track_counts[key] = track_counts.get(key, 0) + 1
        return {
            "global_person_id": gal.global_person_id,
            "status": gal.status,
            "first_camera_id": gal.first_camera_id,
            "last_camera_id": gal.last_camera_id,
            "last_seen_at": gal.last_seen_at,
            "body_count": len(gal.body),
            "face_count": len(gal.face),
            "tracks": [
                {"track_id": t, "video_name": v, "embeddings": c}
                for (v, t), c in sorted(track_counts.items())
            ],
            "videos": videos,
            "body_image": body_image,
            "face_image": face_image,
            "snapshot_dir": str(snapshot_dir),
            "snapshot_dir_exists": snapshot_dir.exists(),
            "snapshots": [
                {
                    **row,
                    "url": f"/gui/snapshots/{row['path']}",
                }
                for row in snaps
            ],
            "observations": [
                {
                    "modality": e.modality,
                    "camera_id": e.camera_id,
                    "video_name": e.source_video_name,
                    "frame": e.source_frame_number,
                    "quality": e.quality_score,
                    "viewpoint": e.viewpoint.value,
                    "model": e.model_name,
                    "track_id": e.track_id,
                }
                for e in (gal.body + gal.face)
            ],
        }

    @router.post("/reset")
    def reset_all():
        manager.reset()
        snapshots.clear()
        uploads = project_root / "data" / "uploads"
        if uploads.exists():
            shutil.rmtree(uploads, ignore_errors=True)
            uploads.mkdir(parents=True, exist_ok=True)
        runtime.review.clear()
        runtime.job = None
        return {"ok": True, "message": "Gallery, snapshots, reviews, and uploads deleted. Annotated videos in outputs/ were kept."}

    @router.get("/persons")
    def gui_persons():
        return [_person_card(gal) for gal in manager.list_persons()]

    @router.get("/persons/{person_id}")
    def gui_person(person_id: str):
        for gal in manager.list_persons():
            if gal.global_person_id == person_id:
                return _person_card(gal)
        raise HTTPException(404, "Person not found")

    @router.delete("/persons/{person_id}")
    def delete_person(person_id: str):
        """Explicit user deletion of one identity + its saved crops."""
        if manager.show_person(person_id) is None:
            raise HTTPException(404, "Person not found")
        if hasattr(store, "delete_person"):
            store.delete_person(person_id)
        snapshots.delete_person(person_id)
        manager.reload()
        return {"deleted": person_id}

    @router.post("/persons/delete_inactive")
    def delete_inactive_persons():
        """Explicit user deletion of every inactive identity + its crops."""
        removed = [g.global_person_id for g in manager.list_persons() if g.status != "active"]
        for gid in removed:
            if hasattr(store, "delete_person"):
                store.delete_person(gid)
            snapshots.delete_person(gid)
        manager.reload()
        return {"deleted": removed}

    dup_root = default_snapshot_dir(project_root).parent / "similar_identities"

    @router.post("/persons/merge")
    def merge_person(body: MergeRequest):
        """Explicit user merge: move all evidence from source onto target."""
        if body.source == body.target:
            raise HTTPException(400, "source and target are the same")
        if manager.show_person(body.source) is None or manager.show_person(body.target) is None:
            raise HTTPException(404, "source or target identity not found")
        seen_together = frozenset((body.source, body.target)) in getattr(
            manager, "distinct_pairs", set()
        )
        if seen_together and not body.force:
            raise HTTPException(
                409,
                f"{body.source} and {body.target} were on screen together in the same "
                "frame, so they are two different people.",
            )
        if hasattr(store, "merge_persons"):
            store.merge_persons(body.source, body.target)
        snapshots.merge_person(body.source, body.target)
        manager.reload()
        return {"merged": body.source, "into": body.target}

    ident_root = default_snapshot_dir(project_root).parent / "identity_audit"

    @router.post("/persons/{person_id}/audit")
    def person_audit(person_id: str):
        """Save/show the most-similar body + face pair across this identity's tracks."""
        gal = manager.show_person(person_id)
        if gal is None:
            raise HTTPException(404, "Person not found")
        from src.gui.duplicates import audit_identity, cooccurring_pairs_from_review

        thr_b = float(cfg.thresholds.get("reid", {}).get("body_similarity_threshold", 0.62))
        thr_f = float(cfg.thresholds.get("reid", {}).get("face_similarity_threshold", 0.45))
        co = cooccurring_pairs_from_review(runtime.review.root, person_id)
        return audit_identity(gal, snapshots.root, ident_root, thr_b, thr_f, co)

    @router.get("/identity_audit/{person_id}/{filename}")
    def identity_audit_file(person_id: str, filename: str):
        root = ident_root.resolve()
        path = (root / person_id / Path(filename).name).resolve()
        if root not in path.parents or not path.exists():
            raise HTTPException(404, "Not found")
        return FileResponse(path, media_type="image/jpeg")

    @router.post("/find_duplicates")
    def find_duplicates():
        """Different Global IDs whose tracks look like the same person."""
        from src.gui.duplicates import find_similar_pairs, save_pairs

        reid_cfg = cfg.thresholds.get("reid", {})
        body_thr = float(reid_cfg.get("body_similarity_threshold", 0.62))
        face_thr = float(reid_cfg.get("face_similarity_threshold", 0.45))
        pairs = find_similar_pairs(
            manager.galleries, body_thr, face_thr,
            distinct_pairs=getattr(manager, "distinct_pairs", None),
        )
        saved = save_pairs(pairs, snapshots.root, dup_root)
        return {
            "body_threshold": body_thr,
            "face_threshold": face_thr,
            "count": len(saved),
            "pairs": saved,
        }

    @router.get("/duplicates/{pair}/{modality}/{filename}")
    def duplicate_file(pair: str, modality: str, filename: str):
        root = dup_root.resolve()
        path = (root / pair / modality / Path(filename).name).resolve()
        if root not in path.parents or not path.exists():
            raise HTTPException(404, "Not found")
        return FileResponse(path, media_type="image/jpeg")

    @router.get("/job")
    def gui_job():
        job = runtime.job
        if job is None:
            return {"status": "idle"}
        return {
            "job_id": job.job_id,
            "status": job.status,
            "video_name": job.video_name,
            "camera_id": job.camera_id,
            "frame": job.frame,
            "total_frames": job.total_frames,
            "reviewed_frames": job.reviewed_frames,
            "error": job.error,
            "stats": job.stats,
            "output_videos": job.output_videos,
            "id_events": job.id_events[-80:],
        }

    @router.get("/events")
    def gui_events(after: int = 0):
        job = runtime.job
        if job is None:
            return {"events": []}
        return {"events": job.wait_after(after, timeout=2.0)}

    @router.get("/settings")
    def gui_settings():
        return _id_gates()

    @router.post("/settings")
    def update_settings(body: IdGateSettings):
        return _apply_id_gates(body.require_body, body.require_face)

    @router.post("/visibility")
    def set_visibility(body: VisibilitySettings):
        hidden = {
            gid
            for gid in body.hidden
            if gid and re.fullmatch(r"[A-Za-z0-9_.-]+", gid)
        }
        runtime.hidden_ids = hidden
        if runtime.processor is not None:
            runtime.processor.hidden_ids = set(hidden)
        # Apply immediately to the frame currently on screen (works while paused).
        runtime.refresh_live_frame()
        return {"hidden": sorted(hidden)}

    @router.post("/pause")
    def pause_job():
        try:
            job = runtime.pause()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"status": job.status, "frame": job.frame}

    @router.post("/resume")
    def resume_job():
        try:
            job = runtime.resume()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"status": job.status, "frame": job.frame}

    @router.post("/stop")
    def stop_job():
        try:
            job = runtime.stop()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"status": job.status, "frame": job.frame}

    @router.post("/process")
    async def process_video(
        video: UploadFile = File(...),
        max_frames: int | None = Form(default=None),
        require_body: bool = Form(default=False),
        require_face: bool = Form(default=False),
    ):
        if runtime.job and runtime.job.status in {"queued", "running", "paused", "stopping"}:
            raise HTTPException(409, "A video is already running")
        _apply_id_gates(require_body, require_face)
        runtime.ensure_processor(cfg, manager, store, snapshots)
        dest = runtime.upload_dir / Path(video.filename or "upload.mp4").name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as fh:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
        limit = max_frames if max_frames and max_frames > 0 else None
        job = runtime.start(dest, max_frames=limit)
        return {
            "job_id": job.job_id,
            "video_name": job.video_name,
            "camera_id": job.camera_id,
            "total_frames": job.total_frames,
        }

    @router.post("/snapshots/{person_id}/open")
    def open_snapshot_folder(person_id: str):
        """Open this identity's saved-crop folder in the OS file manager."""
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", person_id or ""):
            raise HTTPException(400, "invalid person id")
        root = snapshots.root.resolve()
        folder = (root / person_id).resolve()
        if folder != root and root not in folder.parents:
            raise HTTPException(400, "invalid person id")
        if not folder.exists():
            return {"ok": False, "error": "folder does not exist", "path": str(folder)}
        if sys.platform.startswith("darwin"):
            cmd = ["open", str(folder)]
        elif os.name == "nt":
            cmd = ["explorer", str(folder)]
        else:
            cmd = ["xdg-open", str(folder)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            if proc.returncode != 0:
                return {
                    "ok": False,
                    "path": str(folder),
                    "error": (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip(),
                }
        except subprocess.TimeoutExpired:
            pass  # file manager launched and is still running
        except Exception as exc:  # file manager unavailable (e.g. headless server)
            return {"ok": False, "error": str(exc), "path": str(folder)}
        return {"ok": True, "path": str(folder)}

    @router.get("/snapshots/{person_id}/{filename}")
    def snapshot_file(person_id: str, filename: str):
        root = snapshots.root.resolve()
        path = (root / person_id / Path(filename).name).resolve()
        # person_id ".." would otherwise serve any file next to the snapshot
        # folder (e.g. the database).
        if root not in path.parents or not path.is_file():
            raise HTTPException(404, "Snapshot not found")
        return FileResponse(path, media_type="image/jpeg")

    @router.get("/review/{job_id}/timeline")
    def review_timeline(job_id: str):
        return {
            "job_id": job_id,
            "last_frame": runtime.review.last_frame(job_id),
            "events": runtime.review.events(job_id),
        }

    @router.get("/review/{job_id}/at/{frame_idx}")
    def review_at(job_id: str, frame_idx: int):
        row = runtime.review.load_frame(job_id, frame_idx)
        if row is None:
            raise HTTPException(404, "Review frame not found")
        return row

    @router.get("/review/{job_id}/frame/{frame_idx}")
    def review_frame(job_id: str, frame_idx: int):
        path = runtime.review.frame_path(job_id, frame_idx)
        if not path.exists():
            raise HTTPException(404, "Review frame not found")
        return FileResponse(path, media_type="image/jpeg")

    @router.get("/outputs")
    def list_outputs():
        root = runtime.outputs_dir
        items = []
        if root.exists():
            for path in sorted(root.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
                items.append(
                    {
                        "name": path.name,
                        "url": f"/gui/outputs/{path.name}",
                        "bytes": path.stat().st_size,
                    }
                )
        return items

    @router.get("/outputs/{filename}")
    def output_file(filename: str):
        path = (runtime.outputs_dir / Path(filename).name).resolve()
        if path.parent != runtime.outputs_dir.resolve() or not path.exists():
            raise HTTPException(404, "Output video not found")
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    app.include_router(router)

    if web_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(web_dir), html=True), name="ui")

        @app.get("/", include_in_schema=False)
        def home():
            index = web_dir / "index.html"
            return HTMLResponse(index.read_text(encoding="utf-8"))

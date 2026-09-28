"""HTTP API plus the local operator GUI."""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.api.auth import api_key_dep
from src.config import load_config
from src.metrics.collectors import Metrics


def create_app(
    config=None,
    store=None,
    manager=None,
    metrics: Optional[Metrics] = None,
    live_workers: Optional[dict] = None,
) -> FastAPI:
    cfg = config or load_config()
    if store is None or manager is None:
        from src.bootstrap import build_manager, build_store

        store = store or build_store(cfg)
        manager = manager or build_manager(cfg, store=store)
    metrics = metrics or Metrics()
    live_workers = live_workers if live_workers is not None else {}
    auth = api_key_dep(cfg.api_key)

    app = FastAPI(title="Multi-Camera Person ReID", version="0.1.0")

    @app.middleware("http")
    async def _no_cache_ui(request, call_next):
        # Always serve the latest HTML/CSS/JS so UI edits are picked up on reload.
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/ui"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/cameras")
    def cameras(_=Depends(auth)):
        cams = cfg.cameras.get("cameras", [])
        return [
            {
                "camera_id": c.get("camera_id"),
                "name": c.get("name"),
                "source_type": c.get("source_type"),
                "enabled": c.get("enabled"),
                "running": c.get("camera_id") in live_workers,
            }
            for c in cams
        ]

    @app.post("/cameras/{camera_id}/start")
    def start_camera(camera_id: str, _=Depends(auth)):
        live_workers[camera_id] = {"state": "running"}
        return {"camera_id": camera_id, "state": "running"}

    @app.post("/cameras/{camera_id}/stop")
    def stop_camera(camera_id: str, _=Depends(auth)):
        live_workers.pop(camera_id, None)
        return {"camera_id": camera_id, "state": "stopped"}

    @app.get("/cameras/{camera_id}/tracks")
    def camera_tracks(camera_id: str, _=Depends(auth)):
        if not hasattr(store, "list_tracks"):
            return []
        rows = store.list_tracks(camera_id)
        return [
            {
                "track_id": r.track_id,
                "global_person_id": r.global_person_id,
                "status": r.status,
                "confidence": r.confidence,
                "source_video_name": r.source_video_name,
            }
            for r in rows
        ]

    @app.get("/persons")
    def persons(_=Depends(auth)):
        out = []
        for gal in manager.list_persons():
            out.append(
                {
                    "global_person_id": gal.global_person_id,
                    "status": gal.status,
                    "first_camera_id": gal.first_camera_id,
                    "last_camera_id": gal.last_camera_id,
                    "last_seen_at": gal.last_seen_at,
                    "body_count": len(gal.body),
                    "face_count": len(gal.face),
                }
            )
        return out

    @app.get("/persons/{global_id}")
    def person(global_id: str, _=Depends(auth)):
        gal = manager.show_person(global_id)
        if gal is None:
            raise HTTPException(404, "Unknown person")
        return {
            "global_person_id": gal.global_person_id,
            "status": gal.status,
            "first_camera_id": gal.first_camera_id,
            "last_camera_id": gal.last_camera_id,
            "last_seen_at": gal.last_seen_at,
            "body_observations": [
                {
                    "camera_id": e.camera_id,
                    "timestamp": e.timestamp,
                    "quality_score": e.quality_score,
                    "viewpoint": e.viewpoint.value,
                    "source_video_name": e.source_video_name,
                    "source_frame_number": e.source_frame_number,
                    "model_name": e.model_name,
                    "model_version": e.model_version,
                }
                for e in gal.body
            ],
            "face_observations": [
                {
                    "camera_id": e.camera_id,
                    "timestamp": e.timestamp,
                    "quality_score": e.quality_score,
                    "source_video_name": e.source_video_name,
                    "source_frame_number": e.source_frame_number,
                    "model_name": e.model_name,
                }
                for e in gal.face
            ],
        }

    @app.get("/persons/{global_id}/timeline")
    def timeline(global_id: str, _=Depends(auth)):
        if not hasattr(store, "person_timeline"):
            return []
        rows = store.person_timeline(global_id)
        return [
            {
                "track_id": r.track_id,
                "camera_id": r.camera_id,
                "start_time": r.start_time,
                "end_time": r.end_time,
                "source_video_name": r.source_video_name,
            }
            for r in rows
        ]

    @app.get("/persons/{global_id}/cameras")
    def person_cameras(global_id: str, _=Depends(auth)):
        gal = manager.show_person(global_id)
        if gal is None:
            raise HTTPException(404, "Unknown person")
        cams = sorted({e.camera_id for e in gal.body + gal.face})
        return {"global_person_id": global_id, "cameras": cams}

    @app.get("/transitions")
    def transitions(_=Depends(auth)):
        if not hasattr(store, "list_transitions"):
            return []
        return [
            {
                "global_person_id": r.global_person_id,
                "from_camera_id": r.from_camera_id,
                "to_camera_id": r.to_camera_id,
                "from_time": r.from_time,
                "to_time": r.to_time,
                "match_score": r.match_score,
                "confidence": r.confidence,
            }
            for r in store.list_transitions()
        ]

    @app.get("/metrics")
    def metrics_endpoint(_=Depends(auth)):
        return metrics.snapshot()

    @app.delete("/persons/{global_id}")
    def delete_person(global_id: str, _=Depends(auth)):
        if hasattr(store, "delete_person"):
            store.delete_person(global_id)
        manager.reload()
        return {"deleted": global_id}

    @app.exception_handler(Exception)
    async def _unhandled(request, exc: Exception):
        if isinstance(exc, HTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    from src.gui.routes import attach_gui

    attach_gui(app, project_root=cfg.project_root, manager=manager, store=store, cfg=cfg)
    app.state.manager = manager
    app.state.store = store
    app.state.config = cfg
    return app


def main() -> None:
    import uvicorn

    uvicorn.run("src.api.app:create_app", factory=True, host="0.0.0.0", port=8000)

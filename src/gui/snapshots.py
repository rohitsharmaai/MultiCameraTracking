"""Save person-crop JPEGs for the operator GUI. Embeddings stay in the database."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Crop quality can change between releases (e.g. adding sharpness/size terms).
# Rows are tagged with the metric that scored them so a newer metric is never
# blocked by an older, non-comparable high score.
SNAPSHOT_METRIC = "quality-v2"

SNAPSHOT_DIR_ENV = "SNAPSHOT_DIR"


def default_snapshot_dir(project_root: Path | str) -> Path:
    """Where person crops are stored.

    Defaults to ``<project>/data/gui_snapshots``.  Set ``SNAPSHOT_DIR`` to an
    absolute (or ``~``) path to keep crops alongside the database instead of in
    the repository, which matters when the working tree is cleaned/restored.
    """
    override = os.environ.get(SNAPSHOT_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path(project_root) / "data" / "gui_snapshots"


class SnapshotStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index = self.root / "index.json"
        self._rows: list[dict] = self._load()
        pruned = self._prune_missing()
        # Re-register any crops that exist on disk but are missing from the
        # index, so images are never hidden by a lost/empty index file.
        added = self.rebuild_from_disk()
        if pruned and not added:
            self._flush()

    def _row_path(self, row: dict) -> Path:
        return self.root / str(row.get("path", ""))

    def _prune_missing(self) -> bool:
        """Drop index rows whose JPEG is gone so a rerun can re-save them.

        A stale row would otherwise win ``save_if_better`` and the GUI would
        keep showing a broken image forever.
        """
        kept = [row for row in self._rows if row.get("path") and self._row_path(row).exists()]
        if len(kept) != len(self._rows):
            self._rows = kept
            return True
        return False

    def _load(self) -> list[dict]:
        if not self._index.exists():
            return []
        try:
            rows = json.loads(self._index.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(rows, list):
            return []
        return [row for row in rows if isinstance(row, dict)]

    def _flush(self) -> None:
        # Write atomically so a crash mid-write cannot corrupt the index and
        # silently hide every snapshot on the next run.
        fd, tmp_name = tempfile.mkstemp(dir=str(self.root), prefix=".index-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(self._rows, fh, indent=2)
            os.replace(tmp_name, self._index)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def save(
        self,
        *,
        global_id: str,
        crop: np.ndarray,
        modality: str,
        video_name: str,
        camera_id: str,
        frame_number: int,
        match_kind: str,
        quality: float,
        viewpoint: str,
        local_track: str,
    ) -> Optional[str]:
        if crop is None or crop.size == 0:
            return None
        person_dir = self.root / global_id
        person_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{modality}_{Path(video_name).stem}_{frame_number}_{local_track}.jpg"
        dest = person_dir / fname
        h, w = crop.shape[:2]
        scale = 160.0 / max(w, 1)
        if scale < 1:
            crop = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))))
        if not cv2.imwrite(str(dest), crop):
            return None
        rel = f"{global_id}/{fname}"
        row = {
            "global_id": global_id,
            "path": rel,
            "modality": modality,
            "video_name": video_name,
            "camera_id": camera_id,
            "frame_number": frame_number,
            "match_kind": match_kind,
            "quality": float(quality),
            "viewpoint": viewpoint,
            "local_track": local_track,
            "metric": SNAPSHOT_METRIC,
        }
        self._rows.append(row)
        self._flush()
        return rel

    def save_if_better(
        self,
        *,
        global_id: str,
        crop: np.ndarray,
        modality: str,
        video_name: str,
        camera_id: str,
        frame_number: int,
        match_kind: str,
        quality: float,
        viewpoint: str,
        local_track: str,
    ) -> Optional[str]:
        current = self.best(global_id, modality, current_metric_only=True)
        if current is not None and float(current.get("quality") or 0.0) >= float(quality):
            return None
        return self.save(
            global_id=global_id,
            crop=crop,
            modality=modality,
            video_name=video_name,
            camera_id=camera_id,
            frame_number=frame_number,
            match_kind=match_kind,
            quality=quality,
            viewpoint=viewpoint,
            local_track=local_track,
        )

    def save_track_best(
        self,
        *,
        global_id: str,
        crop: np.ndarray,
        modality: str,
        video_name: str,
        camera_id: str,
        frame_number: int,
        match_kind: str,
        quality: float,
        viewpoint: str,
        local_track: str,
    ) -> Optional[str]:
        """Save if this is the first crop for the track, or better than the
        track's best so far. Guarantees every local track keeps at least one
        body and one face crop, so the track-comparison view always has images.
        """
        rows = [
            r
            for r in self.for_person(global_id)
            if r.get("modality") == modality and r.get("local_track") == local_track
        ]
        if rows:
            best = max(float(r.get("quality") or 0.0) for r in rows)
            if best >= float(quality):
                return None
        return self.save(
            global_id=global_id,
            crop=crop,
            modality=modality,
            video_name=video_name,
            camera_id=camera_id,
            frame_number=frame_number,
            match_kind=match_kind,
            quality=quality,
            viewpoint=viewpoint,
            local_track=local_track,
        )

    def for_person(self, global_id: str) -> list[dict]:
        rows = [r for r in self._rows if r["global_id"] == global_id]
        # JPEGs can be deleted while the server runs. Drop those rows now:
        # otherwise the GUI shows broken images and the stale best-quality row
        # blocks save_if_better / save_track_best from saving a replacement.
        missing = [r for r in rows if not self._row_path(r).exists()]
        if missing:
            gone = {id(r) for r in missing}
            self._rows = [r for r in self._rows if id(r) not in gone]
            self._flush()
            rows = [r for r in rows if id(r) not in gone]
        return rows

    def latest(self, global_id: str, modality: str) -> Optional[dict]:
        """Reference crop for the dashboard.

        Prefer the best crop scored by the current metric; fall back to older
        rows only until a current-metric crop exists.
        """
        row = self.best(global_id, modality, current_metric_only=True)
        if row is not None:
            return row
        return self.best(global_id, modality)

    def best(
        self,
        global_id: str,
        modality: str,
        *,
        current_metric_only: bool = False,
    ) -> Optional[dict]:
        rows = [r for r in self.for_person(global_id) if r.get("modality") == modality]
        if current_metric_only:
            rows = [r for r in rows if r.get("metric") == SNAPSHOT_METRIC]
        if not rows:
            return None
        return max(rows, key=lambda r: float(r.get("quality") or 0.0))

    def videos_for(self, global_id: str) -> list[str]:
        seen = []
        for row in self.for_person(global_id):
            name = row.get("video_name")
            if name and name not in seen:
                seen.append(name)
        return seen

    def clear(self) -> None:
        for path in self.root.glob("**/*"):
            if path.is_file():
                path.unlink()
        for path in sorted(self.root.glob("*"), reverse=True):
            if path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        self._rows = []
        self._flush()

    def delete_person(self, global_id: str) -> None:
        """Delete all saved crops for one identity.

        Explicit user action only (GUI delete button / API). Never called by
        the pipeline.
        """
        person_dir = self.root / global_id
        if person_dir.exists():
            for path in sorted(person_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass
            try:
                person_dir.rmdir()
            except OSError:
                pass
        self._rows = [r for r in self._rows if r.get("global_id") != global_id]
        self._flush()

    def merge_person(self, source: str, target: str) -> None:
        """Move all saved crops of one identity onto another (user merge)."""
        if source == target:
            return
        src_dir = self.root / source
        dst_dir = self.root / target
        dst_dir.mkdir(parents=True, exist_ok=True)
        for row in self._rows:
            if row.get("global_id") != source:
                continue
            old = self.root / str(row.get("path") or "")
            if old.exists():
                new = dst_dir / old.name
                if new.exists():
                    new = dst_dir / f"{source}__{old.name}"
                shutil.move(str(old), str(new))
                row["global_id"] = target
                row["path"] = f"{target}/{new.name}"
            else:
                row["global_id"] = target
        if src_dir.exists():
            for path in sorted(src_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass
        self._flush()

    def move_track(
        self,
        source: str,
        target: str,
        local_track: str,
        video_name: Optional[str] = None,
    ) -> int:
        """Move one local track's crops from one identity to another.

        Used when the pipeline splits a track off an ID it wrongly shared with
        another person on screen at the same time. Files are moved, not deleted.
        """
        if source == target:
            return 0
        dst_dir = self.root / target
        moved = 0
        for row in self._rows:
            if row.get("global_id") != source or row.get("local_track") != local_track:
                continue
            if video_name and row.get("video_name") not in {video_name, Path(video_name).stem}:
                continue
            old = self.root / str(row.get("path") or "")
            if old.exists():
                dst_dir.mkdir(parents=True, exist_ok=True)
                new = dst_dir / old.name
                if new.exists():
                    new = dst_dir / f"{source}__{old.name}"
                shutil.move(str(old), str(new))
                row["path"] = f"{target}/{new.name}"
            row["global_id"] = target
            moved += 1
        if moved:
            self._flush()
        return moved

    def rebuild_from_disk(self) -> int:
        """Re-register crop JPEGs that exist on disk but are missing from the index.

        Filenames are ``<modality>_<video>_<frame>_<track>.jpg`` inside a
        folder named after the global id. This guarantees images are never
        hidden just because the index was lost/emptied.
        """
        known = {str(r.get("path") or "") for r in self._rows}
        added = 0
        for person_dir in sorted(self.root.glob("*")):
            if not person_dir.is_dir():
                continue
            gid = person_dir.name
            for f in sorted(person_dir.glob("*.jpg")):
                rel = f"{gid}/{f.name}"
                if rel in known:
                    continue
                from src.gui.duplicates import parse_snapshot_name

                parsed = parse_snapshot_name(f.stem)
                if parsed is not None:
                    modality, video, frame, track = parsed
                else:
                    parts = f.stem.split("_")
                    modality = parts[0] if parts and parts[0] in {"body", "face"} else "body"
                    video, frame, track = "", None, ""
                self._rows.append({
                    "global_id": gid,
                    "path": rel,
                    "modality": modality,
                    "video_name": video,
                    "camera_id": "",
                    "frame_number": frame,
                    "match_kind": "recovered",
                    "quality": 0.0,
                    "viewpoint": "unknown",
                    "local_track": track,
                    "metric": SNAPSHOT_METRIC,
                })
                known.add(rel)
                added += 1
        if added:
            self._flush()
        return added

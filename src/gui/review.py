"""Per-job annotated frames and ID-assignment timeline for GUI scrubbing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np


class ReviewRecorder:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._at: dict[tuple[str, int], dict[str, Any]] = {}

    def job_dir(self, job_id: str) -> Path:
        path = self.root / job_id
        (path / "frames").mkdir(parents=True, exist_ok=True)
        return path

    def save_frame(
        self,
        job_id: str,
        frame_idx: int,
        annotated: np.ndarray,
        people: list[dict[str, Any]],
        events: list[dict[str, Any]],
    ) -> Path:
        dest = self.frame_path(job_id, frame_idx)
        preview = _preview(annotated)
        cv2.imwrite(str(dest), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
        row = {"frame": frame_idx, "people": people, "events": events}
        self._at[(job_id, int(frame_idx))] = row
        index = self._index_path(job_id)
        with index.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        return dest

    def frame_path(self, job_id: str, frame_idx: int) -> Path:
        return self.job_dir(job_id) / "frames" / f"{int(frame_idx):06d}.jpg"

    def load_frame(self, job_id: str, frame_idx: int) -> Optional[dict[str, Any]]:
        path = self.frame_path(job_id, frame_idx)
        if not path.exists():
            return None
        row = self._at.get((job_id, int(frame_idx)))
        if row is None:
            for item in reversed(self.timeline(job_id)):
                if int(item.get("frame", -1)) == int(frame_idx):
                    row = item
                    break
        if row is None:
            row = {"frame": frame_idx, "people": [], "events": []}
        return {**row, "url": f"/gui/review/{job_id}/frame/{frame_idx}"}

    def timeline(self, job_id: str) -> list[dict[str, Any]]:
        index = self._index_path(job_id)
        if not index.exists():
            return []
        rows = []
        for line in index.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def events(self, job_id: str) -> list[dict[str, Any]]:
        found = []
        for row in self.timeline(job_id):
            for ev in row.get("events") or []:
                found.append({**ev, "frame": row.get("frame")})
        return found

    def last_frame(self, job_id: str) -> int:
        frames = self.job_dir(job_id) / "frames"
        nums = [int(p.stem) for p in frames.glob("*.jpg") if p.stem.isdigit()]
        return max(nums) if nums else -1

    def clear(self) -> None:
        self._at.clear()
        if not self.root.exists():
            return
        for path in sorted(self.root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass

    def _index_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "timeline.jsonl"


def _preview(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= 960:
        return frame
    scale = 960 / w
    return cv2.resize(frame, (960, max(1, int(h * scale))))

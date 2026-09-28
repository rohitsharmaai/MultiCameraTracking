"""Annotated output video overlay (Section 14A)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from src.types import LocalTrack, OverlayInfo


def annotate_frame(
    frame: np.ndarray,
    tracks: Iterable[tuple[LocalTrack, OverlayInfo]],
) -> np.ndarray:
    out = frame.copy()
    for track, info in tracks:
        x1, y1, x2, y2 = (int(v) for v in track.bbox.as_xyxy())
        color = {
            "existing": (40, 200, 40),
            "new": (40, 160, 220),
            "unresolved": (40, 40, 220),
            "conflict": (0, 140, 255),
        }.get(info.match.value, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        gid = info.global_id or "UNRESOLVED"
        lines = [
            f"Global ID: {gid}",
            f"Local Track: {info.local_track}",
            f"Source: {info.source}",
            f"Match: {info.match.value}",
            f"Confidence: {info.confidence:.2f}",
        ]
        text_h = 16 * len(lines)
        y = y1 - 8 - text_h
        if y < 16:
            y = min(out.shape[0] - 8, y2 + 16)
        x = max(0, min(x1, out.shape[1] - 8))
        for i, line in enumerate(lines):
            cv2.putText(
                out,
                line,
                (x, y + i * 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    return out


def write_video(
    path: str | Path,
    frames: list[np.ndarray],
    fps: float,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames to write")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return path

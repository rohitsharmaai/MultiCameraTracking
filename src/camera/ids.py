"""Canonical camera IDs. EPFL Lab files map to topology A–D."""

from __future__ import annotations

from pathlib import Path

CAMERA_ALIASES = {
    "lab-c0": "A",
    "lab-c1": "B",
    "lab-c2": "C",
    "lab-c3": "D",
    "4p-c0": "A",
    "4p-c1": "B",
    "4p-c2": "C",
    "4p-c3": "D",
}


def canonical_camera(camera_id: str) -> str:
    return CAMERA_ALIASES.get(camera_id.strip(), camera_id.strip())


def camera_from_filename(path: str | Path) -> tuple[str, str]:
    """Return (topology camera id, original filename for display)."""
    name = Path(path).name
    stem = Path(path).stem
    return canonical_camera(stem), name

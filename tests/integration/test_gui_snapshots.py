"""Body/face crops must be persisted for every processing entry point.

Regression: the CLI processor never received a SnapshotStore, so identities
were written to the database while the operator GUI showed ``no image`` for
both the body and the face after a rerun.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.bootstrap import build_processor
from src.config import load_config
from src.database.repository import InMemoryStore
from src.gui.snapshots import SnapshotStore, default_snapshot_dir
from src.types import IdentityDecision, IdentityAssignmentState, MatchKind


class _FakeDecision:
    def __init__(self) -> None:
        self.global_person_id = "P0001"
        self.track_id = "A_1"
        self.match_kind = MatchKind.NEW


def _crop(w: int = 40, h: int = 80) -> np.ndarray:
    return np.full((h, w, 3), 127, dtype=np.uint8)


def test_build_processor_creates_default_snapshot_store(monkeypatch):
    """Any caller of build_processor (e.g. the CLI) gets a snapshot store."""
    created = {}

    real_store = SnapshotStore

    def fake_store(root):
        created["root"] = Path(root)
        return real_store(root)

    monkeypatch.setattr("src.gui.snapshots.SnapshotStore", fake_store)
    cfg = load_config()
    proc = build_processor(cfg, load_weights=False, store=InMemoryStore())
    assert proc.snapshot_store is not None
    # conftest points SNAPSHOT_DIR at a temp folder so tests never touch real crops.
    assert created["root"] == default_snapshot_dir(cfg.project_root)
    assert cfg.project_root not in created["root"].parents


def test_processor_persists_body_and_face_crops(tmp_path: Path):
    snapshots = SnapshotStore(tmp_path / "snaps")
    cfg = load_config()
    proc = build_processor(
        cfg, load_weights=False, store=InMemoryStore(), snapshot_store=snapshots
    )
    proc.on_frame = None

    decision = _FakeDecision()
    assert proc._maybe_snapshot(
        decision,
        _crop(),
        _crop(20, 20),
        "video_01.mp4",
        "A",
        12,
        0.8,
        0.7,
        "front",
    )

    body = snapshots.latest("P0001", "body")
    face = snapshots.latest("P0001", "face")
    assert body is not None and face is not None
    assert (snapshots.root / body["path"]).exists()
    assert (snapshots.root / face["path"]).exists()

    # A worse crop must not overwrite the reference images.
    assert not proc._maybe_snapshot(
        decision,
        _crop(),
        _crop(20, 20),
        "video_01.mp4",
        "A",
        40,
        0.1,
        0.1,
        "back",
    )

    # A reopened store (process restart / rerun) must still find the images.
    reloaded = SnapshotStore(tmp_path / "snaps")
    assert reloaded.latest("P0001", "body") is not None
    assert reloaded.latest("P0001", "face") is not None


def test_decisions_without_identity_are_not_saved(tmp_path: Path):
    snapshots = SnapshotStore(tmp_path / "snaps")
    cfg = load_config()
    proc = build_processor(
        cfg, load_weights=False, store=InMemoryStore(), snapshot_store=snapshots
    )
    decision = IdentityDecision(
        track_id="A_1",
        camera_id="A",
        state=IdentityAssignmentState.UNRESOLVED,
        global_person_id=None,
        match_kind=MatchKind.UNRESOLVED,
        score=0.0,
        second_best_score=0.0,
        margin=0.0,
        reason="stabilizing",
    )
    proc._maybe_snapshot(decision, _crop(), _crop(20, 20), "video_01.mp4", "A", 1, 0.5, 0.5, "front")
    assert snapshots.for_person("P0001") == []


def test_deleted_jpeg_prunes_stale_index_and_can_be_resaved(tmp_path: Path):
    root = tmp_path / "snaps"
    snapshots = SnapshotStore(root)
    snapshots.save(
        global_id="P0001",
        crop=_crop(),
        modality="body",
        video_name="video_01.mp4",
        camera_id="A",
        frame_number=1,
        match_kind="new",
        quality=0.9,
        viewpoint="front",
        local_track="A_1",
    )
    for jpg in (root / "P0001").glob("*.jpg"):
        jpg.unlink()

    reloaded = SnapshotStore(root)
    assert reloaded.latest("P0001", "body") is None
    # A lower-quality crop must still be accepted because the old file is gone.
    assert (
        reloaded.save_if_better(
            global_id="P0001",
            crop=_crop(),
            modality="body",
            video_name="video_01.mp4",
            camera_id="A",
            frame_number=2,
            match_kind="existing",
            quality=0.1,
            viewpoint="back",
            local_track="A_1",
        )
        is not None
    )



def test_jpeg_deleted_while_running_is_pruned_without_restart(tmp_path: Path):
    """Crops wiped while the server runs must not leave broken images or
    block replacements until the next restart."""
    root = tmp_path / "snaps"
    snapshots = SnapshotStore(root)
    for track in ("A_1", "A_2"):
        snapshots.save_track_best(
            global_id="P0001",
            crop=_crop(),
            modality="body",
            video_name="video_01.mp4",
            camera_id="A",
            frame_number=1,
            match_kind="new",
            quality=0.9,
            viewpoint="front",
            local_track=track,
        )
    for jpg in (root / "P0001").glob("*.jpg"):
        jpg.unlink()

    # Same instance, no reload.
    assert snapshots.latest("P0001", "body") is None
    assert snapshots.for_person("P0001") == []
    assert snapshots.save_if_better(
        global_id="P0001", crop=_crop(), modality="body", video_name="video_01.mp4",
        camera_id="A", frame_number=2, match_kind="existing", quality=0.1,
        viewpoint="back", local_track="A_1",
    ) is not None
    assert snapshots.save_track_best(
        global_id="P0001", crop=_crop(), modality="body", video_name="video_01.mp4",
        camera_id="A", frame_number=3, match_kind="existing", quality=0.1,
        viewpoint="back", local_track="A_2",
    ) is not None
    assert len(SnapshotStore(root).for_person("P0001")) == 2

def _save(snapshots: SnapshotStore, *, frame: int, quality: float, metric: str | None = None) -> None:
    rel = snapshots.save(
        global_id="P0001",
        crop=_crop(),
        modality="body",
        video_name="video_01.mp4",
        camera_id="A",
        frame_number=frame,
        match_kind="new",
        quality=quality,
        viewpoint="front",
        local_track="A_1",
    )
    if metric is not None:
        # Simulate a row written by an older metric version.
        for row in snapshots._rows:
            if row["path"] == rel:
                row["metric"] = metric


def test_old_metric_row_does_not_block_new_metric_crop(tmp_path: Path):
    from src.gui.snapshots import SNAPSHOT_METRIC

    snapshots = SnapshotStore(tmp_path / "snaps")
    # An old metric awarded a very high score; the new metric must ignore it.
    _save(snapshots, frame=10, quality=0.99, metric="legacy-v1")
    assert snapshots.save_if_better(
        global_id="P0001",
        crop=_crop(),
        modality="body",
        video_name="video_01.mp4",
        camera_id="A",
        frame_number=20,
        match_kind="existing",
        quality=0.70,
        viewpoint="front",
        local_track="A_1",
    )
    best = snapshots.latest("P0001", "body")
    assert best is not None
    assert best["metric"] == SNAPSHOT_METRIC
    assert best["quality"] == pytest.approx(0.70)


def test_snapshot_dir_env_override(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("SNAPSHOT_DIR", raising=False)
    assert default_snapshot_dir(tmp_path) == tmp_path / "data" / "gui_snapshots"
    target = tmp_path / "durable" / "snaps"
    monkeypatch.setenv("SNAPSHOT_DIR", str(target))
    assert default_snapshot_dir(tmp_path) == target


def test_move_track_moves_only_that_tracks_crops(tmp_path: Path):
    snapshots = SnapshotStore(tmp_path / "snaps")
    common = dict(modality="body", camera_id="A", match_kind="new", viewpoint="front")
    snapshots.save(global_id="P0001", crop=_crop(), video_name="4p-c0.avi",
                   frame_number=439, quality=0.8, local_track="A_1", **common)
    snapshots.save(global_id="P0001", crop=_crop(), video_name="4p-c0.avi",
                   frame_number=744, quality=0.8, local_track="A_4", **common)
    snapshots.save(global_id="P0001", crop=_crop(), video_name="other.avi",
                   frame_number=10, quality=0.8, local_track="A_4", **common)

    moved = snapshots.move_track("P0001", "P0002", "A_4", "4p-c0.avi")

    assert moved == 1
    assert {r["local_track"] for r in snapshots.for_person("P0001")} == {"A_1", "A_4"}
    [row] = snapshots.for_person("P0002")
    assert row["frame_number"] == 744
    assert (tmp_path / "snaps" / row["path"]).exists()
    assert not (tmp_path / "snaps" / "P0001" / "body_4p-c0_744_A_4.jpg").exists()

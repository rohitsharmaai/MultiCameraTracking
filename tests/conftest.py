from __future__ import annotations

import numpy as np
import pytest

from src.database.repository import InMemoryStore
from src.events import EventBus
from src.identity.manager import GlobalIdentityManager
from src.identity.matcher import IdentityMatcher, MatcherConfig
from src.identity.topology import CameraTopology


def unit_vec(seed: int, dim: int = 512) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v.tolist()


def near(vec: list[float], noise: float = 0.02, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(vec, dtype=np.float32) + rng.normal(scale=noise, size=len(vec)).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v.tolist()


@pytest.fixture(autouse=True)
def _isolated_snapshot_dir(tmp_path, monkeypatch):
    """Never let a test read or wipe the operator's real crop folder.

    ``SNAPSHOT_DIR`` from ``.env`` (or the default ``data/gui_snapshots``)
    would otherwise be used, and ``POST /gui/reset`` in a test deleted every
    saved body/face JPEG while the real gallery database kept its identities.
    """
    monkeypatch.setenv("SNAPSHOT_DIR", str(tmp_path / "gui_snapshots"))


@pytest.fixture
def topology() -> CameraTopology:
    return CameraTopology(
        {
            "edges": [
                {
                    "from_camera": "A",
                    "to_camera": "B",
                    "min_transition_seconds": 1,
                    "max_transition_seconds": 90,
                    "expected_direction": "east",
                    "confidence": 0.9,
                },
                {
                    "from_camera": "B",
                    "to_camera": "A",
                    "min_transition_seconds": 1,
                    "max_transition_seconds": 90,
                    "expected_direction": "west",
                    "confidence": 0.9,
                },
            ]
        }
    )


@pytest.fixture
def thresholds() -> dict:
    return {
        "reid": {
            "body_similarity_threshold": 0.55,
            "minimum_score_margin": 0.08,
            "gallery_size": 4,
            "candidate_k": 16,
        },
        "identity": {
            "create_new_id_only_after_rejection": True,
            "hysteresis_enabled": True,
            "hysteresis_switch_margin": 0.15,
            "stabilization_frames": 1,
            "new_id_prefix": "P",
            "new_id_width": 4,
        },
        "scoring": {
            "w_body": 0.5,
            "w_face": 0.3,
            "w_time": 0.1,
            "w_camera": 0.1,
            "w_motion": 0.0,
            "body_quality_gate": 0.3,
            "face_quality_gate": 0.4,
        },
        "camera_transition": {
            "topology_enabled": True,
            "temporal_constraints_enabled": True,
            "default_max_transition_seconds": 180.0,
            "unknown_edge_penalty": 1.0,
        },
        "overlap": {
            "enabled_only_when_cameras_are_simultaneous": True,
            "score_margin": 0.10,
        },
    }


@pytest.fixture
def manager(topology, thresholds) -> GlobalIdentityManager:
    store = InMemoryStore()
    matcher = IdentityMatcher(
        MatcherConfig(
            accept_threshold=0.55,
            minimum_score_margin=0.08,
            scoring=thresholds["scoring"],
            topology_enabled=True,
            temporal_enabled=True,
            unknown_edge_penalty=1.0,
        ),
        topology,
    )
    return GlobalIdentityManager(store, matcher, topology, thresholds, events=EventBus())

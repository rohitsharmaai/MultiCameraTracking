from pathlib import Path

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import load_config
from src.database.repository import InMemoryStore
from src.identity.manager import GlobalIdentityManager
from src.identity.matcher import IdentityMatcher, MatcherConfig
from src.identity.topology import CameraTopology


def _app(tmp_path: Path):
    cfg = load_config()
    cfg.api_key = "secret"
    # Point the GUI at a throwaway project root: /gui/reset deletes the
    # snapshots, uploads and review folders under it.
    (tmp_path / "web").symlink_to(cfg.project_root / "web", target_is_directory=True)
    cfg.project_root = tmp_path
    store = InMemoryStore()
    manager = GlobalIdentityManager(
        store,
        IdentityMatcher(
            MatcherConfig(0.5, 0.08, cfg.thresholds["scoring"], True, True, 1.0),
            CameraTopology(cfg.topology),
        ),
        CameraTopology(cfg.topology),
        cfg.thresholds,
    )
    return create_app(config=cfg, store=store, manager=manager)


def test_health_open(tmp_path):
    client = TestClient(_app(tmp_path))
    assert client.get("/health").status_code == 200


def test_persons_requires_auth(tmp_path):
    client = TestClient(_app(tmp_path))
    assert client.get("/persons").status_code == 401
    r = client.get("/persons", headers={"X-API-Key": "secret"})
    assert r.status_code == 200
    assert r.json() == []


def test_gui_home_and_reset(tmp_path):
    client = TestClient(_app(tmp_path))
    home = client.get("/")
    assert home.status_code == 200
    assert b"Global ID" in home.content
    reset = client.post("/gui/reset")
    assert reset.status_code == 200
    assert reset.json()["ok"] is True
    assert client.get("/gui/persons").json() == []


def test_gui_settings_and_idle_transport(tmp_path):
    client = TestClient(_app(tmp_path))
    settings = client.get("/gui/settings").json()
    assert settings["require_body"] is False
    assert settings["require_face"] is False
    updated = client.post(
        "/gui/settings", json={"require_body": True, "require_face": False}
    )
    assert updated.status_code == 200
    assert updated.json()["require_body"] is True
    assert client.get("/gui/settings").json()["require_body"] is True
    assert client.post("/gui/pause").status_code == 409
    assert client.post("/gui/resume").status_code == 409
    assert client.post("/gui/stop").status_code == 409
    home = client.get("/")
    assert b"playPause" in home.content
    assert b"Follow live" in home.content
    assert b"Body embedding required" in home.content
    listed = client.get("/gui/outputs")
    assert listed.status_code == 200
    assert isinstance(listed.json(), list)
    assert client.get("/gui/review/missing/at/0").status_code == 404


def test_no_embeddings_in_person_payload(tmp_path):
    client = TestClient(_app(tmp_path))
    r = client.get("/persons/P0001", headers={"X-API-Key": "secret"})
    assert r.status_code == 404
    assert r.json()["detail"] == "Unknown person"


def test_gui_person_detail_route(tmp_path):
    client = TestClient(_app(tmp_path))
    assert client.get("/gui/persons/P0001").status_code == 404
    assert client.get("/ui/person.html").status_code == 200
    assert b"Global Identity" in client.get("/ui/person.html").content


def test_snapshot_route_rejects_path_traversal(tmp_path, monkeypatch):
    snaps = tmp_path / "snaps"
    (snaps / "P0001").mkdir(parents=True)
    (snaps / "P0001" / "body_v_1_A_1.jpg").write_bytes(b"jpg")
    (tmp_path / "secret.txt").write_text("do not serve")
    monkeypatch.setenv("SNAPSHOT_DIR", str(snaps))
    root = tmp_path / "root"
    root.mkdir()
    client = TestClient(_app(root))
    assert client.get("/gui/snapshots/P0001/body_v_1_A_1.jpg").status_code == 200
    assert client.get("/gui/snapshots/%2E%2E/secret.txt").status_code == 404


def test_merge_of_ids_seen_together_needs_force(tmp_path, monkeypatch):
    from src.identity.gallery import PersonGallery

    monkeypatch.setenv("SNAPSHOT_DIR", str(tmp_path / "snaps"))
    root = tmp_path / "root"
    root.mkdir()
    app = _app(root)
    manager = app.state.manager
    for gid in ("P0001", "P0002"):
        manager.galleries[gid] = PersonGallery(gid)
    manager.distinct_pairs.add(frozenset(("P0001", "P0002")))
    client = TestClient(app)
    r = client.post("/gui/persons/merge", json={"source": "P0002", "target": "P0001"})
    assert r.status_code == 409
    assert "on screen together" in r.json()["detail"]
    r = client.post("/gui/persons/merge", json={"source": "P0002", "target": "P0001", "force": True})
    assert r.status_code == 200

from src.config import _anchor_sqlite_path


def test_relative_sqlite_url_is_anchored_to_project_root(tmp_path):
    assert _anchor_sqlite_path("sqlite:///./data/g.db", tmp_path) == f"sqlite:///{tmp_path / 'data' / 'g.db'}"
    assert _anchor_sqlite_path("sqlite:////abs/g.db", tmp_path) == "sqlite:////abs/g.db"
    assert _anchor_sqlite_path("sqlite:///:memory:", tmp_path) == "sqlite:///:memory:"
    pg = "postgresql+psycopg2://u:p@localhost/db"
    assert _anchor_sqlite_path(pg, tmp_path) == pg

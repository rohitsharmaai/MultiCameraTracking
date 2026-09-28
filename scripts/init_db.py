#!/usr/bin/env python3
"""Create PostgreSQL/SQLite schema."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.database.session import create_engine_from_url, init_db  # noqa: E402


def main() -> int:
    cfg = load_config()
    engine = create_engine_from_url(cfg.database_url, environment=cfg.environment)
    init_db(engine)
    print(f"initialized {cfg.database_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Load YAML configuration. Thresholds are never hard-coded in matchers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.exceptions import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


def _read_yaml(path: Path) -> Any:
    if not path.exists():
        raise ConfigurationError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _anchor_sqlite_path(url: str, root: Path) -> str:
    """Resolve a relative SQLite path against the project root, not the CWD.

    ``sqlite:///./data/gallery.db`` otherwise points at a different (empty)
    database whenever a script is started from another directory.
    """
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return url
    raw = url[len(prefix):]
    if not raw or raw.startswith(":memory:") or raw.startswith("/") or Path(raw).is_absolute():
        return url
    return prefix + str((root / raw).resolve())


@dataclass
class AppConfig:
    thresholds: dict[str, Any]
    topology: dict[str, Any]
    cameras: dict[str, Any]
    models: dict[str, Any]
    environment: str
    database_url: str
    api_key: str
    allow_cpu_fallback: bool
    models_dir: Path
    project_root: Path = PROJECT_ROOT

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.thresholds
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def is_production(self) -> bool:
        return self.environment.lower() in {"prod", "production"}


def load_config(
    config_dir: Path | None = None,
    env_file: Path | None = None,
) -> AppConfig:
    load_dotenv(env_file or PROJECT_ROOT / ".env")
    root = config_dir or CONFIG_DIR
    thresholds = _read_yaml(root / "thresholds.yaml")
    topology = _read_yaml(root / "topology.yaml")
    cameras = _read_yaml(root / "cameras.yaml")
    models = _read_yaml(root / "models.yaml")

    database_url = os.environ.get(
        "DATABASE_URL",
        "sqlite:///./data/gallery.db",
    )
    database_url = _anchor_sqlite_path(database_url, PROJECT_ROOT)
    environment = os.environ.get("ENVIRONMENT", "development")
    api_key = os.environ.get("API_KEY", "")
    allow_cpu = os.environ.get("ALLOW_CPU_FALLBACK", "true").lower() in {
        "1",
        "true",
        "yes",
    }
    models_dir = Path(os.environ.get("MODELS_DIR", PROJECT_ROOT / "models"))
    if not models_dir.is_absolute():
        models_dir = PROJECT_ROOT / models_dir

    return AppConfig(
        thresholds=thresholds,
        topology=topology,
        cameras=cameras,
        models=models,
        environment=environment,
        database_url=database_url,
        api_key=api_key,
        allow_cpu_fallback=allow_cpu,
        models_dir=models_dir,
    )

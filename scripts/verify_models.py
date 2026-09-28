#!/usr/bin/env python3
"""Post-download VERIFY. Fail -> REJECTED. Never silently substitute."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.exceptions import ModelVerificationError  # noqa: E402
from src.registry import ModelRegistry, sha256_file  # noqa: E402


def main() -> int:
    cfg = load_config()
    registry = ModelRegistry(
        ROOT / "configs" / "models.yaml",
        ROOT / "models" / "MANIFEST.yaml",
        cfg.models_dir,
    )
    failed = 0
    for entry in registry.all_entries():
        if entry.name in {"RF-DETR-L", "BoT-SORT"}:
            print(f"SKIP on-disk VERIFY {entry.name} (package/code artifact)")
            continue
        path = registry.resolve_path(entry)
        if path is None or not path.exists():
            print(f"MISSING {entry.name}: {path}")
            continue
        digest = sha256_file(path)
        if not entry.sha256:
            registry.record_sha256(entry.name, digest)
            print(f"RECORDED {entry.name} sha256={digest}")
        try:
            registry.verify_on_disk(entry.name)
            print(f"PASS {entry.name}")
        except ModelVerificationError as exc:
            print(f"FAIL {entry.name}: {exc}")
            registry.mark_rejected(entry.name, str(exc))
            print("DO NOT LOAD / DO NOT SHIP / DO NOT SILENTLY SUBSTITUTE / MARK REJECTED")
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

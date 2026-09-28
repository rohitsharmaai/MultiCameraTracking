"""Model registry. Licensing gate is a build requirement, not documentation.

APPROVED (default in this spec)
   ↓
DOWNLOAD
   ↓
VERIFY exact artifact
   ├── PASS → stay APPROVED (record SHA-256, filename, version, license)
   └── FAIL → REJECTED (never use / never ship)

If VERIFY fails: DO NOT LOAD, DO NOT SHIP, DO NOT SILENTLY SUBSTITUTE, MARK REJECTED.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

import yaml

from src.exceptions import (
    ModelNotApprovedError,
    ModelRegistryError,
    ModelRejectedError,
    ModelVerificationError,
)

logger = logging.getLogger(__name__)

COMPARE_FIELDS = ("name", "artifact", "version", "sha256", "status")


@dataclass
class ModelEntry:
    name: str
    stage: str
    source_url: str
    artifact: str
    version: str
    commit: str
    license: str
    commercial_use_verified: bool
    redistribution_verified: bool
    training_provenance_verified: bool
    sha256: str
    verification_date: str
    verification_source: str
    status: str
    local_path: Optional[str] = None
    notes: str = ""

    @property
    def approved(self) -> bool:
        return self.status.upper() == "APPROVED"

    def require_approved(self) -> None:
        if self.status.upper() == "REJECTED":
            raise ModelRejectedError(
                f"{self.name} is REJECTED. Do not load, ship, or substitute."
            )
        if not self.approved:
            raise ModelNotApprovedError(
                f"{self.name} status={self.status}. Production may only load APPROVED models."
            )


def _as_entry(raw: dict[str, Any]) -> ModelEntry:
    required = [
        "name",
        "stage",
        "source_url",
        "artifact",
        "version",
        "license",
        "sha256",
        "status",
    ]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ModelRegistryError(f"Model entry missing fields {missing}: {raw.get('name')}")
    return ModelEntry(
        name=raw["name"],
        stage=raw["stage"],
        source_url=raw["source_url"],
        artifact=raw["artifact"],
        version=str(raw.get("version", "")),
        commit=str(raw.get("commit", "")),
        license=raw["license"],
        commercial_use_verified=bool(raw.get("commercial_use_verified", False)),
        redistribution_verified=bool(raw.get("redistribution_verified", False)),
        training_provenance_verified=bool(raw.get("training_provenance_verified", False)),
        sha256=str(raw.get("sha256") or ""),
        verification_date=str(raw.get("verification_date") or ""),
        verification_source=str(raw.get("verification_source") or ""),
        status=str(raw["status"]).upper(),
        local_path=raw.get("local_path"),
        notes=str(raw.get("notes") or ""),
    )


def load_yaml_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise ModelRegistryError(f"Missing registry file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if isinstance(data, dict) and "models" in data:
        return list(data["models"])
    if isinstance(data, list):
        return data
    raise ModelRegistryError(f"Unrecognized registry format: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelRegistry:
    def __init__(
        self,
        models_yaml: Path,
        manifest_yaml: Path,
        models_dir: Path,
        fail_on_mismatch: bool = True,
    ) -> None:
        self.models_yaml = models_yaml
        self.manifest_yaml = manifest_yaml
        self.models_dir = models_dir
        runtime = [_as_entry(x) for x in load_yaml_list(models_yaml)]
        manifest = [_as_entry(x) for x in load_yaml_list(manifest_yaml)]
        self._runtime = {e.name: e for e in runtime}
        self._manifest = {e.name: e for e in manifest}
        if fail_on_mismatch:
            self._assert_agreement()

    def _assert_agreement(self) -> None:
        runtime_names = set(self._runtime)
        manifest_names = set(self._manifest)
        if runtime_names != manifest_names:
            raise ModelRegistryError(
                "Registry name mismatch between configs/models.yaml and "
                f"models/MANIFEST.yaml: {runtime_names ^ manifest_names}"
            )
        for name in runtime_names:
            a = self._runtime[name]
            b = self._manifest[name]
            for field in COMPARE_FIELDS:
                av = getattr(a, field)
                bv = getattr(b, field)
                if (av or "") != (bv or ""):
                    raise ModelRegistryError(
                        f"Registry disagreement on {name}.{field}: "
                        f"models.yaml={av!r} MANIFEST.yaml={bv!r}. "
                        "Production must fail fast."
                    )

    def get(self, name: str) -> ModelEntry:
        if name not in self._runtime:
            raise ModelRegistryError(f"Unknown model: {name}")
        return self._runtime[name]

    def by_stage(self, stage: str) -> ModelEntry:
        for entry in self._runtime.values():
            if entry.stage == stage:
                return entry
        raise ModelRegistryError(f"No model registered for stage={stage}")

    def require_loadable(self, name: str) -> ModelEntry:
        entry = self.get(name)
        entry.require_approved()
        return entry

    def resolve_path(self, entry: ModelEntry) -> Optional[Path]:
        if not entry.local_path:
            return None
        path = Path(entry.local_path)
        if not path.is_absolute():
            # relative to project root (parent of models/)
            path = self.models_dir.parent / path
        return path

    def verify_on_disk(self, name: str, *, require_file: bool = True) -> ModelEntry:
        """Post-download VERIFY of the exact artifact.

        Checks filename/artifact, recorded SHA-256, and APPROVED status.
        SHA mismatch or missing commercial license → RAISE (caller marks REJECTED).
        """
        entry = self.get(name)
        if entry.status.upper() == "REJECTED":
            raise ModelRejectedError(
                f"{name} is REJECTED. Do not load, ship, or substitute."
            )
        path = self.resolve_path(entry)
        if path is None:
            if require_file:
                raise ModelVerificationError(
                    f"{name} has no local_path; cannot VERIFY an on-disk artifact."
                )
            return entry
        if not path.exists():
            if require_file:
                raise ModelVerificationError(
                    f"{name} artifact missing on disk: {path}. Download the exact file."
                )
            return entry
        if path.name not in entry.artifact and entry.artifact not in path.name:
            # Allow RF-DETR package-managed names and Qualcomm export aliases.
            if name not in {"RF-DETR-L", "RTMPose-Body2d", "BoT-SORT"}:
                raise ModelVerificationError(
                    f"{name} filename {path.name} does not match artifact {entry.artifact}"
                )
        digest = sha256_file(path)
        if entry.sha256 and entry.sha256 != digest:
            raise ModelVerificationError(
                f"{name} SHA-256 mismatch. expected={entry.sha256} actual={digest}. "
                "MARK REJECTED. Do not load, ship, or silently substitute."
            )
        if not entry.license:
            raise ModelVerificationError(f"{name} has no weight-file license recorded.")
        restricted = ("non-commercial", "research-only", "cc-by-nc", "academic only")
        license_l = entry.license.lower()
        if any(tok in license_l for tok in restricted):
            raise ModelVerificationError(
                f"{name} weight license is not commercially usable: {entry.license}"
            )
        return entry

    def mark_rejected(self, name: str, reason: str) -> None:
        """Persist REJECTED in both registry files. Never leave APPROVED after failed VERIFY."""
        logger.error("MARK REJECTED %s: %s", name, reason)
        for path in (self.models_yaml, self.manifest_yaml):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            items = raw["models"] if isinstance(raw, dict) and "models" in raw else raw
            for item in items:
                if item.get("name") == name:
                    item["status"] = "REJECTED"
                    item["verification_date"] = str(date.today())
                    notes = item.get("notes") or ""
                    item["notes"] = f"{notes} REJECTED: {reason}".strip()
            if isinstance(raw, dict) and "models" in raw:
                path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            else:
                path.write_text(yaml.safe_dump(items, sort_keys=False), encoding="utf-8")
        if name in self._runtime:
            self._runtime[name].status = "REJECTED"
        if name in self._manifest:
            self._manifest[name].status = "REJECTED"

    def record_sha256(self, name: str, digest: str) -> None:
        for path in (self.models_yaml, self.manifest_yaml):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            items = raw["models"] if isinstance(raw, dict) and "models" in raw else raw
            for item in items:
                if item.get("name") == name:
                    item["sha256"] = digest
                    item["verification_date"] = str(date.today())
                    item["status"] = "APPROVED"
            if isinstance(raw, dict) and "models" in raw:
                path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            else:
                path.write_text(yaml.safe_dump(items, sort_keys=False), encoding="utf-8")
        if name in self._runtime:
            self._runtime[name].sha256 = digest
            self._runtime[name].verification_date = str(date.today())
        if name in self._manifest:
            self._manifest[name].sha256 = digest
            self._manifest[name].verification_date = str(date.today())

    def all_entries(self) -> list[ModelEntry]:
        return list(self._runtime.values())

from pathlib import Path

import pytest
import yaml

from src.exceptions import ModelRegistryError, ModelRejectedError, ModelVerificationError
from src.registry import ModelRegistry


def _write_pair(tmp_path: Path, status="APPROVED", sha=""):
    item = {
        "name": "YuNet",
        "stage": "face_detection",
        "source_url": "https://example.com",
        "artifact": "face_detection_yunet_2023mar.onnx",
        "version": "2023mar",
        "commit": "",
        "license": "MIT",
        "commercial_use_verified": True,
        "redistribution_verified": True,
        "training_provenance_verified": False,
        "sha256": sha,
        "verification_date": "",
        "verification_source": "",
        "status": status,
        "local_path": str(tmp_path / "face_detection_yunet_2023mar.onnx"),
    }
    models = tmp_path / "models.yaml"
    manifest = tmp_path / "MANIFEST.yaml"
    models.write_text(yaml.safe_dump({"models": [item]}), encoding="utf-8")
    manifest.write_text(yaml.safe_dump([item]), encoding="utf-8")
    return models, manifest


def test_agreement_required(tmp_path: Path):
    models, manifest = _write_pair(tmp_path)
    data = yaml.safe_load(manifest.read_text())
    data[0]["sha256"] = "abc"
    manifest.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ModelRegistryError):
        ModelRegistry(models, manifest, tmp_path)


def test_rejected_cannot_load(tmp_path: Path):
    models, manifest = _write_pair(tmp_path, status="REJECTED")
    reg = ModelRegistry(models, manifest, tmp_path)
    with pytest.raises(ModelRejectedError):
        reg.require_loadable("YuNet")


def test_sha_mismatch_fails_verify(tmp_path: Path):
    blob = tmp_path / "face_detection_yunet_2023mar.onnx"
    blob.write_bytes(b"weights")
    models, manifest = _write_pair(tmp_path, sha="0" * 64)
    # Rewrite local_path into yaml after pair write
    for path in (models, manifest):
        raw = yaml.safe_load(path.read_text())
        items = raw["models"] if isinstance(raw, dict) else raw
        items[0]["local_path"] = str(blob)
        items[0]["sha256"] = "0" * 64
        if isinstance(raw, dict):
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        else:
            path.write_text(yaml.safe_dump(items), encoding="utf-8")
    reg = ModelRegistry(models, manifest, tmp_path)
    with pytest.raises(ModelVerificationError):
        reg.verify_on_disk("YuNet")

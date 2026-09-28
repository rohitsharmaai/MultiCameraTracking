from pathlib import Path

from src.registry import ModelRegistry


def test_checked_in_registry_agrees():
    root = Path(__file__).resolve().parents[2]
    reg = ModelRegistry(
        root / "configs" / "models.yaml",
        root / "models" / "MANIFEST.yaml",
        root / "models",
    )
    names = {e.name for e in reg.all_entries()}
    assert names == {
        "RF-DETR-L",
        "BoT-SORT",
        "OSNet-x1.0",
        "YoutuReID",
        "RTMPose-Body2d",
        "PPHumanSeg",
        "YuNet",
        "AuraFace-v1",
    }
    for entry in reg.all_entries():
        assert entry.status in {"APPROVED", "REJECTED"}

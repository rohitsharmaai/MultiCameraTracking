#!/usr/bin/env python3
"""Download the exact approved artifacts. Do not substitute another checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.registry import ModelRegistry, sha256_file  # noqa: E402

DOWNLOADS = {
    "OSNet-x1.0": {
        "repo": "kaiyangzhou/osnet",
        "filename": "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
        "dest": ROOT / "models" / "osnet" / "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    },
    "YoutuReID": {
        "repo": "opencv/person_reid_youtureid",
        "filename": "person_reid_youtu_2021nov.onnx",
        "dest": ROOT / "models" / "youtureid" / "person_reid_youtu_2021nov.onnx",
    },
    "PPHumanSeg": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx",
        "dest": ROOT / "models" / "pphumanseg" / "human_segmentation_pphumanseg_2023mar.onnx",
    },
    "YuNet": {
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "dest": ROOT / "models" / "yunet" / "face_detection_yunet_2023mar.onnx",
    },
    "AuraFace-v1": {
        "repo": "fal/AuraFace-v1",
        "filename": "glintr100.onnx",
        "dest": ROOT / "models" / "auraface" / "glintr100.onnx",
    },
    "RTMPose-Body2d": {
        "url": "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models/models/rtmpose_body2d/releases/v0.63.0/rtmpose_body2d-onnx-float.zip",
        "unzip_onnx": True,
        "dest": ROOT / "models" / "rtmpose" / "rtmpose_body2d.onnx",
    },
}


def _hf_download(repo: str, filename: str, dest: Path) -> None:
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=repo, filename=filename)
    data = Path(path).read_bytes()
    dest.write_bytes(data)


def _http_download(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp:
        dest.write_bytes(resp.read())


def _download_zip_onnx(url: str, dest: Path) -> None:
    import tempfile
    import zipfile
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "model.zip"
        with urllib.request.urlopen(url) as resp:
            zpath.write_bytes(resp.read())
        with zipfile.ZipFile(zpath) as zf:
            onnxs = [n for n in zf.namelist() if n.endswith(".onnx") and "w8a16" not in n.lower()]
            if not onnxs:
                raise FileNotFoundError(f"No ONNX float file in {url}")
            chosen = sorted(onnxs, key=len)[0]
            dest.write_bytes(zf.read(chosen))
            print(f"  extracted {chosen}")
            # Qualcomm ONNX float keeps weights in a sidecar .data next to the graph.
            prefix = chosen.rsplit("/", 1)[0]
            stem = Path(chosen).stem
            for name in zf.namelist():
                if name == chosen or name.endswith("/"):
                    continue
                if prefix and not name.startswith(prefix + "/"):
                    continue
                if prefix and name.count("/") != chosen.count("/"):
                    continue
                base = Path(name).name
                if not (base.startswith(stem) or base.endswith(".data")):
                    continue
                out = dest.parent / base
                out.write_bytes(zf.read(name))
                print(f"  extracted {name} -> {out.name}")


def main() -> int:
    cfg = load_config()
    registry = ModelRegistry(
        ROOT / "configs" / "models.yaml",
        ROOT / "models" / "MANIFEST.yaml",
        cfg.models_dir,
    )
    for name, spec in DOWNLOADS.items():
        dest: Path = spec["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        sidecar = dest.with_suffix(".data")
        missing_sidecar = bool(spec.get("unzip_onnx")) and not sidecar.exists()
        if missing_sidecar and dest.exists():
            print(f"REDOWNLOAD {name}: missing sidecar {sidecar.name}")
        if dest.exists() and dest.stat().st_size > 0 and not missing_sidecar:
            digest = sha256_file(dest)
            if not registry.get(name).sha256:
                registry.record_sha256(name, digest)
            try:
                registry.verify_on_disk(name)
                print(f"  VERIFY PASS {name} stays APPROVED")
            except Exception as exc:
                print(f"  VERIFY FAIL {name}: {exc}")
                registry.mark_rejected(name, str(exc))
            continue
        print(f"Downloading {name} -> {dest}")
        try:
            if spec.get("unzip_onnx"):
                _download_zip_onnx(spec["url"], dest)
            elif "repo" in spec:
                _hf_download(spec["repo"], spec["filename"], dest)
            else:
                _http_download(spec["url"], dest)
        except Exception as exc:
            print(f"DOWNLOAD FAILED {name}: {exc}")
            print("MARK REJECTED is only applied after VERIFY of a downloaded file.")
            continue
        digest = sha256_file(dest)
        print(f"  sha256={digest}")
        registry.record_sha256(name, digest)
        try:
            registry.verify_on_disk(name)
            print(f"  VERIFY PASS {name} stays APPROVED")
        except Exception as exc:
            print(f"  VERIFY FAIL {name}: {exc}")
            registry.mark_rejected(name, str(exc))
            print("  DO NOT LOAD IN PRODUCTION / DO NOT SHIP / DO NOT SILENTLY SUBSTITUTE")
    print("RF-DETR-L weights are fetched by the rfdetr package (RFDETRLarge).")
    print("BoT-SORT is a tracker-only local implementation (no appearance weights).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Process one video or a synchronized multi-camera sequence.

python -m app.process_video --video data/Lab/4p-c0.avi --camera lab-c0
python -m app.process_video --run-name lab-4p --synchronized --fps 25 \
    --camera A=data/Lab/4p-c0.avi --camera B=data/Lab/4p-c1.avi
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.bootstrap import build_processor
from src.camera.ids import canonical_camera
from src.config import load_config
from src.logging_setup import setup_logging


def _parse_camera(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("camera must be CAMERA_ID=path")
    cam, path = value.split("=", 1)
    return cam.strip(), path.strip()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Process video file(s) into the persistent gallery")
    p.add_argument("--input", "--video", dest="input", help="Single video file")
    p.add_argument(
        "--camera",
        action="append",
        default=[],
        help="Camera id for --video, or CAMERA_ID=path (repeatable)",
    )
    p.add_argument("--synchronized", action="store_true", help="Shared clock across cameras")
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--camera-id", default="A", help="Camera id for --input/--video")
    p.add_argument("--no-weights", action="store_true", help="Skip loading vision weights (tests)")
    return p


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    cameras: dict[str, str] = {}
    camera_id = args.camera_id
    for raw in args.camera:
        if "=" in raw:
            cam, path = _parse_camera(raw)
            cameras[canonical_camera(cam)] = path
        else:
            camera_id = canonical_camera(raw)
    camera_id = canonical_camera(camera_id)
    if args.input:
        cameras.setdefault(camera_id, args.input)
    if not cameras:
        print("Provide --video PATH or --camera ID=path", file=sys.stderr)
        return 2
    for path in cameras.values():
        if not Path(path).exists():
            print(f"Video not found: {path}", file=sys.stderr)
            return 2
    cfg = load_config()
    processor = build_processor(cfg, load_weights=not args.no_weights)
    if args.synchronized or len(cameras) > 1:
        stats = processor.process_synchronized(
            cameras,
            fps=args.fps,
            run_name=args.run_name,
            max_frames=args.max_frames,
        )
    else:
        cam, path = next(iter(cameras.items()))
        stats = processor.process_single(
            path,
            camera_id=cam,
            fps=args.fps,
            run_name=args.run_name,
            max_frames=args.max_frames,
        )
    logging.getLogger(__name__).info(
        "completed run_id=%s frames=%s tracks=%s ids=%s",
        stats.run_id,
        stats.frames_processed,
        stats.tracks_detected,
        ",".join(stats.global_ids_seen),
    )
    print(f"run_id={stats.run_id}")
    print(f"frames={stats.frames_processed} tracks={stats.tracks_detected}")
    print(f"global_ids={','.join(stats.global_ids_seen) or '-'}")
    print(f"new={stats.new_global_ids_created} reused={stats.existing_global_ids_reused}")
    print(f"outputs={processor.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

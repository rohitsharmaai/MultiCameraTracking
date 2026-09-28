"""Inspect or reset the persistent identity gallery.

python -m app.gallery --list
python -m app.gallery --show P0002
python -m app.gallery --reset
"""

from __future__ import annotations

import argparse
import sys

from src.bootstrap import build_manager, build_store
from src.config import load_config
from src.logging_setup import setup_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspect or reset the identity gallery")
    p.add_argument("--list", action="store_true")
    p.add_argument("--show", metavar="GLOBAL_ID")
    p.add_argument("--reset", action="store_true", help="DESTRUCTIVE: delete all identities")
    p.add_argument("--yes", action="store_true", help="Confirm --reset")
    return p


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    args = build_parser().parse_args(argv)
    cfg = load_config()
    store = build_store(cfg)
    manager = build_manager(cfg, store=store)
    if args.reset:
        if not args.yes:
            print("Refusing to reset without --yes", file=sys.stderr)
            return 2
        manager.reset()
        print("gallery reset")
        return 0
    if args.show:
        gal = manager.show_person(args.show)
        if gal is None:
            print(f"unknown id {args.show}", file=sys.stderr)
            return 1
        print(f"{gal.global_person_id} status={gal.status} last_camera={gal.last_camera_id}")
        print(f"  body embeddings: {len(gal.body)}")
        for e in gal.body:
            print(
                f"    cam={e.camera_id} q={e.quality_score:.2f} vp={e.viewpoint.value} "
                f"src={e.source_video_name} frame={e.source_frame_number} "
                f"model={e.model_name}"
            )
        print(f"  face embeddings: {len(gal.face)}")
        for e in gal.face:
            print(
                f"    cam={e.camera_id} q={e.quality_score:.2f} "
                f"src={e.source_video_name} frame={e.source_frame_number}"
            )
        return 0
    # default --list
    persons = manager.list_persons()
    if not persons:
        print("gallery empty")
        return 0
    for gal in persons:
        videos = sorted(
            {e.source_video_name for e in gal.body + gal.face if e.source_video_name}
        )
        print(
            f"{gal.global_person_id} body={len(gal.body)} face={len(gal.face)} "
            f"last={gal.last_camera_id} videos={','.join(videos)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Audit helper for identity evidence.

Two jobs:

1. ``find_similar_pairs`` / ``save_pairs`` - identities that got DIFFERENT
   Global IDs but look like the same person (similar body and/or face). Saves
   the most-similar crop from each track into
       similar_identities/<A>_vs_<B>/{body,face}/
   plus summary.json + notes.txt, so the operator can see *which track from
   which Global ID* matched which, and why they were kept apart.

2. ``audit_identity`` - for one identity that has multiple tracks, saves the
   most-similar body/face pair between those tracks, so the operator can see
   why they were given the SAME Global ID.

Nothing here deletes or merges; it only writes evidence.
"""

from __future__ import annotations

import json
import re
import shutil
from itertools import combinations
from pathlib import Path
from typing import Any

from src.reid.colour import appearance_similarity


def _rec_brief(rec, crop_frame: int | None = None) -> dict[str, Any]:
    brief = {
        "global_id": rec.global_person_id,
        "track": rec.track_id,
        "frame": rec.source_frame_number,
        "video": rec.source_video_name,
        "viewpoint": rec.viewpoint.value,
        "quality": rec.quality_score,
    }
    # The saved crop can come from a nearby frame of the same track when no
    # crop was written for the embedding's exact frame; say so.
    brief["crop_frame"] = crop_frame
    return brief


def _track_key(rec) -> tuple[str, str]:
    """Local track IDs restart per video: identify a track by (video, track)."""
    video = Path(rec.source_video_name or rec.source_video_id or "").stem
    return video, rec.track_id


def _best_cross(galleries: list, modality: str):
    """Best (cos, ra, rb) between two galleries for one modality.

    Only compares records from DIFFERENT local tracks, so the evidence really
    shows "this track from <A> vs that different track from <B>".
    """
    if len(galleries) != 2:
        return None
    a, b = galleries
    recs_a, recs_b = a.items(modality), b.items(modality)
    best = None
    for ra in recs_a:
        for rb in recs_b:
            if ra.track_id == rb.track_id:
                continue
            s = float(appearance_similarity(ra.embedding, rb.embedding))
            if best is None or s > best[0]:
                best = (s, ra, rb)
    return best


def find_similar_pairs(
    galleries: dict[str, Any],
    body_threshold: float = 0.62,
    face_threshold: float = 0.45,
    include_inactive: bool = True,
    distinct_pairs: set | None = None,
) -> list[dict[str, Any]]:
    """Pairs of identities (different Global IDs) that look like one person.

    ``distinct_pairs`` holds ID pairs that were on screen together in one
    frame; such a pair is still listed (it looks alike) but flagged
    ``seen_together`` because it cannot be one person.
    """
    items = [
        (gid, gal)
        for gid, gal in galleries.items()
        if include_inactive or getattr(gal, "status", "active") == "active"
    ]
    pairs: list[dict[str, Any]] = []
    for (a, ga), (b, gb) in combinations(sorted(items, key=lambda x: x[0]), 2):
        entry: dict[str, Any] = {
            "a": a, "b": b, "body": None, "face": None,
            "seen_together": frozenset((a, b)) in (distinct_pairs or set()),
        }
        keep = False
        for modality, thr in (("body", body_threshold), ("face", face_threshold)):
            best = _best_cross([ga, gb], modality)
            if best is None:
                continue
            s, ra, rb = best
            entry[modality] = {
                "cosine": s,
                "threshold": thr,
                "similar": s >= thr,
                "ra": ra,
                "rb": rb,
            }
            if s >= thr:
                keep = True
        if keep:
            pairs.append(entry)
    pairs.sort(
        key=lambda p: max(
            (p["body"]["cosine"] if p["body"] else 0.0),
            (p["face"]["cosine"] if p["face"] else 0.0),
        ),
        reverse=True,
    )
    return pairs


_SNAP_RE = re.compile(r"^(?:.+__)?(body|face)_(.*)_(\d+)_([^_]+_\d+)$")


def parse_snapshot_name(stem: str) -> tuple[str, str, int, str] | None:
    """``<modality>_<video stem>_<frame>_<track>`` -> parts, or None.

    The video stem may itself contain underscores; the track is
    ``<camera>_<n>``. A ``<gid>__`` prefix (added by a merge) is ignored.
    """
    m = _SNAP_RE.match(stem)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3)), m.group(4)


def _crop_file(snapshots_root: Path, rec, modality: str = "body") -> tuple[Path, int] | None:
    """Saved crop for an embedding record and the frame it actually shows.

    Prefers the exact frame. Otherwise uses the nearest saved frame of the
    same track *in the same video* - never another video's track that
    happens to reuse the same local ID.
    """
    stem = Path(rec.source_video_name or "").stem
    person = snapshots_root / rec.global_person_id
    if not person.exists():
        return None
    frame = rec.source_frame_number
    exact = person / f"{modality}_{stem}_{frame}_{rec.track_id}.jpg"
    if exact.exists():
        return exact, int(frame)
    best: tuple[int, Path, int] | None = None
    for f in person.glob(f"*{modality}_*_{rec.track_id}.jpg"):
        parts = parse_snapshot_name(f.stem)
        if parts is None or parts[0] != modality or parts[3] != rec.track_id:
            continue
        if stem and parts[1] != stem:
            continue
        dist = abs(parts[2] - int(frame)) if frame is not None else 0
        if best is None or dist < best[0]:
            best = (dist, f, parts[2])
    if best is None:
        return None
    return best[1], best[2]


def save_pairs(
    pairs: list[dict[str, Any]],
    snapshots_root: Path,
    out_root: Path,
) -> list[dict[str, Any]]:
    out_root.mkdir(parents=True, exist_ok=True)
    saved: list[dict[str, Any]] = []
    for p in pairs:
        a, b = p["a"], p["b"]
        pair_name = f"{a}_vs_{b}"
        folder = out_root / pair_name
        folder.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {
            "pair": pair_name, "a": a, "b": b, "body": None, "face": None,
            "seen_together": bool(p.get("seen_together")),
        }
        notes: list[str] = []

        for modality in ("body", "face"):
            info = p.get(modality)
            if not info:
                continue
            ra, rb = info["ra"], info["rb"]
            sub = folder / modality
            sub.mkdir(exist_ok=True)
            images: dict[str, str] = {}
            frames: dict[str, int | None] = {"a": None, "b": None}
            for side, gid_side, rec in (("a", a, ra), ("b", b, rb)):
                found = _crop_file(snapshots_root, rec, modality)
                if found is None:
                    continue
                src, frames[side] = found
                name = f"{gid_side}__{rec.track_id}__f{frames[side]}.jpg"
                shutil.copy2(src, sub / name)
                images[side] = f"{pair_name}/{modality}/{name}"

            entry[modality] = {
                "cosine": round(info["cosine"], 4),
                "similar": info["similar"],
                "threshold": info["threshold"],
                "a": _rec_brief(ra, frames["a"]),
                "b": _rec_brief(rb, frames["b"]),
                "image_a": f"/gui/duplicates/{images['a']}" if "a" in images else None,
                "image_b": f"/gui/duplicates/{images['b']}" if "b" in images else None,
            }
            notes.append(
                f"{modality.upper()}: {a}/{ra.track_id} f{ra.source_frame_number} "
                f"<-> {b}/{rb.track_id} f{rb.source_frame_number} "
                f"cosine={info['cosine']:.3f} "
                f"({'SIMILAR' if info['similar'] else 'not similar'} vs {info['threshold']})"
            )

        heading = (
            f"Different Global IDs ({a} vs {b}) but the tracks look like the "
            f"same person.\nThese images are the most similar crop of each track.\n"
        )
        if entry["seen_together"]:
            heading += (
                f"NOTE: {a} and {b} were on screen together in the same frame, "
                "so they are two different people. Do not merge.\n"
            )
        (folder / "summary.json").write_text(
            json.dumps(
                {"pair": pair_name, **{k: entry[k] for k in ("body", "face", "seen_together")}},
                indent=2,
            ),
            encoding="utf-8",
        )
        (folder / "notes.txt").write_text(heading + "\n".join(notes) + "\n", encoding="utf-8")
        saved.append(entry)
    return saved


def _copy_pair(
    snapshots_root: Path, folder: Path, gid: str, prefix: str, modality: str, ra, rb
) -> tuple[dict[str, str], dict[str, int | None]]:
    images: dict[str, str] = {}
    frames: dict[str, int | None] = {"a": None, "b": None}
    for side, rec in (("a", ra), ("b", rb)):
        found = _crop_file(snapshots_root, rec, modality)
        if found is None:
            continue
        src, frames[side] = found
        dest = folder / f"{prefix}{modality}_{rec.track_id}_f{frames[side]}.jpg"
        shutil.copy2(src, dest)
        images[side] = f"{gid}/{dest.name}"
    return images, frames


def _pair_entry(s: float, threshold: float, ra, rb, images, frames) -> dict[str, Any]:
    return {
        "cosine": round(s, 4),
        "similar": s >= threshold,
        "threshold": threshold,
        "a": _rec_brief(ra, frames["a"]),
        "b": _rec_brief(rb, frames["b"]),
        "image_a": f"/gui/identity_audit/{images['a']}" if "a" in images else None,
        "image_b": f"/gui/identity_audit/{images['b']}" if "b" in images else None,
    }


def _best_between(records_a: list, records_b: list):
    best = None
    for ra in records_a:
        for rb in records_b:
            s = float(appearance_similarity(ra.embedding, rb.embedding))
            if best is None or s > best[0]:
                best = (s, ra, rb)
    return best


def audit_identity(
    gallery,
    snapshots_root: Path,
    out_root: Path,
    body_threshold: float = 0.62,
    face_threshold: float = 0.45,
    cooccurring_pairs: dict | None = None,
) -> dict[str, Any]:
    """For one identity, save the most-similar body/face pair between its
    DIFFERENT local tracks (why they share the same Global ID).

    A track is identified by (video, local track ID): ``A_1`` of one video
    and ``A_1`` of another are different tracks.
    """
    gid = gallery.global_person_id
    folder = out_root / gid
    folder.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"global_person_id": gid, "body": None, "face": None}

    by_track: dict[str, dict[tuple[str, str], list]] = {"body": {}, "face": {}}
    for modality in ("body", "face"):
        for r in gallery.items(modality):
            by_track[modality].setdefault(_track_key(r), []).append(r)

    for modality, threshold in (("body", body_threshold), ("face", face_threshold)):
        tracks = by_track[modality]
        best = None
        for ka, kb in combinations(sorted(tracks), 2):
            cand = _best_between(tracks[ka], tracks[kb])
            if cand is not None and (best is None or cand[0] > best[0]):
                best = cand
        if best is None:
            continue
        s, ra, rb = best
        images, frames = _copy_pair(snapshots_root, folder, gid, "", modality, ra, rb)
        result[modality] = _pair_entry(s, threshold, ra, rb, images, frames)

    # Co-occurring tracks: two different tracks of this SAME identity that are
    # on screen at the same time (in the same video). Candidate pairs come
    # from the review timeline (frames where both tracks share this Global ID)
    # when provided, otherwise from overlapping persisted frame ranges. Show
    # the pair with the highest similarity - one image from each track.
    pairs_ranges: dict[tuple, tuple[int, int]] = {}
    for key, span in (cooccurring_pairs or {}).items():
        if len(key) == 3:  # (video, track_a, track_b)
            video, ta, tb = key
            pairs_ranges[((video, ta), (video, tb))] = tuple(span)

    ranges: dict[tuple[str, str], tuple[int, int]] = {}
    for r in gallery.body + gallery.face:
        f = r.source_frame_number
        if f is None:
            continue
        k = _track_key(r)
        lo, hi = ranges.get(k, (f, f))
        ranges[k] = (min(lo, f), max(hi, f))
    for ka, kb in combinations(sorted(ranges), 2):
        if ka[0] != kb[0]:
            continue  # frame numbers of different videos are not comparable
        a, b = ranges[ka], ranges[kb]
        if max(a[0], b[0]) <= min(a[1], b[1]):
            pairs_ranges.setdefault((ka, kb), (max(a[0], b[0]), min(a[1], b[1])))

    co: dict[str, Any] = {
        "pair": None, "video": None, "overlap_frames": None, "body": None, "face": None,
    }
    co_best = None
    for (ka, kb), span in pairs_ranges.items():
        pair_best = None
        pairs: dict[str, Any] = {}
        for modality, threshold in (("body", body_threshold), ("face", face_threshold)):
            best = _best_between(
                by_track[modality].get(ka, []), by_track[modality].get(kb, [])
            )
            if best is not None:
                pairs[modality] = (best, threshold)
                if pair_best is None or best[0] > pair_best:
                    pair_best = best[0]
        if pair_best is not None and (co_best is None or pair_best > co_best[0]):
            co_best = (pair_best, ka, kb, pairs, span)

    if co_best is not None:
        _, ka, kb, pairs, span = co_best
        co["pair"] = [ka[1], kb[1]]
        co["video"] = ka[0]
        co["overlap_frames"] = [span[0], span[1]]
        for modality, (best, threshold) in pairs.items():
            s, ra, rb = best
            images, frames = _copy_pair(snapshots_root, folder, gid, "co_", modality, ra, rb)
            co[modality] = _pair_entry(s, threshold, ra, rb, images, frames)
    result["cooccurring"] = co

    (folder / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def cooccurring_pairs_from_review(review_root, gid: str, max_jobs: int = 3) -> dict:
    """Tracks of one Global ID that were on screen at the same time.

    Reads the GUI review timeline (per-frame people) and returns
    {(video_stem, trackA, trackB): (first_frame, last_frame)}.
    """
    review_root = Path(review_root)
    if not review_root.exists():
        return {}
    jobs = [d for d in review_root.iterdir() if d.is_dir()]
    jobs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    pairs: dict[tuple[str, str, str], tuple[int, int]] = {}
    for job in jobs[:max_jobs]:
        timeline = job / "timeline.jsonl"
        if not timeline.exists():
            continue
        try:
            lines = timeline.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            f = row.get("frame")
            if f is None:
                continue
            by_video: dict[str, set[str]] = {}
            for p in row.get("people", []):
                if p.get("global_id") == gid and p.get("local_track"):
                    video = Path(p.get("video_name") or "").stem
                    by_video.setdefault(video, set()).add(p["local_track"])
            for video, tracks in by_video.items():
                for ta, tb in combinations(sorted(tracks), 2):
                    lo, hi = pairs.get((video, ta, tb), (f, f))
                    pairs[(video, ta, tb)] = (min(lo, f), max(hi, f))
    return pairs

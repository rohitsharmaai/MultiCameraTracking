#!/usr/bin/env python3
"""Explain why two body crops get a high (or low) body similarity.

The production path compares bodies with
``cosine_similarity(body_embedding_a, body_embedding_b)`` where each embedding
comes from ``OSNetEncoder.embed(crop, mask=segmenter.run(crop).mask)`` -- see
``src/pipeline/features.py`` and ``src/identity/scoring.py``.  This script runs
the same method on two saved crop JPEGs.  The result is close to, not exactly,
the stored value: saved crops are JPEG re-encoded (and downscaled when wider
than 160 px) and the audit may show a nearby frame of the track.  It then
breaks the result down so it is obvious *what* makes the pair look similar.
Every comparison below uses the segmentation mask, like production:

  * masked cosine (the production value) and no-mask cosine
  * person-only cosine (mask bounding box, background zeroed) vs
    background-only cosine (person zeroed)
  * upper-body vs lower-body cosine
  * horizontal-flip cosine (is the match pose direction sensitive?)
  * mean BGR and a 3-D colour-histogram correlation of the person pixels
  * baselines: each crop vs other tracks of the same identity, and vs crops
    of *other* identities (the scores two different people usually get)

Example::

    PYTHONPATH=. python scripts/compare_body_pair.py \\
        --a ~/.multicam_reid/identity_audit/P0001/co_body_A_1_f439.jpg \\
        --b ~/.multicam_reid/identity_audit/P0001/co_body_A_4_f744.jpg

Writes ``report.json`` (and an annotated contact sheet) to ``--out`` so the
finding can be attached to an audit.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import yaml

from src.registry import ModelRegistry
from src.reid.normalize import cosine_similarity
from src.reid.osnet import OSNetEncoder
from src.segmentation.pphumanseg import PPHumanSeg

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Audit crop names: [co_]<modality>_<track>_f<frame>.jpg
_AUDIT_NAME = re.compile(r"^(?:co_)?(body|face)_(.+)_f(\d+)$")

DEFAULT_A = Path.home() / ".multicam_reid/identity_audit/P0001/co_body_A_1_f439.jpg"
DEFAULT_B = Path.home() / ".multicam_reid/identity_audit/P0001/co_body_A_4_f744.jpg"


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"could not read image: {path}")
    return img


def _embed(encoder: OSNetEncoder, img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    return encoder.embed(img, mask=mask)


def _cos_vec(a: np.ndarray, b: np.ndarray) -> float:
    return round(float(cosine_similarity(a, b)), 4)


def _cos(
    encoder: OSNetEncoder,
    a: np.ndarray,
    b: np.ndarray,
    mask_a: np.ndarray | None = None,
    mask_b: np.ndarray | None = None,
) -> float:
    return _cos_vec(_embed(encoder, a, mask_a), _embed(encoder, b, mask_b))


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _mean_bgr(img: np.ndarray, mask: np.ndarray | None = None) -> list[float]:
    pixels = img[mask > 0] if mask is not None and (mask > 0).any() else img.reshape(-1, 3)
    return [round(float(v), 1) for v in pixels.mean(axis=0)]


def _hist_correlation(
    a: np.ndarray,
    b: np.ndarray,
    mask_a: np.ndarray | None = None,
    mask_b: np.ndarray | None = None,
    bins: int = 8,
) -> float:
    ranges = [0, 256] * 3
    ma = None if mask_a is None else (mask_a > 0).astype(np.uint8)
    mb = None if mask_b is None else (mask_b > 0).astype(np.uint8)
    ha = cv2.calcHist([a], [0, 1, 2], ma, [bins] * 3, ranges)
    hb = cv2.calcHist([b], [0, 1, 2], mb, [bins] * 3, ranges)
    ha = cv2.normalize(ha, ha).flatten()
    hb = cv2.normalize(hb, hb).flatten()
    return round(float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)), 4)


def _audit_track(path: Path) -> str | None:
    m = _AUDIT_NAME.match(path.stem)
    return m.group(2) if m else None


def _config_thresholds() -> dict:
    try:
        return yaml.safe_load((PROJECT_ROOT / "configs/thresholds.yaml").read_text()) or {}
    except OSError:
        return {}


def _stored_cosine(image_a: Path) -> float | None:
    """Read the cosine the audit already recorded, when summary.json is nearby.

    ``co_body_*`` crops belong to the ``cooccurring`` section; everything else
    (the best pair across tracks) belongs to ``body``.
    """
    summary = image_a.parent / "summary.json"
    if not summary.exists():
        return None
    try:
        data = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if image_a.name.startswith("co_"):
        section = (data.get("cooccurring") or {}).get("body")
    else:
        section = data.get("body")
    if isinstance(section, dict) and "cosine" in section:
        return float(section["cosine"])
    return None


def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    v = sorted(values)
    return {"n": len(v), "min": v[0], "median": v[len(v) // 2], "max": v[-1]}


def _baseline(
    encoder: OSNetEncoder,
    segmenter: PPHumanSeg,
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    a_path: Path,
    b_path: Path,
) -> dict:
    """Masked cosine of each input against two reference sets.

    * ``same_identity``: body crops of *other* tracks in the same audit folder
      (same Global ID). Crops of the input's own track are skipped - they are
      the same person by construction and would inflate the numbers.
    * ``other_identities``: body crops from sibling identity folders, i.e.
      what two different Global IDs typically score.

    Byte-identical copies of the inputs are skipped as well.
    """
    raw_a, raw_b = a_path.read_bytes(), b_path.read_bytes()
    own_tracks = {_audit_track(a_path), _audit_track(b_path)} - {None}
    folder = a_path.parent

    def embed_all(paths: list[Path]) -> list[tuple[str, np.ndarray]]:
        out = []
        for p in paths:
            data = p.read_bytes()
            if data in (raw_a, raw_b):
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            out.append((str(p.relative_to(folder.parent)), _embed(encoder, img, segmenter.run(img).mask)))
        return out

    same = embed_all(sorted(
        p for p in folder.glob("*body*.jpg")
        if p not in (a_path, b_path) and _audit_track(p) not in own_tracks
    ))
    others = embed_all(sorted(
        p for d in folder.parent.iterdir() if d.is_dir() and d != folder
        for p in d.glob("*body*.jpg")
    ))
    result = {}
    for name, refs in (("same_identity", same), ("other_identities", others)):
        scores = [
            {"name": n, "vs_a": _cos_vec(emb_a, e), "vs_b": _cos_vec(emb_b, e)} for n, e in refs
        ]
        result[name] = {
            "vs_a": _stats([s["vs_a"] for s in scores]),
            "vs_b": _stats([s["vs_b"] for s in scores]),
            "scores": scores,
        }
    return result


def analyze(a_path: Path, b_path: Path, threshold: float, out_dir: Path) -> dict:
    root = PROJECT_ROOT
    registry = ModelRegistry(
        root / "configs/models.yaml", root / "models/MANIFEST.yaml", root / "models"
    )
    encoder = OSNetEncoder(registry)
    encoder.load()
    segmenter = PPHumanSeg(registry)
    segmenter.load()

    a = _load_image(a_path)
    b = _load_image(b_path)
    seg_a = segmenter.run(a)
    seg_b = segmenter.run(b)
    ma, mb = seg_a.mask, seg_b.mask

    ea_masked = _embed(encoder, a, ma)
    eb_masked = _embed(encoder, b, mb)
    masked = _cos_vec(ea_masked, eb_masked)
    plain = _cos(encoder, a, b)

    # Person-only: crop to the mask bounding box, background inside it zeroed.
    person_only: float | None = None
    bb_a, bb_b = _mask_bbox(ma), _mask_bbox(mb)
    if bb_a and bb_b:
        ax1, ay1, ax2, ay2 = bb_a
        bx1, by1, bx2, by2 = bb_b
        person_only = _cos(
            encoder,
            a[ay1:ay2, ax1:ax2], b[by1:by2, bx1:bx2],
            ma[ay1:ay2, ax1:ax2], mb[by1:by2, bx1:bx2],
        )

    # Background-only: zero the person, keep the surroundings.
    bg_only: float | None = None
    if (ma > 0).any() and (mb > 0).any():
        ba = a.copy()
        ba[ma > 0] = 0
        bb = b.copy()
        bb[mb > 0] = 0
        bg_only = _cos(encoder, ba, bb)

    # Upper / lower body split (masked, like production).
    ha, hb = a.shape[0] // 2, b.shape[0] // 2
    upper = _cos(encoder, a[:ha], b[:hb], ma[:ha], mb[:hb])
    lower = _cos(encoder, a[ha:], b[hb:], ma[ha:], mb[hb:])

    flip_b = _cos(encoder, a, cv2.flip(b, 1), ma, cv2.flip(mb, 1))
    flip_a = _cos(encoder, cv2.flip(a, 1), b, cv2.flip(ma, 1), mb)

    min_cov = float(
        (_config_thresholds().get("observation") or {}).get("min_mask_coverage", 0.25)
    )
    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "image_a": str(a_path),
        "image_b": str(b_path),
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "production_body_method": {
            "formula": (
                "cosine(OSNet(crop_a, mask=PPHumanSeg(a)), "
                "OSNet(crop_b, mask=PPHumanSeg(b)))"
            ),
            "body_cosine_masked": masked,
            "body_cosine_no_mask": plain,
            "threshold": threshold,
            "similar": masked >= threshold,
            "mask_coverage_a": round(float(seg_a.coverage), 4),
            "mask_coverage_b": round(float(seg_b.coverage), 4),
            # Production makes no body embedding below this coverage.
            "min_mask_coverage": min_cov,
            "passes_mask_gate_a": float(seg_a.coverage) >= min_cov,
            "passes_mask_gate_b": float(seg_b.coverage) >= min_cov,
        },
        "stored_audit_cosine": _stored_cosine(a_path),
        "explain": {
            "person_only_cosine": person_only,
            "background_only_cosine": bg_only,
            "upper_body_cosine": upper,
            "lower_body_cosine": lower,
            "a_vs_flip_b_cosine": flip_b,
            "flip_a_vs_b_cosine": flip_a,
            "mean_bgr_person_a": _mean_bgr(a, ma),
            "mean_bgr_person_b": _mean_bgr(b, mb),
            "colour_histogram_correlation_person": _hist_correlation(a, b, ma, mb),
        },
        "baselines": _baseline(encoder, segmenter, ea_masked, eb_masked, a_path, b_path),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _save_contact_sheet(a, b, ma, mb, out_dir)
    return report


def _save_contact_sheet(
    a: np.ndarray, b: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray, out_dir: Path
) -> None:
    h = max(a.shape[0], b.shape[0])

    def fit(img: np.ndarray) -> np.ndarray:
        scale = h / img.shape[0]
        return cv2.resize(img, (max(1, int(img.shape[1] * scale)), h))

    overlay_a = a.copy()
    overlay_a[mask_a > 0] = (
        0.5 * overlay_a[mask_a > 0] + 0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)
    overlay_b = b.copy()
    overlay_b[mask_b > 0] = (
        0.5 * overlay_b[mask_b > 0] + 0.5 * np.array([0, 255, 0])
    ).astype(np.uint8)
    sheet = np.hstack([fit(a), fit(b), fit(overlay_a), fit(overlay_b)])
    cv2.imwrite(str(out_dir / "contact_sheet.jpg"), sheet)


def _print_report(report: dict) -> None:
    p = report["production_body_method"]
    e = report["explain"]
    print()
    print(f"A: {report['image_a']}  {tuple(report['shape_a'])}")
    print(f"B: {report['image_b']}  {tuple(report['shape_b'])}")
    print()
    print("== Production body method (same as pipeline) ==")
    print(f"  masked cosine : {p['body_cosine_masked']:.4f}  (threshold {p['threshold']}) "
          f"-> {'SIMILAR' if p['similar'] else 'not similar'}")
    print(f"  no-mask cosine: {p['body_cosine_no_mask']:.4f}")
    print(f"  mask coverage : A={p['mask_coverage_a']}  B={p['mask_coverage_b']}  "
          f"(production gate {p['min_mask_coverage']}: "
          f"A {'ok' if p['passes_mask_gate_a'] else 'FAILS'}, "
          f"B {'ok' if p['passes_mask_gate_b'] else 'FAILS'})")
    if report["stored_audit_cosine"] is not None:
        print(f"  stored audit  : {report['stored_audit_cosine']:.4f}")
    print()
    print("== Why? (all masked) ==")
    print(f"  person-only cosine     : {e['person_only_cosine']}")
    print(f"  background-only cosine : {e['background_only_cosine']}")
    print(f"  upper-body cosine      : {e['upper_body_cosine']}")
    print(f"  lower-body cosine      : {e['lower_body_cosine']}")
    print(f"  A vs flip(B)           : {e['a_vs_flip_b_cosine']}")
    print(f"  flip(A) vs B           : {e['flip_a_vs_b_cosine']}")
    print(f"  mean BGR person A / B  : {e['mean_bgr_person_a']} / {e['mean_bgr_person_b']}")
    print(f"  colour hist correlation: {e['colour_histogram_correlation_person']}")
    labels = {
        "same_identity": "other tracks, same identity",
        "other_identities": "other identities (different people)",
    }
    for key, label in labels.items():
        base = report["baselines"].get(key) or {}
        if not base.get("vs_a"):
            continue
        print()
        print(f"== Baseline vs {label} ==")
        for side in ("vs_a", "vs_b"):
            st = base[side]
            print(f"  {side[-1].upper()}: n={st['n']} min={st['min']} "
                  f"median={st['median']} max={st['max']}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--a", type=Path, default=DEFAULT_A, help="first body crop JPEG")
    ap.add_argument("--b", type=Path, default=DEFAULT_B, help="second body crop JPEG")
    default_thr = float(
        (_config_thresholds().get("reid") or {}).get("body_similarity_threshold", 0.62)
    )
    ap.add_argument("--threshold", type=float, default=default_thr,
                    help="body similarity threshold (default: configs/thresholds.yaml)")
    ap.add_argument("--out", type=Path, default=None,
                    help="folder for report.json and contact_sheet.jpg "
                         "(default: data/reid_audit/body_pair/<a>_vs_<b>)")
    args = ap.parse_args()
    if args.out is None:
        args.out = (
            PROJECT_ROOT / "data/reid_audit/body_pair"
            / f"{args.a.parent.name}_{args.a.stem}_vs_{args.b.stem}"
        )

    report = analyze(args.a, args.b, args.threshold, args.out)
    _print_report(report)
    print(f"wrote {args.out / 'report.json'}")
    print(f"wrote {args.out / 'contact_sheet.jpg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

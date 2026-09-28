"""Async-path feature extraction: crop -> PPHumanSeg -> RTMPose -> OSNet / YuNet / AuraFace.

Do not generate a body embedding from a crop that fails the mask or pose quality gate.
Face quality is independent of body quality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.face.align import align_face
from src.observation.crop import extract_crop
from src.reid.colour import fuse, part_colour
from src.types import LocalTrack, Viewpoint

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFeatures:
    body_embedding: Optional[list[float]]
    face_embedding: Optional[list[float]]
    body_quality: float
    face_quality: float
    viewpoint: Viewpoint
    mask_coverage: float
    occlusion_score: float
    pose_score: float
    body_visibility: float
    crop: Optional[np.ndarray]
    face_crop: Optional[np.ndarray] = None


def overlap_fraction(box, others) -> float:
    """Fraction of ``box`` covered by the union of ``others``."""
    w = int(np.ceil(box.width()))
    h = int(np.ceil(box.height()))
    if w <= 0 or h <= 0 or not others:
        return 0.0
    covered = np.zeros((h, w), dtype=bool)
    for o in others:
        x1 = int(np.floor(max(o.x1, box.x1) - box.x1))
        y1 = int(np.floor(max(o.y1, box.y1) - box.y1))
        x2 = int(np.ceil(min(o.x2, box.x2) - box.x1))
        y2 = int(np.ceil(min(o.y2, box.y2) - box.y1))
        if x2 > x1 and y2 > y1:
            covered[max(0, y1):y2, max(0, x1):x2] = True
    return float(covered.mean())


def _ready(model) -> bool:
    if model is None:
        return False
    if getattr(model, "_session", None) is not None:
        return True
    if getattr(model, "_model", None) is not None:
        return True
    path = getattr(model, "_path", None)
    return path is not None and Path(path).exists()


class FeatureExtractor:
    def __init__(
        self,
        segmenter=None,
        pose=None,
        body_encoder=None,
        face_detector=None,
        face_encoder=None,
        min_mask_coverage: float = 0.25,
        min_pose_score: float = 0.08,
        min_face_quality: float = 0.45,
        require_unique_face: bool = False,
        max_body_overlap: float = 1.0,
        min_face_px: float = 0.0,
        use_colour: bool = False,
    ) -> None:
        self.segmenter = segmenter
        self.pose = pose
        self.body_encoder = body_encoder
        self.face_detector = face_detector
        self.face_encoder = face_encoder
        self.min_mask_coverage = min_mask_coverage
        self.min_pose_score = min_pose_score
        self.min_face_quality = min_face_quality
        self.require_unique_face = require_unique_face
        self.max_body_overlap = max_body_overlap
        self.min_face_px = min_face_px
        self.use_colour = use_colour
        base = getattr(body_encoder, "name", "OSNet-x1.0")
        # Stored with every body embedding; the colour suffix changes the
        # vector layout, so it is part of the model name.
        self.body_model = f"{base}+colour" if use_colour else base
        self.body_model_version = str(getattr(body_encoder, "version", ""))
        self.small_faces_skipped = 0
        self.ambiguous_faces_skipped = 0
        self.crowded_bodies_skipped = 0

    def extract(
        self,
        frame: np.ndarray,
        track: LocalTrack,
        other_bboxes: Optional[list] = None,
    ) -> ExtractedFeatures:
        crop, crop_box = extract_crop(frame, track.bbox)
        mask_coverage = 1.0
        occlusion = 0.0
        mask = None
        if _ready(self.segmenter):
            try:
                seg = self.segmenter.run(crop)
                mask = seg.mask
                mask_coverage = seg.coverage
                occlusion = seg.occlusion_score
            except Exception:
                logger.exception("PPHumanSeg failed")

        pose_score = 0.5 if not _ready(self.pose) else 0.0
        visibility = 1.0 if not _ready(self.pose) else 0.0
        viewpoint = Viewpoint.UNKNOWN
        if _ready(self.pose) and mask_coverage >= self.min_mask_coverage:
            try:
                pose = self.pose.run(crop)
                pose_score = pose.pose_score
                visibility = pose.body_visibility_score
                viewpoint = pose.viewpoint
            except Exception:
                logger.exception("RTMPose failed")

        body_emb = None
        body_quality = 0.0
        body_ok = mask_coverage >= self.min_mask_coverage and pose_score >= self.min_pose_score
        # A crop that another person's box covers substantially also contains
        # that person (PPHumanSeg keeps every human, not just the target), so
        # its body embedding mixes two people. Skip it; a cleaner frame of the
        # same track will provide the body evidence.
        if body_ok and other_bboxes and self.max_body_overlap < 1.0:
            if overlap_fraction(track.bbox, other_bboxes) > self.max_body_overlap:
                body_ok = False
                self.crowded_bodies_skipped += 1
        if body_ok and _ready(self.body_encoder):
            try:
                vec = self.body_encoder.embed(crop, mask=mask)
                body_emb = [float(x) for x in np.asarray(vec).reshape(-1)]
                if self.use_colour:
                    # Clothing colour of torso and legs: what separates people
                    # in similar outfits when OSNet cannot (low resolution).
                    body_emb = fuse(body_emb, part_colour(crop, mask))
                body_quality = min(1.0, 0.4 * mask_coverage + 0.4 * pose_score + 0.2 * visibility)
            except Exception:
                logger.exception("OSNet embed failed")

        face_emb = None
        face_quality = 0.0
        face_crop = None
        face_ambiguous = False
        if _ready(self.face_detector):
            try:
                face = self.face_detector.detect(crop)
            except Exception:
                logger.exception("YuNet failed")
                face = None
            # Experiment: a face that also falls inside another track's
            # bounding box is ambiguous (could belong to that other person), so
            # do not use it for identification. Only use a face that lies in a
            # single bounding box.
            if face is not None and self.require_unique_face and other_bboxes:
                fx = crop_box.x1 + 0.5 * (face.bbox[0] + face.bbox[2])
                fy = crop_box.y1 + 0.5 * (face.bbox[1] + face.bbox[3])
                boxes = list(other_bboxes) + [track.bbox]
                hits = sum(
                    1
                    for b in boxes
                    if b.x1 <= fx <= b.x2 and b.y1 <= fy <= b.y2
                )
                if hits > 1:
                    face_ambiguous = True
                    self.ambiguous_faces_skipped += 1
            if face is not None and not face_ambiguous:
                face_quality = face.quality
                x1, y1, x2, y2 = face.bbox
                # AuraFace works on 112x112 aligned faces. A face only ~15 px
                # tall upsampled to that size gives an embedding that is
                # noise (same-person and different-person cosines overlap
                # completely on the EPFL clips), so it must not vote.
                if min(x2 - x1, y2 - y1) < self.min_face_px:
                    face_quality = 0.0
                    self.small_faces_skipped += 1
                h, w = crop.shape[:2]
                x1i, y1i = int(max(0, x1)), int(max(0, y1))
                x2i, y2i = int(min(w, x2)), int(min(h, y2))
                if y2i > y1i and x2i > x1i:
                    face_crop = crop[y1i:y2i, x1i:x2i]
                if _ready(self.face_encoder) and face_quality >= self.min_face_quality:
                    try:
                        aligned = align_face(crop, face.landmarks)
                        vec = self.face_encoder.embed(aligned)
                        face_emb = [float(x) for x in np.asarray(vec).reshape(-1)]
                    except Exception:
                        logger.exception("AuraFace embed failed")

        return ExtractedFeatures(
            body_embedding=body_emb,
            face_embedding=face_emb,
            body_quality=body_quality,
            face_quality=face_quality,
            viewpoint=viewpoint,
            mask_coverage=mask_coverage,
            occlusion_score=occlusion,
            pose_score=pose_score,
            body_visibility=visibility,
            crop=crop,
            face_crop=face_crop,
        )

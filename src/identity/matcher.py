"""Global identity matcher. Never force the nearest identity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.identity.candidates import Candidate
from src.identity.gallery import PersonGallery, best_cosine_record, supported_similarity
from src.identity.scoring import compute_score, motion_plausibility, temporal_plausibility
from src.identity.topology import CameraTopology
from src.types import IdentityAssignmentState, IdentityDecision, MatchKind, Viewpoint


@dataclass
class MatcherConfig:
    accept_threshold: float
    minimum_score_margin: float
    scoring: dict[str, Any]
    topology_enabled: bool
    temporal_enabled: bool
    unknown_edge_penalty: float
    max_gap: float = 180.0
    body_similarity_threshold: float = 0.55
    face_similarity_threshold: float = 0.45
    # Body similarity to a gallery = k-th best crop match (see
    # gallery.supported_similarity); 1 = best single crop.
    gallery_support_k: int = 1


def _visual(s) -> float:
    """Visual similarity of a score breakdown (time/camera terms excluded)."""
    vals = []
    if s.used_body:
        vals.append(s.body_similarity)
    if s.used_face:
        vals.append(s.face_similarity)
    return max(vals) if vals else 0.0


class IdentityMatcher:
    def __init__(self, config: MatcherConfig, topology: CameraTopology) -> None:
        self.config = config
        self.topology = topology

    def score_candidate(
        self,
        candidate: Candidate,
        gallery: PersonGallery,
        *,
        camera_id: str,
        timestamp: float,
        body_embedding: Optional[list[float]],
        face_embedding: Optional[list[float]],
        body_quality: float,
        face_quality: float,
        observed_velocity: Optional[tuple[float, float]] = None,
        source_video_id: Optional[str] = None,
        run_id: Optional[str] = None,
        query_viewpoint: Optional[Viewpoint] = None,
        recent_bodies: Optional[list[list[float]]] = None,
        exclude_records: Optional[set[int]] = None,
    ):
        dt = timestamp - gallery.last_seen_at if gallery.last_seen_at else 0.0
        body_gallery = gallery.body
        if exclude_records:
            # Never compare a track with crops it added itself in this run:
            # that is self-similarity, not evidence that it is this person.
            # (Matched by object identity, so earlier runs of the same video,
            # which reuse local track IDs, still count as evidence.)
            body_gallery = [r for r in gallery.body if id(r) not in exclude_records]
        body_cos = None
        if body_gallery and (recent_bodies or body_embedding):
            # Median over the track's recent crops of each crop's best match:
            # one lucky crop cannot decide the identity.
            queries = recent_bodies or [body_embedding]
            k = self.config.gallery_support_k
            sims = sorted(supported_similarity(q, body_gallery, k) or 0.0 for q in queries)
            body_cos = sims[len(sims) // 2]
        face_cos = None
        face_rec = None
        if face_embedding and gallery.face:
            face_cos, face_rec = best_cosine_record(face_embedding, gallery.face)

        # A modality whose similarity is below its own match threshold is not
        # usable evidence: drop its weight so it cannot dilute or veto a strong
        # match from the other modality (e.g. body 0.72 + unrelated face 0.15
        # -> keep the body and match). If BOTH are weak we keep both, so a
        # genuinely new person still scores low and gets a new id as before.
        if self.config.scoring.get("drop_weak_modality", True):
            body_thr = self.config.body_similarity_threshold
            face_thr = self.config.face_similarity_threshold
            body_strong = body_cos is not None and body_cos >= body_thr
            face_strong = face_cos is not None and face_cos >= face_thr
            if body_cos is not None and body_cos < body_thr and face_strong:
                body_cos = None
            if face_cos is not None and face_cos < face_thr and body_strong:
                face_cos = None
                face_rec = None

        # If the query pose and the best-matching gallery face were captured
        # from different viewpoints, the face cosine is unreliable.  Reduce the
        # face weight rather than let it veto an otherwise strong body match.
        face_weight_scale = 1.0
        if query_viewpoint is not None and face_rec is not None:
            gallery_view = face_rec.viewpoint
            if (
                query_viewpoint != Viewpoint.UNKNOWN
                and gallery_view != Viewpoint.UNKNOWN
                and query_viewpoint != gallery_view
            ):
                face_weight_scale = float(
                    self.config.scoring.get("face_view_mismatch_scale", 1.0)
                )

        cross_video = gallery.is_from_other_run(source_video_id, run_id)
        # A later video-file run starts its local clock at zero.  Comparing
        # that clock directly with the persisted local clock makes a valid
        # cross-video match look like a negative-time transition and applies
        # an impossible camera penalty.  Provenance tells us when this is a
        # new source context; keep visual evidence, threshold, and margin as
        # the safeguards for that match.
        clock_reset = bool(gallery.last_seen_at and timestamp < gallery.last_seen_at)
        if cross_video or clock_reset:
            temporal = 1.0
            camera = 1.0
        else:
            camera = self.topology.camera_plausibility(
                gallery.last_camera_id,
                camera_id,
                dt,
                enabled=self.config.topology_enabled,
                temporal_enabled=self.config.temporal_enabled,
                unknown_edge_penalty=self.config.unknown_edge_penalty,
            )
            temporal = temporal_plausibility(dt, self.config.max_gap)
        motion = motion_plausibility(None, observed_velocity)
        return compute_score(
            body_cosine=body_cos,
            face_cosine=face_cos,
            body_quality=body_quality,
            face_quality=face_quality,
            temporal=temporal,
            camera=camera,
            motion=motion,
            cfg=self.config.scoring,
            face_weight_scale=face_weight_scale,
        )

    @staticmethod
    def _best_blocked(blocked):
        usable = [
            item
            for item in (blocked or [])
            if not item[1].both_quality_failed and (item[1].used_body or item[1].used_face)
        ]
        return max(usable, key=lambda x: _visual(x[1])) if usable else None

    def decide(
        self,
        track_id: str,
        camera_id: str,
        scored: list[tuple[Candidate, Any]],
        *,
        both_quality_failed: bool,
        blocked: Optional[list[tuple[Candidate, Any]]] = None,
    ) -> IdentityDecision:
        """Pick the identity for one track.

        ``blocked`` are candidates this track cannot be (their ID is held by a
        track that was on screen together with it). They are never chosen, but
        if one of them looks at least as similar as the best allowed candidate
        (within the score margin), the query is ambiguous between people and
        the allowed match is refused: a new ID is safer than a wrong merge.
        """
        if both_quality_failed:
            return IdentityDecision(
                track_id=track_id,
                camera_id=camera_id,
                state=IdentityAssignmentState.UNRESOLVED,
                global_person_id=None,
                match_kind=MatchKind.UNRESOLVED,
                score=0.0,
                second_best_score=0.0,
                margin=0.0,
                reason="body and face failed quality gates",
                candidate_count=len(scored),
            )
        # A candidate with no comparable modality must not outrank (and then
        # poison) a candidate that has usable visual evidence.  Candidates
        # with no comparable modality are retained only for the unresolved
        # diagnostic path when no usable candidate exists.
        usable_scored = [
            item
            for item in scored
            if not item[1].both_quality_failed
            and (item[1].used_body or item[1].used_face)
        ]
        scored_sorted = sorted(
            usable_scored or scored,
            key=lambda x: x[1].S,
            reverse=True,
        )
        if not scored_sorted:
            return IdentityDecision(
                track_id=track_id,
                camera_id=camera_id,
                state=IdentityAssignmentState.NEW,
                global_person_id=None,
                match_kind=MatchKind.NEW,
                score=0.0,
                second_best_score=0.0,
                margin=0.0,
                reason="no candidates",
                candidate_count=0,
            )
        best_c, best_s = scored_sorted[0]
        second = scored_sorted[1][1].S if len(scored_sorted) > 1 else 0.0
        margin = best_s.S - second
        if best_s.both_quality_failed:
            return IdentityDecision(
                track_id=track_id,
                camera_id=camera_id,
                state=IdentityAssignmentState.UNRESOLVED,
                global_person_id=None,
                match_kind=MatchKind.UNRESOLVED,
                score=best_s.S,
                second_best_score=second,
                margin=margin,
                reason="quality gates failed",
                candidate_count=len(scored_sorted),
            )
        visual_ok = (
            best_s.used_body
            and best_s.body_similarity >= self.config.body_similarity_threshold
        ) or (
            best_s.used_face
            and best_s.face_similarity >= self.config.face_similarity_threshold
        )
        camera_transition_ok = (
            not self.config.topology_enabled
            or best_s.camera_plausibility > 0.0
            or "cross_video" in best_c.reasons
        )
        if not camera_transition_ok:
            return IdentityDecision(
                track_id=track_id,
                camera_id=camera_id,
                state=IdentityAssignmentState.UNCERTAIN,
                global_person_id=None,
                match_kind=MatchKind.UNRESOLVED,
                score=best_s.S,
                second_best_score=second,
                margin=margin,
                body_similarity=best_s.body_similarity,
                face_similarity=best_s.face_similarity,
                temporal_plausibility=best_s.temporal_plausibility,
                camera_plausibility=best_s.camera_plausibility,
                motion_plausibility=best_s.motion_plausibility,
                reason="camera transition is not plausible",
                candidate_count=len(scored_sorted),
            )
        rival = self._best_blocked(blocked)
        if visual_ok and rival is not None:
            rival_c, rival_s = rival
            if _visual(rival_s) > _visual(best_s) - self.config.minimum_score_margin:
                return IdentityDecision(
                    track_id=track_id,
                    camera_id=camera_id,
                    state=IdentityAssignmentState.NEW,
                    global_person_id=None,
                    match_kind=MatchKind.NEW,
                    score=best_s.S,
                    second_best_score=rival_s.S,
                    margin=_visual(best_s) - _visual(rival_s),
                    body_similarity=best_s.body_similarity,
                    face_similarity=best_s.face_similarity,
                    reason=(
                        f"looks as much like {rival_c.global_person_id} (on screen "
                        f"together, so a different person) as like "
                        f"{best_c.global_person_id}"
                    ),
                    candidate_count=len(scored_sorted),
                )
        accepted = (
            visual_ok
            and best_s.S >= self.config.accept_threshold
            and margin >= self.config.minimum_score_margin
        )
        if accepted:
            return IdentityDecision(
                track_id=track_id,
                camera_id=camera_id,
                state=IdentityAssignmentState.MATCHED,
                global_person_id=best_c.global_person_id,
                match_kind=MatchKind.EXISTING,
                score=best_s.S,
                second_best_score=second,
                margin=margin,
                body_similarity=best_s.body_similarity,
                face_similarity=best_s.face_similarity,
                temporal_plausibility=best_s.temporal_plausibility,
                camera_plausibility=best_s.camera_plausibility,
                motion_plausibility=best_s.motion_plausibility,
                reason=(
                    "cross-video threshold+margin"
                    if "cross_video" in best_c.reasons
                    else "threshold+margin"
                ),
                candidate_count=len(scored_sorted),
            )
        if best_s.S >= self.config.accept_threshold and margin < self.config.minimum_score_margin:
            return IdentityDecision(
                track_id=track_id,
                camera_id=camera_id,
                state=IdentityAssignmentState.UNCERTAIN,
                global_person_id=None,
                match_kind=MatchKind.UNRESOLVED,
                score=best_s.S,
                second_best_score=second,
                margin=margin,
                reason="best and second-best too close",
                candidate_count=len(scored_sorted),
            )
        return IdentityDecision(
            track_id=track_id,
            camera_id=camera_id,
            state=IdentityAssignmentState.NEW,
            global_person_id=None,
            match_kind=MatchKind.NEW,
            score=best_s.S,
            second_best_score=second,
            margin=margin,
            reason="no reliable match",
            candidate_count=len(scored_sorted),
        )

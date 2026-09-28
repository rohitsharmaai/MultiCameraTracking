"""Global identity manager is the authority.

New local track -> stabilize -> quality observations -> features ->
candidates -> existing gallery -> reuse ID or create ID.

Do not create a new Global ID immediately when the local tracker starts.
Do not reverse this order. Concurrent camera workers must serialize here.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Protocol

from src.events import (
    CAMERA_TRANSITION,
    IDENTITY_CREATED,
    IDENTITY_MATCHED,
    IDENTITY_UNRESOLVED,
    EventBus,
)
from src.identity.candidates import generate_candidates, nearest_neighbors
from src.identity.gallery import PersonGallery, upgrade_or_insert
from src.identity.matcher import IdentityMatcher
from src.identity.state import TrackAssignment, apply_hysteresis
from src.identity.topology import CameraTopology
from src.types import (
    BBox,
    EmbeddingRecord,
    IdentityAssignmentState,
    IdentityDecision,
    MatchKind,
    Viewpoint,
)


class IdentityStore(Protocol):
    def load_galleries(self) -> dict[str, PersonGallery]: ...
    def next_person_id(self, prefix: str, width: int) -> str: ...
    def upsert_person(self, gallery: PersonGallery) -> None: ...
    def add_embedding(self, rec: EmbeddingRecord) -> None: ...
    def remove_embedding(self, rec: EmbeddingRecord) -> None: ...
    def add_track(
        self,
        track_id: str,
        camera_id: str,
        global_person_id: Optional[str],
        start_time: float,
        end_time: Optional[float],
        status: str,
        source_video_id: Optional[str],
        source_video_name: Optional[str],
        confidence: float,
    ) -> None: ...
    def add_transition(
        self,
        global_person_id: str,
        from_camera_id: str,
        to_camera_id: str,
        from_time: float,
        to_time: float,
        match_score: float,
        match_method: str,
        confidence: float,
    ) -> None: ...
    def delete_person(self, global_person_id: str) -> None: ...
    def reset_gallery(self) -> None: ...


@dataclass
class PendingEvidence:
    body: Optional[list[float]] = None
    face: Optional[list[float]] = None
    body_quality: float = 0.0
    face_quality: float = 0.0
    viewpoint: Viewpoint = Viewpoint.UNKNOWN
    source: dict[str, Any] = field(default_factory=dict)
    frames: int = 0
    persisted_body_token: int = 0
    persisted_face_token: int = 0
    # Most recent body crops (not only the best one): identity decisions use
    # the median over these so one unusual crop cannot decide.
    recent_bodies: list = field(default_factory=list)
    # When the track's best match first fell in the "unclear" band, and
    # when the track was first considered for an ID.
    unclear_since: Optional[float] = None
    decide_since: Optional[float] = None


class GlobalIdentityManager:
    def __init__(
        self,
        store: IdentityStore,
        matcher: IdentityMatcher,
        topology: CameraTopology,
        thresholds: dict[str, Any],
        events: Optional[EventBus] = None,
    ) -> None:
        self.store = store
        self.matcher = matcher
        self.topology = topology
        self.thresholds = thresholds
        self.events = events
        self._lock = threading.RLock()
        self.galleries: dict[str, PersonGallery] = {}
        self.assignments: dict[str, TrackAssignment] = {}
        self._pending: dict[str, PendingEvidence] = {}
        self._active_run_id: Optional[str] = None
        # Run-local "cannot-link" constraints: local tracks that were on
        # screen in the same frame are different people for the whole run.
        self._cannot_link: dict[str, set[str]] = {}
        # Embedding records persisted by each local track in this run, so a
        # track split off a shared ID can take its own evidence with it.
        self._run_records: dict[str, list[EmbeddingRecord]] = {}
        self._bind_seq = 0
        # Crossing check: IoU-only local tracking can hand one person's box to
        # another when their boxes overlap. A track that overlapped another
        # is "unverified" until its own fresh crops re-establish who it is;
        # meanwhile it shows no ID and adds nothing to any gallery.
        self._verify: dict[str, dict[str, Any]] = {}
        self._last_body: dict[str, tuple[float, list[float], float]] = {}
        # Drift check for established tracks: (target ID or "" for new, count)
        self._drift: dict[str, tuple[str, int]] = {}
        # Global ID pairs that were seen together in one frame and therefore
        # cannot be the same person (used to warn before a manual merge).
        self.distinct_pairs: set[frozenset[str]] = set()
        self.reload()

    def reload(self) -> None:
        with self._lock:
            self.galleries = self.store.load_galleries()

    def begin_run(self, run_id: Optional[str] = None) -> None:
        """Start a new video run without discarding learned identities.

        Local tracker IDs are only unique within a camera/run (``A_1`` is
        routinely reused by the next file).  Keeping an assignment from a
        previous run would bypass ReID and attach new evidence to the old
        person.  The global gallery remains intact; only run-local state is
        cleared.
        """
        with self._lock:
            self.assignments.clear()
            self._pending.clear()
            self._cannot_link.clear()
            self._run_records.clear()
            self._verify.clear()
            self._last_body.clear()
            self._drift.clear()
            self._active_run_id = run_id

    def assign(
        self,
        *,
        track_id: str,
        camera_id: str,
        timestamp: float,
        body_embedding: Optional[list[float]],
        face_embedding: Optional[list[float]],
        body_quality: float,
        face_quality: float,
        viewpoint: Viewpoint = Viewpoint.UNKNOWN,
        source_video_id: Optional[str] = None,
        source_video_name: Optional[str] = None,
        source_frame_number: Optional[int] = None,
        source_timestamp_ms: Optional[int] = None,
        body_model: str = "OSNet-x1.0",
        body_model_version: str = "",
        face_model: str = "AuraFace-v1",
        face_model_version: str = "",
        velocity: Optional[tuple[float, float]] = None,
        force_decide: bool = False,
    ) -> IdentityDecision:
        with self._lock:
            return self._assign_locked(
                track_id=track_id,
                camera_id=camera_id,
                timestamp=timestamp,
                body_embedding=body_embedding,
                face_embedding=face_embedding,
                body_quality=body_quality,
                face_quality=face_quality,
                viewpoint=viewpoint,
                source_video_id=source_video_id,
                source_video_name=source_video_name,
                source_frame_number=source_frame_number,
                source_timestamp_ms=source_timestamp_ms,
                body_model=body_model,
                body_model_version=body_model_version,
                face_model=face_model,
                face_model_version=face_model_version,
                velocity=velocity,
                force_decide=force_decide,
            )

    def _assign_locked(self, **kw: Any) -> IdentityDecision:
        track_id = kw["track_id"]
        camera_id = kw["camera_id"]
        timestamp = kw["timestamp"]
        identity_cfg = self.thresholds.get("identity", {})
        stab = int(identity_cfg.get("stabilization_frames", 8))
        incoming_source = kw.get("source_video_id")
        incoming_source_name = kw.get("source_video_name")
        existing = self.assignments.get(track_id)
        source_changed = bool(
            incoming_source
            and existing
            and existing.source_video_id
            and existing.source_video_id != incoming_source
        ) or bool(
            incoming_source_name
            and existing
            and existing.source_video_name
            and existing.source_video_name != incoming_source_name
        )
        if existing is not None and (
            existing.camera_id != camera_id or source_changed
        ):
            # Local IDs are camera/run scoped.  Be defensive for callers that
            # invoke ``assign`` directly instead of VideoProcessor.begin_run.
            self.assignments.pop(track_id, None)
            self._pending.pop(track_id, None)
            self._run_records.pop(track_id, None)
            for other in self._cannot_link.pop(track_id, set()):
                self._cannot_link.get(other, set()).discard(track_id)
        pending = self._pending.setdefault(track_id, PendingEvidence())
        if kw["body_embedding"] is not None:
            self._last_body[track_id] = (timestamp, kw["body_embedding"], kw["body_quality"])
            gate = float(self.thresholds.get("scoring", {}).get("body_quality_gate", 0.35))
            if kw["body_quality"] >= gate:
                keep = int(identity_cfg.get("decision_crops", 5))
                pending.recent_bodies = (pending.recent_bodies + [kw["body_embedding"]])[-keep:]
        if kw["body_embedding"] is not None and kw["body_quality"] >= pending.body_quality:
            pending.body = kw["body_embedding"]
            pending.body_quality = kw["body_quality"]
            pending.viewpoint = kw["viewpoint"]
        if kw["face_embedding"] is not None and kw["face_quality"] >= pending.face_quality:
            pending.face = kw["face_embedding"]
            pending.face_quality = kw["face_quality"]
        pending.frames += 1
        pending.source = {
            "source_video_id": kw["source_video_id"],
            "source_video_name": kw["source_video_name"],
            "source_frame_number": kw["source_frame_number"],
            "source_timestamp_ms": kw["source_timestamp_ms"],
            "body_model": kw["body_model"],
            "body_model_version": kw["body_model_version"],
            "face_model": kw["face_model"],
            "face_model_version": kw["face_model_version"],
        }
        current = self.assignments.setdefault(
            track_id,
            TrackAssignment(
                track_id=track_id,
                camera_id=camera_id,
                source_video_id=incoming_source,
                source_video_name=incoming_source_name,
            ),
        )
        if current.global_person_id and not current.provisional:
            return self._refine_assigned(current, pending, timestamp, kw)
        if pending.frames < stab and not kw["force_decide"] and current.state == IdentityAssignmentState.UNRESOLVED:
            return IdentityDecision(
                track_id=track_id,
                camera_id=camera_id,
                state=IdentityAssignmentState.UNRESOLVED,
                global_person_id=None,
                match_kind=MatchKind.UNRESOLVED,
                score=0.0,
                second_best_score=0.0,
                margin=0.0,
                reason="stabilizing",
                candidate_count=0,
            )

        neighbor_ids: list[str] = []
        neighbor_k = int(self.thresholds.get("reid", {}).get("candidate_k", 32))
        if pending.body:
            neighbor_ids.extend(
                nearest_neighbors(
                    pending.body,
                    (
                        (gid, rec)
                        for gid, gal in self.galleries.items()
                        if gal.status == "active"
                        for rec in gal.body
                    ),
                    k=neighbor_k,
                )
            )
        if pending.face:
            neighbor_ids.extend(
                nearest_neighbors(
                    pending.face,
                    (
                        (gid, rec)
                        for gid, gal in self.galleries.items()
                        if gal.status == "active"
                        for rec in gal.face
                    ),
                    k=neighbor_k,
                )
            )
        cam_cfg = self.thresholds.get("camera_transition", {})
        candidates = generate_candidates(
            self.galleries,
            camera_id=camera_id,
            timestamp=timestamp,
            body_embedding=pending.body,
            face_embedding=pending.face,
            topology=self.topology,
            time_window=float(cam_cfg.get("default_max_transition_seconds", 180.0)),
            candidate_k=int(self.thresholds.get("reid", {}).get("candidate_k", 32)),
            topology_enabled=bool(cam_cfg.get("topology_enabled", True)),
            neighbor_hits=neighbor_ids,
            source_video_id=kw.get("source_video_id"),
            run_id=self._active_run_id,
        )
        if current.global_person_id and current.provisional:
            # A newly-created ID is provisional.  Do not let the track match
            # its own first embedding while waiting for better evidence.
            candidates = [
                cand
                for cand in candidates
                if cand.global_person_id != current.global_person_id
            ]
        excluded = self._excluded_ids(track_id)
        # An ID held by a track that was on screen together with this one
        # belongs to a different person, however similar the body looks. It
        # is still scored: if this track looks MORE like that person than
        # like any allowed ID, the allowed match is not trustworthy either.
        scored = []
        blocked = []
        for cand in candidates:
            gal = self.galleries[cand.global_person_id]
            breakdown = self.matcher.score_candidate(
                cand,
                gal,
                camera_id=camera_id,
                timestamp=timestamp,
                body_embedding=pending.body,
                face_embedding=pending.face,
                body_quality=pending.body_quality,
                face_quality=pending.face_quality,
                observed_velocity=kw.get("velocity"),
                source_video_id=kw.get("source_video_id"),
                run_id=self._active_run_id,
                query_viewpoint=pending.viewpoint,
                recent_bodies=pending.recent_bodies or None,
                exclude_records={id(r) for r in self._run_records.get(track_id, [])},
            )
            if cand.global_person_id in excluded:
                blocked.append((cand, breakdown))
            else:
                scored.append((cand, breakdown))
        scoring_cfg = self.thresholds.get("scoring", {})
        body_gate = float(scoring_cfg.get("body_quality_gate", 0.35))
        face_gate = float(scoring_cfg.get("face_quality_gate", 0.45))
        no_visual = pending.body is None and pending.face is None
        quality_failed = (
            (pending.body is None or pending.body_quality < body_gate)
            and (pending.face is None or pending.face_quality < face_gate)
        )
        # This is a property of the query, not of an individual candidate.
        # A gallery that lacks body embeddings must not make an otherwise
        # usable body match fail just because another candidate is face-only.
        both_failed = bool(no_visual or quality_failed)
        require_body = bool(identity_cfg.get("require_body_for_id", False))
        require_face = bool(identity_cfg.get("require_face_for_id", False))
        body_ok = pending.body is not None and pending.body_quality >= body_gate
        face_ok = pending.face is not None and pending.face_quality >= face_gate
        missing = []
        if require_body and not body_ok:
            missing.append("body")
        if require_face and not face_ok:
            missing.append("face")
        if missing:
            reason = "required " + " and ".join(missing) + " evidence missing"
            if current.global_person_id:
                decision = IdentityDecision(
                    track_id=track_id,
                    camera_id=camera_id,
                    state=IdentityAssignmentState.MATCHED,
                    global_person_id=current.global_person_id,
                    match_kind=MatchKind.EXISTING,
                    score=current.score,
                    second_best_score=current.second_best,
                    margin=max(0.0, current.score - current.second_best),
                    reason=reason + "; kept established assignment",
                )
                if current.provisional:
                    self._keep_provisional(current, pending)
                else:
                    current.state = IdentityAssignmentState.MATCHED
                    current.frames_in_state += 1
            else:
                decision = IdentityDecision(
                    track_id=track_id,
                    camera_id=camera_id,
                    state=IdentityAssignmentState.UNRESOLVED,
                    global_person_id=None,
                    match_kind=MatchKind.UNRESOLVED,
                    score=0.0,
                    second_best_score=0.0,
                    margin=0.0,
                    reason=reason,
                )
                current.state = IdentityAssignmentState.UNRESOLVED
                current.frames_in_state += 1
                if self.events:
                    self.events.publish(
                        IDENTITY_UNRESOLVED,
                        camera_id=camera_id,
                        track_id=track_id,
                        reason=decision.reason,
                    )
            self.store.add_track(
                track_id=track_id,
                camera_id=camera_id,
                global_person_id=current.global_person_id,
                start_time=timestamp,
                end_time=None,
                status=current.state.value,
                source_video_id=pending.source.get("source_video_id"),
                source_video_name=pending.source.get("source_video_name"),
                confidence=decision.score,
            )
            return decision

        decision = self.matcher.decide(
            track_id,
            camera_id,
            scored,
            both_quality_failed=both_failed,
            blocked=blocked,
        )

        if decision.state == IdentityAssignmentState.MATCHED:
            if not apply_hysteresis(
                current,
                decision.global_person_id,
                decision.score,
                decision.second_best_score,
                enabled=(
                    bool(identity_cfg.get("hysteresis_enabled", True))
                    and not current.provisional
                ),
                switch_margin=float(identity_cfg.get("hysteresis_switch_margin", 0.15)),
            ):
                switched = (
                    current.global_person_id
                    and decision.global_person_id != current.global_person_id
                )
                decision.global_person_id = current.global_person_id
                decision.state = IdentityAssignmentState.MATCHED
                decision.match_kind = MatchKind.EXISTING
                if switched:
                    decision.reason = "hysteresis held previous assignment"
            if (
                current.global_person_id
                and current.provisional
                and decision.global_person_id != current.global_person_id
            ):
                decision.archived_gid = current.global_person_id
                self._discard_provisional_identity(
                    current.global_person_id, current.track_id
                )
                # The current evidence was persisted under the discarded
                # provisional ID; persist it again under the recovered ID.
                pending.persisted_body_token = 0
                pending.persisted_face_token = 0
            self._bind_existing(current, decision, timestamp, pending)
        elif decision.state == IdentityAssignmentState.NEW:
            if current.global_person_id:
                decision.global_person_id = current.global_person_id
                decision.state = IdentityAssignmentState.MATCHED
                decision.match_kind = MatchKind.EXISTING
                if current.provisional:
                    decision.reason = "kept provisional assignment"
                    self._keep_provisional(current, pending)
                else:
                    decision.reason = "kept established assignment"
                    current.state = IdentityAssignmentState.MATCHED
                self._persist_embeddings(
                    self.galleries[current.global_person_id],
                    current,
                    timestamp,
                    pending,
                )
                self.store.upsert_person(self.galleries[current.global_person_id])
            elif no_visual or quality_failed:
                decision.state = IdentityAssignmentState.UNRESOLVED
                decision.match_kind = MatchKind.UNRESOLVED
                decision.global_person_id = None
                decision.reason = "no reliable visual evidence for a new ID"
                current.state = IdentityAssignmentState.UNRESOLVED
                current.frames_in_state += 1
            elif self._wait_before_new_id(scored, pending, timestamp, decision):
                current.state = IdentityAssignmentState.UNRESOLVED
                current.frames_in_state += 1
            else:
                self._create_new(current, decision, camera_id, timestamp, pending)
        else:
            if current.global_person_id and current.provisional:
                # Keep the provisional ID open even when this observation is
                # uncertain; a later, better crop may provide the match.
                self._keep_provisional(current, pending)
            else:
                current.state = decision.state
                current.frames_in_state += 1
            if self.events:
                self.events.publish(
                    IDENTITY_UNRESOLVED,
                    camera_id=camera_id,
                    track_id=track_id,
                    reason=decision.reason,
                )
        self.store.add_track(
            track_id=track_id,
            camera_id=camera_id,
            global_person_id=current.global_person_id,
            start_time=timestamp,
            end_time=None,
            status=current.state.value,
            source_video_id=pending.source.get("source_video_id"),
            source_video_name=pending.source.get("source_video_name"),
            confidence=decision.score,
        )
        return decision

    def _wait_before_new_id(
        self,
        scored: list,
        pending: PendingEvidence,
        timestamp: float,
        decision: IdentityDecision,
    ) -> bool:
        """Hold a new track as unresolved while its best match is unclear.

        On 4p-c0, genuinely new people scored <= 0.32 against every existing
        ID when their ID was created, while fragments of an existing person
        (new local track after an occlusion) scored 0.42-0.53 against the
        right person - below the threshold, but far from "nobody". Creating
        a new ID for them splits a person. So while the best match lies in
        [threshold - identity.unclear_band, threshold), keep collecting crops
        for up to identity.unclear_wait_seconds; the track then either
        reaches the threshold (old ID) or gets a new ID. Never loosens a match.
        """
        cfg = self.thresholds.get("identity", {})
        band = float(cfg.get("unclear_band", 0.0))
        min_age = float(cfg.get("new_id_min_seconds", 0.0))
        if pending.decide_since is None:
            pending.decide_since = timestamp
        thr = float(self.thresholds.get("reid", {}).get("body_similarity_threshold", 0.62))
        best = max(
            (b.body_similarity for _, b in scored if b.used_body),
            default=0.0,
        )
        reason = None
        if timestamp - pending.decide_since < min_age:
            # The first crops of a new track are often partial or blurred
            # (A_23 on 4p-c0: 0.21 at first, 0.65 three crops later).
            reason = f"new track ({best:.2f}); observing before a new ID"
        if band > 0 and thr - band <= best < thr and pending.unclear_since is None:
            pending.unclear_since = timestamp
        if pending.unclear_since is not None and timestamp - pending.unclear_since < float(
            cfg.get("unclear_wait_seconds", 3.0)
        ):
            # Once unclear, one dip below the band does not end the wait.
            reason = f"unclear match ({best:.2f}); collecting more crops before a new ID"
        if reason is None:
            return False
        decision.state = IdentityAssignmentState.UNRESOLVED
        decision.match_kind = MatchKind.UNRESOLVED
        decision.global_person_id = None
        decision.reason = reason
        return True

    def _keep_provisional(
        self,
        current: TrackAssignment,
        pending: PendingEvidence,
    ) -> None:
        """Keep a newly-created ID open while better evidence can arrive.

        A face can be usable while the body crop is still poor.  In that
        case, do not confirm solely on the face if a body embedding exists;
        wait for the body evidence (or the bounded maximum window) so a later
        full-body crop can recover the correct existing identity.
        """
        current.frames_in_state += 1
        identity_cfg = self.thresholds.get("identity", {})
        confirmation_frames = max(
            1,
            int(identity_cfg.get("new_id_confirmation_frames", 30)),
        )
        max_frames = max(
            confirmation_frames,
            int(identity_cfg.get("new_id_max_frames", 180)),
        )
        scoring_cfg = self.thresholds.get("scoring", {})
        base_body = float(
            identity_cfg.get(
                "base_body_quality",
                scoring_cfg.get("body_quality_gate", 0.45),
            )
        )
        base_face = float(
            identity_cfg.get(
                "base_face_quality",
                scoring_cfg.get("face_quality_gate", 0.45),
            )
        )
        body_ok = pending.body is not None and pending.body_quality >= base_body
        face_ok = pending.face is not None and pending.face_quality >= base_face
        require_body = bool(identity_cfg.get("require_body_for_id", False))
        require_face = bool(identity_cfg.get("require_face_for_id", False))
        if require_body:
            strong_evidence = body_ok
        elif require_face:
            strong_evidence = face_ok
        elif pending.body is not None:
            strong_evidence = body_ok
        else:
            strong_evidence = face_ok
        if current.frames_in_state >= max_frames:
            current.state = IdentityAssignmentState.MATCHED
            # Keep the provisional flag for the lifetime of the local track.
            # The dense extractor will fall back to the normal stride after
            # max_frames, so a later viewpoint can still trigger a final
            # gallery search without making every frame expensive.
        elif current.frames_in_state >= confirmation_frames and strong_evidence:
            # State is MATCHED for callers, but keep the provisional flag so
            # a later, better viewpoint can still recover an existing ID.
            current.state = IdentityAssignmentState.MATCHED

    def track_needs_dense_extract(self, track_id: str) -> bool:
        """Extract every frame until this local track has a usable base body/face."""
        with self._lock:
            identity_cfg = self.thresholds.get("identity", {})
            pending = self._pending.get(track_id)
            current = self.assignments.get(track_id)
            if current is None or not current.global_person_id or pending is None:
                return True
            if track_id in self._verify:
                return True  # re-establish identity after a crossing quickly
            if current.provisional:
                max_frames = max(
                    1,
                    int(identity_cfg.get("new_id_max_frames", 180)),
                )
                return current.frames_in_state < max_frames
            scoring_cfg = self.thresholds.get("scoring", {})
            base_body = float(identity_cfg.get("base_body_quality", scoring_cfg.get("body_quality_gate", 0.45)))
            base_face = float(identity_cfg.get("base_face_quality", scoring_cfg.get("face_quality_gate", 0.45)))
            give_up = int(identity_cfg.get("refine_give_up_frames", 60))
            body_ok = pending.body is not None and pending.body_quality >= base_body
            face_ok = pending.face is not None and pending.face_quality >= base_face
            if body_ok and face_ok:
                return False
            return pending.frames < give_up

    def _refine_assigned(
        self,
        current: TrackAssignment,
        pending: PendingEvidence,
        timestamp: float,
        kw: dict[str, Any],
    ) -> IdentityDecision:
        """Same BoT-SORT track keeps its Global ID; better pose/face updates the gallery."""
        gid = current.global_person_id
        assert gid is not None
        if current.track_id in self._verify:
            return self._verify_track(current, kw, timestamp)
        drifted = self._check_drift(current, kw, timestamp)
        if drifted is not None:
            return drifted
        gal = self.galleries[gid]
        if current.track_id in self._drift:
            # This track's latest crops look like someone else; keep them out
            # of the gallery until that is confirmed or clears.
            upgraded = False
        else:
            upgraded = self._persist_embeddings(gal, current, timestamp, pending)
        gal.last_camera_id = current.camera_id
        gal.last_seen_at = timestamp
        gal.last_run_id = self._active_run_id
        self.store.upsert_person(gal)
        current.state = IdentityAssignmentState.MATCHED
        current.provisional = False
        current.source_video_id = pending.source.get("source_video_id")
        current.source_video_name = pending.source.get("source_video_name")
        current.frames_in_state += 1
        reason = (
            "same local track; upgraded gallery"
            if upgraded
            else "same local track; monitoring for better pose/face"
        )
        decision = IdentityDecision(
            track_id=current.track_id,
            camera_id=current.camera_id,
            state=IdentityAssignmentState.MATCHED,
            global_person_id=gid,
            match_kind=MatchKind.EXISTING,
            score=current.score,
            second_best_score=current.second_best,
            margin=max(0.0, current.score - current.second_best),
            reason=reason,
        )
        self.store.add_track(
            track_id=current.track_id,
            camera_id=current.camera_id,
            global_person_id=gid,
            start_time=timestamp,
            end_time=None,
            status=current.state.value,
            source_video_id=pending.source.get("source_video_id"),
            source_video_name=pending.source.get("source_video_name"),
            confidence=decision.score,
        )
        return decision

    def _check_drift(
        self,
        current: TrackAssignment,
        kw: dict[str, Any],
        timestamp: float,
    ) -> Optional[IdentityDecision]:
        """Re-check an established track against its own ID on every crop.

        IoU-only tracking can hand a track's box to another person (the
        original person continues under a new local track). The track keeps
        its Global ID even though it now shows someone else. Here each fresh
        body crop is compared with the track's ID (excluding what this track
        itself added in the last few seconds) and with every other ID. If it
        matches another ID better by ``identity.drift_margin`` - or matches
        nobody, including its own ID - for ``drift_confirmations`` crops in a
        row, the track moves to that ID (or to a new provisional ID).
        """
        cfg = self.thresholds.get("identity", {})
        if not bool(cfg.get("verify_track_drift", True)):
            return None
        fresh = kw.get("body_embedding")
        gate = float(self.thresholds.get("scoring", {}).get("body_quality_gate", 0.35))
        if fresh is None or float(kw.get("body_quality") or 0.0) < gate:
            return None
        margin = float(cfg.get("drift_margin", 0.15))
        needed = int(cfg.get("drift_confirmations", 3))
        recent = float(cfg.get("drift_recent_seconds", 3.0))
        thr = float(self.thresholds.get("reid", {}).get("body_similarity_threshold", 0.62))
        gid = current.global_person_id
        own_gal = self.galleries.get(gid)
        if own_gal is None:
            return None
        from src.identity.gallery import supported_similarity

        k = int(self.thresholds.get("reid", {}).get("gallery_support_k", 1))
        ref = [
            r
            for r in own_gal.body
            if not (
                r.track_id == current.track_id
                and r.source_video_id == current.source_video_id
                and r.timestamp > timestamp - recent
            )
        ]
        if not ref:
            return None
        own = supported_similarity(fresh, ref, k) or 0.0
        best_g, best_other = None, -1.0
        for g, gal in self.galleries.items():
            if g == gid or gal.status != "active" or not gal.body:
                continue
            sim = supported_similarity(fresh, gal.body, k) or 0.0
            if sim > best_other:
                best_g, best_other = g, sim
        if best_g is not None and best_other >= thr and best_other - own >= margin:
            target = best_g
        elif own < thr - margin and best_other < thr:
            # Looks like nobody known. In a scene of known people that is
            # usually a poor crop, not a new person: withdraw the ID and
            # re-verify from the next crops (a new ID only if it stays so).
            self._drift.pop(current.track_id, None)
            self._verify[current.track_id] = {"since": timestamp, "tries": 0, "clear": True}
            return self._unverified_decision(current, f"looks like nobody known ({own:.2f})")
        else:
            self._drift.pop(current.track_id, None)
            return None
        prev_target, count = self._drift.get(current.track_id, (None, 0))
        count = count + 1 if prev_target == target else 1
        self._drift[current.track_id] = (target, count)
        if count < needed:
            return None
        self._drift.pop(current.track_id, None)
        return self._rebind_drifted(current, gid, target, fresh, kw, timestamp, own, best_other)

    def _rebind_drifted(
        self,
        current: TrackAssignment,
        old_gid: str,
        target: str,
        fresh: list[float],
        kw: dict[str, Any],
        timestamp: float,
        own: float,
        other: float,
    ) -> IdentityDecision:
        pending = self._pending.get(current.track_id)
        if pending is not None:
            # Evidence gathered before the switch belongs to the old person.
            pending.body = fresh
            pending.body_quality = float(kw.get("body_quality") or 0.0)
            pending.persisted_body_token = 0
            pending.face = None
            pending.face_quality = 0.0
            pending.persisted_face_token = 0
            pending.viewpoint = kw.get("viewpoint", pending.viewpoint)
        if target:
            new_gid = target
            current.provisional = False
            kind = MatchKind.EXISTING
            reason = (
                f"track drifted: now looks like {target} ({other:.2f}) "
                f"not {old_gid} ({own:.2f})"
            )
        else:
            identity_cfg = self.thresholds.get("identity", {})
            new_gid = self.store.next_person_id(
                prefix=str(identity_cfg.get("new_id_prefix", "P")),
                width=int(identity_cfg.get("new_id_width", 4)),
            )
            gal = PersonGallery(
                global_person_id=new_gid,
                last_camera_id=current.camera_id,
                last_seen_at=timestamp,
                first_camera_id=current.camera_id,
                last_run_id=self._active_run_id,
            )
            self.galleries[new_gid] = gal
            self.store.upsert_person(gal)
            current.provisional = True
            current.frames_in_state = 0
            kind = MatchKind.NEW
            reason = f"track drifted: no longer looks like {old_gid} ({own:.2f}) or anyone known"
            if self.events:
                self.events.publish(
                    IDENTITY_CREATED,
                    camera_id=current.camera_id,
                    track_id=current.track_id,
                    global_person_id=new_gid,
                )
        current.global_person_id = new_gid
        current.state = IdentityAssignmentState.MATCHED
        current.score = other if target else 0.0
        current.second_best = own
        self._mark_bound(current)
        gal = self.galleries[new_gid]
        if pending is not None:
            self._persist_embeddings(gal, current, timestamp, pending)
        gal.last_camera_id = current.camera_id
        gal.last_seen_at = timestamp
        gal.last_run_id = self._active_run_id
        self.store.upsert_person(gal)
        self.store.add_track(
            track_id=current.track_id,
            camera_id=current.camera_id,
            global_person_id=new_gid,
            start_time=timestamp,
            end_time=None,
            status=current.state.value,
            source_video_id=current.source_video_id,
            source_video_name=current.source_video_name,
            confidence=current.score,
        )
        return IdentityDecision(
            track_id=current.track_id,
            camera_id=current.camera_id,
            state=IdentityAssignmentState.MATCHED,
            global_person_id=new_gid,
            match_kind=kind,
            score=current.score,
            second_best_score=own,
            margin=other - own,
            reason=reason,
        )

    def _discard_provisional_identity(
        self, global_person_id: str, current_track_id: str
    ) -> None:
        """Archive a provisional ID after a later ReID match.

        The person is *never deleted* here.  It is marked ``inactive`` so
        candidate generation skips it, but its embeddings, tracks and history
        stay in the store so an operator can audit why the tracks were merged.
        Only ``reset()`` ("delete all") or an explicit ``delete_person`` call
        removes people.

        ``current_track_id`` is the track being rebound to the recovered
        identity; its old provisional id is what gets archived.
        """
        if not global_person_id:
            return
        gallery = self.galleries.get(global_person_id)
        if gallery is None:
            return
        gallery.status = "inactive"
        # Persist so a process restart does not silently reactivate it.
        self.store.upsert_person(gallery)

    def _bind_existing(
        self,
        current: TrackAssignment,
        decision: IdentityDecision,
        timestamp: float,
        pending: PendingEvidence,
    ) -> None:
        gid = decision.global_person_id
        assert gid is not None
        gal = self.galleries[gid]
        prev_cam = gal.last_camera_id
        overlap_cfg = self.thresholds.get("overlap", {})
        same_time = abs(timestamp - (gal.last_seen_at or timestamp)) <= float(
            overlap_cfg.get("timestamp_tolerance_seconds", 0.04)
        )
        cross_video = gal.is_from_other_run(
            pending.source.get("source_video_id"), self._active_run_id
        )
        if prev_cam and prev_cam != current.camera_id and not same_time and not cross_video:
            if self.events:
                self.events.publish(
                    CAMERA_TRANSITION,
                    camera_id=current.camera_id,
                    track_id=current.track_id,
                    global_person_id=gid,
                    from_camera_id=prev_cam,
                    to_camera_id=current.camera_id,
                )
            self.store.add_transition(
                global_person_id=gid,
                from_camera_id=prev_cam,
                to_camera_id=current.camera_id,
                from_time=gal.last_seen_at,
                to_time=timestamp,
                match_score=decision.score,
                match_method="global_matcher",
                confidence=decision.score,
            )
        current.state = IdentityAssignmentState.MATCHED
        current.provisional = False
        if current.global_person_id != gid:
            self._mark_bound(current)
        current.global_person_id = gid
        current.source_video_id = pending.source.get("source_video_id")
        current.source_video_name = pending.source.get("source_video_name")
        current.score = decision.score
        current.second_best = decision.second_best_score
        gal.last_camera_id = current.camera_id
        gal.last_seen_at = timestamp
        gal.last_run_id = self._active_run_id
        self._persist_embeddings(gal, current, timestamp, pending)
        self.store.upsert_person(gal)
        if self.events:
            self.events.publish(
                IDENTITY_MATCHED,
                camera_id=current.camera_id,
                track_id=current.track_id,
                global_person_id=gid,
                score=decision.score,
            )

    def _create_new(
        self,
        current: TrackAssignment,
        decision: IdentityDecision,
        camera_id: str,
        timestamp: float,
        pending: PendingEvidence,
    ) -> None:
        identity_cfg = self.thresholds.get("identity", {})
        gid = self.store.next_person_id(
            prefix=str(identity_cfg.get("new_id_prefix", "P")),
            width=int(identity_cfg.get("new_id_width", 4)),
        )
        gal = PersonGallery(
            global_person_id=gid,
            last_camera_id=camera_id,
            last_seen_at=timestamp,
            first_camera_id=camera_id,
            last_run_id=self._active_run_id,
        )
        self.galleries[gid] = gal
        current.state = IdentityAssignmentState.NEW
        current.provisional = True
        current.global_person_id = gid
        self._mark_bound(current)
        current.source_video_id = pending.source.get("source_video_id")
        current.source_video_name = pending.source.get("source_video_name")
        current.score = 1.0
        decision.global_person_id = gid
        decision.state = IdentityAssignmentState.NEW
        decision.match_kind = MatchKind.NEW
        self._persist_embeddings(gal, current, timestamp, pending)
        self.store.upsert_person(gal)
        # Keep the assignment provisional for a short window.  The first
        # acceptable crop is often a side/occluded entrance view; a later
        # front/back crop may match an existing global identity.
        decision.state = IdentityAssignmentState.MATCHED
        if self.events:
            self.events.publish(
                IDENTITY_CREATED,
                camera_id=camera_id,
                track_id=current.track_id,
                global_person_id=gid,
            )

    def _enroll_similarity(
        self,
        gal: PersonGallery,
        current: TrackAssignment,
        pending: PendingEvidence,
    ) -> Optional[float]:
        """Median similarity of the track's recent crops to the ID's *other* evidence.

        None when the gallery holds nothing but this track's own crops from
        this run (the track founded the ID).
        """
        own = {id(r) for r in self._run_records.get(current.track_id, [])}
        others = [r for r in gal.body if id(r) not in own]
        if not others:
            return None
        crops = pending.recent_bodies or ([pending.body] if pending.body is not None else [])
        if not crops:
            return None
        from src.identity.gallery import supported_similarity

        k = int(self.thresholds.get("reid", {}).get("gallery_support_k", 1))
        sims = sorted(supported_similarity(c, others, k) or 0.0 for c in crops)
        return sims[len(sims) // 2]

    def _may_enroll(
        self,
        gal: PersonGallery,
        current: TrackAssignment,
        pending: PendingEvidence,
    ) -> bool:
        """Gallery purity: only confidently matched tracks add crops to an ID.

        A track may be *assigned* an ID once its median similarity passes
        reid.body_similarity_threshold, but it only *adds* its crops to that
        ID's gallery once the median is at least identity.enroll_threshold.
        Otherwise one borderline match (e.g. a different person in similar
        clothes at 0.52) widens the gallery, and every later look-alike
        matches those added crops instead of the real person - the snowball
        that merged 4p-c0 into two IDs. Face evidence is not gated here.
        """
        thr = self.thresholds.get("identity", {}).get("enroll_threshold")
        if thr is None or pending.body is None:
            return True
        sim = self._enroll_similarity(gal, current, pending)
        return sim is None or sim >= float(thr)

    def _persist_embeddings(
        self,
        gal: PersonGallery,
        current: TrackAssignment,
        timestamp: float,
        pending: PendingEvidence,
    ) -> bool:
        if current.track_id in self._verify:
            # Crossed another person; wait until this track is verified.
            return False
        enroll_body = self._may_enroll(gal, current, pending)
        gallery_size = int(self.thresholds.get("reid", {}).get("gallery_size", 20))
        # Safety default: never delete/overwrite embeddings automatically.
        evict = bool(self.thresholds.get("reid", {}).get("evict_embeddings", False))
        src = pending.source
        upgraded = False

        def _rec(emb: list[float], quality: float, modality: str, model: str, version: str) -> EmbeddingRecord:
            return EmbeddingRecord(
                global_person_id=gal.global_person_id,
                track_id=current.track_id,
                camera_id=current.camera_id,
                timestamp=timestamp,
                embedding=list(emb),
                quality_score=quality,
                viewpoint=pending.viewpoint,
                source_frame_id=int(src.get("source_frame_number") or 0),
                source_video_id=src.get("source_video_id"),
                source_video_name=src.get("source_video_name"),
                source_frame_number=src.get("source_frame_number"),
                source_timestamp_ms=src.get("source_timestamp_ms"),
                model_name=model,
                model_version=version,
                modality=modality,
            )

        body_token = id(pending.body) if pending.body is not None else 0
        if (
            enroll_body
            and pending.body is not None
            and body_token != pending.persisted_body_token
        ):
            rec = _rec(
                pending.body,
                pending.body_quality,
                "body",
                src.get("body_model", "OSNet-x1.0"),
                src.get("body_model_version", ""),
            )
            old_body = list(gal.body)
            if evict:
                gal.body, victim = upgrade_or_insert(gal.body, rec, gallery_size)
            else:
                victim = None
                if rec not in gal.body:
                    gal.body = gal.body + [rec]
            inserted = (
                len(gal.body) > len(old_body)
                or (rec not in old_body and rec in gal.body)
            )
            if victim is not None:
                self.store.remove_embedding(victim)
            if inserted:
                self.store.add_embedding(rec)
                self._run_records.setdefault(current.track_id, []).append(rec)
                upgraded = True
            pending.persisted_body_token = body_token
        face_token = id(pending.face) if pending.face is not None else 0
        if pending.face is not None and face_token != pending.persisted_face_token:
            rec = _rec(
                pending.face,
                pending.face_quality,
                "face",
                src.get("face_model", "AuraFace-v1"),
                src.get("face_model_version", ""),
            )
            old_face = list(gal.face)
            if evict:
                gal.face, victim = upgrade_or_insert(gal.face, rec, gallery_size)
            else:
                victim = None
                if rec not in gal.face:
                    gal.face = gal.face + [rec]
            inserted = (
                len(gal.face) > len(old_face)
                or (rec not in old_face and rec in gal.face)
            )
            if victim is not None:
                self.store.remove_embedding(victim)
            if inserted:
                self.store.add_embedding(rec)
                self._run_records.setdefault(current.track_id, []).append(rec)
                upgraded = True
            pending.persisted_face_token = face_token
        return upgraded

    def _mark_bound(self, current: TrackAssignment) -> None:
        self._bind_seq += 1
        current.bound_seq = self._bind_seq

    def _excluded_ids(self, track_id: str) -> set[str]:
        excluded = set()
        for other in self._cannot_link.get(track_id, ()):
            held = self.assignments.get(other)
            if held is not None and held.global_person_id:
                excluded.add(held.global_person_id)
        return excluded

    def observe_frame(
        self,
        camera_id: str,
        tracks: list[tuple[str, BBox]],
        timestamp: float = 0.0,
    ) -> list[IdentityDecision]:
        """Record which confirmed tracks of one camera are on screen together.

        Call once per camera per frame, before ``assign`` for that frame.  Two
        tracks visible in the same frame are two different people for the rest
        of the run, so neither may take the other's Global ID later - even in
        frames where only one of them is visible.  Boxes that overlap by at
        least ``identity.cannot_link_max_iou`` are skipped because they are
        more likely a duplicate detection of one person.

        If two tracks in this frame already share an ID (it was bound while
        the other was not visible), the track that took it first keeps it and
        the other is split onto a new provisional ID, taking the embeddings it
        added this run with it.  Returns the decisions for split tracks.
        """
        with self._lock:
            identity_cfg = self.thresholds.get("identity", {})
            changed: list[IdentityDecision] = []
            if bool(identity_cfg.get("verify_crossings", True)):
                changed.extend(self._mark_crossings(camera_id, tracks, timestamp))
            if not bool(identity_cfg.get("exclusive_per_frame", True)):
                return changed
            max_iou = float(identity_cfg.get("cannot_link_max_iou", 0.5))
            for i, (ta, ba) in enumerate(tracks):
                for tb, bb in tracks[i + 1 :]:
                    if ta == tb or ba.iou(bb) >= max_iou:
                        continue
                    self._cannot_link.setdefault(ta, set()).add(tb)
                    self._cannot_link.setdefault(tb, set()).add(ta)

            visible = [t for t, _ in tracks]
            holders: dict[str, list[TrackAssignment]] = {}
            for t in visible:
                cur = self.assignments.get(t)
                if cur is not None and cur.global_person_id and cur.camera_id == camera_id:
                    holders.setdefault(cur.global_person_id, []).append(cur)
            split = bool(identity_cfg.get("split_conflicting_tracks", True))
            for gid, group in holders.items():
                if len(group) < 2:
                    continue
                group.sort(key=lambda a: a.bound_seq)
                keeper = group[0]
                for other in group[1:]:
                    if other.track_id not in self._cannot_link.get(keeper.track_id, ()):
                        continue
                    if split:
                        changed.append(self._split_track(other, gid, keeper.track_id, timestamp))

            for t in visible:
                gid_t = self.assignments[t].global_person_id if t in self.assignments else None
                if not gid_t:
                    continue
                for o in self._cannot_link.get(t, ()):
                    held = self.assignments.get(o)
                    if held is not None and held.global_person_id and held.global_person_id != gid_t:
                        self.distinct_pairs.add(frozenset((gid_t, held.global_person_id)))
            return changed

    def _mark_crossings(
        self,
        camera_id: str,
        tracks: list[tuple[str, BBox]],
        timestamp: float,
    ) -> list[IdentityDecision]:
        """Mark tracks whose box overlapped another person's as unverified.

        While two boxes overlap by >= ``identity.crossing_iou`` the tracker may
        hand one person's box to the other track (or to a person that then
        continues under a new track). From that moment the track's identity
        is unknown until its own fresh crops re-establish it
        (``_verify_track``). Returns an "unverified" decision for each newly
        marked track so the displayed ID is withdrawn immediately.
        """
        cfg = self.thresholds.get("identity", {})
        cross_iou = float(cfg.get("crossing_iou", 0.3))
        # A crop may verify once the boxes are below this overlap; crops that
        # still contain two people are dropped by the crowded-crop check.
        clear_iou = float(cfg.get("crossing_clear_iou", cross_iou))
        boxes = dict(tracks)
        overlapping: set[str] = set()
        crossed: set[str] = set()
        for i, (ta, ba) in enumerate(tracks):
            for tb, bb in tracks[i + 1 :]:
                iou = ba.iou(bb)
                if iou >= clear_iou:
                    overlapping.update((ta, tb))
                if iou >= cross_iou:
                    crossed.update((ta, tb))
        changed: list[IdentityDecision] = []
        for t in crossed:
            cur = self.assignments.get(t)
            if cur is None or not cur.global_person_id or cur.camera_id != camera_id:
                continue
            if cur.provisional:
                continue  # provisional IDs are re-matched on every crop anyway
            entry = self._verify.get(t)
            if entry is None:
                self._verify[t] = {"since": timestamp, "tries": 0}
                changed.append(self._unverified_decision(cur, "crossed another person"))
            else:
                entry["since"] = timestamp  # still overlapping: restart
        for t, entry in self._verify.items():
            entry["clear"] = t in boxes and t not in overlapping
        for t in [t for t in self._verify if t not in self.assignments]:
            del self._verify[t]
        return changed

    def _unverified_decision(self, cur: TrackAssignment, why: str) -> IdentityDecision:
        return IdentityDecision(
            track_id=cur.track_id,
            camera_id=cur.camera_id,
            state=IdentityAssignmentState.UNCERTAIN,
            global_person_id=None,
            match_kind=MatchKind.UNRESOLVED,
            score=0.0,
            second_best_score=0.0,
            margin=0.0,
            reason=f"identity unverified: {why}",
        )

    def _verify_track(
        self,
        current: TrackAssignment,
        kw: dict[str, Any],
        timestamp: float,
    ) -> IdentityDecision:
        """Re-establish a track's identity from its own crops after a crossing.

        Only crops taken while the box overlaps nobody count. Each is compared
        with the track's ID and with every other ID (k-th best gallery
        support, like matching):
        - own ID fits and beats every other ID by ``drift_margin``: confirmed;
        - another ID fits and beats the own ID by the margin: move there;
        - otherwise try again; after ``drift_confirmations`` unclear crops,
          keep the own ID if it still fits, else start a new ID.
        """
        entry = self._verify[current.track_id]
        fresh = kw.get("body_embedding")
        gate = float(self.thresholds.get("scoring", {}).get("body_quality_gate", 0.35))
        if (
            not entry.get("clear")
            or fresh is None
            or float(kw.get("body_quality") or 0.0) < gate
        ):
            return self._unverified_decision(current, "waiting for a clear crop")
        cfg = self.thresholds.get("identity", {})
        margin = float(cfg.get("drift_margin", 0.15))
        # Keeping the own ID is the default outcome: it needs the normal
        # match margin. Moving to another ID keeps the stricter drift_margin.
        keep_margin = float(self.thresholds.get("reid", {}).get("minimum_score_margin", 0.08))
        needed = int(cfg.get("drift_confirmations", 3))
        thr = float(self.thresholds.get("reid", {}).get("body_similarity_threshold", 0.62))
        k = int(self.thresholds.get("reid", {}).get("gallery_support_k", 1))
        from src.identity.gallery import supported_similarity

        gid = current.global_person_id
        own_recs = [
            r for r in self.galleries[gid].body
            if not (
                r.track_id == current.track_id
                and r.source_video_id == current.source_video_id
                and r.timestamp >= entry["since"]
            )
        ]
        # No evidence of the own ID left to compare with is not a mismatch.
        own = (supported_similarity(fresh, own_recs, k) or 0.0) if own_recs else None
        best_g, other = None, 0.0
        for g, gal in self.galleries.items():
            if g == gid or gal.status != "active" or not gal.body:
                continue
            sim = supported_similarity(fresh, gal.body, k) or 0.0
            if sim > other:
                best_g, other = g, sim
        entry["tries"] += 1
        own_v = own if own is not None else 0.0
        if own is not None and own >= thr and own >= other + keep_margin:
            del self._verify[current.track_id]
            return self._confirmed_decision(current, own, other)
        if best_g is not None and other >= thr and other >= own_v + margin:
            del self._verify[current.track_id]
            return self._rebind_drifted(current, gid, best_g, fresh, kw, timestamp, own_v, other)
        if entry["tries"] < needed:
            return self._unverified_decision(current, f"unclear ({gid} {own_v:.2f}, best other {other:.2f})")
        # No other ID fits clearly better. Keep the own ID unless it clearly
        # does not fit either (same rule as the drift check): a pose change
        # can pull a genuine match a little under the threshold, and a new
        # ID for an existing person is exactly what must not happen.
        if own is None or own >= thr - margin or other >= thr - margin:
            del self._verify[current.track_id]
            return self._confirmed_decision(current, own_v, other)
        wait = float(cfg.get("unclear_wait_seconds", 3.0))
        if timestamp - entry["since"] < wait:
            return self._unverified_decision(current, f"looks like nobody known ({own_v:.2f})")
        del self._verify[current.track_id]
        return self._rebind_drifted(current, gid, "", fresh, kw, timestamp, own_v, other)

    def _confirmed_decision(self, cur: TrackAssignment, own: float, other: float) -> IdentityDecision:
        cur.state = IdentityAssignmentState.MATCHED
        return IdentityDecision(
            track_id=cur.track_id,
            camera_id=cur.camera_id,
            state=IdentityAssignmentState.MATCHED,
            global_person_id=cur.global_person_id,
            match_kind=MatchKind.EXISTING,
            score=own,
            second_best_score=other,
            margin=own - other,
            reason=f"identity re-verified after crossing ({own:.2f} vs {other:.2f})",
        )

    def _split_track(
        self,
        current: TrackAssignment,
        old_gid: str,
        keeper_track: str,
        timestamp: float,
    ) -> IdentityDecision:
        """Move one track off a shared ID onto a new provisional ID.

        The embeddings this track added under ``old_gid`` during this run are
        re-labelled to the new ID (moved, not discarded).  Evidence from other
        tracks and earlier runs is untouched.
        """
        identity_cfg = self.thresholds.get("identity", {})
        old = self.galleries.get(old_gid)
        gid = self.store.next_person_id(
            prefix=str(identity_cfg.get("new_id_prefix", "P")),
            width=int(identity_cfg.get("new_id_width", 4)),
        )
        gal = PersonGallery(
            global_person_id=gid,
            last_camera_id=current.camera_id,
            last_seen_at=timestamp or (old.last_seen_at if old else 0.0),
            first_camera_id=current.camera_id,
            last_run_id=self._active_run_id,
        )
        self.galleries[gid] = gal
        self.store.upsert_person(gal)
        moved: dict[str, int] = {"body": 0, "face": 0}
        kept_records: list[EmbeddingRecord] = []
        for rec in self._run_records.get(current.track_id, []):
            if rec.global_person_id != old_gid:
                kept_records.append(rec)
                continue
            if old is not None:
                if rec.modality == "body":
                    old.body = [r for r in old.body if r is not rec]
                else:
                    old.face = [r for r in old.face if r is not rec]
            self.store.remove_embedding(rec)
            new_rec = replace(rec, global_person_id=gid)
            if new_rec.modality == "body":
                gal.body = gal.body + [new_rec]
            else:
                gal.face = gal.face + [new_rec]
            self.store.add_embedding(new_rec)
            kept_records.append(new_rec)
            moved[new_rec.modality] += 1
        self._run_records[current.track_id] = kept_records
        if old is not None:
            self.store.upsert_person(old)
        self.store.upsert_person(gal)

        pending = self._pending.get(current.track_id)
        if pending is not None:
            # Re-persist the current best evidence under the new ID only for
            # a modality that had nothing to move.
            if not moved["body"]:
                pending.persisted_body_token = 0
            if not moved["face"]:
                pending.persisted_face_token = 0
        current.global_person_id = gid
        current.provisional = True
        current.state = IdentityAssignmentState.MATCHED
        current.frames_in_state = 0
        current.score = 0.0
        current.second_best = 0.0
        self._mark_bound(current)
        self.distinct_pairs.add(frozenset((old_gid, gid)))
        reason = f"split from {old_gid}: on screen together with {keeper_track}"
        self.store.add_track(
            track_id=current.track_id,
            camera_id=current.camera_id,
            global_person_id=gid,
            start_time=timestamp,
            end_time=None,
            status=current.state.value,
            source_video_id=current.source_video_id,
            source_video_name=current.source_video_name,
            confidence=0.0,
        )
        if self.events:
            self.events.publish(
                IDENTITY_CREATED,
                camera_id=current.camera_id,
                track_id=current.track_id,
                global_person_id=gid,
            )
        return IdentityDecision(
            track_id=current.track_id,
            camera_id=current.camera_id,
            state=IdentityAssignmentState.MATCHED,
            global_person_id=gid,
            match_kind=MatchKind.NEW,
            score=0.0,
            second_best_score=0.0,
            margin=0.0,
            reason=reason,
            split_from=old_gid,
        )

    def resolve_overlap(
        self,
        proposals: list[IdentityDecision],
        simultaneous: bool,
        embeddings: Optional[dict[str, list[float]]] = None,
    ) -> list[IdentityDecision]:
        """Section 17: only when two cameras see what appears to be the same person."""
        overlap_cfg = self.thresholds.get("overlap", {})
        if not simultaneous or not overlap_cfg.get("enabled_only_when_cameras_are_simultaneous", True):
            return proposals
        if len(proposals) < 2 or not embeddings:
            return proposals
        from src.reid.colour import appearance_similarity

        margin = float(overlap_cfg.get("score_margin", 0.10))
        for i, a in enumerate(proposals):
            for b in proposals[i + 1 :]:
                if a.camera_id == b.camera_id:
                    continue
                ea, eb = embeddings.get(a.track_id), embeddings.get(b.track_id)
                if not ea or not eb:
                    continue
                if appearance_similarity(ea, eb) < 0.70:
                    continue
                if not a.global_person_id or not b.global_person_id:
                    continue
                if a.global_person_id == b.global_person_id:
                    continue
                if abs(a.score - b.score) >= margin:
                    loser = a if a.score < b.score else b
                    loser.state = IdentityAssignmentState.UNCERTAIN
                    loser.match_kind = MatchKind.UNRESOLVED
                    loser.reason = "overlap: kept higher-confidence assignment"
                else:
                    for p in (a, b):
                        p.state = IdentityAssignmentState.CONFLICT
                        p.match_kind = MatchKind.CONFLICT
                        p.reason = "overlap conflict: scores too close"
        return proposals

    def list_persons(self) -> list[PersonGallery]:
        with self._lock:
            return list(self.galleries.values())

    def show_person(self, gid: str) -> Optional[PersonGallery]:
        with self._lock:
            return self.galleries.get(gid)

    def reset(self) -> None:
        with self._lock:
            self.store.reset_gallery()
            self.galleries.clear()
            self.assignments.clear()
            self._pending.clear()
            self._cannot_link.clear()
            self._run_records.clear()
            self._verify.clear()
            self._last_body.clear()
            self._drift.clear()
            self.distinct_pairs.clear()

"""Candidate generation. Never compare every new track against every identity.

Order:
1. Camera topology
2. Time window
3. Recently active identities
4. Recently missing identities
5. Same-camera historical identities
6. Approximate embedding nearest-neighbor search
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from src.identity.gallery import PersonGallery, best_cosine
from src.identity.topology import CameraTopology
from src.reid.colour import appearance_similarity
from src.types import EmbeddingRecord


@dataclass
class Candidate:
    global_person_id: str
    approx_body: float
    approx_face: float
    last_camera_id: Optional[str]
    last_seen_at: float
    reasons: list[str]


def generate_candidates(
    galleries: dict[str, PersonGallery],
    *,
    camera_id: str,
    timestamp: float,
    body_embedding: Optional[list[float]],
    face_embedding: Optional[list[float]],
    topology: CameraTopology,
    time_window: float = 180.0,
    candidate_k: int = 32,
    topology_enabled: bool = True,
    neighbor_hits: Optional[Iterable[str]] = None,
    source_video_id: Optional[str] = None,
    run_id: Optional[str] = None,
) -> list[Candidate]:
    scored: list[Candidate] = []
    neighbor_set = set(neighbor_hits or [])
    for gid, gal in galleries.items():
        if gal.status != "active":
            continue
        dt = timestamp - gal.last_seen_at if gal.last_seen_at else 0.0
        reasons: list[str] = []
        last_cam = gal.last_camera_id
        cross_video = gal.is_from_other_run(source_video_id, run_id)
        if cross_video:
            # Video-file timestamps restart at zero for a later run.  Keep
            # the identity candidate visible even when the old timestamp is
            # numerically in the future; the matcher will apply a neutral
            # temporal/topology context for this source boundary.
            reasons.append("cross_video")
        if last_cam == camera_id:
            reasons.append("same_camera")
        if 0 <= dt <= time_window:
            reasons.append("recent")
        if last_cam and last_cam != camera_id:
            if topology.is_plausible(last_cam, camera_id, dt, enabled=topology_enabled):
                reasons.append("topology")
            elif topology_enabled and topology.edge(last_cam, camera_id) is None:
                # Still consider via embedding NN; topology will down-weight later.
                reasons.append("no_edge")
        if gid in neighbor_set:
            reasons.append("nn")
        body_sim = best_cosine(body_embedding, gal.body) if body_embedding and gal.body else -1.0
        face_sim = best_cosine(face_embedding, gal.face) if face_embedding and gal.face else -1.0
        if body_sim is None:
            body_sim = -1.0
        if face_sim is None:
            face_sim = -1.0
        if not reasons and body_sim < 0.2 and face_sim < 0.2:
            continue
        scored.append(
            Candidate(
                global_person_id=gid,
                approx_body=float(body_sim),
                approx_face=float(face_sim),
                last_camera_id=last_cam,
                last_seen_at=gal.last_seen_at,
                reasons=reasons or ["embedding"],
            )
        )
    scored.sort(key=lambda c: max(c.approx_body, c.approx_face), reverse=True)
    return scored[:candidate_k]


def nearest_neighbors(
    query: list[float],
    records: Iterable[tuple[str, EmbeddingRecord]],
    k: int = 16,
) -> list[str]:
    ranked: list[tuple[float, str]] = []
    for gid, rec in records:
        ranked.append((appearance_similarity(query, rec.embedding), gid))
    ranked.sort(reverse=True)
    seen = []
    for _, gid in ranked:
        if gid not in seen:
            seen.append(gid)
        if len(seen) >= k:
            break
    return seen

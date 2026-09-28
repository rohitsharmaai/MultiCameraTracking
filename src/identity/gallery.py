"""Per-identity multi-embedding gallery and eviction policy (Section 14).

gallery_size is a hard cap per identity, applied separately to body and face.

When the cap is exceeded:
- never overwrite an existing embedding;
- never drop the last embedding of a viewpoint if another viewpoint still has extras;
- evict the lowest-quality item first;
- if quality is tied, evict the oldest item.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from src.types import EmbeddingRecord, Viewpoint


@dataclass
class PersonGallery:
    global_person_id: str
    body: list[EmbeddingRecord] = field(default_factory=list)
    face: list[EmbeddingRecord] = field(default_factory=list)
    last_camera_id: str | None = None
    last_seen_at: float = 0.0
    first_camera_id: str | None = None
    status: str = "active"
    # Run-local cache metadata; persisted embeddings remain the source of
    # truth and older databases do not need a migration.
    last_run_id: str | None = None

    def items(self, modality: str) -> list[EmbeddingRecord]:
        return self.body if modality == "body" else self.face

    def is_from_other_source(self, source_video_id: Optional[str]) -> bool:
        """Return true when the query comes from a different known video run.

        ``None`` means the caller did not provide provenance, and an empty
        gallery has no source to compare.  In both cases the caller should fall
        back to the normal timestamp/topology logic.
        """
        if not source_video_id:
            return False
        known = {
            rec.source_video_id
            for rec in self.body + self.face
            if rec.source_video_id
        }
        return bool(known and source_video_id not in known)

    def is_from_other_run(
        self,
        source_video_id: Optional[str],
        run_id: Optional[str],
    ) -> bool:
        """Prefer the active run token when it is available.

        A run token distinguishes synchronized cameras (different source
        files, but one logical run) from a later video execution.  After a
        process restart the token is unavailable, so source provenance and
        the timestamp-reset check provide the fallback.
        """
        if run_id and self.last_run_id:
            return run_id != self.last_run_id
        return self.is_from_other_source(source_video_id)


def _same_source(a: EmbeddingRecord, b: EmbeddingRecord) -> bool:
    if a.source_video_id and b.source_video_id:
        return a.source_video_id == b.source_video_id
    if a.source_video_name or b.source_video_name:
        return a.source_video_name == b.source_video_name
    return True


def evict_then_insert(
    items: list[EmbeddingRecord],
    new_item: EmbeddingRecord,
    gallery_size: int,
) -> list[EmbeddingRecord]:
    """Delete the lowest-value item if needed, then insert. Never overwrite in place."""
    upgraded, _victim = upgrade_or_insert(items, new_item, gallery_size)
    return upgraded


def upgrade_or_insert(
    items: list[EmbeddingRecord],
    new_item: EmbeddingRecord,
    gallery_size: int,
) -> tuple[list[EmbeddingRecord], EmbeddingRecord | None]:
    """Insert, or replace a weaker same-pose / unknown entrance crop.

    Keep different viewpoints. Update the stored base only when a later
    frame of the *same local track* has a better pose or a real face.
    Evidence from another track or video is inserted, not overwritten.
    """
    same_track = [
        (i, it)
        for i, it in enumerate(items)
        if it.track_id == new_item.track_id
        and _same_source(it, new_item)
    ]
    same = [(i, it) for i, it in same_track if it.viewpoint == new_item.viewpoint]
    worse_same = [i for i, it in same if it.quality_score < new_item.quality_score]
    better_same = [it for _, it in same if it.quality_score >= new_item.quality_score]
    if better_same and not worse_same:
        return items, None
    if worse_same:
        victim_i = min(worse_same, key=lambda i: items[i].quality_score)
        victim = items[victim_i]
        kept = [it for i, it in enumerate(items) if i != victim_i]
        return kept + [new_item], victim

    if new_item.viewpoint != Viewpoint.UNKNOWN:
        unknown_worse = [
            i
            for i, it in same_track
            if it.viewpoint == Viewpoint.UNKNOWN and it.quality_score < new_item.quality_score
        ]
        if unknown_worse:
            victim_i = min(unknown_worse, key=lambda i: items[i].quality_score)
            victim = items[victim_i]
            kept = [it for i, it in enumerate(items) if i != victim_i]
            return kept + [new_item], victim

    if len(items) < gallery_size:
        return items + [new_item], None
    victim_idx = _choose_victim(items)
    victim = items[victim_idx]
    kept = [it for i, it in enumerate(items) if i != victim_idx]
    return kept + [new_item], victim


def _choose_victim(items: list[EmbeddingRecord]) -> int:
    counts: dict[Viewpoint, int] = {}
    for it in items:
        counts[it.viewpoint] = counts.get(it.viewpoint, 0) + 1
    candidates = []
    for i, it in enumerate(items):
        # Never drop the last embedding of a viewpoint if another viewpoint has extras.
        extras_exist = any(v != it.viewpoint and c > 1 for v, c in counts.items())
        last_of_kind = counts.get(it.viewpoint, 0) == 1
        if last_of_kind and extras_exist:
            continue
        candidates.append(i)
    if not candidates:
        candidates = list(range(len(items)))
    # Lowest quality first; if tied, oldest timestamp.
    candidates.sort(key=lambda i: (items[i].quality_score, items[i].timestamp))
    return candidates[0]


def supported_similarity(
    query: list[float],
    gallery: Iterable[EmbeddingRecord],
    k: int = 1,
) -> float | None:
    """k-th best similarity of ``query`` to ``gallery`` (k=1: the best).

    With k>1 a match needs support from several gallery crops. A single
    contaminated crop in a gallery (another person walking in front, the
    moment of a tracker swap) otherwise attracts everyone who resembles
    that other person: on 4p-c0 the best-single-crop score let different
    people reach 0.80, the 3rd-best at most 0.55. For a gallery smaller
    than k the weakest match is used.
    """
    from src.reid.colour import appearance_similarity

    sims = sorted((appearance_similarity(query, rec.embedding) for rec in gallery), reverse=True)
    if not sims:
        return None
    return sims[min(max(1, k), len(sims)) - 1]


def best_cosine(query: list[float], gallery: Iterable[EmbeddingRecord]) -> float | None:
    sim, _rec = best_cosine_record(query, gallery)
    return sim


def best_cosine_record(
    query: list[float],
    gallery: Iterable[EmbeddingRecord],
) -> tuple[float | None, EmbeddingRecord | None]:
    """Best similarity and the record that produced it.

    The record is returned so callers can inspect its viewpoint (used to
    decide whether a cross-view face match is reliable enough to trust).
    """
    from src.reid.colour import appearance_similarity

    best = None
    best_rec = None
    for rec in gallery:
        sim = appearance_similarity(query, rec.embedding)
        if best is None or sim > best:
            best = sim
            best_rec = rec
    return best, best_rec

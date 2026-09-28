"""Directed camera graph. Topology is a major false-match prevention mechanism."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class TopologyEdge:
    from_camera: str
    to_camera: str
    min_transition_seconds: float
    max_transition_seconds: float
    expected_direction: str
    confidence: float


class CameraTopology:
    def __init__(self, spec: dict[str, Any] | None = None) -> None:
        spec = spec or {}
        self.edges: dict[tuple[str, str], TopologyEdge] = {}
        for raw in spec.get("edges") or []:
            edge = TopologyEdge(
                from_camera=str(raw["from_camera"]),
                to_camera=str(raw["to_camera"]),
                min_transition_seconds=float(raw.get("min_transition_seconds", 0.5)),
                max_transition_seconds=float(raw.get("max_transition_seconds", 180.0)),
                expected_direction=str(raw.get("expected_direction", "")),
                confidence=float(raw.get("confidence", 0.5)),
            )
            self.edges[(edge.from_camera, edge.to_camera)] = edge

    def edge(self, src: str, dst: str) -> Optional[TopologyEdge]:
        return self.edges.get((src, dst))

    def is_plausible(
        self,
        src: str,
        dst: str,
        dt_seconds: float,
        *,
        enabled: bool = True,
        temporal_enabled: bool = True,
        same_camera_ok: bool = True,
    ) -> bool:
        if not enabled:
            return True
        if src == dst:
            return same_camera_ok
        edge = self.edge(src, dst)
        if edge is None:
            return False
        if not temporal_enabled:
            return True
        return edge.min_transition_seconds <= dt_seconds <= edge.max_transition_seconds

    def camera_plausibility(
        self,
        src: Optional[str],
        dst: str,
        dt_seconds: float,
        *,
        enabled: bool = True,
        temporal_enabled: bool = True,
        unknown_edge_penalty: float = 0.15,
    ) -> float:
        """Return a [0, 1] term. Same camera re-entry is fully plausible."""
        if not enabled or src is None:
            return 1.0
        if src == dst:
            return 1.0
        edge = self.edge(src, dst)
        if edge is None:
            # No directed edge: transition is not physically modeled.
            return 0.0
        if not temporal_enabled:
            return float(edge.confidence)
        if dt_seconds < edge.min_transition_seconds or dt_seconds > edge.max_transition_seconds:
            return 0.0
        # Peak at the midpoint of the allowed window.
        mid = 0.5 * (edge.min_transition_seconds + edge.max_transition_seconds)
        span = max(edge.max_transition_seconds - edge.min_transition_seconds, 1e-3)
        closeness = 1.0 - abs(dt_seconds - mid) / span
        return max(0.0, min(1.0, edge.confidence * (0.5 + 0.5 * closeness)))

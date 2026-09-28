"""Identity evaluation helpers. False merges are particularly important."""

from __future__ import annotations

from collections import defaultdict


def id_switch_count(assignments: list[tuple[str, str]]) -> int:
    """assignments: (ground_truth_id, predicted_global_id) in time order per track."""
    last: dict[str, str] = {}
    switches = 0
    for gt, pred in assignments:
        if gt in last and last[gt] != pred and pred and last[gt]:
            switches += 1
        if pred:
            last[gt] = pred
    return switches


def unresolved_rate(states: list[str]) -> float:
    if not states:
        return 0.0
    return sum(1 for s in states if s in {"UNRESOLVED", "UNCERTAIN", "CONFLICT"}) / len(states)


def false_merge_rate(pred_to_gts: dict[str, set[str]]) -> float:
    if not pred_to_gts:
        return 0.0
    merges = sum(1 for gts in pred_to_gts.values() if len(gts) > 1)
    return merges / len(pred_to_gts)

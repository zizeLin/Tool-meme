"""Risk-aware STOP gate used by the RL-ATR router."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from tools.utils import clamp


def executable_outputs(outputs: Iterable[Dict]) -> List[Dict]:
    return [
        output
        for output in outputs or []
        if not output.get("was_skipped")
        and not output.get("was_short_circuited")
        and not output.get("was_not_selected")
    ]


def confidence_stats(outputs: Iterable[Dict]) -> Dict[str, float]:
    values = [clamp(float(output.get("confidence_score", output.get("conf", 0.0)))) for output in executable_outputs(outputs)]
    if not values:
        return {"max_conf": 0.0, "avg_conf": 0.0, "spread": 0.0}
    return {"max_conf": max(values), "avg_conf": sum(values) / len(values), "spread": max(values) - min(values)}


def conflict_score(outputs: Iterable[Dict]) -> float:
    valid = [
        output for output in executable_outputs(outputs)
        if output.get("pred") in {"harmful", "harmless"}
    ]
    if len(valid) < 2:
        return 0.0
    conflict = 0.0
    total = 0.0
    for i, left in enumerate(valid):
        for right in valid[i + 1:]:
            weight = clamp(float(left.get("confidence_score", left.get("conf", 0.0)))) * clamp(
                float(right.get("confidence_score", right.get("conf", 0.0)))
            )
            total += weight
            if left.get("pred") != right.get("pred"):
                conflict += weight
    return conflict / max(total, 1e-8)


def stop_is_legal(
    outputs: Sequence[Dict],
    unresolved_mandatory: Sequence[str],
    tau_stop: float = 0.80,
    tau_conflict: float = 0.30,
    weak_retrieval_only: bool = False,
) -> bool:
    stats = confidence_stats(outputs)
    return (
        stats["max_conf"] >= tau_stop
        and not unresolved_mandatory
        and conflict_score(outputs) <= tau_conflict
        and not weak_retrieval_only
    )

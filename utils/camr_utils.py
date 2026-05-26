"""Helpers for Context-Augmented Multimodal Retrieval outputs."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import numpy as np

from utils.tool_meme_config import (
    RETRIEVAL_CONTEXT_THRESHOLD,
    RETRIEVAL_DELTA,
    RETRIEVAL_NEIGHBOR_THRESHOLD,
    RETRIEVAL_TEMPERATURE,
    RETRIEVAL_TOP_THRESHOLD,
)


EVENT_KEYS = ("event_context", "event", "context", "event_contexts", "events")


def extract_event_context(raw_item: Optional[Dict]) -> Optional[str]:
    if not raw_item:
        return None
    for key in EVENT_KEYS:
        value = raw_item.get(key)
        if isinstance(value, list):
            value = "; ".join(str(item) for item in value if item)
        if value:
            return str(value)
    return None


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(value)))


def compute_base_rate(labels: Iterable[Optional[int]]) -> float:
    valid = [int(label) for label in labels if label is not None]
    if not valid:
        return 0.5
    return float(sum(valid) / len(valid))


def calibrate_retrieval(
    labels: List[int],
    scores: List[float],
    base_rate: float,
    temperature: float = RETRIEVAL_TEMPERATURE,
    top_threshold: float = RETRIEVAL_TOP_THRESHOLD,
    delta: float = RETRIEVAL_DELTA,
    neighbor_threshold: float = RETRIEVAL_NEIGHBOR_THRESHOLD,
    context_threshold: float = RETRIEVAL_CONTEXT_THRESHOLD,
) -> Dict[str, float | str]:
    if not labels or not scores:
        return {
            "raw_label_prior": float(base_rate),
            "label_prior": float(base_rate),
            "retrieval_reliability": 0.0,
            "context_mode": "no-context",
        }

    clipped_scores = np.asarray(scores[: len(labels)], dtype=np.float64)
    clipped_scores = np.nan_to_num(clipped_scores, nan=0.0, posinf=0.0, neginf=0.0)
    label_array = np.asarray(labels[: len(clipped_scores)], dtype=np.float64)

    stable_scores = clipped_scores / max(temperature, 1e-6)
    stable_scores = stable_scores - np.max(stable_scores)
    weights = np.exp(stable_scores)
    raw_prior = float(np.sum(weights * label_array) / max(float(np.sum(weights)), 1e-8))

    best_score = float(np.max(clipped_scores)) if clipped_scores.size else 0.0
    strong_neighbor_fraction = float(np.mean(clipped_scores > neighbor_threshold))
    reliability = sigmoid((best_score - top_threshold) / max(delta, 1e-6))
    reliability *= strong_neighbor_fraction
    reliability = float(np.clip(reliability, 0.0, 1.0))

    prior = reliability * raw_prior + (1.0 - reliability) * float(base_rate)
    mode = "context" if reliability >= context_threshold else "no-context"
    return {
        "raw_label_prior": raw_prior,
        "label_prior": float(prior),
        "retrieval_reliability": reliability,
        "context_mode": mode,
    }

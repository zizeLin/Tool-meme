"""Structured feature encoder for the lightweight Tool-meme router policy."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from tools.utils import clamp, normalize, tokenize

RHETORICAL_MARKERS = ["/s", "yeah right", "sure", "totally", "as if", "literally", "obviously", "clearly", "lol", "lmao", "jk"]
CULTURAL_MARKERS = ["pepe", "wojak", "doomer", "boomer", "kek", "npc", "chad", "soyboy", "meme", "rickroll", "doge"]


def get_feature_names(tool_names: Sequence[str]) -> List[str]:
    base = [
        "bias",
        "token_len",
        "char_len",
        "exclamation_count",
        "question_count",
        "uppercase_ratio",
        "digit_ratio",
        "retrieved_count",
        "event_count",
        "harmful_prior",
        "label_entropy",
        "avg_similarity",
        "alignment_score",
        "prev_mean_conf",
        "prev_max_conf",
        "prev_last_conf",
        "prev_conf_std",
        "prev_output_count",
        "step_progress",
        "remaining_ratio",
        "rhetorical_marker_ratio",
        "cultural_marker_ratio",
        "retrieval_conflict",
        "confidence_trend",
    ]
    return base + [f"executed::{tool}" for tool in tool_names] + [f"available::{tool}" for tool in tool_names]


def build_router_feature_vector(
    query_text: str,
    retrieved_context: Optional[Dict],
    previous_outputs: Optional[Dict[str, Dict]],
    executed_tools: Sequence[str],
    available_tools: Sequence[str],
    tool_names: Sequence[str],
    step_index: int,
    total_tools: int,
) -> np.ndarray:
    query_text = str(query_text or "")
    retrieved_context = retrieved_context or {}
    previous_outputs = previous_outputs or {}
    tokens = tokenize(query_text)
    alpha_chars = [char for char in query_text if char.isalpha()]
    digit_chars = [char for char in query_text if char.isdigit()]
    labels = retrieved_context.get("labels") or []
    scores = [float(score) for score in retrieved_context.get("scores", []) or [] if score is not None]
    harmful_prior = _safe_mean([float(label) for label in labels], default=0.5) if labels else 0.5
    alignment_score = retrieved_context.get("alignment_score")
    alignment_norm = 0.5 if alignment_score is None else clamp((float(alignment_score) + 1.0) / 2.0)
    prev_conf = [
        clamp(float(out.get("confidence_score", out.get("conf", 0.0))))
        for out in previous_outputs.values()
        if not out.get("was_skipped") and not out.get("was_short_circuited") and not out.get("was_not_selected")
    ]
    prev_mean = _safe_mean(prev_conf)
    prev_last = prev_conf[-1] if prev_conf else 0.0
    values = [
        1.0,
        _ratio(len(tokens), 64.0),
        _ratio(len(query_text), 300.0),
        _ratio(query_text.count("!"), 5.0),
        _ratio(query_text.count("?"), 5.0),
        len([char for char in alpha_chars if char.isupper()]) / max(1, len(alpha_chars)),
        len(digit_chars) / max(1, len(query_text)),
        _ratio(len(retrieved_context.get("retrieved_texts") or []), 10.0),
        _ratio(len(retrieved_context.get("event_contexts") or []), 5.0),
        harmful_prior,
        _binary_entropy(harmful_prior),
        clamp(_safe_mean(scores, default=0.5)),
        alignment_norm,
        prev_mean,
        max(prev_conf) if prev_conf else 0.0,
        prev_last,
        _safe_std(prev_conf),
        _ratio(len(previous_outputs), max(1, total_tools)),
        step_index / max(1, total_tools),
        len(available_tools) / max(1, total_tools),
        _ratio(_count_markers(query_text, RHETORICAL_MARKERS), 4.0),
        _ratio(_count_markers(query_text, CULTURAL_MARKERS), 4.0),
        1.0 - abs(2.0 * harmful_prior - 1.0),
        clamp(prev_last - prev_mean + 0.5),
    ]
    values.extend(_flags(tool_names, executed_tools))
    values.extend(_flags(tool_names, available_tools))
    return np.asarray(values, dtype=np.float32)


def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    return float(sum(values) / len(values)) if values else default


def _safe_std(values: Sequence[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float32))) if len(values) > 1 else 0.0


def _ratio(value: float, scale: float) -> float:
    return 0.0 if scale <= 0 else clamp(value / scale)


def _binary_entropy(prob: float) -> float:
    prob = clamp(prob, 1e-6, 1.0 - 1e-6)
    return float(-(prob * math.log(prob) + (1.0 - prob) * math.log(1.0 - prob)) / math.log(2.0))


def _count_markers(text: str, markers: Sequence[str]) -> int:
    lowered = normalize(text)
    return sum(1 for marker in markers if marker in lowered)


def _flags(tool_names: Sequence[str], selected: Sequence[str]) -> List[float]:
    selected_set = set(selected)
    return [1.0 if tool in selected_set else 0.0 for tool in tool_names]

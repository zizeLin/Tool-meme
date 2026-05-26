"""Multi-Perspective Reasoning Ensemble for Tool-meme."""

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional

from utils.openai_client import OpenAIModelConfig, get_openai_response_with_parts
from utils.prompting import HARMFUL_MEME_RUBRIC, JSON_ONLY_RULE


MPRE_SYSTEM_PROMPT = """You are the Multi-Perspective Reasoning Ensemble (MPRE) module for harmful meme detection.
You receive structured tool outputs and CAMR context. Separate harmful evidence, harmless evidence, conflicts, and unresolved uncertainty.
Use this rubric consistently:
""" + HARMFUL_MEME_RUBRIC + """

Return ONLY JSON:
{
  "harmful_evidence": ["..."],
  "harmless_evidence": ["..."],
  "unresolved": ["..."],
  "conflicts": ["..."],
  "consensus_summary": "...",
  "confidence": 0.0
}
Evidence rules:
- Prefer target-specific evidence over retrieved-label majority.
- Put weak or out-of-scope signals in unresolved, not harmful_evidence.
- A high CAMR harmful prior is supporting evidence only when retrieved texts are semantically close to the target.
- Keep evidence concise and inspectable. Do not expose hidden chain-of-thought.
""" + JSON_ONLY_RULE


def _parse_json(text: str) -> Dict:
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
    return {}


def _valid_outputs(atr_output: Dict) -> List[Dict]:
    outputs = atr_output.get("outputs") if isinstance(atr_output, dict) else []
    return [
        output for output in outputs or []
        if output.get("tool")
        and not output.get("was_skipped")
        and not output.get("was_short_circuited")
        and not output.get("was_not_selected")
        and output.get("status", "valid") != "failed"
    ]


def _softmax(values: List[float]) -> List[float]:
    if not values:
        return []
    scale = max(values)
    exp_values = [math.exp(value - scale) for value in values]
    total = sum(exp_values) or 1.0
    return [value / total for value in exp_values]


def _conflict_score(outputs: List[Dict], weights: List[float]) -> float:
    labeled = [
        (idx, output) for idx, output in enumerate(outputs)
        if output.get("pred") in {"harmful", "harmless"}
    ]
    if len(labeled) < 2:
        return 0.0
    numerator = 0.0
    denominator = 0.0
    for left_pos, (left_idx, left) in enumerate(labeled):
        for right_idx, right in labeled[left_pos + 1:]:
            pair_weight = weights[left_idx] * weights[right_idx]
            denominator += pair_weight
            if left.get("pred") != right.get("pred"):
                numerator += pair_weight
    return float(numerator / max(denominator, 1e-8))


def _fallback_summary(outputs: List[Dict], weights: List[float], conflict: float) -> Dict:
    harmful = []
    harmless = []
    unresolved = []
    for output, weight in zip(outputs, weights):
        trace = output.get("trace") or output.get("reasoning_trace") or output.get("tool", "")
        item = f"{output.get('tool')}: {trace}"
        pred = output.get("pred")
        if pred == "harmful":
            harmful.append(item)
        elif pred == "harmless":
            harmless.append(item)
        else:
            unresolved.append(item)
        output["mpre_weight"] = weight
    summary_parts = []
    if harmful:
        summary_parts.append(f"harmful evidence from {len(harmful)} tool(s)")
    if harmless:
        summary_parts.append(f"harmless evidence from {len(harmless)} tool(s)")
    if unresolved:
        summary_parts.append(f"{len(unresolved)} unresolved signal(s)")
    consensus = "; ".join(summary_parts) or "No valid tool evidence."
    avg_conf = sum(float(output.get("confidence_score", 0.0)) * weight for output, weight in zip(outputs, weights))
    return {
        "harmful_evidence": harmful[:4],
        "harmless_evidence": harmless[:4],
        "unresolved": unresolved[:4],
        "conflicts": [f"weighted conflict={conflict:.3f}"] if conflict > 0 else [],
        "consensus_summary": consensus,
        "confidence": float(avg_conf),
    }


def run_mpre(
    atr_output: Dict,
    config: Optional[OpenAIModelConfig] = None,
    camr_package: Optional[Dict] = None,
) -> Dict:
    outputs = _valid_outputs(atr_output)
    if not outputs:
        return {
            "harmful_evidence": [],
            "harmless_evidence": [],
            "unresolved": ["No valid ATR tool output."],
            "conflicts": [],
            "consensus_summary": "No valid tool evidence.",
            "confidence": 0.0,
            "conflict_score": 0.0,
            "tool_weights": {},
        }

    relevance = [1.0 for _ in outputs]
    scores = [
        float(output.get("confidence_score", output.get("conf", 0.0))) + rel - 0.5
        for output, rel in zip(outputs, relevance)
    ]
    weights = _softmax(scores)
    conflict = _conflict_score(outputs, weights)
    fallback = _fallback_summary(outputs, weights, conflict)

    tool_lines = []
    for output, weight in zip(outputs, weights):
        tool_lines.append(json.dumps({
            "tool": output.get("tool"),
            "pred": output.get("pred", "unknown"),
            "confidence": output.get("confidence_score", 0.0),
            "weight": round(weight, 4),
            "trace": output.get("trace") or output.get("reasoning_trace", ""),
            "evidence": output.get("evidence", []),
        }, ensure_ascii=False))

    parts = [
        {"type": "text", "text": MPRE_SYSTEM_PROMPT},
        {"type": "text", "text": (
            "ATR tool outputs:\n"
            + "\n".join(tool_lines)
            + "\n\nCAMR package:\n"
            + json.dumps(camr_package or {}, ensure_ascii=False)
            + f"\n\nWeighted conflict score: {conflict:.4f}\n"
            "Return JSON only."
        )},
    ]
    cfg = config or OpenAIModelConfig.from_env()
    try:
        parsed = _parse_json(get_openai_response_with_parts(parts, cfg))
    except Exception:
        parsed = {}

    result = {**fallback, **{key: value for key, value in parsed.items() if value not in (None, "")}}
    result["conflict_score"] = conflict
    result["tool_weights"] = {
        output.get("tool"): round(weight, 6)
        for output, weight in zip(outputs, weights)
    }
    result["used_tools"] = [output.get("tool") for output in outputs]
    result["short_summary"] = result.get("consensus_summary", "")
    return result

"""Consensus-Based Decision Fusion for Tool-meme."""

from __future__ import annotations

import json
from typing import Dict, Iterable, Optional

from utils.tool_meme_config import (
    TAU_CONFIDENCE_SPLIT,
    TAU_CONFLICT,
    TAU_DECISION,
    label_to_text,
)
from utils.openai_client import OpenAIModelConfig, get_openai_response_with_parts
from utils.prompting import HARMFUL_MEME_RUBRIC, JSON_ONLY_RULE


CBDF_SYSTEM_PROMPT = """You are the Consensus-Based Decision Fusion (CBDF) module for harmful meme detection.
You receive only structured evidence. Produce a calibrated harmful/harmless decision and an inspectable evidence chain.
Use this rubric:
""" + HARMFUL_MEME_RUBRIC + """

Return ONLY JSON:
{
  "label": "harmful|harmless",
  "confidence": 0.0,
  "evidence_chain": ["..."],
  "reason": "...",
  "human_review": false
}
Rules:
- Prioritize target-specific harmful/harmless evidence from MPRE.
- Use CAMR label prior as calibration, not as a replacement for target evidence.
- If evidence is ambiguous, choose the label with stronger concrete evidence and set lower confidence.
- Do not use any other label space.
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


def _label_stats(labels: Iterable[int]) -> Dict[str, float]:
    labels = [int(label) for label in labels if label is not None]
    total = len(labels)
    harmful = sum(1 for label in labels if label == 1)
    return {
        "total": total,
        "harmful": harmful,
        "harmless": total - harmful,
        "ratio_harmful": harmful / total if total else 0.5,
    }


def _tool_vote(mpre_output: Dict) -> float:
    harmful = len(mpre_output.get("harmful_evidence") or [])
    harmless = len(mpre_output.get("harmless_evidence") or [])
    total = harmful + harmless
    if total == 0:
        return 0.5
    return harmful / total


def _bounded(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _deterministic_decision(
    mpre_output: Dict,
    labels,
    camr_package: Optional[Dict],
    atr_output: Optional[Dict],
) -> Dict:
    stats = _label_stats(labels)
    camr_package = camr_package or {}
    atr_output = atr_output or {}
    prior = float(camr_package.get("label_prior", stats["ratio_harmful"]))
    reliability = float(camr_package.get("retrieval_reliability", 0.0))
    context_mode = camr_package.get("context_mode", "no-context")
    tool_vote = _tool_vote(mpre_output)
    tool_conf = float(mpre_output.get("confidence", atr_output.get("confidence_mean", 0.0)))
    conflict = float(mpre_output.get("conflict_score", atr_output.get("conflict_score", 0.0)))
    weighted_harm = 0.55 * tool_vote + 0.45 * (reliability * prior + (1.0 - reliability) * 0.5)
    label = "harmful" if weighted_harm >= 0.5 else "harmless"
    confidence = 0.5 + abs(weighted_harm - 0.5)
    confidence = 0.65 * confidence + 0.35 * tool_conf
    confidence = _bounded(confidence)
    confidence_spread = float(atr_output.get("confidence_spread", 0.0))
    unresolved = mpre_output.get("unresolved") or []
    human_review = (
        conflict > TAU_CONFLICT
        or confidence_spread > TAU_CONFIDENCE_SPLIT
        or (context_mode == "no-context" and confidence < TAU_DECISION)
        or bool(unresolved and confidence < TAU_DECISION)
    )
    evidence_chain = []
    for key in ("harmful_evidence", "harmless_evidence", "conflicts", "unresolved"):
        evidence_chain.extend(str(item) for item in (mpre_output.get(key) or [])[:2])
    if not evidence_chain:
        evidence_chain.append("Decision relies on CAMR label prior and available tool confidence.")
    return {
        "label": label,
        "label_id": 1 if label == "harmful" else 0,
        "confidence": confidence,
        "evidence_chain": evidence_chain[:6],
        "reason": mpre_output.get("consensus_summary", ""),
        "human_review": human_review,
        "signals": {
            "camr_label_prior": prior,
            "camr_reliability": reliability,
            "context_mode": context_mode,
            "tool_vote_harmful": tool_vote,
            "tool_confidence": tool_conf,
            "conflict_score": conflict,
            "confidence_spread": confidence_spread,
            "retrieved_label_stats": stats,
        },
        "short_summary": "; ".join(evidence_chain[:3]),
    }


def run_cbdf(
    mpre_output: Dict,
    labels,
    config: Optional[OpenAIModelConfig] = None,
    camr_package: Optional[Dict] = None,
    atr_output: Optional[Dict] = None,
    use_lmm: bool = False,
) -> Dict:
    decision = _deterministic_decision(
        mpre_output=mpre_output or {},
        labels=labels,
        camr_package=camr_package,
        atr_output=atr_output,
    )
    if not use_lmm:
        return decision

    payload = {
        "mpre": mpre_output,
        "camr": camr_package or {},
        "atr_diagnostics": {
            "routing_mode": (atr_output or {}).get("routing_mode"),
            "tools_executed": (atr_output or {}).get("tools_executed"),
            "conflict_score": (atr_output or {}).get("conflict_score"),
        },
        "deterministic_decision": {
            key: value for key, value in decision.items()
            if key not in {"signals"}
        },
    }
    parts = [
        {"type": "text", "text": CBDF_SYSTEM_PROMPT},
        {"type": "text", "text": json.dumps(payload, ensure_ascii=False) + "\nReturn JSON only."},
    ]
    cfg = config or OpenAIModelConfig.from_env()
    try:
        parsed = _parse_json(get_openai_response_with_parts(parts, cfg))
    except Exception:
        parsed = {}
    label = parsed.get("label")
    if label not in {"harmful", "harmless"}:
        return decision
    parsed["label_id"] = 1 if label == "harmful" else 0
    parsed["confidence"] = _bounded(parsed.get("confidence", decision["confidence"]))
    parsed.setdefault("human_review", decision["human_review"])
    parsed.setdefault("evidence_chain", decision["evidence_chain"])
    parsed.setdefault("reason", decision["reason"])
    parsed.setdefault("short_summary", decision["short_summary"])
    parsed["signals"] = decision["signals"]
    parsed["label_text"] = label_to_text(parsed["label_id"])
    return parsed

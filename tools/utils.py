import re
from typing import Dict, List, Optional


def clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def tokenize(text: str) -> List[str]:
    text = normalize(text)
    return re.findall(r"[a-z0-9']+", text)


def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def summarize_previous(previous_outputs) -> str:
    if not previous_outputs:
        return "None."
    lines = []
    for name, out in previous_outputs.items():
        if out.get("was_skipped") or out.get("was_short_circuited") or out.get("was_not_selected"):
            continue
        trace = out.get("trace") or out.get("reasoning_trace", "")
        score = out.get("conf", out.get("confidence_score", 0.0))
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = 0.0
        pred = out.get("pred", "unknown")
        lines.append(f"- {name}: pred={pred}; conf={score_value:.2f}; {trace}")
    return "\n".join(lines) if lines else "None."


def normalize_tool_schema(
    result: Optional[Dict],
    tool_name: str,
    default_trace: str,
    default_conf: float = 0.5,
    default_pred: str = "unknown",
) -> Dict:
    result = result if isinstance(result, dict) else {}
    trace = str(result.get("trace") or result.get("reasoning_trace") or default_trace)
    pred = str(result.get("pred") or result.get("prediction") or default_pred).strip().lower()
    if pred not in {"harmful", "harmless", "unknown"}:
        pred = "unknown"
    try:
        conf = clamp(float(result.get("conf", result.get("confidence_score", default_conf))))
    except (TypeError, ValueError):
        conf = clamp(default_conf)
    evidence = result.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        evidence = []
    status = str(result.get("status") or "valid").strip().lower()
    if status not in {"valid", "abstain", "failed"}:
        status = "valid"
    return {
        "name": result.get("name") or tool_name,
        "trace": trace,
        "pred": pred,
        "conf": conf,
        "evidence": [str(item) for item in evidence[:5]],
        "status": status,
        "reasoning_trace": trace,
        "confidence_score": conf,
    }

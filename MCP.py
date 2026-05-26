"""Meta-Cognitive Planning for Tool-meme."""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from utils.tool_meme_config import DEFAULT_MCP_MAX_DEPTH, TOOL_NAMES
from utils.openai_client import OpenAIModelConfig, get_openai_response_with_parts
from utils.prompting import HARMFUL_MEME_RUBRIC, JSON_ONLY_RULE


MCP_SYSTEM_PROMPT = """You are the Meta-Cognitive Planning (MCP) module for harmful meme detection.
Read the target meme text and CAMR context, diagnose which reasoning tools are needed for the specific target, and return a compact JSON task DAG.

""" + HARMFUL_MEME_RUBRIC + """

Available tools:
- semantic_dissector: literal/implied meaning, slang, coded wording, target references
- rhetorical_scanner: irony, sarcasm, exaggeration, metaphor, reversal
- knowledge_grounding: event context, public figures, time-sensitive facts, social background
- expectation_deviator: expectation violation between visible content and hostile text
- cross_modal_aligner: image-text consistency and visual-textual grounding
- inconsistency_amplifier: contradictions across modalities, retrieval, and local judgments
- cultural_decoder: meme templates, subcultural symbols, political references, community slang
- evidence_comparator: retrieved-neighbor label prior, reliability, and evidence strength

Return ONLY JSON:
{
  "semantic_layers": ["..."],
  "tool_dag": [
    {
      "tool": "cross_modal_aligner",
      "deps": [],
      "layer": "visual_textual_alignment",
      "priority": 1,
      "mandatory": true
    }
  ]
}

Rules:
1. Each tool appears at most once.
2. Dependencies must point to earlier tools.
3. Prefer at most four dependency levels.
4. Always include semantic_dissector unless the target text is empty.
5. Include cross_modal_aligner when image-text mismatch, object/person grounding, or visual target identity may affect harm.
6. Include rhetorical_scanner for sarcasm, irony, dark humor, reversals, or exaggerated praise/condemnation.
7. Include cultural_decoder or knowledge_grounding only when cultural/event context is necessary.
8. Include evidence_comparator when retrieved labels are mixed or highly reliable.
9. Mark a layer mandatory only when it is essential to avoid a likely label error.
10. Do not add semantic layers unsupported by the target or retrieved context.
""" + JSON_ONLY_RULE + """
"""


FALLBACK_DAG = {
    "semantic_layers": ["visual_textual_alignment", "semantic_reading", "rhetorical_intent"],
    "tool_dag": [
        {
            "tool": "cross_modal_aligner",
            "deps": [],
            "layer": "visual_textual_alignment",
            "priority": 1,
            "mandatory": True,
        },
        {
            "tool": "semantic_dissector",
            "deps": ["cross_modal_aligner"],
            "layer": "semantic_reading",
            "priority": 2,
            "mandatory": True,
        },
        {
            "tool": "rhetorical_scanner",
            "deps": ["semantic_dissector"],
            "layer": "rhetorical_intent",
            "priority": 3,
            "mandatory": False,
        },
    ],
}


def _build_prompt(
    target_text: str,
    retrieved_texts: List[str],
    labels: List[int],
    event_contexts: List[str],
    camr_package: Optional[Dict] = None,
) -> str:
    harmful_count = sum(1 for label in labels if label == 1)
    label_hint = f"Harmful labels in retrieved samples: {harmful_count}/{len(labels)}" if labels else "No labels."
    context_mode = (camr_package or {}).get("context_mode", "unknown")
    reliability = (camr_package or {}).get("retrieval_reliability", "unknown")
    label_prior = (camr_package or {}).get("label_prior", "unknown")
    ctx = "\n".join(f"- {text}" for text in retrieved_texts[:5]) or "None"
    events = "\n".join(f"- {event}" for event in event_contexts[:3]) or "None"
    return f"""Task: plan the minimum useful reasoning path for classifying this meme.

Target text:
{target_text}

CAMR retrieved texts:
{ctx}

CAMR event contexts:
{events}

CAMR calibration:
- context_mode: {context_mode}
- retrieval_reliability: {reliability}
- label_prior_harmful: {label_prior}
- {label_hint}

Return the JSON DAG only."""


def _parse_json(text: str) -> Dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _depth(tool: str, deps_map: Dict[str, List[str]], memo: Dict[str, int]) -> int:
    if tool in memo:
        return memo[tool]
    deps = deps_map.get(tool, [])
    if not deps:
        memo[tool] = 1
    else:
        memo[tool] = 1 + max(_depth(dep, deps_map, memo) for dep in deps if dep in deps_map)
    return memo[tool]


def repair_dag(payload: Dict, max_depth: int = DEFAULT_MCP_MAX_DEPTH) -> Dict:
    if not isinstance(payload, dict):
        return dict(FALLBACK_DAG)
    raw_nodes = payload.get("tool_dag") or []
    seen = set()
    nodes = []
    for index, raw in enumerate(raw_nodes):
        if not isinstance(raw, dict):
            continue
        tool = raw.get("tool")
        if tool not in TOOL_NAMES or tool in seen:
            continue
        seen.add(tool)
        deps = [
            dep for dep in raw.get("deps", []) or []
            if dep in seen and dep != tool
        ]
        nodes.append({
            "tool": tool,
            "deps": deps,
            "layer": str(raw.get("layer") or tool),
            "priority": int(raw.get("priority", index + 1)),
            "mandatory": bool(raw.get("mandatory", False)),
        })

    if not nodes:
        return dict(FALLBACK_DAG)

    nodes.sort(key=lambda item: item["priority"])
    ordered = []
    available = set()
    for node in nodes:
        deps = [dep for dep in node["deps"] if dep in available]
        node = {**node, "deps": deps}
        ordered.append(node)
        available.add(node["tool"])

    deps_map = {node["tool"]: list(node["deps"]) for node in ordered}
    memo: Dict[str, int] = {}
    repaired = []
    for node in ordered:
        if _depth(node["tool"], deps_map, memo) <= max_depth:
            repaired.append(node)

    if not repaired:
        return dict(FALLBACK_DAG)
    layers = payload.get("semantic_layers") or [node["layer"] for node in repaired]
    return {
        "semantic_layers": [str(layer) for layer in layers[: len(repaired)]],
        "tool_dag": repaired,
        "dag_repaired": True,
        "max_depth": max_depth,
    }


def run_mcp(
    target_text: str,
    retrieved_texts: List[str],
    labels: List[int],
    event_contexts: Optional[List[str]] = None,
    config: Optional[OpenAIModelConfig] = None,
    camr_package: Optional[Dict] = None,
    max_depth: int = DEFAULT_MCP_MAX_DEPTH,
) -> Dict:
    prompt = _build_prompt(
        target_text=target_text,
        retrieved_texts=retrieved_texts,
        labels=labels,
        event_contexts=event_contexts or [],
        camr_package=camr_package,
    )
    parts = [
        {"type": "text", "text": MCP_SYSTEM_PROMPT},
        {"type": "text", "text": prompt},
    ]
    cfg = config or OpenAIModelConfig.from_env()
    try:
        output = get_openai_response_with_parts(parts, cfg)
        payload = _parse_json(output)
    except Exception:
        payload = dict(FALLBACK_DAG)
    return repair_dag(payload, max_depth=max_depth)

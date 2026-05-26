"""Adaptive Tool Routing for Tool-meme."""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Set

from utils.tool_meme_config import (
    DEFAULT_SHORT_CIRCUIT_THRESHOLD,
    DEFAULT_TOOL_BUDGET,
    TAU_CONFLICT,
    TAU_STOP,
    TOOL_NAMES,
)
from rl_atr import (
    RLToolRouter,
    dependency_map,
    node_map,
    repair_tool_dag,
    topological_tools,
)
from tools.base_tool import ToolRegistry
from tools import (  # noqa: F401
    cross_modal_aligner,
    cultural_decoder,
    evidence_comparator,
    expectation_deviator,
    inconsistency_amplifier,
    knowledge_grounding,
    rhetorical_scanner,
    semantic_dissector,
)
from tools.utils import clamp


def _build_stub_output(tool: str, note: Optional[str] = None) -> Dict:
    msg = note or "not executed."
    return {
        "name": tool,
        "trace": f"{tool} {msg}",
        "pred": "unknown",
        "conf": 0.0,
        "evidence": [],
        "status": "abstain",
        "reasoning_trace": f"{tool} {msg}",
        "confidence_score": 0.0,
    }


def _normalize_tool_result(tool: str, result: Dict) -> Dict:
    result = result or {}
    trace = str(result.get("trace") or result.get("reasoning_trace") or "")
    pred = str(result.get("pred") or result.get("prediction") or "unknown").lower()
    if pred not in {"harmful", "harmless", "unknown"}:
        pred = "unknown"
    raw_conf = result.get("conf", result.get("confidence_score", 0.0))
    try:
        conf = clamp(float(raw_conf))
    except (TypeError, ValueError):
        conf = 0.0
    evidence = result.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        evidence = []
    status = str(result.get("status") or "valid").lower()
    if status not in {"valid", "abstain", "failed"}:
        status = "valid"
    return {
        **result,
        "tool": tool,
        "name": result.get("name") or tool,
        "trace": trace or f"{tool} completed.",
        "pred": pred,
        "conf": conf,
        "evidence": [str(item) for item in evidence[:5]],
        "status": status,
        "reasoning_trace": trace or result.get("reasoning_trace", f"{tool} completed."),
        "confidence_score": conf,
    }


def _execute_tool(
    tool: str,
    query_text: str,
    image_embedding,
    retrieved_context: Dict,
    outputs: List[Dict],
) -> Dict:
    runner = ToolRegistry.create(tool)
    start_time = time.time()
    if runner is None:
        result = _build_stub_output(tool or "unknown_tool", "is not registered.")
    else:
        try:
            result = runner.analyze(
                query_text=query_text,
                image_embedding=image_embedding,
                retrieved_context=retrieved_context,
                previous_outputs={o["tool"]: o for o in outputs if o.get("tool")},
            )
        except Exception as exc:
            result = _build_stub_output(tool, f"failed: {exc}")
            result["status"] = "failed"
    exec_time = time.time() - start_time
    return {**_normalize_tool_result(tool, result), "execution_time": exec_time}


def _build_deps_map(tool_dag: List[Dict]) -> Dict[str, Set[str]]:
    deps_map: Dict[str, Set[str]] = {}
    for node in tool_dag or []:
        tool = node.get("tool")
        if tool:
            deps_map[tool] = set(node.get("deps", []) or [])
    return deps_map


def _node_map(tool_dag: List[Dict]) -> Dict[str, Dict]:
    return {node.get("tool"): node for node in tool_dag or [] if node.get("tool")}


def _toposort_tools(tool_dag: List[Dict]) -> List[str]:
    if not tool_dag:
        return []
    tools = [node.get("tool") for node in tool_dag if node.get("tool")]
    deps_map = _build_deps_map(tool_dag)
    in_degree = {tool: 0 for tool in tools}
    children = {tool: [] for tool in tools}
    for tool, deps in deps_map.items():
        for dep in deps:
            if dep in in_degree:
                in_degree[tool] += 1
                children[dep].append(tool)
    queue = [tool for tool in tools if in_degree[tool] == 0]
    order: List[str] = []
    seen: Set[str] = set()
    while queue:
        current = queue.pop(0)
        if current in seen:
            continue
        seen.add(current)
        order.append(current)
        for child in children.get(current, []):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    return order if len(order) == len(tools) else tools


def _get_available_tools(
    tool_order: List[str],
    deps_map: Dict[str, Set[str]],
    executed_tools: Set[str],
    disabled_tools: Set[str],
) -> List[str]:
    available = []
    for tool in tool_order:
        if tool in executed_tools or tool in disabled_tools:
            continue
        if deps_map.get(tool, set()).issubset(executed_tools):
            available.append(tool)
    return available


def _confidence_stats(outputs: List[Dict]) -> Dict[str, float]:
    values = [
        clamp(float(output.get("confidence_score", 0.0)))
        for output in outputs
        if not output.get("was_skipped")
        and not output.get("was_short_circuited")
        and not output.get("was_not_selected")
    ]
    if not values:
        return {"max_conf": 0.0, "avg_conf": 0.0, "spread": 0.0}
    return {
        "max_conf": max(values),
        "avg_conf": sum(values) / len(values),
        "spread": max(values) - min(values),
    }


def _conflict_score(outputs: List[Dict]) -> float:
    valid = [
        output for output in outputs
        if output.get("pred") in {"harmful", "harmless"}
        and not output.get("was_skipped")
        and not output.get("was_short_circuited")
        and not output.get("was_not_selected")
    ]
    if len(valid) < 2:
        return 0.0
    total = 0.0
    conflict = 0.0
    for i, left in enumerate(valid):
        for right in valid[i + 1:]:
            weight = float(left.get("confidence_score", 0.0)) * float(right.get("confidence_score", 0.0))
            total += weight
            if left.get("pred") != right.get("pred"):
                conflict += weight
    return float(conflict / max(total, 1e-8))


def _unresolved_mandatory(
    nodes: Dict[str, Dict],
    executed_tools: Set[str],
    disabled_tools: Set[str],
) -> List[str]:
    unresolved = []
    for tool, node in nodes.items():
        if tool in disabled_tools:
            continue
        if bool(node.get("mandatory", False)) and tool not in executed_tools:
            unresolved.append(tool)
    return unresolved


def _stop_gate(
    outputs: List[Dict],
    nodes: Dict[str, Dict],
    executed_tools: Set[str],
    disabled_tools: Set[str],
    tau_stop: float,
    tau_conflict: float,
) -> bool:
    stats = _confidence_stats(outputs)
    return (
        stats["max_conf"] >= tau_stop
        and not _unresolved_mandatory(nodes, executed_tools, disabled_tools)
        and _conflict_score(outputs) <= tau_conflict
    )


def _mark_remaining(
    outputs: List[Dict],
    tool_order: List[str],
    executed_tools: Set[str],
    disabled_tools: Set[str],
    reason: str,
) -> None:
    for tool in tool_order:
        if tool in executed_tools or tool in disabled_tools:
            continue
        outputs.append({
            "tool": tool,
            **_build_stub_output(tool, reason),
            "was_not_selected": True,
            "execution_time": 0.0,
        })


def run_atr(
    task_dag: Dict,
    query_text: str,
    image_embedding,
    retrieved_context: Dict,
    context_summary: str = "",
    short_circuit_threshold: Optional[float] = DEFAULT_SHORT_CIRCUIT_THRESHOLD,
    disabled_tools: Optional[Set[str]] = None,
    routing_mode: str = "dag",
    router_path: Optional[str] = None,
    router_temperature: float = 1.0,
    router_max_steps: Optional[int] = None,
    router_greedy: bool = True,
    tool_budget: int = DEFAULT_TOOL_BUDGET,
    tau_stop: float = TAU_STOP,
    tau_conflict: float = TAU_CONFLICT,
) -> Dict:
    _ = context_summary
    disabled_tools = disabled_tools or set()
    routing_mode = (routing_mode or "dag").lower().replace("-", "_")
    if routing_mode == "static":
        routing_mode = "dag"

    tool_dag = repair_tool_dag(task_dag)
    if routing_mode == "all_tools":
        tool_order = [tool for tool in TOOL_NAMES if tool not in disabled_tools]
        deps_map: Dict[str, Set[str]] = {tool: set() for tool in tool_order}
    else:
        tool_order = topological_tools(tool_dag)
        deps_map = dependency_map(tool_dag)
    nodes = node_map(tool_dag)
    outputs: List[Dict] = []
    executed_tools: Set[str] = set()
    short_circuited: Set[str] = set()
    routed_tools: List[str] = []
    router_trace: List[Dict] = []
    router_stop_reason = ""
    route_source = routing_mode
    budget = max(1, int(tool_budget))
    if router_max_steps is not None:
        budget = min(budget, max(0, int(router_max_steps)))

    for tool in tool_order:
        if tool in disabled_tools:
            outputs.append({
                "tool": tool,
                **_build_stub_output(tool, "skipped by flag."),
                "was_skipped": True,
                "execution_time": 0.0,
            })

    router: Optional[RLToolRouter] = None
    if routing_mode == "rl":
        try:
            router = RLToolRouter.load(router_path) if router_path else RLToolRouter()
            route_source = "rl_trained" if router_path else "rl_untrained"
        except Exception as exc:
            router_trace.append({
                "selected_action": None,
                "available_tools": list(tool_order),
                "reason": f"router_load_failed: {exc}",
            })
            routing_mode = "dag"
            route_source = "dag_fallback"

    while len(routed_tools) < budget:
        if routing_mode == "random":
            available = _get_available_tools(tool_order, deps_map, executed_tools, disabled_tools)
            if not available:
                router_stop_reason = "dag_exhausted"
                break
            next_tool = random.choice(available)
        elif routing_mode == "heuristic":
            available = _get_available_tools(tool_order, deps_map, executed_tools, disabled_tools)
            if not available:
                router_stop_reason = "dag_exhausted"
                break
            next_tool = available[0]
        elif routing_mode == "all_tools":
            available = _get_available_tools(tool_order, deps_map, executed_tools, disabled_tools)
            if not available:
                router_stop_reason = "dag_exhausted"
                break
            next_tool = available[0]
        elif routing_mode == "rl" and router is not None:
            available = _get_available_tools(tool_order, deps_map, executed_tools, disabled_tools)
            if not available:
                router_stop_reason = "dag_exhausted"
                break
            allow_stop = _stop_gate(outputs, nodes, executed_tools, disabled_tools, tau_stop, tau_conflict)
            next_tool, step_trace = router.select_next_tool(
                query_text=query_text,
                retrieved_context=retrieved_context,
                previous_outputs={o["tool"]: o for o in outputs if o.get("tool")},
                executed_tools=routed_tools,
                available_tools=available,
                step_index=len(routed_tools),
                total_tools=max(1, len(tool_order)),
                temperature=router_temperature,
                greedy=router_greedy,
                allow_stop=allow_stop,
            )
            step_trace["stop_gate"] = allow_stop
            router_trace.append(step_trace)
            if next_tool is None:
                router_stop_reason = "policy_stop"
                break
        else:
            available = _get_available_tools(tool_order, deps_map, executed_tools, disabled_tools)
            if not available:
                router_stop_reason = "dag_exhausted"
                break
            if short_circuit_threshold is not None and outputs:
                if _confidence_stats(outputs)["max_conf"] >= short_circuit_threshold:
                    router_stop_reason = "short_circuit"
                    for tool in available:
                        short_circuited.add(tool)
                        outputs.append({
                            "tool": tool,
                            **_build_stub_output(tool, "short-circuited due to sufficient confidence."),
                            "was_short_circuited": True,
                            "execution_time": 0.0,
                        })
                    break
            next_tool = available[0]

        record = _execute_tool(
            tool=next_tool,
            query_text=query_text,
            image_embedding=image_embedding,
            retrieved_context=retrieved_context,
            outputs=outputs,
        )
        outputs.append(record)
        executed_tools.add(next_tool)
        routed_tools.append(next_tool)

    if not router_stop_reason:
        router_stop_reason = "budget_exhausted" if len(routed_tools) >= budget else "dag_exhausted"
    if routing_mode in {"rl", "random", "heuristic", "all_tools"} and router_stop_reason != "dag_exhausted":
        _mark_remaining(outputs, tool_order, executed_tools, disabled_tools, router_stop_reason)

    stats = _confidence_stats(outputs)
    return {
        "tool_order": tool_order,
        "tool_order_planned": tool_order,
        "tool_order_routed": routed_tools,
        "routing_mode": routing_mode,
        "route_source": route_source,
        "tool_budget": budget,
        "short_circuit_threshold": short_circuit_threshold,
        "router_path": router_path or "",
        "router_trace": router_trace,
        "router_stop_reason": router_stop_reason,
        "outputs": outputs,
        "tools_executed": len(routed_tools),
        "tools_short_circuited": len(short_circuited),
        "conflict_score": _conflict_score(outputs),
        "confidence_mean": stats["avg_conf"],
        "confidence_spread": stats["spread"],
        "unresolved_mandatory": _unresolved_mandatory(nodes, executed_tools, disabled_tools),
    }

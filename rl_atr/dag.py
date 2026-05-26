"""DAG repair and frontier utilities for RL-guided ATR.

The MCP module returns a compact tool DAG. This adapter keeps only executable
Tool-meme tools, removes unsafe dependencies, repairs simple cycles, and exposes the
legal DAG frontier used by both heuristic and RL routing.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Sequence, Set

from .schemas import TOOL_INDEX, VALID_TOOLS, ToolNode


def fallback_tool_dag() -> List[Dict[str, Any]]:
    return [
        ToolNode("cross_modal_aligner", [], "visual_textual_alignment", 1, True).to_dict(),
        ToolNode("semantic_dissector", ["cross_modal_aligner"], "semantic_reading", 2, True).to_dict(),
        ToolNode("rhetorical_scanner", ["semantic_dissector"], "rhetorical_intent", 3, False).to_dict(),
    ]


def _raw_nodes(task_dag: Dict[str, Any] | Sequence[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    if isinstance(task_dag, list):
        return [row for row in task_dag if isinstance(row, dict)]
    if not isinstance(task_dag, dict):
        return []
    rows = task_dag.get("tool_dag", task_dag.get("nodes", []))
    return [row for row in rows if isinstance(row, dict)]


def repair_tool_dag(
    task_dag: Dict[str, Any] | Sequence[Dict[str, Any]] | None,
    disabled_tools: Iterable[str] | None = None,
) -> List[Dict[str, Any]]:
    disabled = set(disabled_tools or [])
    nodes: Dict[str, ToolNode] = {}
    alias_to_tool: Dict[str, str] = {}

    for index, raw in enumerate(_raw_nodes(task_dag), start=1):
        try:
            raw_id, node = ToolNode.from_raw(raw, fallback_priority=index)
        except (TypeError, ValueError):
            continue
        if node.tool not in VALID_TOOLS or node.tool in disabled or node.tool in nodes:
            continue
        nodes[node.tool] = node
        alias_to_tool[raw_id] = node.tool
        alias_to_tool[node.tool] = node.tool

    if not nodes:
        return fallback_tool_dag()

    for node in nodes.values():
        clean: List[str] = []
        for dep in node.deps:
            dep_tool = alias_to_tool.get(dep, dep)
            if dep_tool == node.tool or dep_tool not in nodes or dep_tool in clean:
                continue
            if nodes[dep_tool].priority >= node.priority:
                continue
            clean.append(dep_tool)
        node.deps = clean

    _break_cycles(nodes)
    return [node.to_dict() for node in sorted(nodes.values(), key=lambda item: (item.priority, TOOL_INDEX[item.tool]))]


def _break_cycles(nodes: Dict[str, ToolNode]) -> None:
    for _ in range(max(1, len(nodes) * len(nodes))):
        cyclic = _cyclic_tools(nodes)
        if not cyclic:
            return
        child = sorted(cyclic, key=lambda tool: (nodes[tool].priority, TOOL_INDEX[tool]), reverse=True)[0]
        cyclic_deps = [dep for dep in nodes[child].deps if dep in cyclic]
        if not cyclic_deps:
            return
        drop = sorted(cyclic_deps, key=lambda tool: (nodes[tool].priority, TOOL_INDEX[tool]), reverse=True)[0]
        nodes[child].deps = [dep for dep in nodes[child].deps if dep != drop]


def _cyclic_tools(nodes: Dict[str, ToolNode]) -> Set[str]:
    indeg = {tool: 0 for tool in nodes}
    children = {tool: [] for tool in nodes}
    for tool, node in nodes.items():
        for dep in node.deps:
            if dep in nodes:
                indeg[tool] += 1
                children[dep].append(tool)
    queue: deque[str] = deque([tool for tool, degree in indeg.items() if degree == 0])
    seen: Set[str] = set()
    while queue:
        tool = queue.popleft()
        seen.add(tool)
        for child in children[tool]:
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
    return set(nodes) - seen


def node_map(tool_dag: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(node.get("tool")): dict(node) for node in tool_dag if node.get("tool")}


def dependency_map(tool_dag: Sequence[Dict[str, Any]]) -> Dict[str, Set[str]]:
    tools = {str(node.get("tool")) for node in tool_dag if node.get("tool")}
    return {
        str(node.get("tool")): {str(dep) for dep in node.get("deps", []) or [] if str(dep) in tools}
        for node in tool_dag
        if node.get("tool")
    }


def topological_tools(tool_dag: Sequence[Dict[str, Any]]) -> List[str]:
    nodes = node_map(tool_dag)
    deps = dependency_map(tool_dag)
    indeg = {tool: len(deps.get(tool, set())) for tool in nodes}
    children = {tool: [] for tool in nodes}
    for tool, parents in deps.items():
        for parent in parents:
            children.setdefault(parent, []).append(tool)
    queue = sorted([tool for tool, degree in indeg.items() if degree == 0], key=lambda tool: _sort_key(nodes, tool))
    order: List[str] = []
    while queue:
        tool = queue.pop(0)
        order.append(tool)
        for child in sorted(children.get(tool, []), key=lambda item: _sort_key(nodes, item)):
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
        queue.sort(key=lambda item: _sort_key(nodes, item))
    return order if len(order) == len(nodes) else sorted(nodes, key=lambda tool: _sort_key(nodes, tool))


def _sort_key(nodes: Dict[str, Dict[str, Any]], tool: str) -> tuple:
    return (int(nodes.get(tool, {}).get("priority", 99)), TOOL_INDEX.get(tool, 99))


def frontier_tools(
    tool_order: Sequence[str],
    deps_map: Dict[str, Set[str]],
    executed_tools: Iterable[str],
    disabled_tools: Iterable[str] | None = None,
) -> List[str]:
    executed = set(executed_tools)
    disabled = set(disabled_tools or [])
    return [
        tool
        for tool in tool_order
        if tool not in executed and tool not in disabled and deps_map.get(tool, set()).issubset(executed)
    ]


def unresolved_mandatory_tools(
    nodes: Dict[str, Dict[str, Any]],
    executed_tools: Iterable[str],
    disabled_tools: Iterable[str] | None = None,
) -> List[str]:
    executed = set(executed_tools)
    disabled = set(disabled_tools or [])
    return [
        tool
        for tool, node in sorted(nodes.items(), key=lambda item: _sort_key(nodes, item[0]))
        if bool(node.get("mandatory", False)) and tool not in executed and tool not in disabled
    ]

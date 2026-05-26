"""Small typed objects shared by the Tool-meme RL-ATR adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

from utils.tool_meme_config import TOOL_NAMES

VALID_TOOLS = set(TOOL_NAMES)
TOOL_INDEX = {name: index for index, name in enumerate(TOOL_NAMES)}


@dataclass
class ToolNode:
    """Executable tool node after MCP DAG repair.

    The runtime still routes by tool name, so ``tool`` is the stable node id used
    by masks, traces, and training records.
    """

    tool: str
    deps: List[str] = field(default_factory=list)
    layer: str = "generic"
    priority: int = 1
    mandatory: bool = False
    cost: float = 1.0

    @classmethod
    def from_raw(cls, raw: Dict[str, Any], fallback_priority: int = 1) -> Tuple[str, "ToolNode"]:
        raw_id = str(raw.get("node_id", raw.get("node", raw.get("id", raw.get("tool", ""))))).strip()
        tool = str(raw.get("tool", raw.get("tool_name", ""))).strip()
        deps = raw.get("deps", raw.get("dependencies", [])) or []
        if isinstance(deps, str):
            deps = [deps]
        return raw_id or tool, cls(
            tool=tool,
            deps=[str(dep).strip() for dep in deps if str(dep).strip()],
            layer=str(raw.get("layer", raw.get("semantic_layer", tool or "generic"))).strip() or "generic",
            priority=int(raw.get("priority", fallback_priority)),
            mandatory=bool(raw.get("mandatory", False)),
            cost=max(0.0, float(raw.get("cost", raw.get("estimated_cost", 1.0)))),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

"""RL-guided adaptive tool routing for Tool-meme."""

from .dag import (
    dependency_map,
    fallback_tool_dag,
    frontier_tools,
    node_map,
    repair_tool_dag,
    topological_tools,
    unresolved_mandatory_tools,
)
from .masks import action_mask, legal_actions
from .router import (
    DEFAULT_ROUTABLE_TOOLS,
    RLToolRouter,
    train_router_from_logs,
)
from .rewards import DEFAULT_TOOL_WEIGHTS, RouterRewardConfig, compute_route_reward, route_reward_breakdown
from .masks import STOP_ACTION
from .stop_gate import confidence_stats, conflict_score, stop_is_legal

__all__ = [
    "STOP_ACTION",
    "RLToolRouter",
    "RouterRewardConfig",
    "DEFAULT_ROUTABLE_TOOLS",
    "DEFAULT_TOOL_WEIGHTS",
    "train_router_from_logs",
    "repair_tool_dag",
    "fallback_tool_dag",
    "topological_tools",
    "dependency_map",
    "frontier_tools",
    "node_map",
    "unresolved_mandatory_tools",
    "legal_actions",
    "action_mask",
    "confidence_stats",
    "conflict_score",
    "stop_is_legal",
    "compute_route_reward",
    "route_reward_breakdown",
]

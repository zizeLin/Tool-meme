"""Reward terms used to train the offline RL-ATR router from Tool-meme logs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from utils.tool_meme_config import TOOL_WEIGHTS
from tools.utils import clamp
from .stop_gate import executable_outputs

DEFAULT_TOOL_WEIGHTS = dict(TOOL_WEIGHTS)


@dataclass
class RouterRewardConfig:
    beta_fast: float = 0.6
    beta_reasoning: float = 1.0
    beta_tool: float = 0.25
    beta_confidence: float = 0.2
    beta_budget: float = 0.15
    fast_conf_threshold: float = 0.65
    incorrect_penalty: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def extract_executed_outputs(atr_output: Dict) -> List[Dict]:
    return [output for output in executable_outputs((atr_output or {}).get("outputs") or []) if output.get("tool")]


def route_reward_breakdown(
    record: Dict,
    tool_names: Sequence[str],
    reward_config: Optional[RouterRewardConfig] = None,
    tool_weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    cfg = reward_config or RouterRewardConfig()
    weights = tool_weights or DEFAULT_TOOL_WEIGHTS
    outputs = extract_executed_outputs(record.get("atr_output") or {})
    if not outputs:
        return -1.0, {"empty_route": 1.0, "total": -1.0}

    correct = float(record.get("actual") is not None and record.get("predict") is not None and record.get("actual") == record.get("predict"))
    confidences = [clamp(float(output.get("confidence_score", output.get("conf", 0.0)))) for output in outputs]
    unique_tools = list(dict.fromkeys(output.get("tool") for output in outputs if output.get("tool")))
    fast = float(correct and confidences[0] >= cfg.fast_conf_threshold)
    reasoning = correct
    tool_use = sum(weights.get(tool, 1.0) for tool in unique_tools) * correct
    confidence = sum(confidences) / max(1, len(confidences))
    budget = len(unique_tools) / max(1, len(tool_names))
    total = (
        cfg.beta_fast * fast
        + cfg.beta_reasoning * reasoning
        + cfg.beta_tool * tool_use
        + cfg.beta_confidence * confidence
        - cfg.beta_budget * budget
        - (1.0 - correct) * cfg.incorrect_penalty
    )
    return float(total), {
        "fast": fast,
        "reasoning": reasoning,
        "tool_use": tool_use,
        "confidence": confidence,
        "budget": budget,
        "correct": correct,
        "total": float(total),
    }


def compute_route_reward(
    record: Dict,
    tool_names: Sequence[str],
    reward_config: Optional[RouterRewardConfig] = None,
    tool_weights: Optional[Dict[str, float]] = None,
) -> float:
    reward, _ = route_reward_breakdown(record, tool_names, reward_config, tool_weights)
    return reward

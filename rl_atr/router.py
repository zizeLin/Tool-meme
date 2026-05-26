"""Lightweight masked policy router for Tool-meme ATR."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

from utils.tool_meme_config import TOOL_NAMES
from .features import build_router_feature_vector, get_feature_names
from .masks import STOP_ACTION, action_mask
from .rewards import (
    DEFAULT_TOOL_WEIGHTS,
    RouterRewardConfig,
    compute_route_reward,
    extract_executed_outputs,
)

DEFAULT_ROUTABLE_TOOLS = list(TOOL_NAMES)


class ToolRoutingPolicy(nn.Module):
    """Small actor-critic MLP. The DAG legality mask is applied outside it."""

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(features)
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)


class RLToolRouter:
    """Runtime router used by ``ATR.run_atr`` and offline log training."""

    def __init__(
        self,
        tool_names: Optional[Sequence[str]] = None,
        hidden_dim: int = 96,
        device: Optional[str] = None,
    ) -> None:
        self.tool_names = list(tool_names or DEFAULT_ROUTABLE_TOOLS)
        self.action_names = self.tool_names + [STOP_ACTION]
        self.feature_names = get_feature_names(self.tool_names)
        self.hidden_dim = hidden_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.policy = ToolRoutingPolicy(len(self.feature_names), len(self.action_names), hidden_dim).to(self.device)
        self.policy.eval()

    def select_next_tool(
        self,
        query_text: str,
        retrieved_context: Optional[Dict],
        previous_outputs: Optional[Dict[str, Dict]],
        executed_tools: Sequence[str],
        available_tools: Sequence[str],
        step_index: int,
        total_tools: int,
        temperature: float = 1.0,
        greedy: bool = True,
        allow_stop: bool = True,
    ) -> Tuple[Optional[str], Dict]:
        valid_tools = [tool for tool in available_tools if tool in self.tool_names and tool not in executed_tools]
        if not valid_tools and not allow_stop:
            return None, {"selected_action": None, "available_tools": list(available_tools), "reason": "no_valid_action"}

        features = build_router_feature_vector(
            query_text=query_text,
            retrieved_context=retrieved_context,
            previous_outputs=previous_outputs,
            executed_tools=executed_tools,
            available_tools=valid_tools,
            tool_names=self.tool_names,
            step_index=step_index,
            total_tools=total_tools,
        )
        mask = action_mask(self.action_names, valid_tools, allow_stop=allow_stop)
        with torch.no_grad():
            logits, _ = self.policy(torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0))
            masked_logits = logits.squeeze(0).masked_fill(
                ~torch.tensor(mask, dtype=torch.bool, device=self.device),
                -1e9,
            )
            if greedy or temperature <= 0:
                action_index = int(torch.argmax(masked_logits).item())
            else:
                action_index = int(Categorical(logits=masked_logits / max(temperature, 1e-6)).sample().item())
            probs = torch.softmax(masked_logits, dim=-1)

        selected = self.action_names[action_index]
        legal = set(valid_tools)
        if allow_stop:
            legal.add(STOP_ACTION)
        trace = {
            "selected_action": selected,
            "available_tools": list(valid_tools),
            "step_index": step_index,
            "action_scores": {
                action: round(float(probs[index].item()), 4)
                for index, action in enumerate(self.action_names)
                if action in legal
            },
        }
        return (None, trace) if selected == STOP_ACTION else (selected, trace)

    def fit_from_records(
        self,
        records: Sequence[Dict],
        epochs: int = 25,
        learning_rate: float = 1e-3,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        reward_config: Optional[RouterRewardConfig] = None,
    ) -> Dict[str, float]:
        dataset = build_training_samples(records, self.tool_names, reward_config)
        if not dataset:
            raise ValueError("No valid ATR trajectories were found for router training.")

        states = torch.tensor(np.stack([row["features"] for row in dataset]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([row["action_index"] for row in dataset], dtype=torch.long, device=self.device)
        masks = torch.tensor(np.stack([row["mask"] for row in dataset]), dtype=torch.bool, device=self.device)
        returns = torch.tensor([row["return"] for row in dataset], dtype=torch.float32, device=self.device)
        advantages = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-6)

        optimizer = torch.optim.AdamW(self.policy.parameters(), lr=learning_rate)
        self.policy.train()
        last_loss = 0.0
        last_entropy = 0.0
        for _ in range(max(1, epochs)):
            logits, values = self.policy(states)
            masked_logits = logits.masked_fill(~masks, -1e9)
            dist = Categorical(logits=masked_logits)
            log_probs = F.log_softmax(masked_logits, dim=-1).gather(1, actions.unsqueeze(1)).squeeze(1)
            policy_loss = -(log_probs * (advantages - values).detach()).mean()
            value_loss = F.mse_loss(values, advantages)
            entropy = dist.entropy().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            optimizer.step()
            last_loss = float(loss.item())
            last_entropy = float(entropy.item())

        self.policy.eval()
        rewards = [row["return"] for row in dataset]
        return {
            "num_records": float(len(records)),
            "num_samples": float(len(dataset)),
            "avg_reward": float(np.mean(rewards)),
            "max_reward": float(np.max(rewards)),
            "min_reward": float(np.min(rewards)),
            "loss": last_loss,
            "entropy": last_entropy,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "format": "tool_meme_rl_atr_v2",
                "tool_names": self.tool_names,
                "feature_names": self.feature_names,
                "hidden_dim": self.hidden_dim,
                "state_dict": self.policy.state_dict(),
            },
            path,
        )

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Expose the actor-critic weights for tests and diagnostics."""

        return self.policy.state_dict()

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "RLToolRouter":
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            expected = checkpoint_path.as_posix()
            parts = list(checkpoint_path.parts)
            if len(parts) >= 3 and parts[-1] == "checkpoint.pt" and parts[-3] == "rl_atr":
                expected = Path("rl_atr", parts[-2], "checkpoint.pt").as_posix()
            raise FileNotFoundError(f"Router checkpoint not found at {expected}.")
        bundle = _load_checkpoint(path, map_location=device or "cpu")
        state_keys = (bundle.get("state_dict") or {}).keys() if isinstance(bundle, dict) else []
        if isinstance(bundle, dict) and (
            any(str(key).startswith("net.") for key in state_keys)
        ):
            raise ValueError(
                "This checkpoint belongs to an incompatible candidate-action RouterPolicy "
                "format and cannot be loaded by RLToolRouter. "
                "Use a checkpoint produced by rl_atr/train_tool_router.py."
            )
        router = cls(
            tool_names=bundle.get("tool_names") or DEFAULT_ROUTABLE_TOOLS,
            hidden_dim=int(bundle.get("hidden_dim", 96)),
            device=device,
        )
        router.policy.load_state_dict(bundle["state_dict"])
        router.policy.eval()
        return router


def _load_checkpoint(path: str, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_training_samples(
    records: Sequence[Dict],
    tool_names: Sequence[str],
    reward_config: Optional[RouterRewardConfig] = None,
) -> List[Dict]:
    samples: List[Dict] = []
    action_names = list(tool_names) + [STOP_ACTION]
    for record in records:
        atr_output = record.get("atr_output")
        if not isinstance(atr_output, dict):
            continue
        planned = _planned_tools(atr_output, tool_names)
        route_outputs = extract_executed_outputs(atr_output)
        route = [tool for tool in (atr_output.get("tool_order_routed") or [out.get("tool") for out in route_outputs]) if tool in tool_names]
        if not planned or not route:
            continue

        episode_return = compute_route_reward(record, tool_names, reward_config)
        previous_outputs: Dict[str, Dict] = {}
        executed: List[str] = []
        output_by_tool = {out.get("tool"): out for out in route_outputs if out.get("tool")}

        for step_index, action_tool in enumerate(route):
            available = [tool for tool in planned if tool not in executed]
            if action_tool not in available:
                available.append(action_tool)
            samples.append(_sample_row(record, tool_names, action_names, previous_outputs, executed, available, action_tool, step_index, len(planned), episode_return))
            executed.append(action_tool)
            if action_tool in output_by_tool:
                previous_outputs[action_tool] = output_by_tool[action_tool]

        remaining = [tool for tool in planned if tool not in executed]
        samples.append(_sample_row(record, tool_names, action_names, previous_outputs, executed, remaining, STOP_ACTION, len(executed), len(planned), episode_return))
    return samples


def _planned_tools(atr_output: Dict, tool_names: Sequence[str]) -> List[str]:
    planned: List[str] = []
    for tool in atr_output.get("tool_order_planned") or atr_output.get("tool_order") or atr_output.get("tool_order_routed") or []:
        if tool in tool_names and tool not in planned:
            planned.append(tool)
    return planned


def _sample_row(
    record: Dict,
    tool_names: Sequence[str],
    action_names: Sequence[str],
    previous_outputs: Dict[str, Dict],
    executed: Sequence[str],
    available: Sequence[str],
    action: str,
    step_index: int,
    total_tools: int,
    episode_return: float,
) -> Dict:
    return {
        "features": build_router_feature_vector(
            query_text=record.get("text", ""),
            retrieved_context={},
            previous_outputs=previous_outputs,
            executed_tools=executed,
            available_tools=available,
            tool_names=tool_names,
            step_index=step_index,
            total_tools=total_tools,
        ),
        "action_index": list(action_names).index(action),
        "mask": action_mask(action_names, available, allow_stop=True),
        "return": episode_return,
    }


def load_router_training_records(paths: Sequence[str]) -> List[Dict]:
    records: List[Dict] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and "atr_output" in payload and "text" in payload:
                    records.append(payload)
    return records


def train_router_from_logs(
    log_paths: Sequence[str],
    output_path: str,
    tool_names: Optional[Sequence[str]] = None,
    epochs: int = 25,
    learning_rate: float = 1e-3,
    reward_config: Optional[RouterRewardConfig] = None,
    hidden_dim: int = 96,
    seed: int = 42,
) -> Dict[str, float]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    router = RLToolRouter(tool_names=tool_names or DEFAULT_ROUTABLE_TOOLS, hidden_dim=hidden_dim)
    stats = router.fit_from_records(load_router_training_records(log_paths), epochs, learning_rate, reward_config=reward_config)
    router.save(output_path)
    stats.update({
        "num_logs": float(len(log_paths)),
        "model_path": output_path,
        "tool_names": ",".join(router.tool_names),
        "reward_config": json.dumps(asdict(reward_config or RouterRewardConfig())),
    })
    return stats

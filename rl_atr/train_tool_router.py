from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl_atr import (
    DEFAULT_ROUTABLE_TOOLS,
    RouterRewardConfig,
    train_router_from_logs,
)
from utils.tool_meme_config import DEFAULT_ROUTER_RUN_ID


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an RL-based tool router from ATR trajectories logged by Tool_meme.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="One or more Tool-meme result JSONL files containing atr_output trajectories.",
    )
    parser.add_argument(
        "--run-id",
        default=DEFAULT_ROUTER_RUN_ID,
        help="Router run id used when --output is not provided.",
    )
    parser.add_argument(
        "--dest-root",
        default=str(PROJECT_ROOT / "rl_atr"),
        help="Root directory for RL-ATR training runs.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save the trained router model. Defaults to rl_atr/<run_id>/checkpoint.pt.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="Number of offline policy-optimization epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Learning rate for the router policy.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=96,
        help="Hidden size of the router MLP policy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for offline training.",
    )
    parser.add_argument(
        "--tool_names",
        nargs="+",
        default=DEFAULT_ROUTABLE_TOOLS,
        help="Optional tool whitelist/order used by the router.",
    )
    parser.add_argument("--beta_fast", type=float, default=0.6)
    parser.add_argument("--beta_reasoning", type=float, default=1.0)
    parser.add_argument("--beta_tool", type=float, default=0.25)
    parser.add_argument("--beta_confidence", type=float, default=0.2)
    parser.add_argument("--beta_budget", type=float, default=0.15)
    parser.add_argument("--fast_conf_threshold", type=float, default=0.65)
    parser.add_argument("--incorrect_penalty", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or str(Path(args.dest_root) / args.run_id / "checkpoint.pt")
    reward_config = RouterRewardConfig(
        beta_fast=args.beta_fast,
        beta_reasoning=args.beta_reasoning,
        beta_tool=args.beta_tool,
        beta_confidence=args.beta_confidence,
        beta_budget=args.beta_budget,
        fast_conf_threshold=args.fast_conf_threshold,
        incorrect_penalty=args.incorrect_penalty,
    )
    stats = train_router_from_logs(
        log_paths=args.logs,
        output_path=output_path,
        tool_names=args.tool_names,
        epochs=args.epochs,
        learning_rate=args.lr,
        reward_config=reward_config,
        hidden_dim=args.hidden_dim,
        seed=args.seed,
    )
    run_dir = Path(output_path).resolve().parent
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.txt").write_text(" ".join(sys.argv).rstrip() + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

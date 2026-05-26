"""Legal action masking for DAG-constrained tool routing."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

STOP_ACTION = "__stop__"


def legal_actions(available_tools: Iterable[str], allow_stop: bool = True) -> List[str]:
    actions = list(dict.fromkeys(available_tools))
    if allow_stop:
        actions.append(STOP_ACTION)
    return actions


def action_mask(action_names: Sequence[str], available_tools: Iterable[str], allow_stop: bool = True) -> np.ndarray:
    legal = set(legal_actions(available_tools, allow_stop=allow_stop))
    return np.asarray([name in legal for name in action_names], dtype=np.bool_)

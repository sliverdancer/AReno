"""Strict terminal reward for the controlled CARe bifurcation pilot."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task  # noqa: E402


def reward_fn(record) -> float:
    """Return zero unless exactly four valid ordered tool actions exist."""

    actions = []
    for event in record.trace:
        if event.type != "assistant_tool_call":
            continue
        if event.name != "choose_bit" or not isinstance(event.arguments, dict):
            return 0.0
        bit = event.arguments.get("bit")
        if isinstance(bit, bool) or bit not in (0, 1):
            return 0.0
        actions.append(int(bit))
    if len(actions) != task.HORIZON:
        return 0.0
    source = record.source_record
    weights = task.validate_task(source["weights"], source["threshold"])
    return float(task.terminal_reward(weights, int(source["threshold"]), actions))

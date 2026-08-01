"""Canonical JSON-object reward contract used as the P4 intervention."""

from __future__ import annotations

from examples.agentic.care_bifurcation import task
from research.silent_reward_contracts.auditor import canonical_action_arguments


def reward_fn(record) -> float:
    actions = []
    for event in record.trace:
        if event.type != "assistant_tool_call":
            continue
        if event.name != "choose_bit":
            return 0.0
        try:
            arguments = canonical_action_arguments(event.arguments)
        except ValueError:
            return 0.0
        bit = arguments.get("bit")
        if isinstance(bit, bool) or bit not in (0, 1):
            return 0.0
        actions.append(int(bit))
    if len(actions) != task.HORIZON:
        return 0.0
    source = record.source_record
    weights = task.validate_task(source["weights"], source["threshold"])
    return float(task.terminal_reward(weights, int(source["threshold"]), actions))

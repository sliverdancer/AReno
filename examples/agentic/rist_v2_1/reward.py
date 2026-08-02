"""Strict binary exact-oracle reward for RIST-v2.1."""

from __future__ import annotations

import json
import os
from pathlib import Path


def reward_fn(record) -> float:
    """Return one only for the exact complete oracle action sequence."""

    def finish(value: float) -> float:
        _append_reward_event(record, value)
        return value

    expected = list(record.source_record["oracle_actions"])
    observed = []
    for call in record.tool_calls:
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return finish(0.0)
        if not isinstance(arguments, dict):
            return finish(0.0)
        observed.append({"name": call.get("name"), "arguments": arguments})
    reward = float(observed == expected)
    return finish(reward)


def _append_reward_event(record, reward: float) -> None:
    journal = os.environ.get("RIST_REWARD_JOURNAL_PATH")
    if not journal:
        return
    source = record.source_record
    payload = {
        "task_id": source.get("id"),
        "task_signature": source.get("task_signature"),
        "structural_cell": source.get("structural_cell"),
        "resolution_band": source.get("resolution_band"),
        "prompt_index": int(record.metadata["prompt_index"]),
        "sample_index": int(record.metadata["sample_index"]),
        "reward": reward,
    }
    descriptor = os.open(
        Path(journal), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
    )
    try:
        os.write(descriptor, (json.dumps(payload, sort_keys=True) + "\n").encode())
    finally:
        os.close(descriptor)

"""Strict binary exact-oracle reward for RIST-v2.1."""

from __future__ import annotations

import json
from typing import Any


def reward_fn(record) -> float:
    """Return one only for the exact complete oracle action sequence."""

    expected = list(record.source_record["oracle_actions"])
    observed = []
    for call in record.tool_calls:
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return 0.0
        if not isinstance(arguments, dict):
            return 0.0
        observed.append({"name": call.get("name"), "arguments": arguments})
    return float(observed == expected)

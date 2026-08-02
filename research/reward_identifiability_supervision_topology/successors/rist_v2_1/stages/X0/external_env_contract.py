"""Strict JSON transcript contract for future Tau3 and BFCL adapters."""

from __future__ import annotations

import math
from typing import Any

STRICT_REWARD_SOURCES = {
    "db_and_communicate",
    "bfcl_ast_execution",
    "strict_state_transition",
}


def _valid_hash(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_transcript(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate an append-only reset/step transcript without environment imports."""

    if len(events) < 2 or events[0].get("type") != "reset":
        raise ValueError("transcript must begin with reset and contain at least one step")
    reset = events[0]
    if not isinstance(reset.get("episode_id"), str) or not reset["episode_id"]:
        raise ValueError("reset requires a non-empty episode_id")
    if not _valid_hash(reset.get("state_hash")):
        raise ValueError("reset requires a lowercase SHA-256 state_hash")
    if not isinstance(reset.get("observation"), (str, dict, list)):
        raise ValueError("reset observation must be JSON-compatible")

    previous_hash = reset["state_hash"]
    terminal_seen = False
    for expected_index, event in enumerate(events[1:]):
        if event.get("type") != "step" or event.get("step_index") != expected_index:
            raise ValueError("step indices must be contiguous from zero")
        if terminal_seen:
            raise ValueError("no events may follow a terminal step")
        action = event.get("action")
        if not isinstance(action, dict) or set(action) != {"name", "arguments"}:
            raise ValueError("step action must contain only name and arguments")
        if not isinstance(action["name"], str) or not isinstance(action["arguments"], dict):
            raise ValueError("step action schema is invalid")
        if not _valid_hash(event.get("state_hash_before")) or event["state_hash_before"] != previous_hash:
            raise ValueError("state_hash_before must chain from the prior event")
        if not _valid_hash(event.get("state_hash_after")):
            raise ValueError("step requires state_hash_after")
        reward = event.get("reward")
        if not isinstance(reward, (int, float)) or isinstance(reward, bool) or not math.isfinite(reward):
            raise ValueError("step reward must be finite numeric")
        if event.get("reward_source") not in STRICT_REWARD_SOURCES:
            raise ValueError("primary reward source is not strict or is LLM-judged")
        if not isinstance(event.get("raw_tool_result"), (str, dict, list)):
            raise ValueError("raw_tool_result must be preserved")
        if not isinstance(event.get("done"), bool):
            raise ValueError("done must be boolean")
        previous_hash = event["state_hash_after"]
        terminal_seen = event["done"]
    if not terminal_seen:
        raise ValueError("transcript must end in a terminal step")
    return {
        "valid": True,
        "episode_id": reset["episode_id"],
        "step_count": len(events) - 1,
        "final_state_hash": previous_hash,
        "llm_judge_used": False,
    }

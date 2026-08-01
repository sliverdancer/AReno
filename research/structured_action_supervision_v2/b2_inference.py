"""Pure-Python validation and scoring helpers for SAS-TR-v2.0 B2."""

from __future__ import annotations

import hashlib
import json
from typing import Any

PROTOCOL_ID = "SAS-TR-v2.0"
EXPECTED_TOOLS = (
    "search_catalog",
    "inspect_items",
    "check_kit",
    "submit_bundle",
)
SYSTEM_PROMPT = (
    "You are a shopping kit planner. Follow the requested four tool turns. "
    "Return exactly one tool call on every turn and never answer in plain text."
)
TURN_PROMPTS = {
    "search_catalog": "Turn 1: search all required task categories.",
    "inspect_items": "Turn 2: inspect the candidate item ids needed to decide.",
    "check_kit": "Turn 3: check exactly one complete proposed kit.",
    "submit_bundle": "Turn 4: submit exactly one final kit.",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": "Search compact catalog results for one or more categories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "categories": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["jacket", "shoes", "bottle"],
                        },
                    },
                    "max_price": {"type": "integer"},
                },
                "required": ["categories"],
                "additionalProperties": False,
            },
        },
    },
    *[
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["item_ids"],
                    "additionalProperties": False,
                },
            },
        }
        for name, description in (
            ("inspect_items", "Inspect full details for candidate item ids."),
            ("check_kit", "Check one proposed kit against the task constraints."),
            ("submit_bundle", "Submit the final kit item ids."),
        )
    ],
]
TOOL_BY_NAME = {tool["function"]["name"]: tool for tool in TOOLS}


def request_seed(base_seed: int, row_index: int, turn_index: int) -> int:
    """Derive a stable signed-31-bit request seed from the frozen factors."""

    material = f"{PROTOCOL_ID}\x1f{base_seed}\x1f{row_index}\x1f{turn_index}"
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big") % (2**31)


def validate_response(response: dict[str, Any], expected_name: str) -> dict[str, Any]:
    """Validate exactly one parsed call while retaining the untouched response."""

    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return _invalid("INVALID_RESPONSE_CHOICES")
    choice = choices[0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        return _invalid("INVALID_RESPONSE_MESSAGE")
    message = choice["message"]
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list) or not calls:
        return _invalid("MISSING_TOOL_CALL", choice)
    if len(calls) != 1:
        return _invalid("MULTIPLE_TOOL_CALLS", choice)
    call = calls[0]
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict) or function.get("name") != expected_name:
        return _invalid("UNEXPECTED_TOOL_NAME", choice)
    raw_arguments = function.get("arguments")
    try:
        arguments = json.loads(raw_arguments)
    except (TypeError, json.JSONDecodeError):
        return _invalid("INVALID_JSON_ARGUMENTS", choice)
    if not isinstance(arguments, dict):
        return _invalid("ARGUMENTS_NOT_OBJECT", choice)
    if not valid_argument_schema(expected_name, arguments):
        return _invalid("INVALID_ARGUMENT_SCHEMA", choice)
    return {
        "valid": True,
        "reason": "VALID",
        "finish_reason": choice.get("finish_reason"),
        "assistant_message": message,
        "arguments": arguments,
    }


def _invalid(reason: str, choice: dict[str, Any] | None = None) -> dict[str, Any]:
    choice = choice or {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    return {
        "valid": False,
        "reason": reason,
        "finish_reason": choice.get("finish_reason"),
        "assistant_message": message,
        "arguments": None,
    }


def valid_argument_schema(name: str, arguments: dict[str, Any]) -> bool:
    if name == "search_catalog":
        if set(arguments) - {"categories", "max_price"}:
            return False
        categories = arguments.get("categories")
        if not unique_nonempty_strings(categories):
            return False
        if not set(categories).issubset({"jacket", "shoes", "bottle"}):
            return False
        max_price = arguments.get("max_price")
        return max_price is None or (
            isinstance(max_price, int)
            and not isinstance(max_price, bool)
            and max_price >= 0
        )
    if name in {"inspect_items", "check_kit", "submit_bundle"}:
        return set(arguments) == {"item_ids"} and unique_nonempty_strings(
            arguments.get("item_ids")
        )
    return False


def unique_nonempty_strings(values: Any) -> bool:
    return (
        isinstance(values, list)
        and bool(values)
        and all(isinstance(value, str) and bool(value.strip()) for value in values)
        and len(values) == len(set(values))
    )


def execute_call(game: Any, name: str, arguments: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Execute one validated call against the immutable shopping game."""

    if name == "search_catalog":
        return {
            "results_by_category": game.search_catalog_many(
                arguments["categories"], max_price=arguments.get("max_price")
            )
        }
    if name == "inspect_items":
        return {"items": game.inspect_items(arguments["item_ids"])}
    if name == "check_kit":
        return {"kit": game.check_kit(record, arguments["item_ids"])}
    if name == "submit_bundle":
        return {"submitted": arguments["item_ids"]}
    raise ValueError(f"unsupported SAS tool: {name}")


def strict_reward(game: Any, record: dict[str, Any], calls: list[dict[str, Any]]) -> float:
    """Mirror the frozen reward contract without importing AReno or Torch."""

    if [call.get("name") for call in calls] != list(EXPECTED_TOOLS):
        return -1.0
    arguments = [call.get("arguments") for call in calls]
    if not all(isinstance(value, dict) for value in arguments):
        return -1.0
    search_args, inspect_args, check_args, submit_args = arguments
    if set(search_args.get("categories", [])) != set(record["categories"]):
        return -1.0
    for value in (inspect_args, check_args, submit_args):
        if not unique_nonempty_strings(value.get("item_ids")):
            return -1.0
    return float(game.score_bundle(record, submit_args["item_ids"]))


def summarize(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute frozen qualification metrics from complete evidence records."""

    count = len(trajectories)
    first = sum(bool(item.get("first_turn_executable")) for item in trajectories)
    complete = sum(bool(item.get("four_turn_complete")) for item in trajectories)
    positive = sum(float(item.get("strict_reward", -1.0)) > 0 for item in trajectories)
    fabricated = sum(int(item.get("fabricated_call_count", 0)) for item in trajectories)
    raw_complete = all(
        isinstance(item.get("turns"), list)
        and bool(item["turns"])
        and all(isinstance(turn.get("raw_response"), dict) for turn in item["turns"])
        for item in trajectories
    )
    return {
        "trajectory_count": count,
        "first_turn_executable_rate": first / count if count else 0.0,
        "four_turn_completion_rate": complete / count if count else 0.0,
        "positive_reward_rate": positive / count if count else 0.0,
        "fabricated_call_count": fabricated,
        "raw_evidence_complete": raw_complete,
    }

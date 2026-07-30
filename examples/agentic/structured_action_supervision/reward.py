"""Strict outcome reward for the SAS shopping instrument."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

SHOPPING_DIR = Path(__file__).resolve().parents[1] / "shopping"
sys.path.insert(0, str(SHOPPING_DIR))
from game import score_bundle  # noqa: E402

EXPECTED_TOOLS = (
    "search_catalog",
    "inspect_items",
    "check_kit",
    "submit_bundle",
)


def reward_fn(record) -> float:
    """Return outcome reward only for an exact valid four-call protocol."""

    source = dict(record.source_record)
    calls = list(record.tool_calls)
    if [call.get("name") for call in calls] != list(EXPECTED_TOOLS):
        return -1.0
    parsed = [_arguments(call) for call in calls]
    if any(arguments is None for arguments in parsed):
        return -1.0
    search_args, inspect_args, check_args, submit_args = parsed
    if set(search_args.get("categories", [])) != set(source["categories"]):
        return -1.0
    for arguments in (inspect_args, check_args, submit_args):
        if not _item_ids(arguments):
            return -1.0
    return score_bundle(source, _item_ids(submit_args))


def _arguments(call: dict[str, Any]) -> dict[str, Any] | None:
    arguments = call.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    return arguments if isinstance(arguments, dict) else None


def _item_ids(arguments: dict[str, Any]) -> list[str] | None:
    values = arguments.get("item_ids")
    if not isinstance(values, list) or not values:
        return None
    if not all(isinstance(value, str) and value for value in values):
        return None
    if len(values) != len(set(values)):
        return None
    return values


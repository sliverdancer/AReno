"""Minimal typed and identifiability checks for silent reward contracts.

This module intentionally depends only on the Python standard library.  It is
research-local and does not alter AReno's public API or runtime behavior.
"""

from __future__ import annotations

import json
from typing import Any


def canonical_action_arguments(arguments: Any) -> dict[str, Any]:
    """Return equivalent JSON-object arguments or raise a contract error."""

    value = arguments
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("tool arguments are not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("tool arguments must encode a JSON object")
    return dict(value)


def masks_are_distinguishable(left: list[bool], right: list[bool]) -> bool:
    """Return whether two registered treatments select different tokens."""

    if len(left) != len(right):
        raise ValueError("mask lengths differ")
    return left != right


def reward_is_informative(rewards: list[float]) -> bool:
    """Return whether a finite reward fixture contains more than one value."""

    if not rewards:
        return False
    return len({float(value) for value in rewards}) > 1

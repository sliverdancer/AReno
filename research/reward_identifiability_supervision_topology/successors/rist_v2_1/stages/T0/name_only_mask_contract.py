"""Fail-closed exact tool-name-only token mask prototype.

This module is deliberately research-local.  It proves the token-offset
contract needed by the future public AReno option without changing that API.
"""

from __future__ import annotations

import json
import re
from typing import Any

_NAME_FIELD = re.compile(r'"name"\s*:\s*("(?:\\.|[^"\\])*")')


def exact_name_only_mask(
    tokenizer: Any,
    raw_call: str,
    response_tokens: list[int],
    base_mask: list[bool],
) -> list[bool]:
    """Keep only exact tool-name tokens while preserving prior suppression.

    A token that touches both the name and JSON syntax makes the treatment
    non-identifiable at this tokenization and is rejected rather than rounded.
    """

    if not response_tokens or len(response_tokens) != len(base_mask):
        raise ValueError("response_tokens and base_mask must be non-empty and aligned")
    if any(type(bit) is not bool for bit in base_mask):
        raise ValueError("base_mask must contain booleans")
    try:
        payload = json.loads(raw_call)
    except json.JSONDecodeError as exc:
        raise ValueError("raw_call must be valid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("name"), str):
        raise ValueError("raw_call requires a string name field")
    matches = list(_NAME_FIELD.finditer(raw_call))
    matching = [
        match
        for match in matches
        if json.loads(match.group(1)) == payload["name"]
    ]
    if len(matching) != 1:
        raise ValueError("tool name span must be unique and exact")
    quoted_start, quoted_end = matching[0].span(1)
    name_span = (quoted_start + 1, quoted_end - 1)
    if name_span[0] >= name_span[1]:
        raise ValueError("tool name must be non-empty")

    encoded = tokenizer(
        raw_call,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [tuple(int(value) for value in pair) for pair in encoded["offset_mapping"]]
    if token_ids != [int(value) for value in response_tokens]:
        raise ValueError("tokenizer encoding does not match response_tokens")
    if len(offsets) != len(token_ids):
        raise ValueError("offset mapping must align with response_tokens")

    selected: list[int] = []
    covered: set[int] = set()
    for index, (start, end) in enumerate(offsets):
        if start < 0 or end < start or end > len(raw_call):
            raise ValueError("invalid tokenizer offset")
        overlaps = start < name_span[1] and name_span[0] < end
        inside = name_span[0] <= start and end <= name_span[1]
        if overlaps and not inside:
            raise ValueError("token mixes tool-name characters with syntax")
        if inside and end > start:
            selected.append(index)
            covered.update(range(start, end))
    if covered != set(range(*name_span)):
        raise ValueError("tool-name characters are not exactly token-covered")
    if not selected:
        raise ValueError("tool name has no independently trainable token")

    selected_set = set(selected)
    return [
        bool(enabled and index in selected_set)
        for index, enabled in enumerate(base_mask)
    ]

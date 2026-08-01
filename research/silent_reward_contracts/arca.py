"""Frozen framework-independent ARCA contract rules."""

from __future__ import annotations

from collections import Counter
import json
import math
from typing import Any


RULE_CODES = (
    "ARCA-F1-TYPE",
    "ARCA-F2-PAIRING",
    "ARCA-F2-REWARD-FLOW",
    "ARCA-F3-INFORMATIVE",
    "ARCA-F4-ADVANTAGE",
    "ARCA-F5-TREATMENT",
    "ARCA-F6-STATUS",
    "ARCA-F6-PROVENANCE",
)


def _canonical_json_object(value: Any) -> str:
    candidate = value
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSON") from exc
    if not isinstance(candidate, dict):
        raise ValueError("not a JSON object")
    return json.dumps(candidate, sort_keys=True, separators=(",", ":"))


def audit_case(case: dict[str, Any]) -> list[str]:
    """Return sorted contract violations for one schema-valid case payload."""

    payload = case["payload"]
    violations: set[str] = set()

    variants = payload.get("argument_variants")
    if variants is not None:
        try:
            canonical = {_canonical_json_object(value) for value in variants}
        except ValueError:
            violations.add("ARCA-F1-TYPE")
        else:
            if len(canonical) != 1:
                violations.add("ARCA-F1-TYPE")

    call_ids = payload.get("call_ids")
    result_call_ids = payload.get("result_call_ids")
    if call_ids is not None or result_call_ids is not None:
        if Counter(call_ids or []) != Counter(result_call_ids or []):
            violations.add("ARCA-F2-PAIRING")

    intended_keys = set(payload.get("intended_reward_keys", []))
    if intended_keys:
        produced = set(payload.get("produced_reward_keys", []))
        consumed = set(payload.get("consumed_reward_keys", []))
        if not intended_keys.issubset(produced & consumed):
            violations.add("ARCA-F2-REWARD-FLOW")

    controls = payload.get("reward_controls")
    if controls is not None:
        values = [float(value) for value in controls]
        if not values or any(not math.isfinite(value) for value in values):
            violations.add("ARCA-F3-INFORMATIVE")
        elif len(set(values)) < 2:
            violations.add("ARCA-F3-INFORMATIVE")

    groups = payload.get("group_rewards")
    if groups is not None:
        if not groups or any(len(group) < 2 for group in groups):
            violations.add("ARCA-F4-ADVANTAGE")
        elif all(len({float(value) for value in group}) < 2 for group in groups):
            violations.add("ARCA-F4-ADVANTAGE")

    masks = payload.get("treatment_masks")
    if masks is not None:
        declared = {
            tuple(sorted(pair)) for pair in payload.get("declared_aliases", [])
        }
        names = sorted(masks)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                if masks[left] == masks[right] and (left, right) not in declared:
                    violations.add("ARCA-F5-TREATMENT")

    for failure in payload.get("execution_failures", []):
        if (
            failure.get("numeric_reward") is not None
            and not failure.get("structured_status")
        ):
            violations.add("ARCA-F6-STATUS")

    provenance = payload.get("provenance")
    if provenance is not None:
        required = provenance.get("required", {})
        observed = provenance.get("observed", {})
        if any(observed.get(key) != value for key, value in required.items()):
            violations.add("ARCA-F6-PROVENANCE")

    return sorted(violations)

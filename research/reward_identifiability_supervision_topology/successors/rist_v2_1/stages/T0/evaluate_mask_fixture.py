"""Evaluate captured response-token masks for exact RIST treatment semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROLES = ("name_indices", "argument_indices", "other_indices", "shared_indices")


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    """Return exact full/argument-masked/name-only treatment checks."""

    token_ids = list(case["token_ids"])
    mask = list(case["loss_mask"])
    if len(token_ids) == 0 or len(mask) != len(token_ids):
        raise ValueError("token_ids and loss_mask must be non-empty and aligned")
    if any(type(bit) is not bool for bit in mask):
        raise ValueError("loss_mask values must be booleans")

    role_sets = {
        role: {int(index) for index in case.get(role, [])} for role in ROLES
    }
    universe = set(range(len(token_ids)))
    if any(not indices <= universe for indices in role_sets.values()):
        raise ValueError("role index lies outside token_ids")
    if any(
        role_sets[left] & role_sets[right]
        for i, left in enumerate(ROLES)
        for right in ROLES[i + 1 :]
    ):
        raise ValueError("token roles must be disjoint; shared tokens are not exact")
    if set().union(*role_sets.values()) != universe:
        raise ValueError("token roles must partition every response token")
    if not role_sets["name_indices"] or not role_sets["argument_indices"]:
        raise ValueError("each case requires name and argument tokens")

    enabled = {index for index, bit in enumerate(mask) if bit}
    name = role_sets["name_indices"]
    arguments = role_sets["argument_indices"]
    other = role_sets["other_indices"]
    shared = role_sets["shared_indices"]
    return {
        "case_id": str(case["case_id"]),
        "localization_pass": case.get("localization_pass") is True,
        "full_call_exact": enabled == universe,
        "argument_mask_exact": not shared and name <= enabled and not (arguments & enabled),
        "name_only_exact": not shared and enabled == name and not ((arguments | other) & enabled),
        "enabled_indices": sorted(enabled),
        "name_indices": sorted(name),
        "argument_indices": sorted(arguments),
        "other_indices": sorted(other),
        "shared_indices": sorted(shared),
    }


def evaluate_fixture(fixture: dict[str, Any]) -> dict[str, Any]:
    cases = fixture.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("fixture requires non-empty cases")
    results = [evaluate_case(case) for case in cases]
    localization_all = all(result["localization_pass"] for result in results)
    production_mask = (
        fixture.get("mask_implementation")
        == "areno.api.agentic._tool_call_name_only_loss_mask"
    )
    canonical_case_count = fixture.get("case_count") == 32 == len(results)
    tokenizer_only = (
        fixture.get("local_files_only") is True
        and fixture.get("model_weights_present") is False
        and isinstance(fixture.get("tokenizer_snapshot"), dict)
    )
    runtime_capture = fixture.get("runtime_response_tokens") is True
    turn_case_counts = fixture.get("turn_case_counts")
    balanced_four_turns = turn_case_counts == {str(turn): 8 for turn in range(4)}
    name_only_all = all(result["name_only_exact"] for result in results)
    return {
        "checkpoint": fixture["checkpoint"],
        "tokenizer_revision": fixture.get("tokenizer_revision"),
        "tokenizer_sha256": fixture["tokenizer_sha256"],
        "case_count": len(results),
        "canonical_case_count": canonical_case_count,
        "production_mask": production_mask,
        "tokenizer_only": tokenizer_only,
        "runtime_capture": runtime_capture,
        "balanced_four_turns": balanced_four_turns,
        "localization_all": localization_all,
        "full_call_all": all(result["full_call_exact"] for result in results),
        "argument_mask_all": all(result["argument_mask_exact"] for result in results),
        "name_only_all": name_only_all,
        "qualification_pass": all(
            (
                canonical_case_count,
                production_mask,
                tokenizer_only,
                runtime_capture,
                balanced_four_turns,
                localization_all,
                name_only_all,
            )
        ),
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    result = evaluate_fixture(fixture)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if result["qualification_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

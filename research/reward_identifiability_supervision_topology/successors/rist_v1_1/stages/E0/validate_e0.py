"""Apply the frozen RIST-E0 cross-model infrastructure gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_MODELS = {"qwen3_0_6b", "gemma4_e2b_it"}


def validate(
    results: list[dict[str, Any]], *, preflight_passed: bool = True
) -> dict[str, Any]:
    """Return a mechanical E0 decision; never infer scientific performance."""

    if not preflight_passed:
        return {
            "schema_version": 1,
            "protocol": "RIST-E0-v1.1",
            "stage": "E0",
            "stage_status": "INVALID",
            "decision": "INVALID_E0_PREFLIGHT_STOP",
            "scientific_interpretation": "FORBIDDEN_INFRASTRUCTURE_ONLY",
            "models": [],
            "training_performed": False,
        }

    by_model = {result.get("model_cell"): result for result in results}
    if set(by_model) != EXPECTED_MODELS:
        raise ValueError("E0 validator requires exactly both frozen model cells")

    summaries = []
    any_infrastructure_error = False
    all_interface_pass = True
    for model_cell in sorted(EXPECTED_MODELS):
        result = by_model[model_cell]
        infrastructure_error = result.get("infrastructure_error")
        any_infrastructure_error |= infrastructure_error is not None
        trajectories = result.get("trajectories") or []
        records = [
            record
            for trajectory in trajectories
            for record in trajectory.get("records", [])
        ]
        raw_response_count = sum(
            int(trajectory.get("raw_response_count", 0))
            for trajectory in trajectories
        )
        parse_valid_count = sum(bool(record.get("parse_valid")) for record in records)
        exact_instruction_count = sum(
            bool(record.get("exact_instruction")) for record in records
        )
        passed = (
            infrastructure_error is None
            and len(trajectories) == 2
            and len(records) == 8
            and raw_response_count == 8
            and parse_valid_count == 8
            and exact_instruction_count == 8
            and int(result.get("fabricated_call_count", -1)) == 0
            and int(result.get("retry_count", -1)) == 0
        )
        all_interface_pass &= passed
        summaries.append(
            {
                "model_cell": model_cell,
                "trajectory_count": len(trajectories),
                "raw_response_count": raw_response_count,
                "parse_valid_count": parse_valid_count,
                "exact_instruction_count": exact_instruction_count,
                "infrastructure_error": infrastructure_error,
                "passed": passed,
            }
        )

    if any_infrastructure_error:
        decision = "INVALID_E0_INFRASTRUCTURE_STOP"
        status = "INVALID"
    elif all_interface_pass:
        decision = "PASS_E0_INFRASTRUCTURE_TO_P2_1_PROTOCOL_FREEZE"
        status = "PASS"
    else:
        decision = "FAIL_E0_MODEL_INTERFACE_STOP"
        status = "FAIL"
    return {
        "schema_version": 1,
        "protocol": "RIST-E0-v1.1",
        "stage": "E0",
        "stage_status": status,
        "decision": decision,
        "scientific_interpretation": "FORBIDDEN_INFRASTRUCTURE_ONLY",
        "models": summaries,
        "training_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-result", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preflight = json.loads(args.preflight_result.read_text(encoding="utf-8"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in args.result]
    outcome = validate(results, preflight_passed=bool(preflight.get("passed")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(outcome, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(outcome, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if outcome["stage_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

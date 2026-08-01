"""Apply the RIST main-conference gate after a completed research stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def assess(stage_result: dict[str, Any]) -> dict[str, Any]:
    """Return the mechanical main-track decision for one terminal stage result."""

    stage = str(stage_result.get("stage", ""))
    status = str(stage_result.get("stage_status", ""))
    if status == "INVALID":
        decision = "INVALID_PROTOCOL_STOP"
        upgraded = False
    elif status != "PASS":
        decision = "KILL_CURRENT_ROUTE"
        upgraded = False
    elif stage == "P0":
        decision = "STAY_DIAGNOSTIC_OPEN_P1_CPU_FREEZE"
        upgraded = False
    elif stage == "P1":
        decision = "STAY_DIAGNOSTIC_OPEN_P2_GPU_AUTHORIZATION"
        upgraded = False
    elif stage == "P2":
        decision = "STAY_DIAGNOSTIC_OPEN_P3_FACTORIAL_PILOT"
        upgraded = False
    elif stage == "P3":
        upgraded = bool(stage_result.get("power_feasible", False))
        decision = (
            "UPGRADE_MAIN_TRACK_CANDIDATE_OPEN_P4_POWER_FREEZE"
            if upgraded
            else "KILL_UNDERPOWERED_MAIN_TRACK"
        )
    elif stage == "P5":
        required = stage_result.get("main_track_evidence") or {}
        required_flags = (
            "primary_interaction_practical",
            "primary_interval_excludes_zero",
            "multi_environment_complete",
            "multi_model_complete",
            "multi_algorithm_complete",
            "direct_baselines_complete",
            "artifact_reproducible",
            "no_material_protocol_deviation",
        )
        upgraded = all(bool(required.get(flag, False)) for flag in required_flags)
        decision = "GO_MAIN_TRACK" if upgraded else "STAY_DIAGNOSTIC"
    else:
        decision = "INVALID_UNKNOWN_STAGE_STOP"
        upgraded = False
    return {
        "schema_version": 1,
        "protocol_family": "RIST-v1",
        "stage": stage,
        "source_stage_decision": stage_result.get("decision"),
        "decision": decision,
        "upgraded": upgraded,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-result", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stage_result = json.loads(args.stage_result.read_text(encoding="utf-8"))
    result = assess(stage_result)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


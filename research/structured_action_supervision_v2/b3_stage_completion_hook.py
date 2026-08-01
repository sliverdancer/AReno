"""Assess main-track status when the SAS-B3-v3.0 signal gate closes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CRITERIA = (
    "tool_interface_valid",
    "within_group_learning_signal_valid",
    "af_lf_contrast_instantiated",
    "full_vs_name_contrast_instantiated",
    "primary_effect_practical",
    "primary_interval_excludes_zero",
    "independent_seeds_sufficient",
    "multi_environment_complete",
    "multi_model_complete",
    "multi_algorithm_complete",
    "direct_baselines_complete",
    "artifact_reproducible",
    "novel_benchmark_or_method",
    "no_material_protocol_deviation",
)


def assess(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if payload.get("protocol_id") != "SAS-B3-v3.0":
        raise ValueError("unsupported B3 protocol")
    if payload.get("stage") != "B3-A":
        raise ValueError("unsupported B3 stage")
    if payload.get("stage_status") not in {"PASS", "KILL", "INVALID"}:
        raise ValueError("unsupported B3 stage status")
    evidence = payload.get("main_track_evidence", {})
    unknown = sorted(set(evidence) - set(CRITERIA))
    if unknown:
        raise ValueError(f"unknown main-track evidence keys: {unknown}")
    if payload["stage_status"] == "PASS":
        protocol_action = "OPEN_FACTORIAL_TRAINING"
        reasons = [
            "the within-group signal gate passed, so a bounded factorial training pilot may open"
        ]
    else:
        protocol_action = "KILL_CURRENT_PROTOCOL"
        reasons = [
            "the current GSPO pilot has no qualified within-group learning signal"
            if payload["stage_status"] == "KILL"
            else "the B3 protocol closed invalid"
        ]
    return {
        "schema_version": 1,
        "protocol_id": payload["protocol_id"],
        "stage": payload["stage"],
        "stage_status": payload["stage_status"],
        "decision": "STAY_DIAGNOSTIC",
        "protocol_action": protocol_action,
        "reasons": reasons
        + [
            "B3-A estimates no supervision effect and cannot support a main-conference upgrade"
        ],
        "criteria": list(CRITERIA),
        "unmet_criteria": [
            criterion for criterion in CRITERIA if not evidence.get(criterion, False)
        ],
    }


def finalize(stage_result: Path, output: Path | None = None) -> Path:
    source = stage_result.read_bytes()
    result = assess(json.loads(source))
    result["stage_result_sha256"] = hashlib.sha256(source).hexdigest()
    destination = output or stage_result.with_name("main_track_assessment.json")
    serialized = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(f"refusing to overwrite a different assessment: {destination}")
    destination.write_text(serialized, encoding="utf-8")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = finalize(args.stage_result, args.output)
    print(json.dumps({"assessment": str(destination), "ok": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

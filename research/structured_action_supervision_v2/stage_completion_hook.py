"""Assess main-track eligibility when a SAS tool-readiness stage closes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROTOCOL_IDS = {"SAS-TR-v2.0", "SAS-TR-v2.1"}
STAGES = {"B0", "B1", "B2", "B3"}
STATUSES = {"PASS", "BLOCKED", "KILL", "INVALID"}
TERMINAL = {"KILL", "INVALID"}
CRITERIA = (
    "tool_readiness_causally_validated",
    "af_lf_contrast_instantiated",
    "primary_effect_practical",
    "primary_interval_excludes_zero",
    "multi_environment_complete",
    "multi_model_complete",
    "multi_algorithm_complete",
    "direct_baselines_complete",
    "independent_seeds_sufficient",
    "artifact_reproducible",
    "novel_benchmark_or_method",
    "no_material_protocol_deviation",
)


def assess(payload: dict[str, Any]) -> dict[str, Any]:
    _validate(payload)
    status = payload["stage_status"]
    evidence = payload.get("main_track_evidence", {})
    if status in TERMINAL:
        decision = "KILL_MAIN_TRACK"
        reasons = [f"bridge stage closed with terminal status {status}"]
    else:
        decision = "STAY_DIAGNOSTIC"
        reasons = [
            f"{payload['protocol_id']} is an instrument-qualification bridge and cannot "
            "produce main-track efficacy evidence"
        ]
        if status == "BLOCKED":
            reasons.append("the current bridge stage remains blocked")
    return {
        "schema_version": 1,
        "protocol_id": payload["protocol_id"],
        "stage": payload["stage"],
        "stage_status": status,
        "decision": decision,
        "reasons": reasons,
        "criteria": list(CRITERIA),
        "unmet_criteria": [
            criterion for criterion in CRITERIA if not evidence.get(criterion, False)
        ],
    }


def finalize(stage_result: Path, output: Path | None = None) -> Path:
    source = stage_result.read_bytes()
    payload = json.loads(source)
    result = assess(payload)
    result["stage_result_sha256"] = hashlib.sha256(source).hexdigest()
    destination = output or stage_result.with_name("main_track_assessment.json")
    serialized = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(f"refusing to overwrite a different assessment: {destination}")
        return destination
    destination.write_text(serialized, encoding="utf-8")
    return destination


def _validate(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if payload.get("protocol_id") not in PROTOCOL_IDS:
        raise ValueError(f"unsupported protocol_id: {payload.get('protocol_id')!r}")
    if payload.get("stage") not in STAGES:
        raise ValueError(f"unsupported stage: {payload.get('stage')!r}")
    if payload.get("stage_status") not in STATUSES:
        raise ValueError(f"unsupported stage_status: {payload.get('stage_status')!r}")
    evidence = payload.get("main_track_evidence", {})
    if not isinstance(evidence, dict):
        raise ValueError("main_track_evidence must be an object")
    unknown = sorted(set(evidence) - set(CRITERIA))
    if unknown:
        raise ValueError(f"unknown main-track evidence keys: {unknown}")
    non_boolean = sorted(
        key for key, value in evidence.items() if not isinstance(value, bool)
    )
    if non_boolean:
        raise ValueError(f"main-track evidence values must be boolean: {non_boolean}")


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

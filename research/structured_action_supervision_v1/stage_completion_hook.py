"""Evaluate main-track eligibility whenever an SAS research stage closes.

The hook consumes one immutable ``stage_result.json`` and writes a deterministic
``main_track_assessment.json`` beside it. It never upgrades a pilot to a
confirmatory result: only a valid Q3 result can return ``GO_MAIN_TRACK``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DECISIONS = {"GO_MAIN_TRACK", "STAY_DIAGNOSTIC", "KILL_MAIN_TRACK"}
STAGES = {"P0", "Q0", "Q1", "Q2", "Q3", "Q4"}
PROTOCOL_IDS = {"SAS-P0-v1.0", "SAS-P0-v1.1"}
TERMINAL_FAILURES = {"KILL", "INVALID"}
MAIN_TRACK_CRITERIA = (
    "protocol_valid",
    "primary_effect_practical",
    "primary_interval_excludes_zero",
    "multi_environment_complete",
    "multi_model_complete",
    "multi_algorithm_complete",
    "direct_baselines_complete",
    "independent_seeds_sufficient",
    "artifact_reproducible",
    "novel_method_or_benchmark",
    "no_material_protocol_deviation",
)


def assess_stage_result(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic main-track assessment for one stage result."""

    _validate_stage_result(payload)
    stage = str(payload["stage"])
    status = str(payload["stage_status"])
    evidence = payload.get("main_track_evidence") or {}

    if status in TERMINAL_FAILURES:
        decision = "KILL_MAIN_TRACK"
        reasons = [f"stage closed with terminal status {status}"]
        unmet = list(MAIN_TRACK_CRITERIA)
    elif stage != "Q3" or status != "PASS":
        decision = "STAY_DIAGNOSTIC"
        reasons = [
            "main-track upgrade is forbidden before a valid Q3 confirmatory result"
        ]
        if status == "BLOCKED":
            reasons.append("the current stage remains blocked")
        unmet = [criterion for criterion in MAIN_TRACK_CRITERIA if not evidence.get(criterion, False)]
    else:
        unmet = [criterion for criterion in MAIN_TRACK_CRITERIA if not evidence.get(criterion, False)]
        if unmet:
            decision = "STAY_DIAGNOSTIC"
            reasons = ["Q3 passed, but one or more main-track criteria remain unmet"]
        else:
            decision = "GO_MAIN_TRACK"
            reasons = ["Q3 passed and every frozen main-track criterion is satisfied"]

    result = {
        "schema_version": 1,
        "protocol_id": payload["protocol_id"],
        "stage": stage,
        "stage_status": status,
        "decision": decision,
        "reasons": reasons,
        "unmet_criteria": unmet,
        "criteria": list(MAIN_TRACK_CRITERIA),
    }
    assert result["decision"] in DECISIONS
    return result


def finalize_stage(stage_result_path: Path, output_path: Path | None = None) -> Path:
    """Validate a stage result and idempotently write its main-track assessment."""

    source_bytes = stage_result_path.read_bytes()
    payload = json.loads(source_bytes)
    assessment = assess_stage_result(payload)
    assessment["stage_result_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    destination = output_path or stage_result_path.with_name("main_track_assessment.json")
    serialized = (
        json.dumps(assessment, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    if destination.exists():
        existing = destination.read_text(encoding="utf-8")
        if existing != serialized:
            raise RuntimeError(
                f"refusing to overwrite a different stage assessment: {destination}"
            )
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(serialized, encoding="utf-8")
    return destination


def _validate_stage_result(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != 1:
        raise ValueError("stage result schema_version must be 1")
    if payload.get("protocol_id") not in PROTOCOL_IDS:
        raise ValueError(
            "stage result protocol_id must be one of: "
            + ", ".join(sorted(PROTOCOL_IDS))
        )
    if payload.get("stage") not in STAGES:
        raise ValueError(f"unsupported stage: {payload.get('stage')!r}")
    if payload.get("stage_status") not in {
        "PASS",
        "BLOCKED",
        "KILL",
        "INVALID",
    }:
        raise ValueError(f"unsupported stage_status: {payload.get('stage_status')!r}")
    evidence = payload.get("main_track_evidence", {})
    if not isinstance(evidence, dict):
        raise ValueError("main_track_evidence must be an object")
    unknown = sorted(set(evidence) - set(MAIN_TRACK_CRITERIA))
    if unknown:
        raise ValueError(f"unknown main-track evidence keys: {unknown}")
    non_boolean = sorted(key for key, value in evidence.items() if not isinstance(value, bool))
    if non_boolean:
        raise ValueError(f"main-track evidence values must be boolean: {non_boolean}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage_result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = finalize_stage(args.stage_result, args.output)
    print(json.dumps({"ok": True, "assessment": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

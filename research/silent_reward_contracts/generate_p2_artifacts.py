"""Generate structured P2 evidence from AReno and a pinned veRL checkout."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from research.silent_reward_contracts.upstream_audit import (
    analyze_verl_source,
    execute_verl_reward_manager_probe,
)


SCHEMA_VERSION = "arca.p2.v1"
VERL_COMMIT = "e9618406de5bad40041d7612554e465ec2003ec1"
CASE_FIELDS = (
    "case_id",
    "framework",
    "failure_class",
    "failure_family",
    "severity",
    "natural_case",
    "ordinary_exit_ok",
    "detected",
    "conclusion_flip",
    "observed_contract",
    "oracle_contract",
)


def _load_p1(repo_root: Path) -> dict[str, Any]:
    path = (
        repo_root
        / "research/silent_reward_contracts/p1/artifacts/production_contract_cases.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def build_evidence(repo_root: Path, verl_root: Path) -> dict[str, Any]:
    p1 = _load_p1(repo_root)
    verl = analyze_verl_source(verl_root)
    verl["dynamic_probe"] = execute_verl_reward_manager_probe(verl_root)
    facts = verl["facts"]
    probe = verl["dynamic_probe"]
    cases = [
        {
            "case_id": "ARCA-P2-ARENO-ARG-TYPE",
            "framework": "AReno",
            "failure_class": "serialized_argument_type_mismatch",
            "failure_family": "F1_serialization_type",
            "severity": "high",
            "natural_case": True,
            "ordinary_exit_ok": True,
            "detected": bool(p1["cases"][0]["detected"]),
            "conclusion_flip": True,
            "observed_contract": "production_reward=0.0",
            "oracle_contract": "canonical_object_reward=1.0",
        },
        {
            "case_id": "ARCA-P2-VERL-TOOL-REWARD-KEY",
            "framework": "veRL",
            "failure_class": "tool_reward_namespace_mismatch",
            "failure_family": "F2_trace_reward_flow",
            "severity": "high",
            "natural_case": True,
            "ordinary_exit_ok": True,
            "detected": bool(
                facts["tool_reward_namespace_mismatch"]
                and not probe["tool_rewards_visible_to_compute_score"]
                and probe["rollout_reward_scores_seen"] == {}
                and probe["tool_reward_observed_score"] == 0.0
                and probe["tool_reward_oracle_score"] == 1.0
            ),
            "conclusion_flip": True,
            "observed_contract": "tool_rewards serialized but default reward manager reads reward_scores",
            "oracle_contract": "documented tool reward reaches compute_score under one stable key",
        },
        {
            "case_id": "ARCA-P2-VERL-TIMEOUT-ZERO",
            "framework": "veRL",
            "failure_class": "timeout_zero_status_conflation",
            "failure_family": "F6_artifact_provenance",
            "severity": "medium",
            "natural_case": True,
            "ordinary_exit_ok": True,
            "detected": bool(
                facts["timeout_maps_to_numeric_zero"]
                and not facts["timeout_status_structured"]
                and probe["timeout_observed_score"] == 0.0
                and not probe["timeout_structured"]
                and probe["timeout_console_marker"]
            ),
            "conclusion_flip": False,
            "observed_contract": "TimeoutError becomes score=0.0 with console text only",
            "oracle_contract": "structured timeout status remains distinguishable from legitimate zero",
        },
    ]
    natural_detected = [
        case for case in cases if case["natural_case"] and case["detected"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": "ARCA-CPU-AUDIT-v0.1",
        "gpu_executed": False,
        "frameworks": {
            "AReno": {"commit": "c96bcf2da464dff36593d43c8d29991d4b998059"},
            "veRL": {"commit": VERL_COMMIT, **verl},
        },
        "cases": cases,
        "summary": {
            "natural_medium_or_high_detected": len(natural_detected),
            "independent_frameworks": len(
                {case["framework"] for case in natural_detected}
            ),
            "distinct_failure_classes": len(
                {case["failure_class"] for case in natural_detected}
            ),
            "conclusion_flips": sum(
                bool(case["conclusion_flip"]) for case in natural_detected
            ),
        },
    }


def write_evidence(output_dir: Path, repo_root: Path, verl_root: Path) -> dict[str, Any]:
    evidence = build_evidence(repo_root, verl_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cross_system_cases.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "cross_system_cases.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CASE_FIELDS)
        writer.writeheader()
        writer.writerows(evidence["cases"])
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    write_evidence(
        args.output_dir,
        args.repo_root.resolve(),
        args.verl_root.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

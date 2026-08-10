"""Validate one RIST C0 v3 collection job without using outcome values."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

MANIFEST_PROTOCOL = "RIST-C0-v3.0-STAGE-MANIFEST-v1"
JOB_PROTOCOL = "RIST-C0-v3.0-SCIENTIFIC-JOB-EVIDENCE-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _job(manifest: dict[str, Any], job_id: str) -> dict[str, Any]:
    matches = [row for row in manifest.get("jobs", []) if row.get("job_id") == job_id]
    if len(matches) != 1:
        raise ValueError("RIST C0 v3 job id must resolve exactly once")
    return matches[0]


def validate_job(
    *,
    manifest: dict[str, Any],
    manifest_sha256: str,
    source_rows: list[dict[str, Any]],
    job_id: str,
    result: dict[str, Any],
    trajectories: list[dict[str, Any]],
    trajectory_sha256: str,
    journal_rows: list[dict[str, Any]],
    journal_sha256: str,
) -> dict[str, Any]:
    if manifest.get("protocol") != MANIFEST_PROTOCOL:
        raise ValueError("unexpected RIST C0 v3 scientific manifest")
    job = _job(manifest, job_id)
    split = str(job["split"])
    seeds = [int(seed) for seed in job["rollout_seeds"]]
    expected_pairs = {
        (str(task["id"]), int(seed))
        for task in source_rows
        for seed in seeds
    }
    signatures = {str(task["id"]): str(task["task_signature"]) for task in source_rows}
    observed_pairs = [
        (str(row.get("task_id")), int(row.get("rollout_seed", -1)))
        for row in trajectories
    ]
    duplicate_pairs = len(observed_pairs) != len(set(observed_pairs))
    identity_complete = (
        len(trajectories) == len(expected_pairs)
        and len(expected_pairs) == int(job["trajectory_count"])
        and set(observed_pairs) == expected_pairs
        and not duplicate_pairs
        and all(
            row.get("split") == split
            and str(row.get("task_signature")) == signatures.get(str(row.get("task_id")))
            and 1 <= int(row.get("raw_response_count", 0)) <= 4
            for row in trajectories
        )
    )

    journal_by_pair: dict[tuple[str, int], list[int]] = defaultdict(list)
    journal_schema_pass = True
    for row in journal_rows:
        try:
            pair = (str(row["task_id"]), int(row["rollout_seed"]))
            turn_index = int(row["turn_index"])
        except (KeyError, TypeError, ValueError):
            journal_schema_pass = False
            continue
        if (
            pair not in expected_pairs
            or str(row.get("task_signature")) != signatures.get(pair[0])
            or not isinstance(row.get("raw_response"), dict)
        ):
            journal_schema_pass = False
        journal_by_pair[pair].append(turn_index)
    rows_by_pair = {pair: row for pair, row in zip(observed_pairs, trajectories, strict=True)}
    contiguous_journal = journal_schema_pass and all(
        sorted(journal_by_pair.get(pair, []))
        == list(range(int(row.get("raw_response_count", 0))))
        for pair, row in rows_by_pair.items()
    )
    journal_count_pass = len(journal_rows) == sum(
        int(row.get("raw_response_count", 0)) for row in trajectories
    )

    expected_identity = job.get("runtime_identity")
    identity = result.get("runtime_identity")
    frozen_required = {
        "family",
        "source_commit",
        "model_revision",
        "gpu_uuid",
        "interpreter_realpath",
        "interpreter_sha256",
        "interpreter_version",
    }
    live_receipt_sha = identity.get("deployment_receipt_sha256") if isinstance(identity, dict) else None
    runtime_identity_pass = (
        isinstance(identity, dict)
        and isinstance(expected_identity, dict)
        and frozen_required.issubset(expected_identity)
        and "deployment_receipt_sha256" not in expected_identity
        and all(identity.get(key) == expected_identity[key] for key in frozen_required)
        and isinstance(live_receipt_sha, str)
        and len(live_receipt_sha) == 64
    )
    result_contract_pass = all(
        (
            result.get("protocol") == JOB_PROTOCOL,
            result.get("job_id") == job_id,
            result.get("family") == job["family"],
            result.get("split") == split,
            result.get("collection_manifest_sha256") == manifest_sha256,
            int(result.get("expected_trajectory_count", -1)) == int(job["trajectory_count"]),
            int(result.get("trajectory_count", -1)) == int(job["trajectory_count"]),
            int(result.get("request_concurrency", -1)) == int(job["concurrency"]),
            result.get("retry_count") == 0,
            result.get("infrastructure_error") is None,
            result.get("complete") is True,
            result.get("outcomes_inspected_by_evidence_chain") is False,
            result.get("trajectory_artifact_sha256") == trajectory_sha256,
            result.get("raw_journal_sha256") == journal_sha256,
            runtime_identity_pass,
        )
    )
    passed = all((result_contract_pass, identity_complete, contiguous_journal, journal_count_pass))
    return {
        "protocol": "RIST-C0-v3.0-SCIENTIFIC-JOB-VALIDATION-v1",
        "job_id": job_id,
        "family": job["family"],
        "split": split,
        "expected_trajectory_count": int(job["trajectory_count"]),
        "trajectory_count": len(trajectories),
        "raw_response_count": len(journal_rows),
        "retry_count": 0,
        "result_contract_pass": result_contract_pass,
        "runtime_identity_pass": runtime_identity_pass,
        "task_seed_identity_complete": identity_complete,
        "duplicate_task_seed_identity": duplicate_pairs,
        "journal_contiguous_pass": contiguous_journal,
        "journal_count_pass": journal_count_pass,
        "trajectory_artifact_sha256": trajectory_sha256,
        "raw_journal_sha256": journal_sha256,
        "outcomes_inspected": False,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    job = _job(manifest, args.job_id)
    source_path = Path(job["task_file"])
    if _sha256(source_path) != job["task_file_sha256"]:
        raise ValueError("v3 source split hash mismatch")
    result = validate_job(
        manifest=manifest,
        manifest_sha256=_sha256(args.manifest),
        source_rows=_read_jsonl(source_path),
        job_id=args.job_id,
        result=json.loads(args.result.read_text(encoding="utf-8")),
        trajectories=_read_jsonl(args.trajectories),
        trajectory_sha256=_sha256(args.trajectories),
        journal_rows=_read_jsonl(args.journal),
        journal_sha256=_sha256(args.journal),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

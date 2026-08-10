"""Finalize one two-family RIST C0 v4 split without interpreting outcomes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

STAGE_ROOT = Path(__file__).resolve().parent
VALIDATOR_PATH = STAGE_ROOT / "validate_scientific_job.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _load_validator():
    spec = importlib.util.spec_from_file_location("rist_c0_v4_job_validator", VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("RIST C0 v4 job validator is unavailable")
    spec.loader.exec_module(module)
    return module


def finalize(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "RIST-C0-v4.0-STAGE-MANIFEST-v1":
        raise ValueError("unexpected RIST C0 v4 scientific manifest")
    split = manifest.get("split")
    if split not in {"calibration", "qualification"}:
        raise ValueError("v4 finalization requires one frozen split")
    jobs = manifest.get("jobs", [])
    expected_job_ids = {f"qwen3-{split}", f"gemma4-{split}"}
    if {str(job.get("job_id")) for job in jobs} != expected_job_ids:
        raise ValueError("v4 finalization requires both families for one split")
    if int(manifest.get("job_count", -1)) != 2:
        raise ValueError("v4 finalization requires two jobs")
    if any(
        manifest.get(key) is not False
        for key in (
            "heldout_permitted",
            "bfcl_permitted",
            "training_permitted",
            "prior_lineage_outcomes_permitted",
            "retry_permitted",
        )
    ):
        raise PermissionError("v4 finalization crosses a forbidden boundary")

    validator = _load_validator()
    manifest_sha256 = _sha256(manifest_path)
    rows = []
    for job in jobs:
        result_root = Path(job["result_root"])
        result_path = result_root / "result.json"
        trajectory_path = result_root / "trajectories.jsonl"
        journal_path = result_root / "raw_journal.jsonl"
        source_path = Path(job["task_file"])
        if _sha256(source_path) != job["task_file_sha256"]:
            raise ValueError("v4 source split hash mismatch")
        validation = validator.validate_job(
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            source_rows=_read_jsonl(source_path),
            job_id=job["job_id"],
            result=json.loads(result_path.read_text(encoding="utf-8")),
            trajectories=_read_jsonl(trajectory_path),
            trajectory_sha256=_sha256(trajectory_path),
            journal_rows=_read_jsonl(journal_path),
            journal_sha256=_sha256(journal_path),
        )
        if validation["passed"] is not True:
            raise ValueError(f"v4 scientific job validation failed: {job['job_id']}")
        rows.append(
            {
                "job_id": job["job_id"],
                "family": job["family"],
                "split": job["split"],
                "trajectory_count": validation["trajectory_count"],
                "raw_response_count": validation["raw_response_count"],
                "retry_count": validation["retry_count"],
                "result_sha256": _sha256(result_path),
                "trajectory_artifact_sha256": validation["trajectory_artifact_sha256"],
                "raw_journal_sha256": validation["raw_journal_sha256"],
                "outcomes_inspected": False,
                "passed": validation["passed"],
            }
        )
    total = sum(int(row["trajectory_count"]) for row in rows)
    passed = (
        total == int(manifest["trajectory_count"])
        and all(row["passed"] for row in rows)
        and all(row["retry_count"] == 0 for row in rows)
    )
    return {
        "protocol": "RIST-C0-v4.0-SCIENTIFIC-COLLECTION-FINAL-v1",
        "collection_manifest_sha256": manifest_sha256,
        "source_commit": manifest["source_commit"],
        "split": split,
        "job_count": len(rows),
        "trajectory_count": total,
        "retry_count": 0,
        "jobs": rows,
        "outcomes_inspected": False,
        "passed": passed,
        "decision": (
            f"PASS_{str(split).upper()}_COLLECTION_TO_SEPARATE_ANALYSIS"
            if passed
            else f"KILL_C0_V4_{str(split).upper()}_POOL"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

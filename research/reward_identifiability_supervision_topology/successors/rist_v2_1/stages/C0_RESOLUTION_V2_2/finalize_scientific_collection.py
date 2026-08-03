"""Finalize four C0 v2.2 jobs as a content-blind 4,096-trajectory receipt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


STAGE_ROOT = Path(__file__).resolve().parent
VALIDATOR_PATH = STAGE_ROOT / "validate_scientific_job.py"
EXPECTED_JOB_IDS = {
    "qwen3-calibration",
    "qwen3-qualification",
    "gemma4-calibration",
    "gemma4-qualification",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _load_validator():
    spec = importlib.util.spec_from_file_location("rist_c0_v2_2_job_validator", VALIDATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("C0 v2.2 job validator is unavailable")
    spec.loader.exec_module(module)
    return module


def finalize(manifest_path: Path) -> dict[str, Any]:
    """Revalidate all raw artifacts; never inspect rewards, actions, or success."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "RIST-C0-v2.2-SCIENTIFIC-COLLECTION-MANIFEST-v1":
        raise ValueError("unexpected C0 v2.2 scientific manifest")
    if manifest.get("trajectory_count") != 4096 or manifest.get("job_count") != 4:
        raise ValueError("C0 v2.2 finalization requires four jobs and 4,096 trajectories")
    jobs = manifest.get("jobs", [])
    if {str(job.get("job_id")) for job in jobs} != EXPECTED_JOB_IDS:
        raise ValueError("C0 v2.2 finalization requires the full family/split factorial")
    gate = manifest.get("canary_gate")
    if not isinstance(gate, dict) or gate.get("passed") is not True:
        raise PermissionError("C0 v2.2 finalization is blocked without canary PASS")

    pool_path = Path(manifest["pool_manifest"])
    if _sha256(pool_path) != manifest["pool_manifest_sha256"]:
        raise ValueError("C0 v2.2 pool manifest hash mismatch")
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    validator = _load_validator()
    manifest_sha256 = _sha256(manifest_path)
    rows = []
    for job in jobs:
        result_path = Path(job["result"])
        trajectory_path = Path(job["trajectories"])
        journal_path = Path(job["raw_journal"])
        split_spec = pool["splits"][job["split"]]
        source_path = pool_path.parent / split_spec["file"]
        if _sha256(source_path) != split_spec["sha256"]:
            raise ValueError("C0 v2.2 source split hash mismatch")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        validation = validator.validate_job(
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            pool_manifest=pool,
            source_rows=_read_jsonl(source_path),
            job_id=job["job_id"],
            result=result,
            trajectories=_read_jsonl(trajectory_path),
            trajectory_sha256=_sha256(trajectory_path),
            journal_rows=_read_jsonl(journal_path),
            journal_sha256=_sha256(journal_path),
        )
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
        total == 4096
        and all(row["passed"] for row in rows)
        and all(row["retry_count"] == 0 for row in rows)
    )
    return {
        "protocol": "RIST-C0-v2.2-SCIENTIFIC-COLLECTION-FINAL-v1",
        "collection_manifest_sha256": manifest_sha256,
        "collection_source_commit": manifest["collection_source_commit"],
        "gpu_uuid": gate["gpu_identity"]["gpu_uuid"],
        "models": manifest["models"],
        "model_revisions": manifest["model_revisions"],
        "tokenizer_snapshot_sha256": manifest["tokenizer_snapshot_sha256"],
        "job_count": len(rows),
        "trajectory_count": total,
        "retry_count": 0,
        "jobs": rows,
        "outcomes_inspected": False,
        "passed": passed,
        "decision": "PASS_COLLECTION_TO_SEPARATE_RESOLUTION_ANALYSIS" if passed else "KILL_C0_V2_2_POOL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

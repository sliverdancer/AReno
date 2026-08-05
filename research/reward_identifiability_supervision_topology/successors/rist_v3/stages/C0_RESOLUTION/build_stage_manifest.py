"""Build one split-scoped v3 C0 manifest without crossing its access boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROTOCOL = "RIST-C0-v3.0-STAGE-MANIFEST-v1"
ADMISSION_PROTOCOL = "RIST-C0-v3.0-CALIBRATION-ADMISSION-v1"
FAMILIES = ("qwen3", "gemma4")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_admission(path: Path, pool_sha256: str) -> dict[str, Any]:
    value = _load_json(path)
    expected_keys = {
        "protocol", "decision", "passed", "pool_manifest_sha256",
        "calibration_result_sha256", "frozen_whole_cell_map",
        "qualification_accessed",
    }
    if set(value) != expected_keys:
        raise PermissionError("calibration admission schema mismatch")
    if (
        value["protocol"] != ADMISSION_PROTOCOL
        or value["decision"] != "PASS_CALIBRATION_TO_QUALIFICATION"
        or value["passed"] is not True
        or value["pool_manifest_sha256"] != pool_sha256
        or value["qualification_accessed"] is not False
    ):
        raise PermissionError("qualification remains sealed")
    mapping = value["frozen_whole_cell_map"]
    if not isinstance(mapping, dict):
        raise PermissionError("frozen whole-cell map is required")
    counts = {
        band: sum(label == band for label in mapping.values())
        for band in ("low", "high")
    }
    if counts["low"] < 2 or counts["high"] < 2:
        raise PermissionError("frozen map lacks two low and two high cells")
    result_sha = value["calibration_result_sha256"]
    if not isinstance(result_sha, str) or len(result_sha) != 64:
        raise PermissionError("calibration result SHA-256 is required")
    return value


def build_manifest(
    *,
    split: str,
    pool_manifest_path: Path,
    output_root: Path,
    source_commit: str,
    calibration_admission_path: Path | None = None,
) -> dict[str, Any]:
    if split not in {"calibration", "qualification"}:
        raise ValueError("split must be calibration or qualification")
    if len(source_commit) != 40 or any(ch not in "0123456789abcdef" for ch in source_commit):
        raise ValueError("exact lowercase source commit is required")
    pool_path = pool_manifest_path.resolve(strict=True)
    pool = _load_json(pool_path)
    if pool.get("protocol") != "RIST-C0-v3.0-FRESH-POOL-v1":
        raise ValueError("wrong v3 pool protocol")
    pool_sha = _sha256(pool_path)
    admission = None
    if split == "qualification":
        if calibration_admission_path is None:
            raise PermissionError("qualification remains sealed")
        admission = _validate_admission(calibration_admission_path, pool_sha)
    elif calibration_admission_path is not None:
        raise PermissionError("calibration cannot consume a qualification admission")

    split_row = pool["splits"][split]
    task_path = pool_path.parent / split_row["file"]
    if _sha256(task_path) != split_row["sha256"]:
        raise ValueError(f"{split} task bytes do not match the pool manifest")
    if split_row["task_count"] != 32 or len(split_row["rollout_seeds"]) != 32:
        raise ValueError(f"{split} shape mismatch")
    jobs = [
        {
            "job_id": f"{family}-{split}",
            "family": family,
            "split": split,
            "task_file": str(task_path),
            "task_file_sha256": split_row["sha256"],
            "rollout_seeds": split_row["rollout_seeds"],
            "trajectory_count": 1024,
            "concurrency": 8,
            "max_retries": 0,
            "result_root": str(output_root / split / family),
        }
        for family in FAMILIES
    ]
    return {
        "protocol": PROTOCOL,
        "split": split,
        "source_commit": source_commit,
        "pool_manifest": str(pool_path),
        "pool_manifest_sha256": pool_sha,
        "job_count": 2,
        "trajectory_count": 2048,
        "jobs": jobs,
        "calibration_permitted": split == "calibration",
        "qualification_permitted": split == "qualification",
        "qualification_admission_sha256": (
            _sha256(calibration_admission_path) if admission is not None else None
        ),
        "heldout_permitted": False,
        "bfcl_permitted": False,
        "training_permitted": False,
        "retry_permitted": False,
        "prior_lineage_outcomes_permitted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("calibration", "qualification"), required=True)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--calibration-admission", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_manifest(
        split=args.split,
        pool_manifest_path=args.pool_manifest,
        output_root=args.output_root,
        source_commit=args.source_commit,
        calibration_admission_path=args.calibration_admission,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

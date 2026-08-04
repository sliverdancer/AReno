"""Verify the commit-backed RIST C0 v2.3 capacity-canary CPU freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[6]
STAGE_ROOT = Path(__file__).resolve().parent
EXPECTED_FILES = {
    "run_capacity_canary.py",
    "verify_capacity_canary_cpu_freeze.py",
    "data/capacity_canary.jsonl",
    "data/manifest.json",
    "gpu_bind_20260804_a800/POST_RENT_BINDING.json",
    "gpu_bind_20260804_a800/qwen3_receipt.json",
    "gpu_bind_20260804_a800/gemma4_receipt.json",
    "../../../../../../tests/test_rist_c0_v2_3_capacity_canary_cpu.py",
}
FAMILIES = ["qwen3", "gemma4"]
SEEDS = list(range(18001, 18009))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _valid_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def verify(freeze_path: Path) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    source_commit = freeze.get("source_commit")
    commit_valid = _valid_commit(source_commit)
    recorded = freeze.get("files", {})
    current_files: dict[str, bool] = {}
    committed_files: dict[str, bool] = {}
    for relative, expected_sha in recorded.items():
        path = (STAGE_ROOT / relative).resolve()
        current_files[relative] = path.is_file() and _sha256(path.read_bytes()) == expected_sha
        if not commit_valid:
            committed_files[relative] = False
            continue
        repository_relative = path.relative_to(REPO_ROOT).as_posix()
        result = subprocess.run(
            ["git", "show", f"{source_commit}:{repository_relative}"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        committed_files[relative] = (
            result.returncode == 0 and _sha256(result.stdout) == expected_sha
        )

    boundary_pass = (
        freeze.get("protocol") == "RIST-C0-v2.3-CAPACITY-CANARY-CPU-FREEZE-v1"
        and commit_valid
        and set(recorded) == EXPECTED_FILES
        and freeze.get("families") == FAMILIES
        and freeze.get("request_seeds") == SEEDS
        and freeze.get("requests_per_family") == 8
        and freeze.get("request_concurrency") == 8
        and freeze.get("retry_count") == 0
        and freeze.get("minimum_gpu_memory_gib") == 48
        and freeze.get("outcomes_inspected") is False
        and freeze.get("scientific_result") is False
        and freeze.get("gpu_execution_permitted_by_freeze") is False
        and freeze.get("training_permitted") is False
        and freeze.get("heldout_permitted") is False
        and freeze.get("bfcl_permitted") is False
        and freeze.get("qwen_previous_startup_valid_as_capacity_evidence") is False
    )
    passed = (
        boundary_pass
        and all(current_files.values())
        and all(committed_files.values())
    )
    return {
        "protocol": "RIST-C0-v2.3-CAPACITY-CANARY-CPU-FREEZE-VERIFY-v1",
        "boundary_pass": boundary_pass,
        "current_files": current_files,
        "committed_files": committed_files,
        "file_count": len(recorded),
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.freeze)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

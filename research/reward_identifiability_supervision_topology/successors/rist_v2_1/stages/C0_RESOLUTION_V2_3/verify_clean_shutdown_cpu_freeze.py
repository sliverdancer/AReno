"""Verify the commit-backed RIST C0 v2.3 clean-shutdown CPU freeze."""

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
    "CLEAN_SHUTDOWN_PROTOCOL.md",
    "deployment_entrypoint.py",
    "graceful_deployment_entrypoint.py",
    "verify_clean_shutdown_cpu_freeze.py",
    "gpu_bind_20260804_a800/capacity_20260804_v1/CAPACITY_CANARY_RESULT.json",
    "../../../../../../tests/test_rist_c0_v2_3_clean_shutdown_cpu.py",
    "../../../../../../tests/test_rist_c0_v2_3_deployment_cpu.py",
}
ORIGINAL_ENTRYPOINT_SHA256 = (
    "e6c3717da4fd5ca3562fd1d27914996dd0ddfdc22e2a4873e5f21dbdc1fdeb7a"
)
CAPACITY_RESULT_SHA256 = (
    "cb32b2f6773ba61a08e7b1d926086980429c9f84a7bc890f52a2fd60d79c1e27"
)


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

    original_preserved = (
        recorded.get("deployment_entrypoint.py") == ORIGINAL_ENTRYPOINT_SHA256
        and recorded.get(
            "gpu_bind_20260804_a800/capacity_20260804_v1/CAPACITY_CANARY_RESULT.json"
        )
        == CAPACITY_RESULT_SHA256
    )
    boundary_pass = (
        freeze.get("protocol") == "RIST-C0-v2.3-CLEAN-SHUTDOWN-CPU-FREEZE-v1"
        and commit_valid
        and set(recorded) == EXPECTED_FILES
        and original_preserved
        and freeze.get("cpu_test_result")
        == "29 passed across clean-shutdown, deployment, capacity, and pool tests"
        and freeze.get("signal_targets") == ["direct_server_child"]
        and freeze.get("process_group_signaling_permitted") is False
        and freeze.get("capacity_result_reinterpreted") is False
        and freeze.get("gpu_validated") is False
        and freeze.get("gpu_execution_permitted_by_freeze") is False
        and freeze.get("model_requests_permitted") is False
        and freeze.get("calibration_permitted") is False
        and freeze.get("qualification_permitted") is False
        and freeze.get("heldout_permitted") is False
        and freeze.get("bfcl_permitted") is False
        and freeze.get("training_permitted") is False
    )
    passed = boundary_pass and all(current_files.values()) and all(committed_files.values())
    return {
        "protocol": "RIST-C0-v2.3-CLEAN-SHUTDOWN-CPU-FREEZE-VERIFY-v1",
        "boundary_pass": boundary_pass,
        "original_artifacts_preserved": original_preserved,
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

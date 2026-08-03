"""Verify the commit-backed E1 v2.2 deployment-manifest CPU freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[6]
EXPECTED_FILES = {
    "build_v2_2_deployment_manifest.py",
    "validate_c0_admission.py",
    "C0_GATE_PROTOCOL.md",
    "build_capacity_manifest.py",
    "monitor_gpu_command.py",
    "run_serving_canary.py",
    "extract_one_step_metrics.py",
    "build_checkpoint_manifest.py",
    "build_directory_manifest.py",
    "assemble_capacity_evidence.py",
    "validate_capacity_evidence.py",
    "../../GPU_AUTHORIZATION_20260803.json",
    "../../GPU_EXECUTION_FREEZE.json",
    "../../verify_gpu_execution_freeze.py",
    "../C0_RESOLUTION_V2_2/RESOLUTION_CPU_FREEZE.json",
    "../C0_RESOLUTION_V2_2/RESOLUTION_EXECUTION_ROOT.json",
    "../../../../../../tests/test_rist_e1_v2_2_deployment_cpu.py",
    "verify_v2_2_deployment_cpu_freeze.py",
}


def verify(freeze_path: Path) -> dict:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    stage_root = freeze_path.parent
    source_commit = freeze.get("source_commit")
    commit_valid = (
        isinstance(source_commit, str)
        and len(source_commit) == 40
        and all(character in "0123456789abcdef" for character in source_commit)
    )
    recorded = freeze.get("files", {})
    files = {}
    committed_files = {}
    for relative, expected in recorded.items():
        path = (stage_root / relative).resolve()
        files[relative] = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
        if not commit_valid:
            committed_files[relative] = False
            continue
        repository_relative = path.relative_to(REPO_ROOT).as_posix()
        result = subprocess.run(
            ["git", "show", f"{source_commit}:{repository_relative}"],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        committed_files[relative] = (
            result.returncode == 0 and hashlib.sha256(result.stdout).hexdigest() == expected
        )
    boundary = (
        freeze.get("protocol") == "RIST-E1-v2.2-DEPLOYMENT-CPU-FREEZE-v1"
        and commit_valid
        and set(recorded) == EXPECTED_FILES
        and freeze.get("families") == ["qwen3", "gemma4"]
        and freeze.get("algorithms") == ["gspo", "grpo"]
        and freeze.get("optimizer_steps_per_algorithm") == 1
        and freeze.get("c0_validation_required") is True
        and freeze.get("actual_gpu_identity_required") is True
        and freeze.get("gpu_execution_permitted_by_freeze") is False
        and freeze.get("heldout_permitted") is False
        and freeze.get("bfcl_permitted") is False
    )
    return {
        "protocol": "RIST-E1-v2.2-DEPLOYMENT-CPU-FREEZE-VERIFY-v1",
        "file_count": len(files),
        "files": files,
        "committed_files": committed_files,
        "boundary_pass": boundary,
        "passed": boundary and all(files.values()) and all(committed_files.values()),
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

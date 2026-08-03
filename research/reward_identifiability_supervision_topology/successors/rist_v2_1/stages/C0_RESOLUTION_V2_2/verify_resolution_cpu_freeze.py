"""Verify the CPU-only C0 v2.2 resolution-analysis source freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


EXPECTED_FILES = {
    "RESOLUTION_ANALYSIS_PROTOCOL.md",
    "replay_resolution_outcomes.py",
    "resolution_analysis.py",
    "run_resolution_analysis.py",
    "validate_resolution_analysis.py",
    "verify_resolution_cpu_freeze.py",
    "../../../../../../tests/test_rist_c0_v2_2_resolution_cpu.py",
    "../E1/validate_c0_admission.py",
    "../E1/C0_GATE_PROTOCOL.md",
    "../D4_EVAL/evaluate_checkpoint.py",
    "finalize_scientific_collection.py",
    "validate_scientific_job.py",
}
REPO_ROOT = Path(__file__).resolve().parents[6]


def verify(freeze_path: Path) -> dict:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    stage_root = freeze_path.parent
    recorded = freeze.get("files", {})
    source_commit = freeze.get("source_commit")
    commit_valid = (
        isinstance(source_commit, str)
        and len(source_commit) == 40
        and all(character in "0123456789abcdef" for character in source_commit)
    )
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
    boundary_pass = (
        freeze.get("schema_version") == "rist-c0-v2.2-resolution-cpu-freeze-v1"
        and commit_valid
        and set(recorded) == EXPECTED_FILES
        and freeze.get("input_trajectory_count") == 4096
        and freeze.get("model_families") == ["qwen3", "gemma4"]
        and freeze.get("training_permitted") is False
        and freeze.get("gpu_permitted") is False
        and freeze.get("heldout_permitted") is False
        and freeze.get("bfcl_permitted") is False
        and freeze.get("terminal_rerun_permitted") is False
    )
    return {
        "protocol": "RIST-C0-v2.2-RESOLUTION-CPU-FREEZE-VERIFY-v1",
        "file_count": len(files),
        "files": files,
        "committed_files": committed_files,
        "boundary_pass": boundary_pass,
        "passed": boundary_pass and all(files.values()) and all(committed_files.values()),
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

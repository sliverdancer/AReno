"""Verify the local, outcome-blind M0 third-checkpoint freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[6]


def verify(path: Path) -> dict:
    freeze = json.loads(path.read_text())
    file_checks = {
        relative: (REPO_ROOT / relative).is_file()
        and hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest() == digest
        for relative, digest in freeze.get("audited_files", {}).items()
    }
    authorization = freeze.get("authorization", {})
    boundary_pass = all(
        (
            freeze.get("schema_version") == "rist-m0-third-checkpoint-selection-v1",
            freeze.get("main_conference_requirement")
            == "AT_LEAST_3_CHECKPOINTS_ACROSS_2_FAMILIES",
            freeze.get("primary_candidate", {}).get("checkpoint") == "Qwen/Qwen3-1.7B",
            freeze.get("primary_candidate", {}).get("exact_revision") is None,
            freeze.get("selection_uses_model_outcomes") is False,
            freeze.get("selection_uses_task_outcomes") is False,
            freeze.get("new_dependency_added") is False,
            freeze.get("counts_as_third_checkpoint") is False,
            set(freeze.get("qualification_gates", []))
            == {
                "M1_TOKENIZER_CONFIG_LICENSE",
                "M2_WEIGHT_IDENTITY",
                "M3_RUNTIME_TREATMENT",
                "M4_C0_RESOLUTION_TRANSPORT",
                "M5_GSPO_GRPO_CAPACITY",
                "M6_POWERED_STEP_AND_TOKEN_MATCHED_REPLICATION",
            },
            all(value is False for value in authorization.values()),
        )
    )
    return {
        "protocol": "RIST-M0-THIRD-CHECKPOINT-VERIFY-v1",
        "audited_file_count": len(file_checks),
        "audited_files": file_checks,
        "outcome_blind_boundary_pass": boundary_pass,
        "passed": boundary_pass and len(file_checks) == 4 and all(file_checks.values()),
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

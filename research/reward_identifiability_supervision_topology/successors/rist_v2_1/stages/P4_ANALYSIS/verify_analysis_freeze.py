"""Verify the frozen P4/P5 analysis and evidence pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def verify(freeze_path: Path) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text())
    stage_root = freeze_path.parent.parent
    v2_root = stage_root.parent
    repo_root = v2_root.parents[3]
    files = {}
    for label, expected in freeze["files"].items():
        if label.startswith("repo:"):
            path = repo_root / label.removeprefix("repo:")
        else:
            path = stage_root / label
        files[label] = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected
    boundary_pass = (
        freeze.get("schema_version") == "rist-p4-analysis-freeze-v5"
        and freeze.get("results_opened") is False
        and freeze.get("gpu_used") is False
        and freeze.get("model_accessed") is False
        and freeze.get("parent_heldout_retired") is True
        and freeze.get("exact_run_evidence_artifact_count") == 22
        and freeze.get("reward_journal_key") == ["training_step", "sample_index"]
        and freeze.get("minimum_total_exposure_support_fraction_per_arm") == 0.5
        and freeze.get("auc_integration_grid")
        == "union_of_all_observed_token_knots"
        and freeze.get("independent_token_budget_schedule_executed") is False
    )
    return {
        "protocol": "RIST-P4-ANALYSIS-FREEZE-VERIFY-v5",
        "file_count": len(files),
        "files": files,
        "outcome_blind_boundary_pass": boundary_pass,
        "passed": boundary_pass and all(files.values()),
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

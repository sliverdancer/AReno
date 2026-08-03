"""Verify the X4 CPU freeze and its non-execution boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(freeze_path: Path) -> dict:
    stage = freeze_path.parent
    x3 = stage.parent / "X3_TAU3"
    freeze = json.loads(freeze_path.read_text())
    paths = {
        "protocol_sha256": stage / "PROTOCOL.md",
        "seed_bank_sha256": stage / "SEED_BANK.json",
        "analysis_sha256": stage / "analyze_stability.py",
        "manifest_builder_sha256": stage / "build_execution_manifest.py",
        "seed_freezer_sha256": stage / "freeze_seed_bank.py",
        "evidence_validator_sha256": stage / "validate_evidence.py",
        "execution_manifest_sha256": stage / "EXECUTION_MANIFEST.json",
        "x3_cpu_freeze_sha256": x3 / "CPU_FREEZE.json",
        "x3_validator_sha256": x3 / "validate_pilot_evidence.py",
    }
    files = {
        field: path.is_file()
        and hashlib.sha256(path.read_bytes()).hexdigest() == freeze.get(field)
        for field, path in paths.items()
    }
    manifest = json.loads((stage / "EXECUTION_MANIFEST.json").read_text())
    bank = json.loads((stage / "SEED_BANK.json").read_text())
    boundary = (
        manifest.get("source_commit", "").startswith(freeze.get("design_commit", "!"))
        and manifest.get("execution_authorized") is False
        and manifest.get("x3_pilot_prerequisite", {}).get("validation_sha256") is None
        and manifest.get("user_simulator_binding") is None
        and all(not all(binding.values()) for binding in manifest["family_bindings"].values())
        and manifest.get("heldout_permitted") is False
        and manifest.get("bfcl_permitted") is False
        and bank.get("outcomes_read") is False
        and freeze.get("model_accessed") is False
        and freeze.get("gpu_used") is False
    )
    counts = (
        len(bank.get("seeds", [])) == 32
        and len(bank.get("planning_seeds", [])) == 12
        and manifest.get("job_count") == 512
        and manifest.get("planning_job_count") == 192
        and manifest.get("maximum_episode_count") == 102400
    )
    return {
        "protocol": "RIST-X4-TAU3-CPU-FREEZE-VERIFY-v1",
        "passed": all(files.values()) and boundary and counts,
        "files": files,
        "authorization_boundary_pass": boundary,
        "count_contract_pass": counts,
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

"""Verify the outcome-blind powered-expansion freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(freeze_path: Path) -> dict:
    freeze = json.loads(freeze_path.read_text())
    stage_root = freeze_path.parent.parent
    files = {
        relative: (stage_root / relative).is_file()
        and hashlib.sha256((stage_root / relative).read_bytes()).hexdigest() == expected
        for relative, expected in freeze["files"].items()
    }
    seed_bank = [7101 + 101 * index for index in range(32)]
    seed_hash = hashlib.sha256(
        json.dumps(seed_bank, separators=(",", ":")).encode()
    ).hexdigest()
    boundary_pass = (
        freeze.get("schema_version") == "rist-p3-powered-expansion-freeze-v1"
        and freeze.get("pilot_results_opened") is False
        and freeze.get("execution_authorized") is False
        and freeze.get("minimum_paired_seeds") == 8
        and freeze.get("maximum_paired_seeds") == 32
        and freeze.get("minimum_absolute_interaction") == 0.1
        and freeze.get("seed_bank_sha256") == seed_hash
    )
    return {
        "protocol": "RIST-P3-POWERED-EXPANSION-FREEZE-VERIFY-v1",
        "file_count": len(files),
        "files": files,
        "seed_bank_pass": freeze.get("seed_bank_sha256") == seed_hash,
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

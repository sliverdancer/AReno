"""Verify the outcome-free C0 v2.2 capacity-canary freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(freeze_path: Path) -> dict:
    freeze = json.loads(freeze_path.read_text())
    stage_root = freeze_path.parent
    files = {
        relative: (stage_root / relative).is_file()
        and hashlib.sha256((stage_root / relative).read_bytes()).hexdigest() == expected
        for relative, expected in freeze["files"].items()
    }
    boundary_pass = (
        freeze.get("schema_version")
        == "rist-c0-v2.2-capacity-canary-freeze-v1"
        and freeze.get("retired_predecessor") == "RIST-C0-v2.1"
        and freeze.get("minimum_gpu_memory_gib") == 48.0
        and freeze.get("recommended_gpu_memory_gib") == 80.0
        and freeze.get("request_concurrency") == 8
        and freeze.get("expected_responses_per_family") == 8
        and freeze.get("model_families") == ["qwen3", "gemma4"]
        and freeze.get("execution_authorized") is False
        and freeze.get("scientific_task_access_permitted") is False
        and freeze.get("outcomes_inspected") is False
        and freeze.get("training_permitted") is False
        and freeze.get("heldout_permitted") is False
    )
    return {
        "protocol": "RIST-C0-v2.2-CAPACITY-CANARY-FREEZE-VERIFY-v1",
        "file_count": len(files),
        "files": files,
        "outcome_free_boundary_pass": boundary_pass,
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

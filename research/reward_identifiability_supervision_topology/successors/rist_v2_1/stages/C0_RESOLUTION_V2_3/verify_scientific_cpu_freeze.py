"""Verify the frozen C0 v2.3 split-collection and resolution artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FREEZE = ROOT / "SCIENTIFIC_CPU_FREEZE.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict:
    value = json.loads(FREEZE.read_text(encoding="utf-8"))
    checks = []
    checks.append(value.get("protocol") == "RIST-C0-v2.3-SCIENTIFIC-CPU-FREEZE-v1")
    checks.append(value.get("status") == "PASS_CPU_FREEZE_GPU_UNEXECUTED")
    checks.append(value.get("calibration_trajectory_count") == 2048)
    checks.append(value.get("qualification_trajectory_count") == 2048)
    checks.append(value.get("request_concurrency") == 8)
    checks.append(value.get("retry_count") == 0)
    checks.append(value.get("qualification_requires_calibration_go") is True)
    checks.append(value.get("heldout_permitted") is False)
    checks.append(value.get("bfcl_permitted") is False)
    checks.append(value.get("training_permitted") is False)
    files = value.get("files", {})
    checks.append(isinstance(files, dict) and len(files) == 12)
    for relative, expected in files.items():
        path = ROOT / relative
        checks.append(path.is_file() and _sha256(path) == expected)
    return {
        "protocol": "RIST-C0-v2.3-SCIENTIFIC-CPU-FREEZE-VERIFICATION-v1",
        "check_count": len(checks),
        "passed_check_count": sum(checks),
        "passed": all(checks),
        "gpu_used": False,
        "model_accessed": False,
        "scientific_outcomes_accessed": False,
        "qualification_accessed": False,
        "heldout_accessed": False,
        "bfcl_accessed": False,
        "training_performed": False,
    }


if __name__ == "__main__":
    result = verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 2)

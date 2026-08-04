"""Verify the C0 v2.3 to E1 admission CPU freeze."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> dict:
    value = json.loads((ROOT / "CPU_FREEZE.json").read_text(encoding="utf-8"))
    checks = [
        value.get("protocol") == "RIST-E1-v2.3-ADMISSION-CPU-FREEZE-v1",
        value.get("status") == "PASS_CPU_ADMISSION_FREEZE_C0_OUTCOME_BLOCKED",
        value.get("required_c0_decision") == "PASS_C0_V2_3_TO_E1_CAPACITY",
        value.get("capacity_task_count") == 4,
        value.get("selection_rule") == "LEXICOGRAPHIC_FIRST_TRANSPORTED_HIGH_CELL",
        value.get("execution_authorized") is False,
        value.get("gpu_used") is False,
        value.get("model_accessed") is False,
        value.get("scientific_outcomes_accessed") is False,
        value.get("training_performed") is False,
        value.get("heldout_accessed") is False,
        value.get("bfcl_accessed") is False,
        set(value.get("files", {})) == {"PROTOCOL.md", "build_admission.py"},
    ]
    for relative, expected in value.get("files", {}).items():
        path = ROOT / relative
        checks.append(path.is_file() and _sha256(path) == expected)
    return {
        "protocol": "RIST-E1-v2.3-ADMISSION-CPU-FREEZE-VERIFICATION-v1",
        "check_count": len(checks),
        "passed_check_count": sum(checks),
        "passed": all(checks),
        "gpu_used": False,
        "model_accessed": False,
        "scientific_outcomes_accessed": False,
        "training_performed": False,
    }


if __name__ == "__main__":
    result = verify()
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["passed"] else 2)

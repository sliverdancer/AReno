"""Fail-closed E1 gate for a sealed C0 v2.2 resolution directory."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


C0_VALIDATOR = Path(__file__).resolve().parents[1] / "C0_RESOLUTION_V2_2/validate_resolution_analysis.py"
C0_FREEZE = C0_VALIDATOR.parent / "RESOLUTION_CPU_FREEZE.json"
C0_FREEZE_VERIFIER = C0_VALIDATOR.parent / "verify_resolution_cpu_freeze.py"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_validator():
    spec = importlib.util.spec_from_file_location("rist_e1_c0_resolution_validator", C0_VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("C0 v2.2 resolution validator is unavailable")
    spec.loader.exec_module(module)
    return module


def _verify_source_freeze() -> None:
    spec = importlib.util.spec_from_file_location("rist_e1_c0_freeze_verifier", C0_FREEZE_VERIFIER)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("C0 v2.2 freeze verifier is unavailable")
    spec.loader.exec_module(module)
    if module.verify(C0_FREEZE).get("passed") is not True:
        raise PermissionError("E1 gate requires the commit-backed C0 analysis freeze")


def validate_c0_admission(resolution_root: Path) -> dict[str, Any]:
    _verify_source_freeze()
    saved_path = resolution_root / "VALIDATION_RESULT.json"
    admission_path = resolution_root / "e1/E1_ADMISSION.json"
    final_path = resolution_root / "FINAL_RESULT.json"
    data_path = resolution_root / "e1/data/train.jsonl"
    saved = _json(saved_path)
    reproduced = _load_validator().validate(resolution_root, write_receipt=False)
    if saved != reproduced or not (
        saved.get("protocol") == "RIST-C0-v2.2-RESOLUTION-VALIDATION-v1"
        and saved.get("valid") is True
        and saved.get("passed") is True
        and saved.get("decision") == "PASS_C0_V2_2_TO_E1_CAPACITY"
        and saved.get("e1_training_permitted") is True
        and saved.get("resolution_final_sha256") == _sha(final_path)
        and saved.get("e1_admission_sha256") == _sha(admission_path)
        and saved.get("capacity_train_sha256") == _sha(data_path)
    ):
        raise PermissionError("E1 is closed without an exact, independently reproduced C0 validation receipt")
    admission = _json(admission_path)
    return {
        "protocol": "RIST-E1-C0-ADMISSION-GATE-v1",
        "passed": True,
        "decision": "PASS_C0_GATE_TO_DEPLOYMENT_BOUND_E1",
        "resolution_validation_sha256": _sha(saved_path),
        "resolution_final_sha256": _sha(final_path),
        "e1_admission_sha256": _sha(admission_path),
        "capacity_train_sha256": _sha(data_path),
        "capacity_train_path": str(data_path),
        "models": admission["models"],
        "model_revisions": admission["model_revisions"],
        "tokenizer_snapshot_sha256": admission["tokenizer_snapshot_sha256"],
        "collection_gpu_uuid": admission["gpu_uuid"],
        "training_execution_authorized": False,
        "deployment_binding_required": True,
        "heldout_permitted": False,
        "bfcl_permitted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_c0_admission(args.resolution_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

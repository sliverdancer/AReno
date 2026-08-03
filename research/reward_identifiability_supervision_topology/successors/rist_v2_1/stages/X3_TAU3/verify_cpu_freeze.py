"""Verify the X3 Tau3 CPU adapter freeze and its authorization boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def verify(freeze_path: Path) -> dict[str, Any]:
    stage = freeze_path.parent
    repo = stage.parents[5]
    freeze = json.loads(freeze_path.read_text())
    paths = {
        "scope_validation_sha256": stage / "SCOPE_VALIDATION.json",
        "dataset_loader_sha256": repo / "examples/agentic/rist_v2_1_tau3/dataset_loader.py",
        "reward_sha256": repo / "examples/agentic/rist_v2_1_tau3/reward.py",
        "run_agent_sha256": repo / "examples/agentic/rist_v2_1_tau3/run_agent.py",
        "rollout_session_sha256": repo / "areno/api/agentic.py",
        "protocol_sha256": stage / "PROTOCOL.md",
        "execution_manifest_sha256": stage / "EXECUTION_MANIFEST.json",
        "result_validator_sha256": stage / "validate_pilot_evidence.py",
        "dataset_manifest_sha256": stage / "data/manifest.json",
    }
    files = {
        field: path.is_file()
        and hashlib.sha256(path.read_bytes()).hexdigest() == freeze.get(field)
        for field, path in paths.items()
    }
    manifest = json.loads((stage / "EXECUTION_MANIFEST.json").read_text())
    boundary_pass = (
        manifest.get("source_commit", "").startswith(freeze.get("adapter_commit", "!"))
        and manifest.get("execution_authorized") is False
        and manifest.get("user_simulator_authorized") is False
        and manifest.get("heldout_permitted") is False
        and manifest.get("bfcl_permitted") is False
        and freeze.get("model_accessed") is False
        and freeze.get("inference_run") is False
        and freeze.get("training_run") is False
        and freeze.get("gpu_used") is False
    )
    passed = freeze.get("protocol") == "RIST-X3-TAU3-CPU-FREEZE-v2" and all(
        files.values()
    ) and boundary_pass
    return {
        "protocol": "RIST-X3-TAU3-CPU-FREEZE-VERIFY-v2",
        "file_count": len(files),
        "files": files,
        "authorization_boundary_pass": boundary_pass,
        "passed": passed,
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

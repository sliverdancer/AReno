"""Verify the commit-backed, pre-rent RIST C0 v2.3 CPU freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[6]
STAGE_ROOT = Path(__file__).resolve().parent
RUNTIME_COMMIT = "f00b688dff3e4ad35229d059536c975d4c56594f"
MODEL_REVISIONS = [
    "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    "c1899de289a04d12100db370d81485cdf75e47ca",
]
EXPECTED_FILES = {
    "PROTOCOL.md",
    "DEPLOYMENT_PROTOCOL.md",
    "build_fresh_pool.py",
    "deployment_entrypoint.py",
    "verify_cpu_freeze.py",
    "data/capacity_canary.jsonl",
    "data/calibration.jsonl",
    "data/manifest.json",
    "data/qualification.jsonl",
    "../../../../../../tests/test_rist_c0_v2_3_deployment_cpu.py",
    "../../../../../../tests/test_rist_c0_v2_3_pool_cpu.py",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _valid_commit(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def verify(freeze_path: Path) -> dict[str, Any]:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    source_commit = freeze.get("source_commit")
    commit_valid = _valid_commit(source_commit)
    recorded = freeze.get("files", {})
    current_files: dict[str, bool] = {}
    committed_files: dict[str, bool] = {}
    for relative, expected_sha in recorded.items():
        path = (STAGE_ROOT / relative).resolve()
        current_files[relative] = (
            path.is_file() and _sha256(path.read_bytes()) == expected_sha
        )
        if not commit_valid:
            committed_files[relative] = False
            continue
        repository_relative = path.relative_to(REPO_ROOT).as_posix()
        result = subprocess.run(
            ["git", "show", f"{source_commit}:{repository_relative}"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        )
        committed_files[relative] = (
            result.returncode == 0 and _sha256(result.stdout) == expected_sha
        )

    authority_path = STAGE_ROOT / "PRE_RENT_AUTHORITY.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority_sha = _sha256(_canonical(authority))
    authority_pass = (
        authority.get("protocol") == "RIST-C0-v2.3-PRE-RENT-AUTHORITY-v1"
        and authority.get("control_commit") == source_commit
        and authority.get("runtime_commit") == RUNTIME_COMMIT
        and authority.get("manifest_sha256")
        == _sha256((STAGE_ROOT / "data/manifest.json").read_bytes())
        and authority.get("model_revisions") == MODEL_REVISIONS
        and freeze.get("authority_canonical_sha256") == authority_sha
    )
    boundary_pass = (
        freeze.get("protocol") == "RIST-C0-v2.3-CPU-FREEZE-v1"
        and commit_valid
        and set(recorded) == EXPECTED_FILES
        and freeze.get("fresh_pool_protocol") == "RIST-C0-v2.3-FRESH-POOL"
        and freeze.get("runtime_manifest_lookup_permitted") is False
        and freeze.get("external_receipt_sha_required") is True
        and freeze.get("receipt_binding_status") == "NOT_BOUND_PRE_RENT"
        and freeze.get("gpu_execution_permitted") is False
        and freeze.get("model_access_permitted") is False
        and freeze.get("training_permitted") is False
        and freeze.get("heldout_permitted") is False
        and freeze.get("bfcl_permitted") is False
    )
    passed = (
        boundary_pass
        and authority_pass
        and all(current_files.values())
        and all(committed_files.values())
    )
    return {
        "protocol": "RIST-C0-v2.3-CPU-FREEZE-VERIFY-v1",
        "authority_pass": authority_pass,
        "boundary_pass": boundary_pass,
        "committed_files": committed_files,
        "current_files": current_files,
        "file_count": len(recorded),
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

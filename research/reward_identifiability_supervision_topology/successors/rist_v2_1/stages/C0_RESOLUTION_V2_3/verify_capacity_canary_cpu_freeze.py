"""Verify the commit-backed RIST C0 v2.3 capacity-canary CPU freeze."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[6]
STAGE_ROOT = Path(__file__).resolve().parent
EXPECTED_FILES = {
    "run_capacity_canary.py",
    "verify_capacity_canary_cpu_freeze.py",
    "data/capacity_canary.jsonl",
    "data/manifest.json",
    "gpu_bind_20260804_a800/POST_RENT_BINDING.json",
    "gpu_bind_20260804_a800/qwen3_receipt.json",
    "gpu_bind_20260804_a800/gemma4_receipt.json",
    "../../../../../../tests/test_rist_c0_v2_3_capacity_canary_cpu.py",
}
FAMILIES = ["qwen3", "gemma4"]
SEEDS = list(range(18001, 18009))
BINDING_COMMIT = "2a31e76b546dfbf9b5c801bc0efad3066254ec9f"
GPU_UUID = "GPU-10daada5-2347-5f45-92ae-cb5c3ddb646c"
RECEIPT_SHA256 = {
    "qwen3": "055269d1ab2ef7d5b02a416c27afaab6b039470a89889286cf8d91b820de5c90",
    "gemma4": "da49af33c36f26ff0c7030c7e929ef1ec0d358dc5dc51d3fbad41bafe21e83a6",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _binding_semantics(freeze: dict[str, Any]) -> bool:
    try:
        binding = json.loads(
            (STAGE_ROOT / "gpu_bind_20260804_a800/POST_RENT_BINDING.json").read_bytes()
        )
        manifest_bytes = (STAGE_ROOT / "data/manifest.json").read_bytes()
        if not (
            freeze.get("binding_commit") == BINDING_COMMIT
            and freeze.get("gpu_uuid") == GPU_UUID
            and binding.get("status")
            == "BOUND_BY_COMMIT_2a31e76b546dfbf9b5c801bc0efad3066254ec9f"
            and binding.get("gpu", {}).get("uuid") == GPU_UUID
            and binding.get("gpu", {}).get("memory_total_mib") == 81920
            and binding.get("manifest_sha256") == _sha256(manifest_bytes)
        ):
            return False
        for family in FAMILIES:
            receipt_path = STAGE_ROOT / f"gpu_bind_20260804_a800/{family}_receipt.json"
            receipt_bytes = receipt_path.read_bytes()
            receipt = json.loads(receipt_bytes)
            spec = binding["receipts"][family]
            if not (
                _sha256(receipt_bytes) == RECEIPT_SHA256[family]
                and spec["artifact_sha256"] == RECEIPT_SHA256[family]
                and receipt["bindings"]["gpu_uuid"] == GPU_UUID
                and receipt["bindings"]["model_revision"] == spec["model_revision"]
                and receipt["bindings"]["manifest_sha256"] == _sha256(manifest_bytes)
                and base64.b64decode(receipt["manifest_base64"], validate=True)
                == manifest_bytes
            ):
                return False
            body = {
                key: receipt[key]
                for key in (
                    "protocol",
                    "authority_sha256",
                    "bindings",
                    "manifest_base64",
                    "launcher_command",
                )
            }
            if receipt["receipt_sha256"] != _sha256(_canonical(body)):
                return False
            for field in (
                "control_commit",
                "runtime_commit",
                "manifest_sha256",
                "extension_sha256",
            ):
                if receipt["bindings"][field] != binding[field]:
                    return False
        return True
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


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
        current_files[relative] = path.is_file() and _sha256(path.read_bytes()) == expected_sha
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

    ancestor = False
    if commit_valid and _valid_commit(freeze.get("binding_commit")):
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", freeze["binding_commit"], source_commit],
            cwd=REPO_ROOT,
            check=False,
        ).returncode == 0
    binding_semantics = _binding_semantics(freeze)

    boundary_pass = (
        freeze.get("protocol") == "RIST-C0-v2.3-CAPACITY-CANARY-CPU-FREEZE-v1"
        and commit_valid
        and ancestor
        and binding_semantics
        and freeze.get("binding_commit") == BINDING_COMMIT
        and freeze.get("gpu_uuid") == GPU_UUID
        and set(recorded) == EXPECTED_FILES
        and freeze.get("families") == FAMILIES
        and freeze.get("request_seeds") == SEEDS
        and freeze.get("requests_per_family") == 8
        and freeze.get("request_concurrency") == 8
        and freeze.get("retry_count") == 0
        and freeze.get("minimum_gpu_memory_gib") == 48
        and freeze.get("outcomes_inspected") is False
        and freeze.get("scientific_result") is False
        and freeze.get("gpu_execution_permitted_by_freeze") is False
        and freeze.get("training_permitted") is False
        and freeze.get("heldout_permitted") is False
        and freeze.get("bfcl_permitted") is False
        and freeze.get("qwen_previous_startup_valid_as_capacity_evidence") is False
    )
    passed = (
        boundary_pass
        and all(current_files.values())
        and all(committed_files.values())
    )
    return {
        "protocol": "RIST-C0-v2.3-CAPACITY-CANARY-CPU-FREEZE-VERIFY-v1",
        "boundary_pass": boundary_pass,
        "binding_ancestor_pass": ancestor,
        "binding_semantics_pass": binding_semantics,
        "current_files": current_files,
        "committed_files": committed_files,
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

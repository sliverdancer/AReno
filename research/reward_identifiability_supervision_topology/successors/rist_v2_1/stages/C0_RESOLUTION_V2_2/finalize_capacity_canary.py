"""Validate both C0 v2.2 capacity canaries without inspecting outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finalize(
    manifest: dict[str, Any],
    bindings: dict[str, tuple[dict[str, Any], Path]],
) -> dict[str, Any]:
    if manifest.get("protocol") != "RIST-C0-v2.2-CANARY-MANIFEST":
        raise ValueError("unexpected C0 v2.2 canary manifest")
    prohibited = (
        "execution_authorized",
        "scientific_task_access_permitted",
        "calibration_permitted",
        "qualification_permitted",
        "training_permitted",
        "heldout_permitted",
    )
    if any(manifest.get(key) is not False for key in prohibited):
        raise ValueError("capacity canary manifest must remain outcome-free")
    if {job.get("family") for job in manifest.get("jobs", [])} != {
        "qwen3",
        "gemma4",
    }:
        raise ValueError("capacity canary manifest must bind both model families")
    rows = []
    gpu_identity = None
    for family in ("qwen3", "gemma4"):
        result, journal_path = bindings[family]
        identity = result.get("runtime_identity")
        if not isinstance(identity, dict):
            raise ValueError(f"{family} lacks runtime identity")
        expected = {
            "family": family,
            "checkpoint": manifest["models"][family],
            "model_revision": manifest["model_revisions"][family],
            "tokenizer_snapshot_sha256": manifest["tokenizer_snapshot_sha256"][family],
            "source_commit": manifest["source_commit"],
        }
        identity_pass = all(identity.get(key) == value for key, value in expected.items())
        gpu = {
            key: identity.get(key)
            for key in ("gpu_name", "gpu_uuid", "gpu_total_memory_gib", "driver_version", "cuda_version", "torch_version")
        }
        if gpu_identity is None:
            gpu_identity = gpu
        same_gpu = gpu == gpu_identity
        journal_hash = _sha256(journal_path)
        passed = all(
            (
                result.get("protocol") == "RIST-C0-v2.2-CAPACITY-CANARY",
                result.get("family") == family,
                result.get("request_concurrency") == manifest["request_concurrency"],
                result.get("expected_response_count") == manifest["expected_responses_per_family"],
                result.get("response_count") == manifest["expected_responses_per_family"],
                result.get("retry_count") == 0,
                result.get("infrastructure_error") is None,
                result.get("complete") is True,
                result.get("outcomes_inspected") is False,
                result.get("scientific_result") is False,
                result.get("raw_journal_sha256") == journal_hash,
                float(identity.get("gpu_total_memory_gib", 0.0)) >= manifest["minimum_gpu_memory_gib"],
                identity_pass,
                same_gpu,
            )
        )
        rows.append(
            {
                "family": family,
                "passed": passed,
                "identity_pass": identity_pass,
                "same_gpu": same_gpu,
                "peak_memory_mib": result.get("peak_memory_mib"),
                "result_sha256": hashlib.sha256(
                    (json.dumps(result, sort_keys=True) + "\n").encode()
                ).hexdigest(),
                "journal_sha256": journal_hash,
            }
        )
    passed = all(row["passed"] for row in rows)
    return {
        "protocol": "RIST-C0-v2.2-CAPACITY-CANARY-FINAL",
        "status": "PASS" if passed else "FAIL",
        "decision": (
            "PASS_CANARY_TO_SEPARATE_C0_AUTHORIZATION"
            if passed
            else "KILL_C0_V2_2_FRESH_POOL_BEFORE_SCIENTIFIC_ACCESS"
        ),
        "outcomes_inspected": False,
        "scientific_result": False,
        "gpu_identity": gpu_identity,
        "families": rows,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    for family in ("qwen", "gemma"):
        parser.add_argument(f"--{family}-result", type=Path, required=True)
        parser.add_argument(f"--{family}-journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bindings = {
        "qwen3": (json.loads(args.qwen_result.read_text()), args.qwen_journal),
        "gemma4": (json.loads(args.gemma_result.read_text()), args.gemma_journal),
    }
    result = finalize(json.loads(args.manifest.read_text()), bindings)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Build one fail-closed C0 v2.3 calibration or qualification manifest."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

STAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[6]
FAMILIES = ("qwen3", "gemma4")
MODELS = {"qwen3": "Qwen/Qwen3-0.6B", "gemma4": "google/gemma-4-E2B-it"}
REVISIONS = {
    "qwen3": "c1899de289a04d12100db370d81485cdf75e47ca",
    "gemma4": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
}
TOKENIZER_SNAPSHOTS = {
    "qwen3": "c83c7f983e1204841852a4cb47cff31dfd829437c80dccc55dd52d0c8fe532b1",
    "gemma4": "7c813a44e67aa09d81001db777c261d858417b45ce0204bc6a32f0b7b96720f7",
}
EXPECTED_TRAJECTORIES_PER_JOB = 1024


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _require_head(commit: str) -> str:
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("source commit must be a full lowercase Git commit")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != head:
        raise ValueError("source commit must equal checked-out HEAD")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("scientific collection requires a clean checked-out worktree")
    return commit


def _validate_capacity(result: dict[str, Any]) -> None:
    if result.get("protocol") != "RIST-C0-v2.3-CAPACITY-CANARY-RESULT-v1":
        raise ValueError("unexpected v2.3 capacity result")
    if result.get("status") != "PASS_RESPONSE_CAPACITY_WITH_POST_COLLECTION_SHUTDOWN_ANOMALY":
        raise PermissionError("capacity response gate did not pass")
    if any(result.get(key) is not False for key in (
        "outcomes_inspected", "scientific_result", "calibration_accessed",
        "qualification_accessed", "heldout_accessed", "bfcl_accessed", "training_performed",
    )):
        raise PermissionError("capacity evidence crossed a forbidden boundary")
    for family in FAMILIES:
        row = result.get("models", {}).get(family, {})
        if not (
            row.get("complete") is True
            and row.get("response_count") == 8
            and row.get("expected_response_count") == 8
            and row.get("request_concurrency") == 8
            and row.get("retry_count") == 0
            and row.get("infrastructure_error") is None
            and row.get("deployment_accepted") is True
            and row.get("exclusive_bound_process_before_requests") is True
        ):
            raise PermissionError(f"capacity evidence failed for {family}")


def _validate_shutdown(result: dict[str, Any], gpu_uuid: str) -> None:
    if result.get("protocol") != "RIST-C0-v2.3-CLEAN-SHUTDOWN-GPU-RESULT-v1":
        raise ValueError("unexpected v2.3 clean-shutdown result")
    if result.get("status") != "PASS_CLEAN_SHUTDOWN_WITH_RESOURCE_TRACKER_WARNING":
        raise PermissionError("clean-shutdown gate did not pass")
    checks = result.get("checks", {})
    if not (
        result.get("gpu_uuid") == gpu_uuid
        and checks.get("deployment_receipt_accepted") is True
        and checks.get("application_startup_complete") is True
        and checks.get("model_http_request_count") == 0
        and checks.get("application_shutdown_complete") is True
        and checks.get("rollout_session_end_error_count") == 0
        and checks.get("supervisor_server_and_worker_exited") is True
        and checks.get("final_gpu_memory_used_mib") == 0
        and checks.get("final_compute_process_count") == 0
        and result.get("model_requests_performed") is False
        and result.get("calibration_accessed") is False
        and result.get("qualification_accessed") is False
        and result.get("heldout_accessed") is False
        and result.get("bfcl_accessed") is False
        and result.get("training_performed") is False
    ):
        raise PermissionError("clean-shutdown evidence is incomplete or bound to another GPU")


def _validate_identity(family: str, identity: dict[str, Any], source_commit: str) -> str:
    expected = {
        "family": family,
        "checkpoint": MODELS[family],
        "model_revision": REVISIONS[family],
        "tokenizer_snapshot_sha256": TOKENIZER_SNAPSHOTS[family],
        "source_commit": source_commit,
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError(f"runtime identity mismatch for {family}")
    gpu_uuid = str(identity.get("gpu_uuid", ""))
    if not gpu_uuid.startswith("GPU-") or float(identity.get("gpu_total_memory_gib", 0.0)) < 79.0:
        raise ValueError("scientific collection requires one bound 80 GB GPU")
    return gpu_uuid


def _validate_deployment_receipt(
    path: Path, family: str, source_commit: str, gpu_uuid: str
) -> str:
    raw = path.read_bytes()
    receipt = json.loads(raw)
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    if raw != canonical or receipt.get("protocol") != "RIST-C0-v2.3-DEPLOYMENT-RECEIPT-v1":
        raise ValueError("deployment receipt must be canonical v2.3 bytes")
    required = {
        "protocol", "authority_sha256", "bindings", "manifest_base64",
        "launcher_command", "receipt_sha256",
    }
    if set(receipt) != required:
        raise ValueError("deployment receipt schema mismatch")
    body = {key: receipt[key] for key in (
        "protocol", "authority_sha256", "bindings", "manifest_base64", "launcher_command"
    )}
    body_sha256 = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if receipt["receipt_sha256"] != body_sha256:
        raise ValueError("deployment receipt internal SHA mismatch")
    bindings = receipt.get("bindings", {})
    if set(bindings) != {
        "control_commit", "runtime_commit", "manifest_sha256", "model_revision",
        "gpu_uuid", "extension_sha256",
    }:
        raise ValueError("deployment receipt identity schema mismatch")
    if not (
        bindings["control_commit"] == source_commit
        and bindings["runtime_commit"] == source_commit
        and bindings["model_revision"] == REVISIONS[family]
        and bindings["gpu_uuid"] == gpu_uuid
        and len(bindings["manifest_sha256"]) == 64
        and len(bindings["extension_sha256"]) == 64
    ):
        raise ValueError("scientific deployment must use one frozen commit and GPU identity")
    try:
        embedded = base64.b64decode(receipt["manifest_base64"], validate=True)
        json.loads(embedded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("deployment receipt contains an invalid embedded manifest") from exc
    if hashlib.sha256(embedded).hexdigest() != bindings["manifest_sha256"]:
        raise ValueError("deployment receipt embedded-manifest SHA mismatch")
    command = receipt.get("launcher_command")
    if not isinstance(command, list) or "serve" not in command or "127.0.0.1" not in command:
        raise ValueError("deployment receipt must launch the loopback serving entrypoint")
    return _sha256(path)


def _validate_calibration_admission(path: Path | None, pool_sha256: str) -> dict[str, Any] | None:
    if path is None:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not (
        value.get("protocol") == "RIST-C0-v2.3-CALIBRATION-ADMISSION-v1"
        and value.get("passed") is True
        and value.get("decision") == "PASS_CALIBRATION_TO_QUALIFICATION"
        and value.get("pool_manifest_sha256") == pool_sha256
        and value.get("qualification_accessed") is False
    ):
        raise PermissionError("qualification requires a frozen calibration-only admission")
    counts = value.get("common_candidate_counts", {})
    if counts.get("low", 0) < 2 or counts.get("high", 0) < 2:
        raise PermissionError("calibration admission lacks two common cells per band")
    return value


def build_manifest(
    *,
    split: str,
    pool_manifest_path: Path,
    capacity_result_path: Path,
    clean_shutdown_result_path: Path,
    runtime_identities: dict[str, dict[str, Any]],
    deployment_receipts: dict[str, Path],
    source_commit: str,
    output_root: Path,
    calibration_admission_path: Path | None = None,
) -> dict[str, Any]:
    if split not in {"calibration", "qualification"}:
        raise ValueError("split must be calibration or qualification")
    pool = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
    if pool.get("protocol") != "RIST-C0-v2.3-FRESH-POOL":
        raise ValueError("unexpected v2.3 pool manifest")
    pool_sha256 = _sha256(pool_manifest_path)
    capacity = json.loads(capacity_result_path.read_text(encoding="utf-8"))
    shutdown = json.loads(clean_shutdown_result_path.read_text(encoding="utf-8"))
    _validate_capacity(capacity)
    commit = _require_head(source_commit)
    if set(runtime_identities) != set(FAMILIES):
        raise ValueError("both runtime identities are required")
    gpu_uuids = {
        _validate_identity(family, runtime_identities[family], commit)
        for family in FAMILIES
    }
    if len(gpu_uuids) != 1:
        raise ValueError("both models must be bound to the same GPU UUID")
    gpu_uuid = next(iter(gpu_uuids))
    _validate_shutdown(shutdown, gpu_uuid)
    if set(deployment_receipts) != set(FAMILIES):
        raise ValueError("both deployment receipts are required")
    receipt_sha256 = {
        family: _validate_deployment_receipt(
            deployment_receipts[family], family, commit, gpu_uuid
        )
        for family in FAMILIES
    }
    admission = _validate_calibration_admission(
        calibration_admission_path, pool_sha256
    )
    if split == "calibration" and admission is not None:
        raise ValueError("calibration cannot consume its own outcome admission")
    if split == "qualification" and admission is None:
        raise PermissionError("qualification remains sealed until calibration GO")

    spec = pool["splits"][split]
    count = int(spec["task_count"]) * len(spec["rollout_seeds"])
    if count != EXPECTED_TRAJECTORIES_PER_JOB:
        raise ValueError("each family/split job must contain 1,024 trajectories")
    jobs = []
    for family in FAMILIES:
        root = output_root / f"{family}-{split}"
        jobs.append({
            "job_id": f"{family}-{split}",
            "family": family,
            "split": split,
            "checkpoint": MODELS[family],
            "model_revision": REVISIONS[family],
            "tokenizer_snapshot_sha256": TOKENIZER_SNAPSHOTS[family],
            "source_commit": commit,
            "gpu_uuid": gpu_uuid,
            "trajectory_count": count,
            "request_concurrency": 8,
            "retry_count": 0,
            "deployment_receipt_sha256": receipt_sha256[family],
            "result": str(root / "result.json"),
            "trajectories": str(root / "trajectories.jsonl"),
            "raw_journal": str(root / "raw_responses.jsonl"),
        })
    evidence = {
        "pool_manifest": _binding(pool_manifest_path),
        "capacity_result": _binding(capacity_result_path),
        "clean_shutdown_result": _binding(clean_shutdown_result_path),
        "qwen3_deployment_receipt": _binding(deployment_receipts["qwen3"]),
        "gemma4_deployment_receipt": _binding(deployment_receipts["gemma4"]),
    }
    if calibration_admission_path is not None:
        evidence["calibration_admission"] = _binding(calibration_admission_path)
    return {
        "protocol": "RIST-C0-v2.3-SCIENTIFIC-COLLECTION-MANIFEST-v1",
        "split": split,
        "collection_source_commit": commit,
        "pool_manifest": str(pool_manifest_path.resolve()),
        "pool_manifest_sha256": pool_sha256,
        "canary_gate": {
            "passed": True,
            "status": "PASS",
            "gpu_identity": {"gpu_uuid": gpu_uuid, "minimum_memory_gib": 79.0},
            "evidence_files": evidence,
            "outcomes_inspected": False,
            "scientific_result": False,
        },
        "models": MODELS,
        "model_revisions": REVISIONS,
        "tokenizer_snapshot_sha256": TOKENIZER_SNAPSHOTS,
        "sampling": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 128, "retry_count": 0},
        "request_concurrency": 8,
        "job_count": 2,
        "trajectory_count": 2048,
        "jobs": jobs,
        "calibration_permitted": split == "calibration",
        "qualification_permitted": split == "qualification",
        "heldout_permitted": False,
        "bfcl_permitted": False,
        "training_permitted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("calibration", "qualification"), required=True)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--capacity-result", type=Path, required=True)
    parser.add_argument("--clean-shutdown-result", type=Path, required=True)
    parser.add_argument("--qwen-runtime-identity", type=Path, required=True)
    parser.add_argument("--gemma-runtime-identity", type=Path, required=True)
    parser.add_argument("--qwen-deployment-receipt", type=Path, required=True)
    parser.add_argument("--gemma-deployment-receipt", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--calibration-admission", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(
        split=args.split,
        pool_manifest_path=args.pool_manifest,
        capacity_result_path=args.capacity_result,
        clean_shutdown_result_path=args.clean_shutdown_result,
        runtime_identities={
            "qwen3": json.loads(args.qwen_runtime_identity.read_text(encoding="utf-8")),
            "gemma4": json.loads(args.gemma_runtime_identity.read_text(encoding="utf-8")),
        },
        deployment_receipts={
            "qwen3": args.qwen_deployment_receipt,
            "gemma4": args.gemma_deployment_receipt,
        },
        source_commit=args.source_commit,
        output_root=args.output_root,
        calibration_admission_path=args.calibration_admission,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

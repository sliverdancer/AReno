"""Run one C0 v2.3 job through the sole receipt-bound graceful supervisor."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

STAGE_ROOT = Path(__file__).resolve().parent


def _load_collector():
    path = STAGE_ROOT / "collect_scientific_job.py"
    spec = importlib.util.spec_from_file_location("rist_c0_v2_3_bound_collector", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("scientific collector is unavailable")
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ledger_accepted(path: Path) -> bool:
    if not path.is_file():
        return False
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    decisions = [row for row in rows if row.get("event") == "prelaunch_decision"]
    return (
        len(decisions) == 1
        and decisions[0].get("decision") == "ACCEPT"
        and not any(row.get("decision") == "REJECT" for row in rows)
        and sum(row.get("event") == "launch_attempt" for row in rows) == 1
    )


def _require_clean_shared_root(root: Path, source_commit: str) -> None:
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if head != source_commit or status:
        raise ValueError("scientific serving requires the exact clean frozen source commit")


def _wait_ready(base_url: str, process: subprocess.Popen[Any], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"{base_url.rstrip('/')}/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"receipt-bound serving exited before readiness: {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:
                if 200 <= response.status < 300:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1.0)
    raise TimeoutError("receipt-bound serving did not become ready")


def run(
    *,
    manifest_path: Path,
    job_id: str,
    runtime_identity_path: Path,
    receipt_path: Path,
    control_root: Path,
    runtime_root: Path,
    model_path: Path,
    extension_path: Path,
    ledger_path: Path,
    base_url: str,
    startup_timeout: float,
    shutdown_timeout: float,
) -> dict[str, Any]:
    if control_root.resolve() != runtime_root.resolve():
        raise ValueError("v2.3 scientific serving requires one shared control/runtime worktree")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    collector = _load_collector()
    job = collector._job(manifest, job_id)
    _require_clean_shared_root(control_root, job["source_commit"])
    receipt_sha256 = _sha256(receipt_path)
    if receipt_sha256 != job.get("deployment_receipt_sha256"):
        raise ValueError("scientific job receipt SHA mismatch")
    if ledger_path.exists():
        raise FileExistsError("deployment ledger must be fresh")
    identity = json.loads(runtime_identity_path.read_text(encoding="utf-8"))
    identity["deployment_receipt_sha256"] = receipt_sha256
    collector._validate_gate(manifest, job, identity)

    command = [
        str(Path(__file__).resolve().with_name("graceful_deployment_entrypoint.py")),
        "--receipt", str(receipt_path),
        "--authorized-receipt-sha256", receipt_sha256,
        "--control-root", str(control_root),
        "--runtime-root", str(runtime_root),
        "--model-path", str(model_path),
        "--extension", str(extension_path),
        "--ledger", str(ledger_path),
    ]
    process = subprocess.Popen(["python3", *command])
    shutdown_clean = False
    result: dict[str, Any] | None = None
    try:
        _wait_ready(base_url, process, startup_timeout)
        if not _ledger_accepted(ledger_path):
            raise RuntimeError("deployment ledger did not record one exact ACCEPT")
        pool_path = Path(manifest["pool_manifest"])
        if _sha256(pool_path) != manifest["pool_manifest_sha256"]:
            raise ValueError("scientific pool manifest hash mismatch")
        evaluator = collector._load_evaluator()
        result = collector.collect_job(
            manifest=manifest,
            manifest_sha256=_sha256(manifest_path),
            job_id=job_id,
            runtime_identity=identity,
            pool_manifest=json.loads(pool_path.read_text(encoding="utf-8")),
            data_dir=pool_path.parent,
            result_path=Path(job["result"]),
            trajectory_path=Path(job["trajectories"]),
            journal_path=Path(job["raw_journal"]),
            post_json=lambda payload: evaluator._post_json(base_url, "EMPTY", 900.0, payload),
        )
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        try:
            returncode = process.wait(timeout=shutdown_timeout)
            shutdown_clean = returncode == 0
        except subprocess.TimeoutExpired:
            returncode = None
        if result is not None:
            result["deployment_receipt_sha256"] = receipt_sha256
            result["deployment_ledger_sha256"] = _sha256(ledger_path) if ledger_path.is_file() else None
            result["server_shutdown_clean"] = shutdown_clean
            result["server_returncode"] = returncode
            if not shutdown_clean:
                result["complete"] = False
                result["infrastructure_error"] = {
                    "error_type": "UncleanScientificServerShutdown",
                    "error": "receipt-bound serving did not exit cleanly",
                }
            collector._write_json(Path(job["result"]), result)
    if result is None:
        raise RuntimeError("scientific collection did not produce a result")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--shutdown-timeout", type=float, default=180.0)
    args = parser.parse_args()
    result = run(
        manifest_path=args.manifest,
        job_id=args.job_id,
        runtime_identity_path=args.runtime_identity,
        receipt_path=args.receipt,
        control_root=args.control_root,
        runtime_root=args.runtime_root,
        model_path=args.model_path,
        extension_path=args.extension,
        ledger_path=args.ledger,
        base_url=args.base_url,
        startup_timeout=args.startup_timeout,
        shutdown_timeout=args.shutdown_timeout,
    )
    return 0 if result.get("complete") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())

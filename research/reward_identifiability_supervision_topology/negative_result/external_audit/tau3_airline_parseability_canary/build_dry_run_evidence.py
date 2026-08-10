"""Build CPU-only dry-run evidence for the Tau3 parseability canary chain."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "dry_run_evidence"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    binder = _load("tau3_binder_for_dry_run_evidence", HERE / "bind_runtime_receipt.py")
    runner = _load("tau3_runner_for_dry_run_evidence", HERE / "run_tau3_single_request_canary.py")
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    receipt = binder.build_bound_receipt(
        source_commit="1" * 40,
        model_repo_or_api_id="Qwen/Qwen3-0.6B",
        model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        user_simulator_model="tau3/synthetic-user-simulator-placeholder",
        user_simulator_revision="tau3-v1.0.1",
        user_simulator_authorization_sha256="2" * 64,
        gpu_uuid_or_api_provider="CPU_SYNTHETIC_DRY_RUN",
        task_id="airline:synthetic-public-task",
    )
    receipt_path = OUT_DIR / "TAU3_RUNTIME_RECEIPT_BOUND.json"
    receipt_digest = binder.write_bound_receipt(receipt, receipt_path)
    plan = runner.build_request_plan(runner.load_receipt(receipt_path))
    runner.write_json(OUT_DIR / "TAU3_REQUEST_PLAN_DRY_RUN.json", plan)
    manifest = {
        "protocol": "RRC-TAU3-AIRLINE-PARSEABILITY-CANARY-DRY-RUN-EVIDENCE-v1",
        "status": "PASS_CPU_ONLY_NO_MODEL_REQUEST_SENT",
        "model_request_sent": False,
        "api_used": False,
        "gpu_used": False,
        "training_used": False,
        "bfcl_used": False,
        "heldout_or_sealed_access_used": False,
        "raw_response_included": False,
        "receipt_sha256": receipt_digest,
        "files": [
            {
                "path": "TAU3_RUNTIME_RECEIPT_BOUND.json",
                "sha256": sha256_file(receipt_path),
            },
            {
                "path": "TAU3_RUNTIME_RECEIPT_BOUND.json.sha256",
                "sha256": sha256_file(receipt_path.with_suffix(receipt_path.suffix + ".sha256")),
            },
            {
                "path": "TAU3_REQUEST_PLAN_DRY_RUN.json",
                "sha256": sha256_file(OUT_DIR / "TAU3_REQUEST_PLAN_DRY_RUN.json"),
            },
        ],
    }
    (OUT_DIR / "DRY_RUN_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

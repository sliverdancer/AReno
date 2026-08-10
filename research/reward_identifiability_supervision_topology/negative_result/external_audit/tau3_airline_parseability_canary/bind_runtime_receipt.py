"""Bind the Tau3 parseability canary runtime receipt before a model request.

This script is CPU-only. It writes a bound receipt plus its SHA-256 sidecar, and
fails before writing if any required runtime identity is missing or malformed.
It does not start Tau3, call a model/API, use GPU, or train.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
TEMPLATE_PATH = HERE / "TAU3_RUNTIME_RECEIPT_TEMPLATE.json"
FINALIZER_PATH = HERE / "tau3_canary_finalizer.py"


def _load_finalizer():
    spec = importlib.util.spec_from_file_location("tau3_canary_finalizer_for_bind", FINALIZER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_template() -> dict[str, Any]:
    value = json.loads(TEMPLATE_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("runtime receipt template must be a JSON object")
    return value


def _require_bound(name: str, value: str) -> str:
    value = str(value).strip()
    if not value or value.startswith("UNBOUND"):
        raise ValueError(f"{name} must be bound before first request")
    return value


def _sha256_json(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_bound_receipt(
    *,
    source_commit: str,
    model_repo_or_api_id: str,
    model_revision: str,
    user_simulator_model: str,
    user_simulator_revision: str,
    user_simulator_authorization_sha256: str,
    gpu_uuid_or_api_provider: str,
    task_id: str,
) -> dict[str, Any]:
    receipt = _load_template()
    receipt.update(
        {
            "status": "BOUND_READY_FOR_SINGLE_REQUEST_CANARY",
            "source_commit": _require_bound("source_commit", source_commit),
            "model_repo_or_api_id": _require_bound("model_repo_or_api_id", model_repo_or_api_id),
            "model_revision": _require_bound("model_revision", model_revision),
            "user_simulator_model": _require_bound("user_simulator_model", user_simulator_model),
            "user_simulator_revision": _require_bound(
                "user_simulator_revision", user_simulator_revision
            ),
            "user_simulator_authorization_sha256": _require_bound(
                "user_simulator_authorization_sha256",
                user_simulator_authorization_sha256,
            ),
            "gpu_uuid_or_api_provider": _require_bound(
                "gpu_uuid_or_api_provider", gpu_uuid_or_api_provider
            ),
            "task_id": _require_bound("task_id", task_id),
            "pre_request_exit_if_any_unbound": True,
        }
    )
    finalizer = _load_finalizer()
    finalizer.validate_runtime_receipt(receipt)
    return receipt


def write_bound_receipt(receipt: dict[str, Any], output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = _sha256_json(receipt)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n",
        encoding="ascii",
    )
    return digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model-repo-or-api-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--user-simulator-model", required=True)
    parser.add_argument("--user-simulator-revision", required=True)
    parser.add_argument("--user-simulator-authorization-sha256", required=True)
    parser.add_argument("--gpu-uuid-or-api-provider", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_bound_receipt(
        source_commit=args.source_commit,
        model_repo_or_api_id=args.model_repo_or_api_id,
        model_revision=args.model_revision,
        user_simulator_model=args.user_simulator_model,
        user_simulator_revision=args.user_simulator_revision,
        user_simulator_authorization_sha256=args.user_simulator_authorization_sha256,
        gpu_uuid_or_api_provider=args.gpu_uuid_or_api_provider,
        task_id=args.task_id,
    )
    write_bound_receipt(receipt, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CPU-only Tau3 parseability canary receipt/finalizer helpers.

This module never sends a model request. It validates runtime binding records
and finalizes a supplied single-response summary into PASS/FAIL evidence for the
Tau3 parseability canary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
TEMPLATE_PATH = HERE / "TAU3_PARSEABILITY_CANARY_TEMPLATE.json"
TEMPLATE_SHA_PATH = HERE / "TAU3_PARSEABILITY_CANARY_TEMPLATE.sha256"


HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def assert_template_hash() -> str:
    expected = TEMPLATE_SHA_PATH.read_text(encoding="ascii").split()[0]
    actual = sha256_file(TEMPLATE_PATH)
    if expected != actual:
        raise ValueError("Tau3 parseability template hash mismatch")
    return actual


def validate_runtime_receipt(receipt: dict[str, Any]) -> None:
    if receipt.get("protocol") != "RRC-TAU3-AIRLINE-PARSEABILITY-CANARY-RUNTIME-RECEIPT-v1":
        raise ValueError("unexpected Tau3 runtime receipt protocol")
    if receipt.get("template_sha256") != assert_template_hash():
        raise ValueError("runtime receipt is not bound to the frozen template")
    if not HEX40.fullmatch(str(receipt.get("source_commit", ""))):
        raise ValueError("source_commit must be 40 lowercase hex characters")
    if not HEX64.fullmatch(str(receipt.get("user_simulator_authorization_sha256", ""))):
        raise ValueError("user simulator authorization hash must be 64 lowercase hex characters")
    if receipt.get("model_request_budget") != 1:
        raise ValueError("Tau3 parseability canary allows exactly one model request")
    if receipt.get("retry_budget") != 0:
        raise ValueError("Tau3 parseability canary requires zero retry")
    if receipt.get("training_authorized") is not False:
        raise ValueError("training must remain unauthorized")
    if receipt.get("bfcl_used") is not False:
        raise ValueError("BFCL must not be used by the Tau3 canary")
    if receipt.get("heldout_or_sealed_accessed") is not False:
        raise ValueError("held-out or sealed data must not be accessed")
    if not str(receipt.get("task_id", "")).strip():
        raise ValueError("task_id must be bound before first request")
    if not str(receipt.get("model_repo_or_api_id", "")).strip():
        raise ValueError("model identity must be bound before first request")


def finalize_canary(receipt: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    validate_runtime_receipt(receipt)
    if observation.get("protocol") != "RRC-TAU3-AIRLINE-PARSEABILITY-CANARY-OBSERVATION-v1":
        raise ValueError("unexpected Tau3 canary observation protocol")
    if observation.get("model_request_count") != 1:
        raise ValueError("observation must summarize exactly one model request")
    if observation.get("retry_count") != 0:
        raise ValueError("observation must use zero retry")
    if observation.get("task_id") != receipt.get("task_id"):
        raise ValueError("observation task_id does not match receipt")
    if observation.get("model_repo_or_api_id") != receipt.get("model_repo_or_api_id"):
        raise ValueError("observation model identity does not match receipt")

    observed_calls = observation.get("observed_tool_calls")
    if not isinstance(observed_calls, list):
        raise ValueError("observed_tool_calls must be a list")
    parseable = False
    parsed_action_json_object = False
    failure_mode = "PARSE_FAILURE"
    if observed_calls:
        first = observed_calls[0]
        if isinstance(first, dict) and isinstance(first.get("name"), str) and isinstance(
            first.get("arguments"), dict
        ):
            parseable = True
            parsed_action_json_object = True
            failure_mode = None

    success_gate_passed = parseable and parsed_action_json_object
    status = "PASS_PARSEABLE_TOOL_CALL" if success_gate_passed else "TERMINAL_PARSE_FAILURE"
    return {
        "protocol": "RRC-TAU3-AIRLINE-PARSEABILITY-CANARY-TERMINAL-FINALIZER-v1",
        "status": status,
        "template_sha256": receipt["template_sha256"],
        "runtime_receipt_sha256": observation.get("runtime_receipt_sha256"),
        "source_commit": receipt["source_commit"],
        "task_id": receipt["task_id"],
        "model_repo_or_api_id": receipt["model_repo_or_api_id"],
        "model_revision": receipt.get("model_revision"),
        "model_request_sent": True,
        "model_request_count": 1,
        "retry_count": 0,
        "parseable_tool_calls": parseable,
        "observed_call_count": len(observed_calls),
        "parsed_action_json_object": parsed_action_json_object,
        "success_gate_passed": success_gate_passed,
        "failure_mode": failure_mode,
        "strict_task_success_required": False,
        "strict_task_success_claimed": False,
        "reward_resolution_claim_allowed": False,
        "reward_resolution_claimed": False,
        "training_authorized": False,
        "bfcl_used": False,
        "heldout_or_sealed_accessed": False,
        "go_to_reward_resolution_calibration": success_gate_passed,
        "full_external_audit_authorized": False,
    }


def build_synthetic_receipt() -> dict[str, Any]:
    return {
        "protocol": "RRC-TAU3-AIRLINE-PARSEABILITY-CANARY-RUNTIME-RECEIPT-v1",
        "template_sha256": assert_template_hash(),
        "source_commit": "a" * 40,
        "model_repo_or_api_id": "synthetic/no-model",
        "model_revision": "synthetic-revision",
        "user_simulator_model": "synthetic/no-user-simulator",
        "user_simulator_revision": "synthetic-revision",
        "user_simulator_authorization_sha256": "b" * 64,
        "gpu_uuid_or_api_provider": "CPU_SYNTHETIC_REPLAY",
        "task_id": "synthetic_tau3_airline_public_task",
        "model_request_budget": 1,
        "retry_budget": 0,
        "training_authorized": False,
        "bfcl_used": False,
        "heldout_or_sealed_accessed": False,
    }


def build_synthetic_replay() -> dict[str, Any]:
    receipt = build_synthetic_receipt()
    receipt_sha = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    base = {
        "protocol": "RRC-TAU3-AIRLINE-PARSEABILITY-CANARY-OBSERVATION-v1",
        "runtime_receipt_sha256": receipt_sha,
        "task_id": receipt["task_id"],
        "model_repo_or_api_id": receipt["model_repo_or_api_id"],
        "model_request_count": 1,
        "retry_count": 0,
    }
    pass_observation = {
        **base,
        "observed_tool_calls": [{"name": "DB", "arguments": {"query": "SELECT 1"}}],
    }
    fail_observation = {**base, "observed_tool_calls": []}
    return {
        "protocol": "RRC-TAU3-AIRLINE-PARSEABILITY-CANARY-FINALIZER-REPLAY-v1",
        "status": "PASS",
        "cpu_only": True,
        "model_inference_used": False,
        "api_used": False,
        "gpu_used": False,
        "training_used": False,
        "heldout_or_sealed_access_used": False,
        "bfcl_used": False,
        "pass_finalizer": finalize_canary(receipt, pass_observation),
        "parse_failure_finalizer": finalize_canary(receipt, fail_observation),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--observation", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-synthetic-replay", action="store_true")
    args = parser.parse_args()
    if args.write_synthetic_replay:
        replay = build_synthetic_replay()
        (args.output or HERE / "TAU3_CANARY_FINALIZER_REPLAY.json").write_text(
            json.dumps(replay, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0
    if not (args.receipt and args.observation and args.output):
        parser.error("--receipt, --observation, and --output are required unless --write-synthetic-replay is set")
    result = finalize_canary(load_json(args.receipt), load_json(args.observation))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

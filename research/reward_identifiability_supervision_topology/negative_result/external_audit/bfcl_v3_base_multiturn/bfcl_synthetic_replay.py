"""CPU-only BFCL external-audit evaluator wiring and synthetic replay.

This module intentionally does not load BFCL prompts, possible answers, model
responses, model weights, GPUs, or APIs. It validates the frozen receipt and
exercises the strict-evaluation/finalization path on synthetic dummy records.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


AUDIT_DIR = Path(__file__).resolve().parent
RECEIPT = AUDIT_DIR / "EXECUTION_RECEIPT_STATIC.json"
RECEIPT_SHA = AUDIT_DIR / "EXECUTION_RECEIPT_STATIC.sha256"
SELECTED_IDS = AUDIT_DIR / "SELECTED_TASK_IDS.txt"
PUBLIC_MANIFEST = AUDIT_DIR / "PUBLIC_FILE_MANIFEST.json"
SCHEMA_SUMMARY = AUDIT_DIR / "SCHEMA_SUMMARY.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_receipt(receipt_path: Path = RECEIPT) -> dict[str, Any]:
    return json.loads(receipt_path.read_text(encoding="utf-8"))


def validate_static_receipt(receipt_path: Path = RECEIPT) -> dict[str, Any]:
    receipt = load_receipt(receipt_path)
    expected_hash = RECEIPT_SHA.read_text(encoding="ascii").split()[0]
    actual_hash = _sha256(receipt_path)
    if actual_hash != expected_hash:
        raise ValueError(f"receipt hash mismatch: expected {expected_hash}, got {actual_hash}")
    scope = receipt["scope"]
    denied = [
        "model_inference_authorized",
        "api_inference_authorized",
        "gpu_authorized",
        "training_authorized",
        "heldout_or_sealed_access_authorized",
    ]
    for key in denied:
        if scope.get(key) is not False:
            raise ValueError(f"receipt must deny {key}")
    if scope.get("cpu_only_freeze") is not True:
        raise ValueError("receipt must be CPU-only")
    if scope.get("raw_bfcl_data_committed") is not False:
        raise ValueError("receipt must not commit raw BFCL data")
    selected_hash = _sha256(SELECTED_IDS)
    if selected_hash != receipt["task_selection"]["selected_task_ids_sha256"]:
        raise ValueError("selected task-id hash mismatch")
    selected = SELECTED_IDS.read_text(encoding="utf-8").splitlines()
    if len(selected) != receipt["task_selection"]["task_count"]:
        raise ValueError("selected task-id count mismatch")
    if selected[0] != receipt["task_selection"]["first_task_id"]:
        raise ValueError("first selected task id mismatch")
    if selected[-1] != receipt["task_selection"]["last_task_id"]:
        raise ValueError("last selected task id mismatch")
    if _sha256(PUBLIC_MANIFEST) != receipt["public_data_identity"]["file_manifest_sha256"]:
        raise ValueError("public file manifest hash mismatch")
    if _sha256(SCHEMA_SUMMARY) != receipt["public_data_identity"]["schema_summary_sha256"]:
        raise ValueError("schema summary hash mismatch")
    rollout = receipt["rollout_design_when_separately_authorized"]
    if rollout["group_size_per_task_per_model"] != 32:
        raise ValueError("group size must remain 32")
    if rollout["retries"] != 0 or receipt["finalization_rules"]["zero_retry"] is not True:
        raise ValueError("zero-retry policy mismatch")
    return receipt


def _normalize_call(call: dict[str, Any]) -> dict[str, Any]:
    args = call.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"__raw_arguments__": args}
    return {"name": call.get("name"), "arguments": args}


def strict_success(expected: list[dict[str, Any]], observed: list[dict[str, Any]]) -> bool:
    return [_normalize_call(call) for call in observed] == [_normalize_call(call) for call in expected]


def synthetic_cases() -> list[dict[str, Any]]:
    expected = [
        {"name": "lookup_order", "arguments": {"order_id": "SYN-001"}},
        {"name": "refund_order", "arguments": {"order_id": "SYN-001", "amount": 17}},
    ]
    return [
        {
            "case_id": "synthetic_pass",
            "task_id": "multi_turn_base_0",
            "model_slot": "synthetic",
            "rollout_index": 0,
            "expected_calls": expected,
            "observed_calls": copy.deepcopy(expected),
        },
        {
            "case_id": "synthetic_wrong_tool",
            "task_id": "multi_turn_base_0",
            "model_slot": "synthetic",
            "rollout_index": 1,
            "expected_calls": expected,
            "observed_calls": [
                {"name": "lookup_order", "arguments": {"order_id": "SYN-001"}},
                {"name": "cancel_order", "arguments": {"order_id": "SYN-001", "amount": 17}},
            ],
        },
        {
            "case_id": "synthetic_wrong_argument",
            "task_id": "multi_turn_base_0",
            "model_slot": "synthetic",
            "rollout_index": 2,
            "expected_calls": expected,
            "observed_calls": [
                {"name": "lookup_order", "arguments": {"order_id": "SYN-001"}},
                {"name": "refund_order", "arguments": {"order_id": "SYN-002", "amount": 17}},
            ],
        },
        {
            "case_id": "synthetic_string_arguments",
            "task_id": "multi_turn_base_1",
            "model_slot": "synthetic",
            "rollout_index": 0,
            "expected_calls": expected,
            "observed_calls": [
                {"name": "lookup_order", "arguments": "{\"order_id\":\"SYN-001\"}"},
                {"name": "refund_order", "arguments": "{\"amount\":17,\"order_id\":\"SYN-001\"}"},
            ],
        },
        {
            "case_id": "synthetic_empty_observed",
            "task_id": "multi_turn_base_2",
            "model_slot": "synthetic",
            "rollout_index": 0,
            "expected_calls": expected,
            "observed_calls": [],
        },
    ]


def evaluate_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        success = strict_success(case["expected_calls"], case["observed_calls"])
        rows.append(
            {
                "case_id": case["case_id"],
                "task_id": case["task_id"],
                "model_slot": case["model_slot"],
                "rollout_index": case["rollout_index"],
                "strict_success": success,
            }
        )
    return rows


def summarize_resolution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[bool]] = {}
    for row in rows:
        grouped.setdefault((row["model_slot"], row["task_id"]), []).append(bool(row["strict_success"]))
    summaries = []
    for (model_slot, task_id), values in sorted(grouped.items()):
        success_count = sum(values)
        group_size = len(values)
        summaries.append(
            {
                "model_slot": model_slot,
                "task_id": task_id,
                "group_size": group_size,
                "strict_success_count": success_count,
                "all_fail": success_count == 0,
                "all_pass": success_count == group_size,
                "mixed_reward": 0 < success_count < group_size,
                "non_zero_advantage": 0 < success_count < group_size,
            }
        )
    return {
        "group_count": len(summaries),
        "mixed_group_count": sum(1 for row in summaries if row["mixed_reward"]),
        "all_fail_group_count": sum(1 for row in summaries if row["all_fail"]),
        "all_pass_group_count": sum(1 for row in summaries if row["all_pass"]),
        "non_zero_advantage_group_count": sum(1 for row in summaries if row["non_zero_advantage"]),
        "groups": summaries,
    }


def run_synthetic_replay() -> dict[str, Any]:
    receipt = validate_static_receipt()
    cases = synthetic_cases()
    rows = evaluate_cases(cases)
    return {
        "protocol": "RRC-BFCL-SYNTHETIC-REPLAY-v1",
        "status": "PASS",
        "cpu_only": True,
        "model_inference_used": False,
        "api_used": False,
        "gpu_used": False,
        "training_used": False,
        "heldout_or_sealed_access_used": False,
        "receipt_sha256": _sha256(RECEIPT),
        "selected_task_count": receipt["task_selection"]["task_count"],
        "case_count": len(cases),
        "evaluation_rows": rows,
        "resolution_summary": summarize_resolution(rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=AUDIT_DIR / "SYNTHETIC_REPLAY_RESULT.json")
    args = parser.parse_args()
    result = run_synthetic_replay()
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

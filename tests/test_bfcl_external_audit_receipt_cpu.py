from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUDIT = (
    ROOT
    / "research/reward_identifiability_supervision_topology/negative_result/external_audit/bfcl_v3_base_multiturn"
)


def _load_replay_module():
    path = AUDIT / "bfcl_synthetic_replay.py"
    spec = importlib.util.spec_from_file_location("bfcl_synthetic_replay_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_bfcl_execution_receipt_is_cpu_only_and_no_inference():
    receipt = json.loads((AUDIT / "EXECUTION_RECEIPT_STATIC.json").read_text(encoding="utf-8"))
    assert receipt["protocol"] == "RRC-EXTERNAL-AUDIT-BFCL-V3-BASE-MULTITURN-EXECUTION-RECEIPT-v1"
    assert receipt["status"] == "FROZEN_CPU_ONLY_NO_INFERENCE_AUTHORIZED"
    assert receipt["scope"]["cpu_only_freeze"] is True
    assert receipt["scope"]["model_inference_authorized"] is False
    assert receipt["scope"]["api_inference_authorized"] is False
    assert receipt["scope"]["gpu_authorized"] is False
    assert receipt["scope"]["training_authorized"] is False
    assert receipt["scope"]["heldout_or_sealed_access_authorized"] is False
    assert receipt["scope"]["raw_bfcl_data_committed"] is False


def test_bfcl_execution_receipt_freezes_selection_and_rollout_policy():
    receipt = json.loads((AUDIT / "EXECUTION_RECEIPT_STATIC.json").read_text(encoding="utf-8"))
    selected = (AUDIT / "SELECTED_TASK_IDS.txt").read_text(encoding="utf-8").splitlines()
    assert receipt["task_selection"]["task_count"] == 64
    assert selected == [f"multi_turn_base_{idx}" for idx in range(64)]
    assert receipt["task_selection"]["selected_task_ids_sha256"] == hashlib.sha256(
        (AUDIT / "SELECTED_TASK_IDS.txt").read_bytes()
    ).hexdigest()
    rollout = receipt["rollout_design_when_separately_authorized"]
    assert rollout["group_size_per_task_per_model"] == 32
    assert rollout["temperature"] == 0.7
    assert rollout["top_p"] == 0.95
    assert rollout["retries"] == 0
    assert receipt["finalization_rules"]["zero_retry"] is True
    assert receipt["finalization_rules"]["no_selective_reruns"] is True
    assert receipt["finalization_rules"]["no_protocol_edit_after_first_model_request"] is True


def test_bfcl_execution_receipt_hash_file_matches_static_receipt():
    expected = (AUDIT / "EXECUTION_RECEIPT_STATIC.sha256").read_text(encoding="ascii").split()[0]
    actual = hashlib.sha256((AUDIT / "EXECUTION_RECEIPT_STATIC.json").read_bytes()).hexdigest()
    assert expected == actual


def test_bfcl_synthetic_replay_is_cpu_only_and_exercises_finalizer():
    replay = _load_replay_module()
    result = replay.run_synthetic_replay()
    assert result["status"] == "PASS"
    assert result["cpu_only"] is True
    assert result["model_inference_used"] is False
    assert result["api_used"] is False
    assert result["gpu_used"] is False
    assert result["training_used"] is False
    assert result["heldout_or_sealed_access_used"] is False
    summary = result["resolution_summary"]
    assert summary["mixed_group_count"] == 1
    assert summary["all_pass_group_count"] == 1
    assert summary["all_fail_group_count"] == 1
    assert summary["non_zero_advantage_group_count"] == 1


def test_bfcl_synthetic_evaluator_rejects_wrong_tool_and_argument():
    replay = _load_replay_module()
    expected = [{"name": "lookup_order", "arguments": {"order_id": "SYN-001"}}]
    assert replay.strict_success(expected, [{"name": "lookup_order", "arguments": {"order_id": "SYN-001"}}])
    assert not replay.strict_success(expected, [{"name": "cancel_order", "arguments": {"order_id": "SYN-001"}}])
    assert not replay.strict_success(expected, [{"name": "lookup_order", "arguments": {"order_id": "SYN-002"}}])
    assert replay.strict_success(expected, [{"name": "lookup_order", "arguments": "{\"order_id\":\"SYN-001\"}"}])


def test_bfcl_synthetic_replay_fails_fast_on_receipt_hash_mismatch(tmp_path, monkeypatch):
    replay = _load_replay_module()
    bad_receipt = tmp_path / "EXECUTION_RECEIPT_STATIC.json"
    receipt = json.loads((AUDIT / "EXECUTION_RECEIPT_STATIC.json").read_text(encoding="utf-8"))
    receipt["scope"]["gpu_authorized"] = True
    bad_receipt.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(replay, "RECEIPT_SHA", AUDIT / "EXECUTION_RECEIPT_STATIC.sha256")
    with pytest.raises(ValueError, match="receipt hash mismatch"):
        replay.validate_static_receipt(bad_receipt)


def test_bfcl_minimal_canary_template_is_single_request_and_not_executable():
    template = json.loads((AUDIT / "MINIMAL_CANARY_RUNTIME_RECEIPT_TEMPLATE.json").read_text(encoding="utf-8"))
    assert template["protocol"] == "RRC-BFCL-V3-BASE-MT-MINIMAL-INFERENCE-CANARY-RUNTIME-RECEIPT-TEMPLATE-v1"
    assert template["status"] == "TEMPLATE_FROZEN_NOT_EXECUTABLE_UNTIL_RUNTIME_BINDINGS_FILLED_AND_SEPARATELY_AUTHORIZED"
    scope = template["scope"]
    assert scope["canary_only"] is True
    assert scope["full_external_audit_authorized"] is False
    assert scope["training_authorized"] is False
    assert scope["heldout_or_sealed_access_authorized"] is False
    assert scope["task_count"] == 1
    assert scope["model_count"] == 1
    assert scope["rollouts_per_task"] == 1
    assert scope["retries"] == 0
    assert template["selected_task"]["task_id"] == "multi_turn_base_0"
    assert template["request_policy"]["zero_retry"] is True
    assert template["request_policy"]["single_model_request_only"] is True
    assert template["request_policy"]["terminal_finalizer_required_even_on_parse_error"] is True
    assert template["request_policy"]["no_protocol_edit_after_first_model_request"] is True
    required_fields = template["model_slot"]["runtime_required_fields"]
    assert required_fields
    assert all(value.startswith("UNBOUND") for value in required_fields.values())
    assert template["decoding"]["max_output"].startswith("UNBOUND")
    assert template["decoding"]["tool_call_format"].startswith("UNBOUND")
    assert template["decoding"]["request_timeout_seconds"].startswith("UNBOUND")


def test_bfcl_minimal_canary_template_hash_file_matches():
    expected = (AUDIT / "MINIMAL_CANARY_RUNTIME_RECEIPT_TEMPLATE.sha256").read_text(encoding="ascii").split()[0]
    actual = hashlib.sha256((AUDIT / "MINIMAL_CANARY_RUNTIME_RECEIPT_TEMPLATE.json").read_bytes()).hexdigest()
    assert expected == actual


def test_bfcl_local_preflight_blocked_before_first_model_request():
    report = json.loads((AUDIT / "MINIMAL_CANARY_LOCAL_PREFLIGHT_BLOCKED.json").read_text(encoding="utf-8"))
    assert report["protocol"] == "RRC-BFCL-MINIMAL-CANARY-LOCAL-PREFLIGHT-v1"
    assert report["status"] == "BLOCKED_BEFORE_FIRST_MODEL_REQUEST"
    scope = report["scope_confirmed"]
    assert scope["model_request_sent"] is False
    assert scope["api_request_sent"] is False
    assert scope["gpu_model_serving_started"] is False
    assert scope["training_started"] is False
    assert scope["heldout_or_sealed_accessed"] is False
    assert scope["raw_bfcl_committed"] is False
    assert report["template"]["sha256"] == hashlib.sha256(
        (AUDIT / "MINIMAL_CANARY_RUNTIME_RECEIPT_TEMPLATE.json").read_bytes()
    ).hexdigest()
    assert report["decision"].startswith("Do not send a model request")


def test_bfcl_minimal_canary_terminal_result_keeps_full_audit_closed():
    finalizer = json.loads((AUDIT / "MINIMAL_CANARY_TERMINAL_FINALIZER.json").read_text(encoding="utf-8"))
    assert finalizer["protocol"] == "RRC-BFCL-MINIMAL-CANARY-TERMINAL-FINALIZER-v1"
    assert finalizer["status"] == "TERMINAL_INTERPRETABLE"
    assert finalizer["model_request_sent"] is True
    assert finalizer["model_request_count"] == 1
    assert finalizer["retry_count"] == 0
    assert finalizer["task_id"] == "multi_turn_base_0"
    assert finalizer["model_repo_id"] == "Qwen/Qwen3-0.6B"
    assert finalizer["model_revision"] == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert finalizer["strict_success"] is False
    assert finalizer["failure_mode"] == "PARSE_FAILURE"
    assert finalizer["parseable_tool_calls"] is False
    assert finalizer["observed_call_count"] == 0
    assert finalizer["expected_call_count"] == 3
    assert finalizer["go_to_two_model_canary"] is False
    assert finalizer["full_audit_authorized"] is False
    assert finalizer["training_authorized"] is False
    assert finalizer["heldout_or_sealed_accessed"] is False


def test_bfcl_minimal_canary_bound_receipt_matches_terminal_hash():
    receipt = json.loads((AUDIT / "MINIMAL_CANARY_RUNTIME_RECEIPT_BOUND.json").read_text(encoding="utf-8"))
    finalizer = json.loads((AUDIT / "MINIMAL_CANARY_TERMINAL_FINALIZER.json").read_text(encoding="utf-8"))
    assert receipt["protocol"] == "RRC-BFCL-V3-BASE-MT-MINIMAL-INFERENCE-CANARY-RUNTIME-RECEIPT-BOUND-v1"
    assert receipt["source_commit"] == "48cacb973cd4d29c6cf39db221e1ba7e7cf27006"
    assert receipt["model_request_budget"] == 1
    assert receipt["model_request_count_before_run"] == 0
    assert receipt["request_policy"]["single_model_request_only"] is True
    assert receipt["request_policy"]["zero_retry"] is True
    assert receipt["serving_backend"] == "direct_transformers_generate_single_call"
    assert receipt["template_sha256"] == hashlib.sha256(
        (AUDIT / "MINIMAL_CANARY_RUNTIME_RECEIPT_TEMPLATE.json").read_bytes()
    ).hexdigest()
    assert finalizer["runtime_receipt_sha256"] == hashlib.sha256(
        (AUDIT / "MINIMAL_CANARY_RUNTIME_RECEIPT_BOUND.json").read_bytes()
    ).hexdigest()

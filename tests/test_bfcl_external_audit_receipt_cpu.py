from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = (
    ROOT
    / "research/reward_identifiability_supervision_topology/negative_result/external_audit/bfcl_v3_base_multiturn"
)


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

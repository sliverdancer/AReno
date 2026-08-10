from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANARY = (
    ROOT
    / "research/reward_identifiability_supervision_topology/negative_result/external_audit"
    / "tau3_airline_parseability_canary"
)


def _load_builder():
    path = CANARY / "build_canary_freeze.py"
    spec = importlib.util.spec_from_file_location("tau3_parseability_canary_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_finalizer():
    path = CANARY / "tau3_canary_finalizer.py"
    spec = importlib.util.spec_from_file_location("tau3_parseability_finalizer_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_binder():
    path = CANARY / "bind_runtime_receipt.py"
    spec = importlib.util.spec_from_file_location("tau3_parseability_binder_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tau3_parseability_template_is_cpu_only_and_unbound():
    template = json.loads((CANARY / "TAU3_PARSEABILITY_CANARY_TEMPLATE.json").read_text(encoding="utf-8"))
    assert template["protocol"] == "RRC-TAU3-AIRLINE-PARSEABILITY-CANARY-v1"
    assert template["status"] == "CPU_FROZEN_AWAITING_SEPARATE_SINGLE_REQUEST_AUTHORIZATION"
    scope = template["scope"]
    assert scope["external_public_environment_candidate"] == "Tau3/Tau2 airline tasks"
    assert scope["tau3_tag"] == "v1.0.1"
    assert scope["tau3_commit"] == "fc0055dc4e0a316c3f83133267fbd6faaa770992"
    assert scope["task_count"] == 1
    assert scope["model_count"] == 1
    assert scope["rollouts_per_task"] == 1
    assert scope["retries"] == 0
    assert scope["training_authorized"] is False
    assert scope["bfcl_used"] is False
    assert scope["heldout_or_sealed_access_authorized"] is False
    assert scope["model_inference_authorized"] is False
    assert scope["api_inference_authorized"] is False
    assert scope["gpu_authorized"] is False
    assert all(value.startswith("UNBOUND") for value in template["runtime_required_bindings"].values())


def test_tau3_parseability_template_hash_matches():
    expected = (CANARY / "TAU3_PARSEABILITY_CANARY_TEMPLATE.sha256").read_text(encoding="ascii").split()[0]
    actual = hashlib.sha256((CANARY / "TAU3_PARSEABILITY_CANARY_TEMPLATE.json").read_bytes()).hexdigest()
    assert expected == actual


def test_tau3_cpu_parser_replay_passes_and_rejects_bad_arguments():
    template = json.loads((CANARY / "TAU3_PARSEABILITY_CANARY_TEMPLATE.json").read_text(encoding="utf-8"))
    replay = template["cpu_parser_replay"]
    assert replay["status"] == "PASS"
    assert replay["cpu_only"] is True
    assert replay["model_inference_used"] is False
    assert replay["api_used"] is False
    assert replay["gpu_used"] is False
    assert replay["training_used"] is False
    assert replay["heldout_or_sealed_access_used"] is False
    assert replay["bfcl_used"] is False
    assert replay["parsed_action"]["name"] == "DB"
    assert isinstance(replay["parsed_action"]["arguments"], dict)
    assert replay["assistant_has_tool_calls"] is True
    assert replay["malformed_non_object_arguments_rejected"] is True


def test_tau3_canary_builder_reproduces_tracked_template_except_self_hash_file():
    builder = _load_builder()
    rebuilt = builder.build_template(builder.run_cpu_parser_replay())
    tracked = json.loads((CANARY / "TAU3_PARSEABILITY_CANARY_TEMPLATE.json").read_text(encoding="utf-8"))
    assert rebuilt == tracked


def test_tau3_finalizer_replay_covers_pass_and_parse_failure():
    replay = json.loads((CANARY / "TAU3_CANARY_FINALIZER_REPLAY.json").read_text(encoding="utf-8"))
    assert replay["protocol"] == "RRC-TAU3-AIRLINE-PARSEABILITY-CANARY-FINALIZER-REPLAY-v1"
    assert replay["status"] == "PASS"
    assert replay["cpu_only"] is True
    assert replay["model_inference_used"] is False
    assert replay["api_used"] is False
    assert replay["gpu_used"] is False
    assert replay["training_used"] is False
    assert replay["heldout_or_sealed_access_used"] is False
    assert replay["bfcl_used"] is False
    passed = replay["pass_finalizer"]
    failed = replay["parse_failure_finalizer"]
    assert passed["status"] == "PASS_PARSEABLE_TOOL_CALL"
    assert passed["success_gate_passed"] is True
    assert passed["parseable_tool_calls"] is True
    assert passed["go_to_reward_resolution_calibration"] is True
    assert failed["status"] == "TERMINAL_PARSE_FAILURE"
    assert failed["success_gate_passed"] is False
    assert failed["parseable_tool_calls"] is False
    assert failed["go_to_reward_resolution_calibration"] is False
    for finalizer in (passed, failed):
        assert finalizer["strict_task_success_required"] is False
        assert finalizer["reward_resolution_claim_allowed"] is False
        assert finalizer["training_authorized"] is False
        assert finalizer["bfcl_used"] is False
        assert finalizer["heldout_or_sealed_accessed"] is False


def test_tau3_finalizer_rejects_unauthorized_training_and_extra_requests():
    finalizer = _load_finalizer()
    receipt = finalizer.build_synthetic_receipt()
    observation = {
        "protocol": "RRC-TAU3-AIRLINE-PARSEABILITY-CANARY-OBSERVATION-v1",
        "runtime_receipt_sha256": "c" * 64,
        "task_id": receipt["task_id"],
        "model_repo_or_api_id": receipt["model_repo_or_api_id"],
        "model_request_count": 1,
        "retry_count": 0,
        "observed_tool_calls": [{"name": "DB", "arguments": {"query": "SELECT 1"}}],
    }
    receipt["training_authorized"] = True
    try:
        finalizer.finalize_canary(receipt, observation)
    except ValueError as exc:
        assert "training" in str(exc)
    else:
        raise AssertionError("training-authorized receipt should be rejected")

    receipt = finalizer.build_synthetic_receipt()
    observation["model_request_count"] = 2
    try:
        finalizer.finalize_canary(receipt, observation)
    except ValueError as exc:
        assert "exactly one model request" in str(exc)
    else:
        raise AssertionError("multi-request observation should be rejected")


def test_tau3_runtime_receipt_template_is_not_executable():
    template = json.loads((CANARY / "TAU3_RUNTIME_RECEIPT_TEMPLATE.json").read_text(encoding="utf-8"))
    finalizer = _load_finalizer()
    assert template["protocol"] == "RRC-TAU3-AIRLINE-PARSEABILITY-CANARY-RUNTIME-RECEIPT-v1"
    assert template["status"] == "TEMPLATE_UNBOUND_NOT_EXECUTABLE"
    assert template["model_request_budget"] == 1
    assert template["retry_budget"] == 0
    assert template["training_authorized"] is False
    assert template["bfcl_used"] is False
    assert template["heldout_or_sealed_accessed"] is False
    assert template["raw_response_committed"] is False
    assert template["pre_request_exit_if_any_unbound"] is True
    assert template["template_sha256"] == hashlib.sha256(
        (CANARY / "TAU3_PARSEABILITY_CANARY_TEMPLATE.json").read_bytes()
    ).hexdigest()
    try:
        finalizer.validate_runtime_receipt(template)
    except ValueError as exc:
        assert "source_commit" in str(exc)
    else:
        raise AssertionError("unbound runtime receipt template should not validate")


def test_tau3_observation_schema_preserves_single_request_boundary():
    schema = json.loads((CANARY / "TAU3_OBSERVATION_SCHEMA.json").read_text(encoding="utf-8"))
    assert schema["protocol"] == "RRC-TAU3-AIRLINE-PARSEABILITY-CANARY-OBSERVATION-SCHEMA-v1"
    assert schema["status"] == "SCHEMA_ONLY_NO_MODEL_RESPONSE_INCLUDED"
    required = schema["required_fields"]
    assert required["protocol"] == "RRC-TAU3-AIRLINE-PARSEABILITY-CANARY-OBSERVATION-v1"
    assert required["model_request_count"] == 1
    assert required["retry_count"] == 0
    assert "raw model response text" in schema["forbidden"]
    assert "additional model requests" in schema["forbidden"]
    assert "reward-resolution claim" in schema["forbidden"]


def test_tau3_runtime_binder_produces_valid_bound_receipt(tmp_path):
    binder = _load_binder()
    finalizer = _load_finalizer()
    receipt = binder.build_bound_receipt(
        source_commit="1" * 40,
        model_repo_or_api_id="Qwen/Qwen3-0.6B",
        model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        user_simulator_model="tau3/synthetic-user-simulator-placeholder",
        user_simulator_revision="tau3-v1.0.1",
        user_simulator_authorization_sha256="2" * 64,
        gpu_uuid_or_api_provider="GPU-synthetic-preflight",
        task_id="airline:synthetic-public-task",
    )
    assert receipt["status"] == "BOUND_READY_FOR_SINGLE_REQUEST_CANARY"
    assert receipt["model_request_budget"] == 1
    assert receipt["retry_budget"] == 0
    assert receipt["training_authorized"] is False
    assert receipt["bfcl_used"] is False
    assert receipt["heldout_or_sealed_accessed"] is False
    finalizer.validate_runtime_receipt(receipt)

    output = tmp_path / "RUNTIME_RECEIPT_BOUND.json"
    digest = binder.write_bound_receipt(receipt, output)
    assert output.exists()
    assert output.with_suffix(output.suffix + ".sha256").exists()
    expected = output.with_suffix(output.suffix + ".sha256").read_text(encoding="ascii").split()[0]
    assert expected == digest


def test_tau3_runtime_binder_rejects_unbound_or_malformed_identity():
    binder = _load_binder()
    kwargs = dict(
        source_commit="1" * 40,
        model_repo_or_api_id="Qwen/Qwen3-0.6B",
        model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        user_simulator_model="tau3/synthetic-user-simulator-placeholder",
        user_simulator_revision="tau3-v1.0.1",
        user_simulator_authorization_sha256="2" * 64,
        gpu_uuid_or_api_provider="GPU-synthetic-preflight",
        task_id="airline:synthetic-public-task",
    )
    bad = dict(kwargs)
    bad["model_revision"] = "UNBOUND_AT_RUNTIME"
    try:
        binder.build_bound_receipt(**bad)
    except ValueError as exc:
        assert "model_revision" in str(exc)
    else:
        raise AssertionError("unbound model revision should be rejected")

    bad = dict(kwargs)
    bad["source_commit"] = "not-a-commit"
    try:
        binder.build_bound_receipt(**bad)
    except ValueError as exc:
        assert "source_commit" in str(exc)
    else:
        raise AssertionError("malformed source commit should be rejected")

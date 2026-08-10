from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROL = (
    ROOT
    / "research/reward_identifiability_supervision_topology/negative_result/external_audit"
    / "agentic_tictactoe_positive_control"
)


def _load_builder():
    path = CONTROL / "build_positive_control.py"
    spec = importlib.util.spec_from_file_location("agentic_tictactoe_positive_control_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_agentic_positive_control_is_cpu_only_and_public_repo_native():
    result = json.loads((CONTROL / "POSITIVE_CONTROL_RESULT.json").read_text(encoding="utf-8"))
    assert result["protocol"] == "RRC-AGENTIC-TICTACTOE-PARSEABLE-POSITIVE-CONTROL-v1"
    assert result["status"] == "PASS_CPU_ONLY_PARSEABLE_TOOL_CALL_POSITIVE_CONTROL"
    scope = result["scope"]
    assert scope["uses_public_repo_native_example"] is True
    assert scope["bfcl_used"] is False
    assert scope["heldout_or_sealed_access_used"] is False
    assert scope["model_inference_used"] is False
    assert scope["api_used"] is False
    assert scope["gpu_used"] is False
    assert scope["training_used"] is False
    assert scope["raw_model_responses_included"] is False


def test_agentic_positive_control_has_parseable_resolution_contrast():
    result = json.loads((CONTROL / "POSITIVE_CONTROL_RESULT.json").read_text(encoding="utf-8"))
    summary = result["resolution_summary"]
    assert summary["group_count"] == 4
    assert summary["rollout_count"] == 16
    assert summary["parseable_tool_call_rate"] == 1.0
    assert summary["mixed_group_count"] >= 2
    assert summary["all_pass_group_count"] >= 1
    assert summary["all_fail_group_count"] >= 1
    assert summary["non_zero_advantage_group_count"] == summary["mixed_group_count"]
    assert summary["high_low_contrast_present"] is True

    categories = {group["group_id"]: group["observed_category"] for group in result["groups"]}
    assert categories["ttt_public_all_pass_win"] == "all_pass"
    assert categories["ttt_public_mixed_win_or_occupied"] == "mixed"
    assert categories["ttt_public_mixed_win_or_nonterminal_best"] == "mixed"
    assert categories["ttt_public_all_fail_occupied"] == "all_fail"


def test_agentic_positive_control_builder_reproduces_tracked_result():
    builder = _load_builder()
    rebuilt = builder.build_result()
    tracked = json.loads((CONTROL / "POSITIVE_CONTROL_RESULT.json").read_text(encoding="utf-8"))
    assert rebuilt == tracked


def test_agentic_positive_control_parser_rejects_prose_and_wrong_shape():
    builder = _load_builder()
    assert builder.parse_tool_calls("choose square 3") == []
    assert builder.parse_tool_calls("<tool_call>{\"name\":\"choose_square\"}</tool_call>") == []
    parsed = builder.parse_tool_calls(
        '<tool_call>{"name":"choose_square","arguments":{"square":3}}</tool_call>'
    )
    assert parsed == [{"name": "choose_square", "arguments": {"square": 3}}]

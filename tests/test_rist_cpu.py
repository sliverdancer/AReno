from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = REPO_ROOT / "research" / "reward_identifiability_supervision_topology"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_rist_factorial_is_balanced_disjoint_and_reproducible(tmp_path):
    generator = _load_module("rist_task_generator", RESEARCH_DIR / "task_generator.py")

    first = generator.generate_splits(seed=17)
    repeated = generator.generate_splits(seed=17)

    assert first == repeated
    assert {split: len(rows) for split, rows in first.items()} == {
        "train": 96,
        "qualification": 32,
        "heldout": 32,
    }
    for split, rows in first.items():
        counts = {}
        for row in rows:
            counts[row["factor_cell"]] = counts.get(row["factor_cell"], 0) + 1
        assert len(counts) == 32
        assert set(counts.values()) == {generator.SPLIT_REPLICATES[split]}
    signatures = {
        split: {row["task_signature"] for row in rows}
        for split, rows in first.items()
    }
    assert signatures["train"].isdisjoint(signatures["qualification"])
    assert signatures["train"].isdisjoint(signatures["heldout"])
    assert signatures["qualification"].isdisjoint(signatures["heldout"])

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    generator.write_splits(first_dir, seed=17)
    generator.write_splits(second_dir, seed=17)
    for filename in ("train.jsonl", "qualification.jsonl", "heldout.jsonl", "manifest.json"):
        assert (first_dir / filename).read_bytes() == (second_dir / filename).read_bytes()


def test_rist_analytic_measure_separates_action_and_reward_diversity():
    generator = _load_module("rist_task_generator_metrics", RESEARCH_DIR / "task_generator.py")
    rows = generator.generate_splits(seed=19)["qualification"]

    assert {row["reward_resolution_stratum"] for row in rows} == {
        "low",
        "intermediate",
        "high",
    }
    assert any(
        row["possible_action_sequences"] >= 256
        and row["mixed_group_probability_g8"] < 0.2
        for row in rows
    )
    for row in rows:
        assert generator.strict_reward(row, row["oracle_actions"]) == 1
        assert generator.strict_reward(row, row["oracle_actions"][:-1]) == 0


def test_rist_train_and_qualification_evaluators_pass(tmp_path):
    generator = _load_module("task_generator", RESEARCH_DIR / "task_generator.py")
    evaluator = _load_module("rist_evaluate_tasks", RESEARCH_DIR / "evaluate_tasks.py")
    generator.write_splits(tmp_path, seed=23)

    train = evaluator.evaluate(tmp_path, "train")
    qualification = evaluator.evaluate(tmp_path, "qualification")

    assert train["passed"] is True
    assert train["score"] == train["max_score"] == 8
    assert qualification["passed"] is True
    assert qualification["score"] == qualification["max_score"] == 8


def test_rist_stage_hook_keeps_p1_diagnostic_and_kills_failed_stages():
    hook = _load_module(
        "rist_stage_completion_hook",
        RESEARCH_DIR / "stage_completion_hook.py",
    )

    passed = hook.assess(
        {
            "stage": "P1",
            "stage_status": "PASS",
            "decision": "PASS_P1_CPU_FREEZE_TO_GPU_AUTHORIZATION_REQUEST",
        }
    )
    killed = hook.assess(
        {"stage": "P2", "stage_status": "KILL", "decision": "NO_SIGNAL"}
    )

    assert passed["decision"] == "STAY_DIAGNOSTIC_OPEN_P2_GPU_AUTHORIZATION"
    assert passed["upgraded"] is False
    assert killed["decision"] == "KILL_CURRENT_ROUTE"
    assert killed["upgraded"] is False


def test_rist_p2_parser_rejects_missing_or_malformed_calls_without_repair():
    client = _load_module("rist_run_p2_client", RESEARCH_DIR / "run_p2_client.py")
    offered = {"scan_registry"}

    missing = client.parse_tool_call({"role": "assistant"}, offered)
    malformed = client.parse_tool_call(
        {
            "tool_calls": [
                {
                    "id": "x",
                    "function": {
                        "name": "scan_registry",
                        "arguments": "{}",
                    },
                }
            ]
        },
        offered,
    )
    multiple = client.parse_tool_call(
        {
            "tool_calls": [
                {
                    "id": "x",
                    "function": {
                        "name": "scan_registry",
                        "arguments": '{"code":"r0-abc"}',
                    },
                },
                {
                    "id": "y",
                    "function": {
                        "name": "scan_registry",
                        "arguments": '{"code":"r0-def"}',
                    },
                },
            ]
        },
        offered,
    )
    unlisted = client.parse_tool_call(
        {
            "tool_calls": [
                {
                    "id": "x",
                    "function": {
                        "name": "unknown_tool",
                        "arguments": '{"code":"r0-abc"}',
                    },
                }
            ]
        },
        offered,
    )
    valid = client.parse_tool_call(
        {
            "tool_calls": [
                {
                    "id": "x",
                    "function": {
                        "name": "scan_registry",
                        "arguments": '{"code":"r0-abc"}',
                    },
                }
            ]
        },
        offered,
    )

    assert missing == {"valid": False, "reason": "MISSING_TOOL_CALL", "call": None}
    assert malformed["reason"] == "INVALID_ARGUMENT_SCHEMA"
    assert multiple["reason"] == "MULTIPLE_TOOL_CALLS"
    assert unlisted["reason"] == "UNLISTED_TOOL"
    assert valid["valid"] is True
    assert valid["call"]["arguments"] == {"code": "r0-abc"}


def test_rist_p2_forced_cells_hide_distractor_tools():
    client = _load_module("rist_run_p2_client_topology", RESEARCH_DIR / "run_p2_client.py")
    turn = {
        "expected_tool": "scan_registry",
        "offered_tools": ["scan_registry", "delete_registry", "export_registry"],
    }

    forced = client._offered_tool_names(
        {"factors": {"tool_choice_mode": "forced"}},
        turn,
    )
    free = client._offered_tool_names(
        {"factors": {"tool_choice_mode": "free"}},
        turn,
    )

    assert forced == ["scan_registry"]
    assert free == turn["offered_tools"]


def test_rist_p2_analyzer_passes_only_cross_family_resolution_signal():
    generator = _load_module("rist_task_generator_p2", RESEARCH_DIR / "task_generator.py")
    analyzer = _load_module("rist_analyze_p2", RESEARCH_DIR / "analyze_p2.py")
    tasks = generator.generate_splits(seed=31)["qualification"]
    selected = []
    for stratum in ("low", "intermediate", "high"):
        selected.extend(
            task for task in tasks if task["reward_resolution_stratum"] == stratum
        )
    mixed_signatures = {task["task_signature"] for task in selected[:2]}
    mixed_signatures.update(
        task["task_signature"]
        for task in [
            *[task for task in tasks if task["reward_resolution_stratum"] == "low"][:2],
            *[task for task in tasks if task["reward_resolution_stratum"] == "high"][:2],
            *[task for task in tasks if task["reward_resolution_stratum"] == "intermediate"][:2],
        ]
    )

    def payload(model_cell: str):
        trajectories = []
        for task in tasks:
            for sample_index in range(8):
                mixed = task["task_signature"] in mixed_signatures
                reward = int(mixed and sample_index % 2 == 0)
                actions = [
                    {
                        "name": action["name"],
                        "arguments": {
                            "code": (
                                action["arguments"]["code"]
                                if reward or sample_index == 0
                                else f"wrong-{sample_index}"
                            )
                        },
                    }
                    for action in task["oracle_actions"]
                ]
                trajectories.append(
                    {
                        "task_signature": task["task_signature"],
                        "analytic_stratum": task["reward_resolution_stratum"],
                        "strict_reward": reward,
                        "actions": actions,
                        "first_turn_executable": True,
                        "four_turn_complete": True,
                        "raw_response_count": 4,
                        "fabricated_call_count": 0,
                    }
                )
        return {
            "model_cell": model_cell,
            "expected_trajectories": 256,
            "infrastructure_error": None,
            "trajectories": trajectories,
        }

    result = analyzer.analyze_cross_family(
        [payload("qwen3_0_6b"), payload("gemma4_e2b_it")]
    )

    assert result["stage_status"] == "PASS"
    assert result["decision"] == "PASS_P2_CROSS_FAMILY_RESOLUTION_TO_P3_PILOT"
    assert all(result["cross_family_gates"].values())
    assert all(
        model["mean_empirical_reward_entropy_bits"] > 0.0
        and model["raw_parsed_agreement_rate"] == 1.0
        for model in result["models"]
    )


def test_rist_e0_canary_is_independent_explicit_and_four_turn():
    e0_dir = RESEARCH_DIR / "successors" / "rist_v1_1" / "stages" / "E0"
    runner = _load_module("rist_run_e0_canary", e0_dir / "run_e0_canary.py")
    payload = __import__("json").loads(
        (e0_dir / "canary_tasks.json").read_text(encoding="utf-8")
    )

    runner.validate_canary_tasks(payload)
    assert {task["mode"] for task in payload["tasks"]} == {"forced", "required"}
    assert all(len(task["turns"]) == 4 for task in payload["tasks"])
    assert all(
        "CANARY" in turn["expected_code"]
        for task in payload["tasks"]
        for turn in task["turns"]
    )

    qualification = {
        __import__("json").loads(line)["id"]
        for line in (RESEARCH_DIR / "stages" / "P1" / "data" / "qualification.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    }
    assert {task["id"] for task in payload["tasks"]}.isdisjoint(qualification)


def test_rist_e0_parser_and_cross_model_gate_are_fail_closed():
    e0_dir = RESEARCH_DIR / "successors" / "rist_v1_1" / "stages" / "E0"
    runner = _load_module("rist_run_e0_canary_parser", e0_dir / "run_e0_canary.py")
    validator = _load_module("rist_validate_e0", e0_dir / "validate_e0.py")

    valid = runner.parse_tool_call(
        {
            "tool_calls": [
                {
                    "id": "canary-1",
                    "function": {
                        "name": "probe_catalog",
                        "arguments": '{"code":"CANARY-F1"}',
                    },
                }
            ]
        },
        {"probe_catalog"},
    )
    missing = runner.parse_tool_call({}, {"probe_catalog"})
    assert valid["valid"] is True
    assert valid["call"]["arguments"] == {"code": "CANARY-F1"}
    assert missing["reason"] == "MISSING_TOOL_CALL"

    def payload(model_cell, *, exact=True, infrastructure_error=None):
        return {
            "model_cell": model_cell,
            "infrastructure_error": infrastructure_error,
            "fabricated_call_count": 0,
            "retry_count": 0,
            "trajectories": [
                {
                    "raw_response_count": 4,
                    "records": [
                        {
                            "parse_valid": True,
                            "exact_instruction": bool(exact),
                        }
                        for _ in range(4)
                    ],
                }
                for _ in range(2)
            ],
        }

    passed = validator.validate(
        [payload("qwen3_0_6b"), payload("gemma4_e2b_it")]
    )
    failed = validator.validate(
        [payload("qwen3_0_6b"), payload("gemma4_e2b_it", exact=False)]
    )
    invalid = validator.validate(
        [
            payload("qwen3_0_6b"),
            payload(
                "gemma4_e2b_it",
                infrastructure_error={"error_type": "RuntimeError"},
            ),
        ]
    )
    preflight_invalid = validator.validate([], preflight_passed=False)

    assert passed["decision"] == "PASS_E0_INFRASTRUCTURE_TO_P2_1_PROTOCOL_FREEZE"
    assert failed["decision"] == "FAIL_E0_MODEL_INTERFACE_STOP"
    assert invalid["decision"] == "INVALID_E0_INFRASTRUCTURE_STOP"
    assert preflight_invalid["decision"] == "INVALID_E0_PREFLIGHT_STOP"


def test_rist_e0_collects_mock_raw_calls_without_retry_or_repair(tmp_path, monkeypatch):
    import hashlib
    import json
    import re

    e0_dir = RESEARCH_DIR / "successors" / "rist_v1_1" / "stages" / "E0"
    runner = _load_module("rist_run_e0_canary_collect", e0_dir / "run_e0_canary.py")
    tasks_path = e0_dir / "canary_tasks.json"
    manifest = {
        "protocol": "RIST-E0-v1.1",
        "canary_tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        "models": [
            {"cell": "qwen3_0_6b"},
            {"cell": "gemma4_e2b_it"},
        ],
        "request_seed": 9101,
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 96,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    calls = []

    def fake_post(url, payload, *, api_key, timeout_seconds):
        del url, api_key, timeout_seconds
        instruction = payload["messages"][-1]["content"]
        matched = re.search(r"call function (\w+) with code ([A-Z0-9-]+)", instruction)
        assert matched is not None
        name, code = matched.groups()
        calls.append((name, code, payload["seed"]))
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": f"mock-{len(calls)}",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps({"code": code}),
                                },
                            }
                        ],
                    }
                }
            ]
        }

    monkeypatch.setattr(runner, "_post_json", fake_post)
    output_path = tmp_path / "qwen.json"
    result = runner.collect(
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        model_cell="qwen3_0_6b",
        tasks_path=tasks_path,
        manifest_path=manifest_path,
        output_path=output_path,
        timeout_seconds=1.0,
    )

    assert result["infrastructure_error"] is None
    assert len(result["trajectories"]) == 2
    assert len(calls) == 8
    assert all(
        record["parse_valid"] and record["exact_instruction"]
        for trajectory in result["trajectories"]
        for record in trajectory["records"]
    )
    assert result["fabricated_call_count"] == 0
    assert result["retry_count"] == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["model_cell"] == "qwen3_0_6b"


def test_rist_e0_v1_2_is_a_minimal_registered_extension_revision():
    import hashlib
    import json

    e0_v1_1 = RESEARCH_DIR / "successors" / "rist_v1_1" / "stages" / "E0"
    e0_v1_2 = RESEARCH_DIR / "successors" / "rist_v1_1" / "stages" / "E0_v1_2"
    manifest = json.loads(
        (e0_v1_2 / "EXECUTION_MANIFEST.json").read_text(encoding="utf-8")
    )
    tasks_path = e0_v1_2 / "canary_tasks.json"
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    protocol = (e0_v1_2 / "PROTOCOL.md").read_text(encoding="utf-8")

    assert manifest["protocol"] == "RIST-E0-v1.2"
    assert tasks["protocol"] == "RIST-E0-v1.2"
    assert manifest["parent_terminal_decision"] == "INVALID_E0_PREFLIGHT_STOP"
    assert manifest["correction_scope"] == "REGISTERED_EXTENSION_IMPORT_ONLY"
    assert manifest["canary_tasks_sha256"] == hashlib.sha256(
        tasks_path.read_bytes()
    ).hexdigest()
    assert 'python -c "import areno.accel._areno_accel"' in protocol
    assert 'python -c "import areno_accel"' not in protocol

    for filename in ("run_e0_canary.py", "validate_e0.py"):
        prior = (e0_v1_1 / filename).read_text(encoding="utf-8")
        revised = (e0_v1_2 / filename).read_text(encoding="utf-8")
        assert revised == prior.replace("RIST-E0-v1.1", "RIST-E0-v1.2")


def test_rist_e0_v1_2_canary_and_validator_remain_fail_closed():
    e0_dir = RESEARCH_DIR / "successors" / "rist_v1_1" / "stages" / "E0_v1_2"
    runner = _load_module("rist_run_e0_v1_2", e0_dir / "run_e0_canary.py")
    validator = _load_module("rist_validate_e0_v1_2", e0_dir / "validate_e0.py")
    payload = __import__("json").loads(
        (e0_dir / "canary_tasks.json").read_text(encoding="utf-8")
    )

    runner.validate_canary_tasks(payload)
    assert validator.validate([], preflight_passed=False)["decision"] == (
        "INVALID_E0_PREFLIGHT_STOP"
    )
    assert all(len(task["turns"]) == 4 for task in payload["tasks"])


def test_rist_p2_1_freeze_is_e0_gated_and_resource_feasible():
    import hashlib
    import json

    stage = RESEARCH_DIR / "successors" / "rist_v1_1" / "stages" / "P2_1"
    e0_evidence = (
        RESEARCH_DIR
        / "successors"
        / "rist_v1_1"
        / "stages"
        / "E0_v1_2"
        / "gpu_run_20260802"
        / "evidence"
    )
    manifest = json.loads(
        (stage / "EXECUTION_MANIFEST.json").read_text(encoding="utf-8")
    )

    assert manifest["protocol"] == "RIST-P2.1-v1.0"
    assert manifest["e0_prerequisite"]["decision"] == (
        "PASS_E0_INFRASTRUCTURE_TO_P2_1_PROTOCOL_FREEZE"
    )
    assert manifest["e0_prerequisite"]["stage_result_sha256"] == hashlib.sha256(
        (e0_evidence / "stage_result.json").read_bytes()
    ).hexdigest()
    assert manifest["e0_prerequisite"]["audit_result_sha256"] == hashlib.sha256(
        (e0_evidence / "audit_result.json").read_bytes()
    ).hexdigest()
    assert manifest["serve"]["attn_backend"] == "native"
    assert manifest["serve"]["max_running_prompts"] == 1
    assert manifest["total_requests"] == 2048
    assert manifest["gpu_time_forecast_seconds"] < manifest["gpu_time_limit_seconds"]
    assert manifest["gpu_time_limit_seconds"] == 5 * 60 * 60


def test_rist_p2_1_client_preserves_data_and_no_retry_contract():
    import json

    stage = RESEARCH_DIR / "successors" / "rist_v1_1" / "stages" / "P2_1"
    client = _load_module("rist_run_p2_1_client", stage / "run_p2_1_client.py")
    manifest = json.loads(
        (stage / "EXECUTION_MANIFEST.json").read_text(encoding="utf-8")
    )
    qualification = RESEARCH_DIR / "stages" / "P1" / "data" / "qualification.jsonl"
    payload = json.loads(qualification.read_text(encoding="utf-8").splitlines()[0])

    assert manifest["retry_limit"] == 0
    assert manifest["tasks_per_model"] == 32
    assert len(payload["turns"]) == manifest["turns_per_trajectory"] == 4
    assert client._offered_tool_names(payload, payload["turns"][0]) == (
        [payload["turns"][0]["expected_tool"]]
        if payload["factors"]["tool_choice_mode"] == "forced"
        else payload["turns"][0]["offered_tools"]
    )
    source = (stage / "run_p2_1_client.py").read_text(encoding="utf-8")
    assert "heldout.jsonl" not in source
    assert "retry_count\": 0" in source


def test_rist_p2_1_analyzer_names_terminal_outcomes_fail_closed():
    stage = RESEARCH_DIR / "successors" / "rist_v1_1" / "stages" / "P2_1"
    analyzer = _load_module("rist_analyze_p2_1", stage / "analyze_p2_1.py")
    qualification_sha = (
        "8018137606e12da0f0096ac86f11312d94d631326965198783ebf9cecc94570f"
    )

    def invalid_payload(model_cell):
        return {
            "protocol": "RIST-P2.1-v1.0",
            "model_cell": model_cell,
            "qualification_sha256": qualification_sha,
            "expected_trajectories": 256,
            "retry_count": 0,
            "infrastructure_error": {"error_type": "RuntimeError"},
            "trajectories": [],
        }

    result = analyzer.analyze_cross_family(
        [invalid_payload("qwen3_0_6b"), invalid_payload("gemma4_e2b_it")]
    )
    assert result["protocol"] == "RIST-P2.1-v1.0"
    assert result["stage_status"] == "INVALID"
    assert result["decision"] == "INVALID_P2_1_PREFLIGHT_OR_INFRASTRUCTURE"

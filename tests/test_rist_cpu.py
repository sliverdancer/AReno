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

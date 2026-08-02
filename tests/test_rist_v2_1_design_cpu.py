from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
V2_1 = (
    REPO_ROOT
    / "research"
    / "reward_identifiability_supervision_topology"
    / "successors"
    / "rist_v2_1"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_v2_1_factorial_matrix_has_48_paired_unauthorized_runs():
    design = _load_module(
        "rist_v2_1_design", V2_1 / "stages" / "P3_DESIGN" / "design_matrix.py"
    )
    rows = design.build_matrix()
    design.validate_matrix(rows)

    assert len(rows) == 48
    assert {row["algorithm"] for row in rows} == {"gspo", "grpo"}
    assert {row["family"] for row in rows} == {"qwen3", "gemma4"}
    assert {row["arm"] for row in rows} == {"AF", "LF", "AN", "LN"}
    assert all(row["execution_authorized"] is False for row in rows)
    assert {
        row["content_claim"] for row in rows if row["arm"] in {"AN", "LN"}
    } == {"argument_masked_not_name_only"}
    for row in rows:
        arguments = design.cli_treatment_args(row)
        assert "heldout" not in " ".join(arguments).lower()
        assert ("--mask-tool-call-args" in arguments) == (row["arm"] in {"AN", "LN"})


def test_v2_1_token_grid_is_outcome_blind_and_uses_common_support():
    matching = _load_module(
        "rist_v2_1_token_matching",
        V2_1 / "stages" / "P3_DESIGN" / "token_matching.py",
    )
    curves = {
        arm: [
            {"cumulative_trainable_tokens": 100.0, "strict_success": offset},
            {"cumulative_trainable_tokens": maximum, "strict_success": offset + 0.2},
        ]
        for arm, maximum, offset in (
            ("AF", 1000.0, 0.1),
            ("LF", 700.0, 0.2),
            ("AN", 800.0, 0.3),
            ("LN", 600.0, 0.4),
        )
    }
    result = matching.match_curves(curves)

    assert result["grid_uses_outcomes"] is False
    assert result["grid"][-1] == 600.0
    assert set(result["matched"]) == {"AF", "LF", "AN", "LN"}


def test_v2_1_external_contract_accepts_strict_stateful_transcript():
    contract = _load_module(
        "rist_v2_1_external_contract",
        V2_1 / "stages" / "X0" / "external_env_contract.py",
    )
    before = "a" * 64
    after = "b" * 64
    result = contract.validate_transcript(
        [
            {
                "type": "reset",
                "episode_id": "mock-airline-001",
                "state_hash": before,
                "observation": {"request": "change flight"},
            },
            {
                "type": "step",
                "step_index": 0,
                "action": {"name": "change_flight", "arguments": {"id": "F1"}},
                "state_hash_before": before,
                "state_hash_after": after,
                "reward": 1.0,
                "reward_source": "db_and_communicate",
                "raw_tool_result": {"changed": True},
                "done": True,
            },
        ]
    )
    assert result["valid"] is True
    assert result["llm_judge_used"] is False


def test_v2_1_external_contract_rejects_llm_judge_and_broken_state_chain():
    contract = _load_module(
        "rist_v2_1_external_contract_invalid",
        V2_1 / "stages" / "X0" / "external_env_contract.py",
    )
    events = [
        {
            "type": "reset",
            "episode_id": "mock-001",
            "state_hash": "a" * 64,
            "observation": "start",
        },
        {
            "type": "step",
            "step_index": 0,
            "action": {"name": "tool", "arguments": {}},
            "state_hash_before": "c" * 64,
            "state_hash_after": "b" * 64,
            "reward": 1.0,
            "reward_source": "llm_judge",
            "raw_tool_result": {},
            "done": True,
        },
    ]
    try:
        contract.validate_transcript(events)
    except ValueError as error:
        assert "state_hash_before" in str(error)
    else:
        raise AssertionError("broken state chain must be rejected")


def test_v2_1_train_split_is_balanced_and_disjoint_from_parent_signatures(tmp_path):
    builder = _load_module(
        "rist_v2_1_train_builder", V2_1 / "stages" / "D3" / "build_train_split.py"
    )
    manifest = builder.write_train(tmp_path / "data")
    rows = builder.build_rows()
    parent_signatures = set()
    parent_data = V2_1.parent / "rist_v2" / "stages" / "D2" / "data"
    for split in ("calibration", "qualification"):
        for line in (parent_data / f"{split}.jsonl").read_text().splitlines():
            parent_signatures.add(__import__("json").loads(line)["task_signature"])

    assert manifest["count"] == 32
    assert manifest["heldout_data_opened"] is False
    assert {row["split"] for row in rows} == {"train"}
    assert {row["task_signature"] for row in rows}.isdisjoint(parent_signatures)


def test_v2_1_dataset_loader_rejects_non_train_and_builds_prompt(tmp_path):
    loader = _load_module(
        "rist_v2_1_dataset_loader",
        REPO_ROOT / "examples" / "agentic" / "rist_v2_1" / "dataset_loader.py",
    )
    builder = _load_module(
        "rist_v2_1_train_builder_loader", V2_1 / "stages" / "D3" / "build_train_split.py"
    )
    row = builder.build_rows()[0]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_path = data_dir / "train.jsonl"
    train_path.write_text("unused")
    loaded = loader.load_training_dataset(
        str(train_path), default_loader=lambda _: [row]
    )
    assert loaded[0]["split"] == "train"
    assert "Target label" not in loaded[0]["prompt"]
    assert "target_label=" in loaded[0]["prompt"]

    try:
        loader.load_training_dataset(
            str(data_dir / "qualification.jsonl"), default_loader=lambda _: [row]
        )
    except ValueError as error:
        assert "train.jsonl" in str(error)
    else:
        raise AssertionError("training loader must reject non-train paths")


def test_v2_1_strict_runner_accepts_exact_call_and_rejects_wrong_tool():
    runner = _load_module(
        "rist_v2_1_runner",
        REPO_ROOT / "examples" / "agentic" / "rist_v2_1" / "run_agent.py",
    )
    builder = _load_module(
        "rist_v2_1_train_builder_runner", V2_1 / "stages" / "D3" / "build_train_split.py"
    )
    turn = next(
        row["turns"][0]
        for row in builder.build_rows()
        if len(row["turns"][0]["offered_tools"]) > 1
    )
    target = next(
        candidate["code"]
        for candidate in turn["candidate_records"]
        if candidate["label"] == turn["target_label"]
    )

    class Function:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class Call:
        id = "call-1"
        type = "function"

        def __init__(self, name, arguments):
            self.function = Function(name, arguments)

    class Message:
        content = None

        def __init__(self, call):
            self.tool_calls = [call]

    class Choice:
        def __init__(self, call):
            self.message = Message(call)

    class Response:
        def __init__(self, call):
            self.choices = [Choice(call)]

    exact = Response(Call(turn["expected_tool"], __import__("json").dumps({"code": target})))
    assert runner.validate_response(exact, turn)["valid"] is True

    wrong_name = next(name for name in turn["offered_tools"] if name != turn["expected_tool"])
    wrong = Response(Call(wrong_name, __import__("json").dumps({"code": target})))
    result = runner.validate_response(wrong, turn)
    assert result["valid"] is False
    assert result["reason"] == "WRONG_TOOL"


def test_v2_1_tokenizer_gate_distinguishes_argument_mask_from_name_only():
    evaluator = _load_module(
        "rist_v2_1_mask_fixture",
        V2_1 / "stages" / "T0" / "evaluate_mask_fixture.py",
    )
    fixture = {
        "checkpoint": "mock",
        "tokenizer_sha256": "a" * 64,
        "cases": [
            {
                "case_id": "argument-mask",
                "token_ids": [10, 11, 12, 13],
                "loss_mask": [True, True, False, True],
                "name_indices": [1],
                "argument_indices": [2],
                "other_indices": [0, 3],
            }
        ],
    }
    result = evaluator.evaluate_fixture(fixture)
    assert result["argument_mask_all"] is True
    assert result["name_only_all"] is False

    fixture["cases"][0]["loss_mask"] = [False, True, False, False]
    result = evaluator.evaluate_fixture(fixture)
    assert result["name_only_all"] is True


def test_v2_1_tokenizer_capture_builds_32_local_canonical_cases():
    capture = _load_module(
        "rist_v2_1_mask_capture",
        V2_1 / "stages" / "T0" / "capture_mask_fixture.py",
    )
    evaluator = _load_module(
        "rist_v2_1_mask_capture_eval",
        V2_1 / "stages" / "T0" / "evaluate_mask_fixture.py",
    )

    class CharacterTokenizer:
        special_tokens_map = {}

        def get_vocab(self):
            return {chr(index): index for index in range(128)}

        def __call__(self, text, **_):
            return {
                "input_ids": [ord(character) for character in text],
                "offset_mapping": [(index, index + 1) for index in range(len(text))],
            }

        def decode(self, token_ids):
            return "".join(chr(token_id) for token_id in token_ids)

    def argument_range(tokenizer, token_ids):
        text = tokenizer.decode(token_ids)
        start = text.index('"arguments":') + len('"arguments":')
        return start, len(text) - 1

    fixture = capture.build_fixture(CharacterTokenizer(), "local/mock", argument_range)
    result = evaluator.evaluate_fixture(fixture)
    assert fixture["case_count"] == 32
    assert fixture["local_files_only"] is True
    assert result["argument_mask_all"] is True
    assert result["name_only_all"] is False


def test_v2_1_capacity_gate_requires_both_algorithms_and_memory_headroom():
    validator = _load_module(
        "rist_v2_1_capacity",
        V2_1 / "stages" / "E1" / "validate_capacity_evidence.py",
    )
    evidence = {
        "checkpoint": "mock/model",
        "model_revision": "b" * 40,
        "tokenizer_sha256": "c" * 64,
        "gpu_name": "Mock GPU",
        "gpu_total_memory_gib": 80,
        "serving_canary": {
            "health_pass": True,
            "task_count": 32,
            "complete_four_turn_count": 32,
            "raw_response_count": 128,
            "peak_memory_gib": 20,
            "oom": False,
            "retry_count": 0,
        },
        "training_canaries": [
            {
                "algorithm": algorithm,
                "optimizer_step_completed": True,
                "trainable_tokens": 100,
                "loss": 0.5,
                "gradient_norm": 1.0,
                "peak_memory_gib": peak,
                "oom": False,
                "checkpoint_roundtrip": True,
            }
            for algorithm, peak in (("gspo", 60), ("grpo", 64))
        ],
    }
    assert validator.validate_capacity(evidence)["passed"] is True
    evidence["training_canaries"][1]["peak_memory_gib"] = 70
    result = validator.validate_capacity(evidence)
    assert result["memory_headroom_pass"] is False
    assert result["passed"] is False


def test_v2_1_external_source_validator_checks_commit_origin_and_license(tmp_path):
    validator = _load_module(
        "rist_v2_1_source_validator",
        V2_1 / "stages" / "X1" / "validate_checkout.py",
    )
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "cpu@test.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "CPU Test"],
        check=True,
    )
    (root / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "LICENSE"], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-q", "-m", "fixture"], check=True
    )
    remote = "https://github.com/example/source.git"
    subprocess.run(
        ["git", "-C", str(root), "remote", "add", "origin", remote], check=True
    )
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    lock = {
        "tau3": {
            "commit": head,
            "repository": "https://github.com/example/source",
            "expected_license": "MIT",
        }
    }
    result = validator.validate_checkout(root, "tau3", lock)
    assert result["commit"] == head
    assert result["benchmark_content_opened"] is False


def test_v2_1_execution_manifest_is_blocked_and_complete(tmp_path):
    builder = _load_module(
        "rist_v2_1_execution_manifest",
        V2_1 / "stages" / "P3_DESIGN" / "build_execution_manifest.py",
    )
    manifest = builder.build_manifest(tmp_path / "run", max_steps=7)
    assert manifest["run_count"] == 48
    assert manifest["execution_authorized"] is False
    assert "T0_EXACT_NAME_ONLY_TREATMENT" in manifest["blocked_by"]
    assert sum(run["scientific_treatment_ready"] for run in manifest["runs"]) == 24
    assert all("--max-steps" in run["command"] for run in manifest["runs"])
    assert all("heldout" not in " ".join(run["command"]).lower() for run in manifest["runs"])


def test_v2_1_power_plan_treats_three_seeds_as_pilot_only():
    planner = _load_module(
        "rist_v2_1_power_plan",
        V2_1 / "stages" / "P3_DESIGN" / "power_plan.py",
    )
    assert planner.required_paired_seeds(0.10, 0.05) == 3
    assert planner.required_paired_seeds(0.10, 0.20) > 20
    plan = planner.sensitivity_plan()
    assert plan["statistical_unit"] == "paired_training_seed"
    assert plan["three_seed_status"] == "pilot_variance_only"

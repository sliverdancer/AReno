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
    } == {"name_only"}
    for row in rows:
        arguments = design.cli_treatment_args(row)
        assert "heldout" not in " ".join(arguments).lower()
        index = arguments.index("--tool-call-supervision")
        expected = "name_only" if row["arm"] in {"AN", "LN"} else "full"
        assert arguments[index + 1] == expected
        assert "--mask-tool-call-args" not in arguments


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


def test_v2_1_reward_journal_keeps_zero_and_one_outcomes(tmp_path):
    reward_module = _load_module(
        "rist_v2_1_reward_journal",
        REPO_ROOT / "examples" / "agentic" / "rist_v2_1" / "reward.py",
    )
    train_builder = _load_module(
        "rist_v2_1_train_builder_reward",
        V2_1 / "stages" / "D3" / "build_train_split.py",
    )
    row = train_builder.build_rows()[0]
    row["resolution_band"] = "low"
    from types import SimpleNamespace
    import os

    journal = tmp_path / "rewards.jsonl"
    previous = os.environ.get("RIST_REWARD_JOURNAL_PATH")
    os.environ["RIST_REWARD_JOURNAL_PATH"] = str(journal)
    try:
        invalid = SimpleNamespace(
            source_record=row,
            metadata={"prompt_index": 0, "sample_index": 0},
            tool_calls=[{"name": "bad", "arguments": "not-json"}],
        )
        exact = SimpleNamespace(
            source_record=row,
            metadata={"prompt_index": 0, "sample_index": 1},
            tool_calls=[
                {"name": action["name"], "arguments": action["arguments"]}
                for action in row["oracle_actions"]
            ],
        )
        assert reward_module.reward_fn(invalid) == 0.0
        assert reward_module.reward_fn(exact) == 1.0
    finally:
        if previous is None:
            os.environ.pop("RIST_REWARD_JOURNAL_PATH", None)
        else:
            os.environ["RIST_REWARD_JOURNAL_PATH"] = previous
    events = [
        __import__("json").loads(line) for line in journal.read_text().splitlines()
    ]
    assert [event["reward"] for event in events] == [0.0, 1.0]
    assert [event["sample_index"] for event in events] == [0, 1]
    assert all(event["resolution_band"] == "low" for event in events)


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
    visible = runner._visible_turn(turn)
    assert "expected_tool" not in visible
    assert "oracle_actions" not in visible
    assert set(visible) == {
        "turn_index",
        "offered_tools",
        "target_label",
        "candidate_records",
        "selection_rule",
        "depends_on_previous_observation",
    }


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


def test_v2_1_exact_name_only_contract_is_offset_exact_and_compositional():
    contract = _load_module(
        "rist_v2_1_name_only_contract",
        V2_1 / "stages" / "T0" / "name_only_mask_contract.py",
    )

    class CharacterTokenizer:
        def __call__(self, text, **_):
            return {
                "input_ids": [ord(character) for character in text],
                "offset_mapping": [(index, index + 1) for index in range(len(text))],
            }

    raw = '{"name":"scan_registry","arguments":{"code":"r0"}}'
    tokens = [ord(character) for character in raw]
    base = [True] * len(tokens)
    name_start = raw.index("scan_registry")
    base[name_start + 2] = False
    result = contract.exact_name_only_mask(CharacterTokenizer(), raw, tokens, base)
    assert sum(result) == len("scan_registry") - 1
    assert all(
        enabled is (name_start <= index < name_start + len("scan_registry") and base[index])
        for index, enabled in enumerate(result)
    )


def test_v2_1_exact_name_only_contract_rejects_syntax_mixed_token():
    contract = _load_module(
        "rist_v2_1_name_only_contract_mixed",
        V2_1 / "stages" / "T0" / "name_only_mask_contract.py",
    )
    raw = '{"name":"scan_registry","arguments":{}}'
    name_start = raw.index("scan_registry")

    class MixedTokenizer:
        def __call__(self, text, **_):
            return {
                "input_ids": [1],
                "offset_mapping": [(name_start - 1, name_start + len("scan_registry"))],
            }

    try:
        contract.exact_name_only_mask(MixedTokenizer(), raw, [1], [True])
    except ValueError as error:
        assert "mixes" in str(error)
    else:
        raise AssertionError("syntax-mixed name token must fail closed")


def _strict_replay(episode_id, *, reward=1.0):
    return [
        {
            "type": "reset",
            "episode_id": episode_id,
            "state_hash": "a" * 64,
            "observation": {"request": "change flight"},
        },
        {
            "type": "step",
            "step_index": 0,
            "action": {"name": "change_flight", "arguments": {"id": "F1"}},
            "state_hash_before": "a" * 64,
            "state_hash_after": "b" * 64,
            "reward": reward,
            "reward_source": "db_and_communicate",
            "raw_tool_result": {"changed": True},
            "done": True,
        },
    ]


def test_v2_1_tau3_replay_gate_requires_two_identical_clean_resets():
    qualifier = _load_module(
        "rist_v2_1_tau3_replay",
        V2_1 / "stages" / "X0" / "qualify_replay.py",
    )
    first = {"dev-001": _strict_replay("dev-001")}
    second = {"dev-001": _strict_replay("dev-001")}
    partitions = {
        "development": ["dev-001"],
        "training": ["train-001"],
        "confirmatory": ["confirm-001"],
    }
    result = qualifier.qualify_replays(first, second, partitions)
    assert result["environment_qualification_pass"] is True
    assert result["clean_reset_replay_count"] == 2
    assert result["llm_judge_used"] is False

    second["dev-001"] = _strict_replay("dev-001", reward=0.0)
    try:
        qualifier.qualify_replays(first, second, partitions)
    except ValueError as error:
        assert "nondeterministic" in str(error)
    else:
        raise AssertionError("different clean-reset replays must be rejected")


def test_v2_1_tau3_replay_gate_rejects_split_overlap():
    qualifier = _load_module(
        "rist_v2_1_tau3_replay_overlap",
        V2_1 / "stages" / "X0" / "qualify_replay.py",
    )
    replay = {"dev-001": _strict_replay("dev-001")}
    partitions = {
        "development": ["dev-001"],
        "training": ["dev-001"],
        "confirmatory": ["confirm-001"],
    }
    try:
        qualifier.qualify_replays(replay, replay, partitions)
    except ValueError as error:
        assert "disjoint" in str(error)
    else:
        raise AssertionError("overlapping real-environment IDs must be rejected")


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
    manifest = builder.build_manifest(tmp_path / "run", max_steps=100)
    assert manifest["run_count"] == 48
    assert manifest["execution_authorized"] is False
    assert manifest["resolution_filtered_dataset_ready"] is False
    assert manifest["train_sha256"] is None
    assert "C0_COMMON_TRANSPORTED_RESOLUTION_BANDS" in manifest["blocked_by"]
    assert "T0_EXACT_NAME_ONLY_TREATMENT" not in manifest["blocked_by"]
    assert sum(run["scientific_treatment_ready"] for run in manifest["runs"]) == 0
    assert all("--tool-call-supervision" in run["command"] for run in manifest["runs"])
    assert all("--max-steps" in run["command"] for run in manifest["runs"])
    assert manifest["save_interval"] == 25
    assert manifest["saved_checkpoint_steps"] == [25, 50, 75, 100]
    assert all("--save-path" in run["command"] for run in manifest["runs"])
    assert all("--save-interval" in run["command"] for run in manifest["runs"])
    assert all(
        set(run["required_environment"])
        == {"RIST_RAW_JOURNAL_PATH", "RIST_REWARD_JOURNAL_PATH"}
        for run in manifest["runs"]
    )
    assert all("heldout" not in " ".join(run["command"]).lower() for run in manifest["runs"])
    assert all("{FILTERED_TRAIN_JSONL}" in run["command"] for run in manifest["runs"])
    assert all(
        str(REPO_ROOT) not in " ".join(run["command"])
        for run in manifest["runs"]
    )


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


def _mock_p4_results(manifest, *, token_reversal: bool = False):
    endpoints = {"AF": 0.45, "LF": 0.35, "AN": 0.75, "LN": 0.45}
    starts = {"AF": 0.30, "LF": 0.25, "AN": 0.55, "LN": 0.30}
    token_support = {
        "AF": (100.0, 1000.0),
        "LF": (80.0, 700.0),
        "AN": (90.0, 5000.0 if token_reversal else 800.0),
        "LN": (70.0, 600.0),
    }
    if token_reversal:
        starts["AN"] = 0.0
    rows = []
    for design in manifest["runs"]:
        arm = design["arm"]
        start = starts[arm]
        endpoint = endpoints[arm]
        low = max(0.0, endpoint - 0.05)
        high = min(1.0, endpoint + 0.05)
        rows.append(
            {
                "run_id": design["run_id"],
                "family": design["family"],
                "algorithm": design["algorithm"],
                "arm": arm,
                "seed": design["seed"],
                "catastrophic": False,
                "confirmatory_strict_success": endpoint,
                "confirmatory_strict_success_by_resolution": {
                    "low": low,
                    "high": high,
                },
                "nonzero_advantage_groups": 10,
                "raw_evidence_sha256": __import__("hashlib")
                .sha256(design["run_id"].encode())
                .hexdigest(),
                "curve": [
                    {
                        "step": 1,
                        "cumulative_trainable_tokens": token_support[arm][0],
                        "strict_success": start,
                        "strict_success_by_resolution": {
                            "low": max(0.0, start - 0.05),
                            "high": min(1.0, start + 0.05),
                        },
                    },
                    {
                        "step": manifest["max_steps"],
                        "cumulative_trainable_tokens": token_support[arm][1],
                        "strict_success": endpoint,
                        "strict_success_by_resolution": {"low": low, "high": high},
                    },
                ],
            }
        )
    return rows


def test_v2_1_p4_analyzer_uses_seed_interactions_and_blocks_three_seed_pilot(
    tmp_path,
):
    analyzer = _load_module(
        "rist_v2_1_p4_analysis",
        V2_1 / "stages" / "P4_ANALYSIS" / "analyze_results.py",
    )
    manifest = __import__("json").loads(
        (V2_1 / "stages" / "P3_DESIGN" / "execution_manifest.json").read_text()
    )
    rows = _mock_p4_results(manifest)
    for row in rows:
        relative = f"{row['run_id']}.json"
        payload = row["run_id"].encode()
        (tmp_path / relative).write_bytes(payload)
        row["raw_evidence_path"] = relative
        row["raw_evidence_sha256"] = __import__("hashlib").sha256(payload).hexdigest()
    result = analyzer.analyze_bundle(manifest, rows, evidence_root=tmp_path)
    assert result["run_count"] == 48
    assert len(result["seed_rows"]) == 12
    assert len(result["blocks"]) == 4
    assert result["cross_block_step_sign_consistent"] is True
    assert all(block["token_sign_consistent"] for block in result["blocks"])
    assert all(
        set(block["resolution_interaction"]) == {"low", "high"}
        for block in result["blocks"]
    )
    assert all(
        "mean" in block["resolution_moderation_high_minus_low"]
        for block in result["blocks"]
    )
    assert result["three_seed_pilot_only"] is True
    assert result["raw_evidence_files_verified"] is True
    assert result["scientific_treatment_ready"] is False
    assert result["execution_authority_recorded"] is False
    assert result["main_track_eligible"] is False
    assert all(block["required_paired_seeds"] >= 8 for block in result["blocks"])


def test_v2_1_p4_analyzer_detects_token_matched_sign_failure():
    analyzer = _load_module(
        "rist_v2_1_p4_analysis_reversal",
        V2_1 / "stages" / "P4_ANALYSIS" / "analyze_results.py",
    )
    manifest = __import__("json").loads(
        (V2_1 / "stages" / "P3_DESIGN" / "execution_manifest.json").read_text()
    )
    result = analyzer.analyze_bundle(
        manifest, _mock_p4_results(manifest, token_reversal=True)
    )
    assert any(not block["token_sign_consistent"] for block in result["blocks"])
    assert result["main_track_eligible"] is False


def test_v2_1_p4_analyzer_rejects_missing_run():
    analyzer = _load_module(
        "rist_v2_1_p4_analysis_missing",
        V2_1 / "stages" / "P4_ANALYSIS" / "analyze_results.py",
    )
    manifest = __import__("json").loads(
        (V2_1 / "stages" / "P3_DESIGN" / "execution_manifest.json").read_text()
    )
    rows = _mock_p4_results(manifest)[:-1]
    try:
        analyzer.analyze_bundle(manifest, rows)
    except ValueError as error:
        assert "run bundle mismatch" in str(error)
    else:
        raise AssertionError("missing frozen run must invalidate the bundle")


def test_v2_1_p4_training_collector_aligns_tokens_and_mixed_groups():
    collector = _load_module(
        "rist_v2_1_p4_training_collector",
        V2_1 / "stages" / "P4_ANALYSIS" / "collect_training_evidence.py",
    )
    series = {
        name: [
            {
                "step": step,
                "value": (
                    10.0
                    if name == "trainable_tokens"
                    else 5.0
                    if name == "masked_response_tokens"
                    else 0.5
                ),
            }
            for step in range(100)
        ]
        for name in collector.TAGS
    }
    rewards = []
    for step in range(100):
        for sample_index in range(8):
            rewards.append(
                {
                    "sample_index": sample_index,
                    "reward": sample_index % 2 if step < 40 else 0,
                }
            )
    result = collector.summarize_training(series, rewards)
    assert [point["cumulative_trainable_tokens"] for point in result["points"]] == [
        250,
        500,
        750,
        1000,
    ]
    assert result["total_nonzero_advantage_groups"] == 40
    assert result["zero_advantage_group_count"] == 60


def test_v2_1_p4_run_assembler_uses_dev_curve_and_confirmatory_endpoint(tmp_path):
    assembler = _load_module(
        "rist_v2_1_p4_run_assembler",
        V2_1 / "stages" / "P4_ANALYSIS" / "assemble_run_result.py",
    )
    run_id = "qwen3-gspo-AF-7101"
    training = {
        "protocol": "RIST-P4-v2.1",
        "total_nonzero_advantage_groups": 40,
        "points": [
            {
                "checkpoint_step": step,
                "cumulative_trainable_tokens": step * 10,
                "mean_training_reward": 0.5,
            }
            for step in (25, 50, 75, 100)
        ],
    }
    resolution_map = {
        "passed": True,
        "common_resolution_map": {
            "c00": "low",
            "c01": "low",
            "c06": "high",
            "c07": "high",
        },
    }

    def evaluation(split, checkpoint_id, low_reward, high_reward):
        trajectories = []
        for cell in ("c00", "c01", "c06", "c07"):
            reward = low_reward if cell in {"c00", "c01"} else high_reward
            for index in range(2):
                trajectories.append(
                    {
                        "structural_cell": cell,
                        "strict_success": int(index / 2 < reward),
                    }
                )
        return {
            "split": split,
            "checkpoint_id": checkpoint_id,
            "complete": True,
            "trajectories": trajectories,
        }

    dev = [
        evaluation("dev_curve", f"{run_id}-dev-step-{step:03d}", 0.0, 0.5)
        for step in (25, 50, 75, 100)
    ]
    confirmatory = evaluation(
        "confirmatory", f"{run_id}-confirmatory-step-100", 0.5, 1.0
    )
    archive = tmp_path / "evidence.tgz"
    archive.write_bytes(b"evidence")
    result = assembler.assemble_run(
        {
            "run_id": run_id,
            "family": "qwen3",
            "algorithm": "gspo",
            "arm": "AF",
            "seed": 7101,
        },
        training,
        dev,
        confirmatory,
        resolution_map,
        archive,
        "runs/qwen3/evidence.tgz",
    )
    assert result["curve"][-1]["strict_success"] == 0.25
    assert result["confirmatory_strict_success"] == 0.75
    assert result["confirmatory_strict_success_by_resolution"] == {
        "low": 0.5,
        "high": 1.0,
    }
    assert result["nonzero_advantage_groups"] == 40


def test_v2_1_p5_cross_setting_gate_requires_real_and_sealed_transport():
    gate = _load_module(
        "rist_v2_1_p5_gate",
        V2_1 / "stages" / "P5_META" / "cross_setting_gate.py",
    )
    summaries = {
        setting: {
            "completed": True,
            "interaction_estimate": 0.15,
            "ci95": [0.05, 0.25],
            "families": ["qwen3", "gemma4"],
            "algorithms": ["gspo", "grpo"],
            "token_sign_consistent": True,
            "catastrophic_run_rate": 0.0,
            "raw_evidence_complete": True,
            "powered": True,
            "analysis_protocol_sha256": "d" * 64,
            "environment_qualification_pass": True,
            "sealed_once": True,
        }
        for setting in ("rist_synthetic", "tau3", "bfcl_sealed")
    }
    assert gate.evaluate_cross_setting(summaries)["main_track_eligible"] is True
    summaries["tau3"]["completed"] = False
    result = gate.evaluate_cross_setting(summaries)
    assert result["main_track_eligible"] is False
    assert result["decision"] == "DO_NOT_UPGRADE_MAIN_TRACK"


def test_v2_1_d4_eval_splits_are_balanced_and_disjoint(tmp_path):
    builder = _load_module(
        "rist_v2_1_d4_builder",
        V2_1 / "stages" / "D4_EVAL" / "build_eval_splits.py",
    )
    manifest = builder.write_splits(tmp_path / "data")
    assert manifest["splits"]["dev_curve"]["count"] == 16
    assert manifest["splits"]["confirmatory"]["count"] == 32
    assert manifest["splits"]["dev_curve"]["trajectory_count"] == 32
    assert manifest["splits"]["confirmatory"]["trajectory_count"] == 128
    assert manifest["splits"]["dev_curve"]["materialized"] is True
    assert manifest["splits"]["confirmatory"]["materialized"] is False
    assert (tmp_path / "data" / "dev_curve.jsonl").is_file()
    assert not (tmp_path / "data" / "confirmatory.jsonl").exists()
    assert manifest["parent_heldout_opened"] is False

    signatures = {
        split: set(manifest["splits"][split]["task_signatures"])
        for split in ("dev_curve", "confirmatory")
    }
    train_builder = _load_module(
        "rist_v2_1_d3_builder_for_d4", V2_1 / "stages" / "D3" / "build_train_split.py"
    )
    train_signatures = {row["task_signature"] for row in train_builder.build_rows()}
    assert signatures["dev_curve"].isdisjoint(signatures["confirmatory"])
    assert signatures["dev_curve"].isdisjoint(train_signatures)
    assert signatures["confirmatory"].isdisjoint(train_signatures)


def test_v2_1_d4_strict_evaluator_stops_without_repair_and_hides_oracle(tmp_path):
    builder = _load_module(
        "rist_v2_1_d4_builder_runner",
        V2_1 / "stages" / "D4_EVAL" / "build_eval_splits.py",
    )
    evaluator = _load_module(
        "rist_v2_1_d4_evaluator",
        V2_1 / "stages" / "D4_EVAL" / "evaluate_checkpoint.py",
    )
    task = builder.build_rows("dev_curve")[0]
    requests = []

    def exact_post(payload):
        requests.append(payload)
        turn = task["turns"][len(requests) - 1]
        target = next(
            candidate["code"]
            for candidate in turn["candidate_records"]
            if candidate["label"] == turn["target_label"]
        )
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call-{len(requests)}",
                                "type": "function",
                                "function": {
                                    "name": turn["expected_tool"],
                                    "arguments": __import__("json").dumps({"code": target}),
                                },
                            }
                        ],
                    }
                }
            ]
        }

    result = evaluator.run_trajectory(
        task,
        8101,
        {"temperature": 0.7, "top_p": 0.95, "max_tokens": 128},
        exact_post,
        tmp_path / "exact.jsonl",
    )
    assert result["strict_success"] == 1
    assert result["completed_turns"] == 4
    assert len((tmp_path / "exact.jsonl").read_text().splitlines()) == 4
    assert "expected_tool" not in __import__("json").dumps(requests[1]["messages"])

    invalid = evaluator.run_trajectory(
        task,
        8101,
        {"temperature": 0.7, "top_p": 0.95, "max_tokens": 128},
        lambda _: {"choices": [{"message": {"role": "assistant", "tool_calls": []}}]},
        tmp_path / "invalid.jsonl",
    )
    assert invalid["strict_success"] == 0
    assert invalid["completed_turns"] == 0
    assert invalid["invalid_reason"] == "CALL_COUNT"
    assert len((tmp_path / "invalid.jsonl").read_text().splitlines()) == 1


def test_v2_1_d4_confirmatory_ledger_is_consumed_before_data(tmp_path):
    builder = _load_module(
        "rist_v2_1_d4_builder_ledger",
        V2_1 / "stages" / "D4_EVAL" / "build_eval_splits.py",
    )
    evaluator = _load_module(
        "rist_v2_1_d4_evaluator_ledger",
        V2_1 / "stages" / "D4_EVAL" / "evaluate_checkpoint.py",
    )
    data_dir = tmp_path / "data"
    builder.write_splits(data_dir)
    ledger = tmp_path / "confirmatory-ledger.json"
    invalid_response = {
        "choices": [{"message": {"role": "assistant", "tool_calls": []}}]
    }
    result = evaluator.collect(
        data_dir=data_dir,
        split="confirmatory",
        checkpoint_id="mock-step-100",
        output_path=tmp_path / "result.json",
        journal_path=tmp_path / "journal.jsonl",
        ledger_path=ledger,
        post_json=lambda _: invalid_response,
    )
    assert result["trajectory_count"] == 128
    assert result["complete"] is True
    assert result["strict_success"] == 0.0
    assert __import__("json").loads(ledger.read_text())["result_complete"] is True
    try:
        evaluator.collect(
            data_dir=data_dir,
            split="confirmatory",
            checkpoint_id="mock-step-100",
            output_path=tmp_path / "result-second.json",
            journal_path=tmp_path / "journal-second.jsonl",
            ledger_path=ledger,
            post_json=lambda _: invalid_response,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("confirmatory ledger must be one-shot")


def test_v2_1_d4_persists_infrastructure_failure_without_fabrication(tmp_path):
    builder = _load_module(
        "rist_v2_1_d4_builder_failure",
        V2_1 / "stages" / "D4_EVAL" / "build_eval_splits.py",
    )
    evaluator = _load_module(
        "rist_v2_1_d4_evaluator_failure",
        V2_1 / "stages" / "D4_EVAL" / "evaluate_checkpoint.py",
    )
    data_dir = tmp_path / "data"
    builder.write_splits(data_dir)

    def fail(_):
        raise RuntimeError("fixture endpoint failure")

    result = evaluator.collect(
        data_dir=data_dir,
        split="dev_curve",
        checkpoint_id="mock-failure",
        output_path=tmp_path / "failure-result.json",
        journal_path=tmp_path / "failure-journal.jsonl",
        post_json=fail,
    )
    assert result["complete"] is False
    assert result["trajectory_count"] == 0
    assert result["strict_success"] is None
    assert result["infrastructure_error"]["error_type"] == "RuntimeError"
    assert result["raw_journal_sha256"] is None
    assert (tmp_path / "failure-result.json").is_file()


def test_v2_1_d4_evaluation_manifest_covers_all_checkpoints_without_authority(
    tmp_path,
):
    training_builder = _load_module(
        "rist_v2_1_training_manifest_for_d4",
        V2_1 / "stages" / "P3_DESIGN" / "build_execution_manifest.py",
    )
    evaluation_builder = _load_module(
        "rist_v2_1_d4_eval_manifest",
        V2_1 / "stages" / "D4_EVAL" / "build_evaluation_manifest.py",
    )
    training = training_builder.build_manifest(tmp_path / "run", max_steps=100)
    evaluation = evaluation_builder.build_evaluation_manifest(
        training, tmp_path / "run"
    )
    assert evaluation["job_count"] == 242
    assert evaluation["development_job_count"] == 194
    assert evaluation["confirmatory_job_count"] == 48
    assert evaluation["execution_authorized"] is False
    assert evaluation["commands_are_templates_only"] is True
    assert all(job["execution_authorized"] is False for job in evaluation["jobs"])
    assert all(
        str(REPO_ROOT) not in " ".join(job["client_command_template"])
        for job in evaluation["jobs"]
    )
    confirmatory = [
        job for job in evaluation["jobs"] if job["split"] == "confirmatory"
    ]
    assert all(job["ledger"] is not None for job in confirmatory)
    assert all(
        job["prerequisite"] == "ALL_TRAINING_AND_ANALYSIS_HASHES_FROZEN"
        for job in confirmatory
    )


def _mock_resolution_rows(split):
    rows = []
    for cell_index in range(8):
        cell = f"c{cell_index:02d}"
        for task_index in range(4):
            for seed_index in range(32):
                reward = 0 if cell_index < 4 else seed_index % 2
                rows.append(
                    {
                        "split": split,
                        "structural_cell": cell,
                        "task_id": f"{split}-{cell}-{task_index}",
                        "rollout_seed": 10000 + seed_index,
                        "strict_success": reward,
                    }
                )
    return rows


def test_v2_1_c0_direct_mixed_group_calibration_transports_across_families():
    calibrator = _load_module(
        "rist_v2_1_c0_calibrator",
        V2_1 / "stages" / "C0_RESOLUTION" / "calibrate_resolution.py",
    )
    combiner = _load_module(
        "rist_v2_1_c0_combiner",
        V2_1 / "stages" / "C0_RESOLUTION" / "combine_family_maps.py",
    )
    calibration = _mock_resolution_rows("calibration")
    qualification = _mock_resolution_rows("qualification")
    qwen = calibrator.calibrate_checkpoint(calibration, qualification, "qwen3")
    gemma = calibrator.calibrate_checkpoint(calibration, qualification, "gemma4")
    assert qwen["passed"] is True
    assert qwen["band_cell_counts"] == {"low": 4, "high": 4}
    assert qwen["calibration"]["cells"]["c00"]["classification"] == "collapsed"
    assert qwen["calibration"]["cells"]["c07"]["classification"] == "resolved"
    combined = combiner.combine_maps([qwen, gemma])
    assert combined["passed"] is True
    assert combined["band_cell_counts"] == {"low": 4, "high": 4}


def test_v2_1_c0_kills_nontransporting_resolution_pool():
    calibrator = _load_module(
        "rist_v2_1_c0_calibrator_kill",
        V2_1 / "stages" / "C0_RESOLUTION" / "calibrate_resolution.py",
    )
    calibration = _mock_resolution_rows("calibration")
    qualification = _mock_resolution_rows("qualification")
    for row in qualification:
        if row["structural_cell"] in {"c00", "c01", "c02"}:
            row["strict_success"] = row["rollout_seed"] % 2
    result = calibrator.calibrate_checkpoint(calibration, qualification, "mock")
    assert result["passed"] is False
    assert result["decision"] == "KILL_C0_TASK_POOL"


def test_v2_1_c0_filter_retains_whole_cells_without_outcome_selection(tmp_path):
    train_builder = _load_module(
        "rist_v2_1_d3_builder_filter", V2_1 / "stages" / "D3" / "build_train_split.py"
    )
    filterer = _load_module(
        "rist_v2_1_c0_filter",
        V2_1 / "stages" / "C0_RESOLUTION" / "filter_train_pool.py",
    )
    source_dir = tmp_path / "source" / "data"
    train_builder.write_train(source_dir)
    map_result = {
        "passed": True,
        "common_resolution_map": {
            "c00": "low",
            "c01": "low",
            "c06": "high",
            "c07": "high",
        },
    }
    result = filterer.filter_train(
        source_dir / "train.jsonl", map_result, tmp_path / "filtered" / "data"
    )
    assert result["task_count"] == 16
    assert result["band_cell_counts"] == {"low": 2, "high": 2}
    assert result["individual_outcome_selection"] is False
    rows = [
        __import__("json").loads(line)
        for line in (tmp_path / "filtered" / "data" / "train.jsonl")
        .read_text()
        .splitlines()
    ]
    assert {row["structural_cell"] for row in rows} == set(
        map_result["common_resolution_map"]
    )
    assert all("resolution_band" in row for row in rows)


def test_v2_1_c0_collection_manifest_freezes_four_unauthorized_jobs(tmp_path):
    builder = _load_module(
        "rist_v2_1_c0_collection_manifest",
        V2_1 / "stages" / "C0_RESOLUTION" / "build_collection_manifest.py",
    )
    manifest = builder.build_manifest(tmp_path / "run")
    assert manifest["job_count"] == 4
    assert manifest["execution_authorized"] is False
    assert manifest["commands_are_templates_only"] is True
    assert manifest["parent_heldout_opened"] is False
    assert manifest["group_size"] == 8
    assert manifest["groups_per_task"] == 4
    assert all(row["trajectory_count"] == 1024 for row in manifest["jobs"])
    assert not Path(manifest["source_data_dir"]).is_absolute()
    assert all(
        str(REPO_ROOT) not in " ".join(row["client_command_template"])
        for row in manifest["jobs"]
    )
    assert len(manifest["splits"]["calibration"]["rollout_seeds"]) == 32


def test_v2_1_c0_collector_persists_complete_zero_reward_rows(tmp_path):
    train_builder = _load_module(
        "rist_v2_1_d3_builder_for_c0", V2_1 / "stages" / "D3" / "build_train_split.py"
    )
    collector = _load_module(
        "rist_v2_1_c0_collector",
        V2_1 / "stages" / "C0_RESOLUTION" / "collect_pretraining.py",
    )
    row = train_builder.build_rows()[0]
    row["split"] = "calibration"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    encoded = (__import__("json").dumps(row, sort_keys=True) + "\n").encode()
    (source_dir / "calibration.jsonl").write_bytes(encoded)
    manifest = {
        "protocol": "RIST-C0-v2.1",
        "source_data_dir": str(source_dir),
        "models": {"qwen3": "mock"},
        "sampling": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 128},
        "splits": {
            "calibration": {
                "file": "calibration.jsonl",
                "sha256": __import__("hashlib").sha256(encoded).hexdigest(),
                "rollout_seeds": [1, 2],
                "trajectory_count": 2,
            }
        },
    }
    invalid_response = {
        "choices": [{"message": {"role": "assistant", "tool_calls": []}}]
    }
    result = collector.collect(
        manifest,
        "qwen3",
        "calibration",
        tmp_path / "result.json",
        tmp_path / "journal.jsonl",
        lambda _: invalid_response,
    )
    assert result["complete"] is True
    assert result["trajectory_count"] == 2
    assert all(row["strict_success"] == 0 for row in result["trajectories"])
    assert all(row["split"] == "calibration" for row in result["trajectories"])

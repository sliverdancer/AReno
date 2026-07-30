from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from areno.api import agentic
from areno.api.agentic import LossMaskPolicy, RolloutSession
from areno.api.config import ArenoConfig
from areno.api.backend.areno.backend import ArenoBackend
from areno.api.backend.areno.backend import _rollout_options
from areno.api.models import SamplingParams as ApiSamplingParams
from areno.api.models import TrainSequence
from areno.api.seeding import derive_seed, epoch_dataset_view, seed_parent_process
from areno.api.trainer_config import PolicyTrainerConfig
from areno.engine.protocol import _seed_worker_process

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTRUMENT_DIR = (
    REPO_ROOT / "examples" / "agentic" / "structured_action_supervision"
)
RESEARCH_DIR = REPO_ROOT / "research" / "structured_action_supervision_v1"


def _load_module(name: str, path: Path):
    previous_game = sys.modules.pop("game", None)
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(path.parent))
        sys.modules.pop(name, None)
        sys.modules.pop("game", None)
        if previous_game is not None:
            sys.modules["game"] = previous_game


def _instrument_module(name: str):
    return _load_module(
        f"sas_{name}_for_tests",
        INSTRUMENT_DIR / f"{name}.py",
    )


def _response(*calls, content="raw assistant content"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=list(calls),
                )
            )
        ]
    )


def _call(name: str, arguments: str, call_id: str = "call-1"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def test_sas_dataset_splits_are_deterministic_disjoint_and_valid(tmp_path):
    generator = _instrument_module("dataset_generator")

    first = generator.generate_splits(seed=17)
    second = generator.generate_splits(seed=17)

    assert first == second
    assert {name: len(rows) for name, rows in first.items()} == {
        "train": 48,
        "qualification": 16,
        "heldout": 16,
    }
    signatures = {
        split: {row["constraint_signature"] for row in rows}
        for split, rows in first.items()
    }
    assert signatures["train"].isdisjoint(signatures["qualification"])
    assert signatures["train"].isdisjoint(signatures["heldout"])
    assert signatures["qualification"].isdisjoint(signatures["heldout"])

    manifest = generator.write_splits(tmp_path, seed=17)

    assert manifest["split_counts"] == {
        "train": 48,
        "qualification": 16,
        "heldout": 16,
    }
    for split in manifest["files"]:
        assert len(manifest["files"][split]["sha256"]) == 64


def test_sas_dataset_loader_preserves_signature_and_builds_prompt():
    generator = _instrument_module("dataset_generator")
    loader = _instrument_module("dataset_loader")
    source = generator.generate_splits(seed=23)["train"][0]

    records = loader.load_training_dataset(
        "unused",
        default_loader=lambda _: [source],
    )

    assert records[0]["constraint_signature"] == source["constraint_signature"]
    assert "submit the final item ids" in records[0]["prompt"]


@pytest.mark.parametrize(
    ("response", "expected_name", "reason"),
    [
        (_response(), "search_catalog", "MISSING_TOOL_CALL"),
        (
            _response(
                _call("search_catalog", '{"categories":["jacket"]}', "a"),
                _call("search_catalog", '{"categories":["bottle"]}', "b"),
            ),
            "search_catalog",
            "MULTIPLE_TOOL_CALLS",
        ),
        (
            _response(_call("inspect_items", '{"item_ids":["x"]}')),
            "search_catalog",
            "UNEXPECTED_TOOL_NAME",
        ),
        (
            _response(_call("search_catalog", "{")),
            "search_catalog",
            "INVALID_JSON_ARGUMENTS",
        ),
        (
            _response(_call("search_catalog", "[]")),
            "search_catalog",
            "ARGUMENTS_NOT_OBJECT",
        ),
        (
            _response(_call("search_catalog", '{"categories":[]}')),
            "search_catalog",
            "INVALID_ARGUMENT_SCHEMA",
        ),
    ],
)
def test_sas_runner_rejects_invalid_calls_without_synthesis(
    response,
    expected_name,
    reason,
):
    runner = _instrument_module("run_agent")

    validation = runner.validate_response(response, expected_name)

    assert validation.valid is False
    assert validation.reason == reason
    assert validation.assistant_message["content"] == "raw assistant content"
    if reason == "MISSING_TOOL_CALL":
        assert "tool_calls" not in validation.assistant_message


def test_sas_runner_preserves_valid_raw_arguments_and_executes():
    runner = _instrument_module("run_agent")
    raw_arguments = '{"categories":["jacket","bottle"],"max_price":140}'
    response = _response(_call("search_catalog", raw_arguments))

    validation = runner.validate_response(response, "search_catalog")
    result = runner.execute_call(
        "search_catalog",
        validation.arguments,
        {},
    )

    assert validation.valid is True
    assert (
        validation.assistant_message["tool_calls"][0]["function"]["arguments"]
        == raw_arguments
    )
    assert set(result["results_by_category"]) == {"jacket", "bottle"}


def test_sas_reward_requires_exact_four_call_protocol():
    generator = _instrument_module("dataset_generator")
    reward = _instrument_module("reward")
    game = _load_module(
        "sas_shopping_game_for_reward_tests",
        REPO_ROOT / "examples" / "agentic" / "shopping" / "game.py",
    )
    source = generator.generate_splits(seed=29)["qualification"][0]
    best = game.best_bundle(source)
    calls = [
        {
            "name": "search_catalog",
            "arguments": json.dumps({"categories": source["categories"]}),
        },
        {"name": "inspect_items", "arguments": json.dumps({"item_ids": best})},
        {"name": "check_kit", "arguments": json.dumps({"item_ids": best})},
        {"name": "submit_bundle", "arguments": json.dumps({"item_ids": best})},
    ]

    complete = SimpleNamespace(source_record=source, tool_calls=calls)
    incomplete = SimpleNamespace(source_record=source, tool_calls=calls[:-1])

    assert reward.reward_fn(complete) == 1.0
    assert reward.reward_fn(incomplete) == -1.0


class _CharTokenizer:
    chat_template = ""

    def encode(self, text):
        return [ord(character) for character in text]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


class _MaskTrainer:
    def __init__(self):
        self.config = SimpleNamespace(world_size=1, tp_size=1)
        self.tokenizer = _CharTokenizer()

    def dp_size(self):
        return 1

    def get_tokenizer(self):
        return self.tokenizer


def _mask_for(mode: str, mask_arguments: bool) -> list[bool]:
    texts = [
        '{"name":"search_catalog","arguments":{"categories":["jacket"]}}',
        '{"name":"inspect_items","arguments":{"item_ids":["a"]}}',
        '{"name":"check_kit","arguments":{"item_ids":["a"]}}',
        '{"name":"submit_bundle","arguments":{"item_ids":["a"]}}',
    ]
    tokenizer = _CharTokenizer()
    tokens = [token for text in texts for token in tokenizer.encode(text)]
    sample = agentic._AgentSample(
        item=agentic.AgentItem(
            record={},
            prompt="p",
            input_tokens=[1],
            prompt_index=0,
            sample_index=0,
        ),
        messages=[],
        response_text="",
        last_response_text="",
        response_tokens=tokens,
        response_logprobs=[0.0] * len(tokens),
        trace=[],
        loss_mask_override=[True] * len(tokens),
        response_spans=[
            agentic.ResponseSpan("assistant_tool_call", len(tokenizer.encode(text)))
            for text in texts
        ],
    )
    session = RolloutSession(
        _MaskTrainer(),
        sampling_params=SimpleNamespace(),
        loss_mask_policy=LossMaskPolicy(
            trainable_turns=mode,
            mask_tool_call_args=mask_arguments,
        ),
        proxy=False,
    )
    session._apply_trainable_turn_mode(sample)
    return sample.loss_mask_override


def test_sas_four_factorial_masks_are_distinct_and_z0_is_empty():
    masks = {
        "AF": _mask_for("all_assistant", False),
        "LF": _mask_for("last_assistant", False),
        "AN": _mask_for("all_assistant", True),
        "LN": _mask_for("last_assistant", True),
        "Z0": _mask_for("final_answer", False),
    }

    assert len({tuple(masks[name]) for name in ("AF", "LF", "AN", "LN")}) == 4
    assert sum(masks["AF"]) > sum(masks["LF"]) > sum(masks["LN"]) > 0
    assert sum(masks["AF"]) > sum(masks["AN"]) > sum(masks["LN"])
    assert not any(masks["Z0"])


def test_backend_skips_explicit_zero_signal_batch_without_parameter_delta():
    class _FakeEngine:
        def __init__(self):
            self.parameter = 3.0
            self.step_calls = 0

        def step(self, packs, gradient_accumulation_steps=None):
            del packs, gradient_accumulation_steps
            self.step_calls += 1
            self.parameter -= 0.1
            return []

    engine = _FakeEngine()
    backend = ArenoBackend()
    backend._engine = engine
    sequence = TrainSequence(
        tokens=[1, 2],
        prompt_mask=[True, False],
        loss_mask=[False, False],
        logprobs=[0.0, 0.0],
        advantages=[0.0, 0.0],
    )

    result = backend.train(
        None,
        [sequence],
        lambda _pack, _logprobs: None,
        mini_bs=1,
    )

    assert engine.step_calls == 0
    assert engine.parameter == 3.0
    assert result["optimizer_step_skipped"] == 1.0
    assert result["trainable_tokens"] == 0.0


def test_seed_is_deterministic_across_parent_data_and_request_paths():
    import random

    seed_parent_process(123)
    first_random = random.random()
    seed_parent_process(123)
    assert random.random() == first_random

    dataset = [{"id": index} for index in range(12)]
    first = [row["id"] for row in epoch_dataset_view(dataset, seed=123, epoch=0)]
    repeated = [row["id"] for row in epoch_dataset_view(dataset, seed=123, epoch=0)]
    next_epoch = [row["id"] for row in epoch_dataset_view(dataset, seed=123, epoch=1)]
    assert first == repeated
    assert first != next_epoch
    assert set(first) == set(range(12))

    assert derive_seed(123, "rollout", 0) == derive_seed(123, "rollout", 0)
    assert derive_seed(123, "rollout", 0) != derive_seed(123, "rollout", 1)


def test_worker_process_seed_replays_torch_rng():
    import torch

    _seed_worker_process(456)
    first = torch.rand(4)
    _seed_worker_process(456)
    repeated = torch.rand(4)

    assert torch.equal(first, repeated)


def test_seed_propagates_from_trainer_config_to_backend_rollout_options():
    config = PolicyTrainerConfig(
        algo="gspo",
        ckpt="model",
        dataset_path="dataset",
        reward_fn_path="reward.py",
        seed=987,
    )
    backend_config = config.areno_config()
    sampling = ApiSamplingParams(temperature=0.8, seed=config.seed)
    ctx = SimpleNamespace(
        eos_token_ids=(2,),
        tokenizer=SimpleNamespace(all_special_ids=[]),
        custom_config=ArenoConfig(seed=config.seed),
    )

    options = _rollout_options(ctx, sampling)

    assert backend_config.seed == 987
    assert options["sampling_params"].seed == 987


def test_agent_request_seeds_are_stable_and_turn_specific():
    params = ApiSamplingParams(seed=321)
    session = RolloutSession(
        _MaskTrainer(),
        sampling_params=params,
        proxy=False,
    )

    first = session.request_seed(0, 0, 0)
    repeated = session.request_seed(0, 0, 0)
    later_turn = session.request_seed(0, 0, 1)

    assert first == repeated
    assert first != later_turn


def test_q1_pilot_dry_run_is_seeded_two_arm_and_qualification_only(tmp_path):
    pilot = _instrument_module("run_q1_pilot")
    dataset_path = (
        RESEARCH_DIR / "stages" / "Q0" / "dataset" / "qualification.jsonl"
    )

    commands = pilot.build_commands(
        repo_root=REPO_ROOT,
        run_root=tmp_path,
        dataset_path=dataset_path,
        seeds=(11, 22),
        max_steps=3,
    )

    assert set(commands) == {
        "AF-seed-11",
        "LF-seed-11",
        "AF-seed-22",
        "LF-seed-22",
    }
    for key, command in commands.items():
        assert "--seed" in command
        assert command[command.index("--seed") + 1] in {"11", "22"}
        assert str(dataset_path) in command
        assert "heldout.jsonl" not in command
        assert "--max-steps" in command
        assert command[command.index("--max-steps") + 1] == "3"
        expected_mode = "all_assistant" if key.startswith("AF") else "last_assistant"
        assert command[command.index("--trainable-turns") + 1] == expected_mode


def test_main_track_hook_requires_q3_and_all_frozen_criteria(tmp_path):
    hook = _load_module(
        "sas_stage_completion_hook_for_tests",
        RESEARCH_DIR / "stage_completion_hook.py",
    )
    q0 = {
        "schema_version": 1,
        "protocol_id": "SAS-P0-v1.0",
        "stage": "Q0",
        "stage_status": "PASS",
        "main_track_evidence": {},
    }
    assert hook.assess_stage_result(q0)["decision"] == "STAY_DIAGNOSTIC"

    q3 = {
        **q0,
        "stage": "Q3",
        "main_track_evidence": {
            criterion: True for criterion in hook.MAIN_TRACK_CRITERIA
        },
    }
    assert hook.assess_stage_result(q3)["decision"] == "GO_MAIN_TRACK"
    q3["main_track_evidence"]["multi_environment_complete"] = False
    assessment = hook.assess_stage_result(q3)
    assert assessment["decision"] == "STAY_DIAGNOSTIC"
    assert "multi_environment_complete" in assessment["unmet_criteria"]

    killed = {**q0, "stage_status": "KILL"}
    assert hook.assess_stage_result(killed)["decision"] == "KILL_MAIN_TRACK"

    stage_result = tmp_path / "stage_result.json"
    stage_result.write_text(json.dumps(q0, sort_keys=True) + "\n", encoding="utf-8")
    first = hook.finalize_stage(stage_result)
    second = hook.finalize_stage(stage_result)
    assert first == second
    assert json.loads(first.read_text(encoding="utf-8"))["decision"] == "STAY_DIAGNOSTIC"

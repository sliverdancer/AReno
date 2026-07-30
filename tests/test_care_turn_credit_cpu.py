from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from areno.api import agentic
from areno.api.rewards import RewardRecord
from areno.api.trainers.policy_only import PolicyOnlyTrainer
from areno.cli import train as train_cli
from areno.experimental.care.turn_credit import (
    SCHEMA_VERSION,
    build_turn_credit_batch,
    load_turn_credit_config,
    load_turn_credit_fn,
    route_turn_credit_batch,
    select_ranked_prefix,
    write_turn_credit_diagnostics,
)


def _record(prompt_index: int = 0, sample_index: int = 0) -> RewardRecord:
    return RewardRecord(
        prompt="p",
        completion="c",
        source_record={"id": "fixture"},
        metadata={"prompt_index": prompt_index, "sample_index": sample_index},
    )


def _credit_batch(
    spans,
    *,
    loss_mask=None,
    reward=1.0,
    advantage=0.5,
    seed=17,
    step=3,
):
    response_len = sum(span.length for span in spans)
    response_mask = [False] + [True] * response_len
    loss_mask = list(loss_mask or ([False] + [True] * response_len))
    batch = build_turn_credit_batch(
        seed=seed,
        step=step,
        response_spans=[spans],
        response_masks=[response_mask],
        loss_masks=[loss_mask],
        rollout_logprobs=[[0.0] + [-0.1] * response_len],
        rewards=[reward],
        outcome_advantages=[advantage],
        reward_records=[_record()],
    )
    return batch, response_mask, loss_mask


def _result(batch, turns, *, selected_tokens, budget_tokens, audit_calls=1):
    return {
        "trajectories": [
            {
                "trajectory_id": batch.trajectories[0].trajectory_id,
                "turns": turns,
                "selected_tokens": selected_tokens,
                "budget_tokens": budget_tokens,
                "audit_calls": audit_calls,
                "diagnostics": {"fixture": True},
            }
        ]
    }


def _turn(index, weight, confidence=0.8):
    sign = 0 if weight == 0 else (1 if weight > 0 else -1)
    return {
        "turn_index": index,
        "weight": weight,
        "sign": sign,
        "abstain": sign == 0,
        "confidence": confidence,
        "diagnostics": {"source": "cpu-fixture"},
    }


def test_response_span_offsets_survive_agent_train_rows():
    item = agentic.AgentItem(
        record={"id": "fixture"},
        prompt="p",
        input_tokens=[1],
        prompt_index=0,
        sample_index=0,
    )
    spans = [
        agentic.ResponseSpan("assistant_text", 2, raw_text="first"),
        agentic.ResponseSpan("assistant_tool_call", 1, raw_text="call"),
        agentic.ResponseSpan("assistant_text", 3, raw_text="final"),
    ]
    sample = agentic._AgentSample(
        item=item,
        messages=[],
        response_text="first call final",
        last_response_text="final",
        response_tokens=[10, 11, 12, 13, 14, 15],
        response_logprobs=[-0.1] * 6,
        trace=[],
        token_row=[1, 10, 11, 12, 13, 14, 15],
        response_mask_row=[False, True, True, True, True, True, True],
        loss_mask_row=[False, True, True, True, True, True, True],
        rollout_logprobs_row=[0.0] + [-0.1] * 6,
        response_spans=spans,
    )
    rows = agentic.RolloutSession(None, sampling_params=None)._train_rows_from_samples([sample])

    assert rows.response_spans[0] == spans
    batch = build_turn_credit_batch(
        seed=7,
        step=4,
        response_spans=rows.response_spans,
        response_masks=rows.response_masks,
        loss_masks=rows.loss_masks,
        rollout_logprobs=rows.rollout_logprobs,
        rewards=[1.0],
        outcome_advantages=[0.25],
        reward_records=[_record()],
    )
    assert [
        (span.response_start, span.response_end, span.raw_text)
        for span in batch.trajectories[0].spans
    ] == [(0, 2, "first"), (2, 3, "call"), (3, 6, "final")]


def test_signed_and_abstained_turns_map_to_expected_tokens():
    spans = [
        agentic.ResponseSpan("assistant_text", 2),
        agentic.ResponseSpan("assistant_tool_call", 2),
        agentic.ResponseSpan("assistant_text", 2),
    ]
    batch, response_mask, loss_mask = _credit_batch(spans)
    result = _result(
        batch,
        [_turn(0, 1.0), _turn(1, -2.0), _turn(2, 0.0)],
        selected_tokens=4,
        budget_tokens=6,
    )

    routed = route_turn_credit_batch(
        batch=batch,
        result=result,
        token_row_lengths=[7],
        response_masks=[response_mask],
        base_loss_masks=[loss_mask],
    )

    scale = 4 / 6
    assert routed.loss_masks == [[False, True, True, True, True, False, False]]
    assert routed.advantages[0] == pytest.approx(
        [0.0, scale, scale, -2 * scale, -2 * scale, 0.0, 0.0]
    )
    assert routed.selected_tokens == 4
    assert routed.budget_tokens == 6


def test_full_abstention_returns_empty_train_batch(tmp_path):
    spans = [agentic.ResponseSpan("assistant_text", 2)]
    reward_record = _record()
    agent_batch = agentic.AgentTrainBatch(
        token_rows=[[1, 10, 11]],
        response_masks=[[False, True, True]],
        loss_masks=[[False, True, True]],
        rollout_logprobs=[[0.0, -0.1, -0.2]],
        rewards=[1.0],
        records=[{"id": "fixture"}],
        reward_records=[reward_record],
        response_spans=[spans],
    )
    trainer = PolicyOnlyTrainer.__new__(PolicyOnlyTrainer)
    trainer.config = SimpleNamespace(seed=9, metrics_log_dir=str(tmp_path))
    trainer._dashboard_step = 5
    trainer._turn_credit_config = {}

    def abstain_hook(batch, *, step, config):
        del step, config
        return _result(
            batch,
            [_turn(0, 0.0)],
            selected_tokens=0,
            budget_tokens=2,
            audit_calls=0,
        )

    trainer._turn_credit_fn = abstain_hook
    trainer.logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    tokenizer = SimpleNamespace(eos_token_id=2)

    train_batch, rewards, rollout_logprobs = trainer._materialize_agentic_train_batch(
        tokenizer,
        None,
        agent_batch,
    )

    assert train_batch == []
    assert rewards == [1.0]
    assert rollout_logprobs == []
    diagnostics = [
        json.loads(line)
        for line in (tmp_path / "turn_credit_diagnostics.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert diagnostics[0]["selected_mass"] == 0


def test_full_abstention_skips_backend_train_and_closes_step(monkeypatch):
    class FakeAreno:
        def __init__(self):
            self.finish_calls = 0

        def get_tokenizer(self):
            return SimpleNamespace(eos_token_id=2)

        def load_prompt_batches(self, dataset, *, batch_size, max_prompt_tokens):
            del dataset, batch_size, max_prompt_tokens
            return [SimpleNamespace(items=[])]

        def train(self, *args, **kwargs):
            raise AssertionError("backend train must not be called")

        def finish_step(self):
            self.finish_calls += 1

    trainer = PolicyOnlyTrainer.__new__(PolicyOnlyTrainer)
    trainer.config = SimpleNamespace(
        seed=7,
        chat_template_enable_thinking=None,
        greedy=False,
        temperature=1.0,
        max_new_tokens=8,
        max_context_len=64,
        max_prompt_tokens=16,
        top_k=-1,
        top_p=1.0,
        epochs=1,
        batch_size=1,
        max_steps=1,
        agent_fn="agent.py",
    )
    trainer.areno = FakeAreno()
    trainer.dataset = [{}]
    trainer._turn_credit_fn = object()
    trainer.logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    async def fake_rollout(sampling_params, prompt_batch):
        del sampling_params, prompt_batch
        return object()

    trainer._run_agentic_rollout = fake_rollout
    trainer._log_agentic_sample_completions = lambda *args: None
    trainer._materialize_agentic_train_batch = lambda *args: ([], [1.0], [])
    monkeypatch.setattr(
        "areno.api.trainers.policy_only.record_dashboard_state",
        lambda *args, **kwargs: None,
    )

    trainer._fit_initialized()

    assert trainer.areno.finish_calls == 1


def test_no_backfill_prefix_is_nested_and_never_exceeds_budget():
    scores = [0.9, 0.8, 0.7]
    masses = [4, 3, 1]
    selections = {
        budget: select_ranked_prefix(scores, masses, budget)
        for budget in range(9)
    }

    for budget, selected in selections.items():
        assert sum(masses[index] for index in selected) <= budget
    assert selections[3] == ()
    assert selections[4] == (0,)
    assert selections[5] == (0,)
    assert selections[7] == (0, 1)
    for lower in range(9):
        for upper in range(lower, 9):
            assert set(selections[lower]).issubset(selections[upper])


def test_fixed_budget_normalization_is_independent_of_selected_denominator():
    spans = [
        agentic.ResponseSpan("assistant_text", 2),
        agentic.ResponseSpan("assistant_text", 2),
    ]
    batch, response_mask, loss_mask = _credit_batch(spans)
    routed_two = route_turn_credit_batch(
        batch=batch,
        result=_result(
            batch,
            [_turn(0, 1.0), _turn(1, 0.0)],
            selected_tokens=2,
            budget_tokens=4,
        ),
        token_row_lengths=[5],
        response_masks=[response_mask],
        base_loss_masks=[loss_mask],
    )
    routed_four = route_turn_credit_batch(
        batch=batch,
        result=_result(
            batch,
            [_turn(0, 1.0), _turn(1, 1.0)],
            selected_tokens=4,
            budget_tokens=4,
        ),
        token_row_lengths=[5],
        response_masks=[response_mask],
        base_loss_masks=[loss_mask],
    )

    objective_two = sum(routed_two.advantages[0]) / routed_two.selected_tokens
    objective_four = sum(routed_four.advantages[0]) / routed_four.selected_tokens
    raw_mass_two = 2.0
    raw_mass_four = 4.0
    assert raw_mass_two / objective_two == pytest.approx(4.0)
    assert raw_mass_four / objective_four == pytest.approx(4.0)


def test_fixed_budget_normalization_matches_microbatch_accumulation_groups():
    spans = [[agentic.ResponseSpan("assistant_text", 2)] for _ in range(4)]
    response_masks = [[False, True, True] for _ in range(4)]
    loss_masks = [
        [False, True, False],
        [False, True, False],
        [False, True, True],
        [False, True, True],
    ]
    batch = build_turn_credit_batch(
        seed=11,
        step=2,
        response_spans=spans,
        response_masks=response_masks,
        loss_masks=loss_masks,
        rollout_logprobs=[[0.0, -0.1, -0.2] for _ in range(4)],
        rewards=[0.0, 1.0, 2.0, 3.0],
        outcome_advantages=[-1.0, -0.5, 0.5, 1.0],
        reward_records=[_record(0, index) for index in range(4)],
    )
    result = {
        "trajectories": [
            {
                "trajectory_id": trajectory.trajectory_id,
                "turns": [_turn(0, 1.0)],
                "selected_tokens": trajectory.spans[0].eligible_token_mass,
                "budget_tokens": 4,
                "audit_calls": 1,
                "diagnostics": {},
            }
            for trajectory in batch.trajectories
        ]
    }
    routed = route_turn_credit_batch(
        batch=batch,
        result=result,
        token_row_lengths=[3] * 4,
        response_masks=response_masks,
        base_loss_masks=loss_masks,
        mini_bs=2,
        gradient_accumulation_steps=2,
    )

    pack_selected = [2, 4]
    pack_objectives = []
    for pack_index, start in enumerate((0, 2)):
        pack_sum = sum(
            sum(routed.advantages[row_index])
            for row_index in range(start, start + 2)
        )
        pack_objectives.append(pack_sum / pack_selected[pack_index])
    engine_accumulated_objective = sum(pack_objectives) / 2
    assert engine_accumulated_objective == pytest.approx(6 / 16)


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda row: row.update(trajectory_id="wrong"), "trajectory_id mismatch"),
        (lambda row: row["turns"][0].update(sign=-1), "sign must match"),
        (lambda row: row.update(selected_tokens=99), "selected_tokens mismatch"),
        (lambda row: row.update(budget_tokens=1), "exceed budget"),
        (lambda row: row["turns"][0].update(confidence=2.0), "confidence"),
    ],
)
def test_malformed_hook_output_is_rejected(mutate, match):
    spans = [agentic.ResponseSpan("assistant_text", 2)]
    batch, response_mask, loss_mask = _credit_batch(spans)
    result = _result(
        batch,
        [_turn(0, 1.0)],
        selected_tokens=2,
        budget_tokens=2,
    )
    mutate(result["trajectories"][0])

    with pytest.raises(ValueError, match=match):
        route_turn_credit_batch(
            batch=batch,
            result=result,
            token_row_lengths=[3],
            response_masks=[response_mask],
            base_loss_masks=[loss_mask],
        )


def test_diagnostics_jsonl_contains_frozen_audit_fields(tmp_path):
    spans = [agentic.ResponseSpan("assistant_text", 2)]
    batch, response_mask, loss_mask = _credit_batch(spans, seed=101, step=12)
    routed = route_turn_credit_batch(
        batch=batch,
        result=_result(
            batch,
            [_turn(0, -1.0, confidence=0.75)],
            selected_tokens=2,
            budget_tokens=2,
            audit_calls=3,
        ),
        token_row_lengths=[3],
        response_masks=[response_mask],
        base_loss_masks=[loss_mask],
    )
    path = tmp_path / "diagnostics.jsonl"
    write_turn_credit_diagnostics(path, routed.diagnostics)
    row = json.loads(path.read_text(encoding="utf-8"))

    required = {
        "schema_version",
        "seed",
        "step",
        "trajectory_id",
        "turn_index",
        "sign",
        "confidence",
        "selected_mass",
        "masked_mass",
        "audit_calls",
    }
    assert required.issubset(row)
    assert row["schema_version"] == SCHEMA_VERSION
    assert row["seed"] == 101
    assert row["step"] == 12
    assert row["sign"] == -1


def test_default_none_hook_preserves_constructed_train_sequences():
    agent_batch = agentic.AgentTrainBatch(
        token_rows=[[1, 10, 11], [1, 20, 21]],
        response_masks=[[False, True, True], [False, True, True]],
        loss_masks=[[False, True, False], [False, True, True]],
        rollout_logprobs=[[0.0, -0.1, -0.2], [0.0, -0.3, -0.4]],
        rewards=[1.0, 3.0],
        records=[{}, {}],
        reward_records=[_record(0, 0), _record(0, 1)],
    )
    trainer = PolicyOnlyTrainer.__new__(PolicyOnlyTrainer)
    trainer._turn_credit_fn = None
    tokenizer = SimpleNamespace(eos_token_id=2)

    train_batch, rewards, rollout_logprobs = trainer._materialize_agentic_train_batch(
        tokenizer,
        None,
        agent_batch,
    )

    assert rewards == [1.0, 3.0]
    assert rollout_logprobs == pytest.approx([-0.1, -0.3, -0.4])
    assert [sequence.model_dump(mode="json") for sequence in train_batch] == [
        {
            "prompt_mask": [True, False, False],
            "loss_mask": [False, True, False],
            "tokens": [1, 10, 11],
            "logprobs": [0.0, -0.1, -0.2],
            "advantages": [0.0, -1.0, 0.0],
            "returns": [],
            "values": [],
            "ref_logprobs": [],
            "reward": 1.0,
            "eos_token_id": 2,
        },
        {
            "prompt_mask": [True, False, False],
            "loss_mask": [False, True, True],
            "tokens": [1, 20, 21],
            "logprobs": [0.0, -0.3, -0.4],
            "advantages": [0.0, 1.0, 1.0],
            "returns": [],
            "values": [],
            "ref_logprobs": [],
            "reward": 3.0,
            "eos_token_id": 2,
        },
    ]


def test_public_config_rejects_unsupported_turn_credit_combinations():
    from areno.api.trainer_config import PolicyTrainerConfig

    common = dict(ckpt="actor", dataset_path="data")
    with pytest.raises(ValueError, match="only --algo grpo"):
        PolicyTrainerConfig(
            algo="gspo",
            agent_fn="agent.py",
            turn_credit_fn_path="credit.py",
            **common,
        )
    with pytest.raises(ValueError, match="requires agent_fn"):
        PolicyTrainerConfig(
            algo="grpo",
            turn_credit_fn_path="credit.py",
            **common,
        )
    with pytest.raises(ValueError, match="requires turn_credit_fn_path"):
        PolicyTrainerConfig(
            algo="grpo",
            turn_credit_config_path="config.json",
            **common,
        )


def test_cli_wires_authorized_turn_credit_paths_before_run(tmp_path, monkeypatch):
    hook_path = tmp_path / "credit.py"
    hook_path.write_text(
        "def route_turn_credit(batch, *, step, config):\n"
        "    return {'trajectories': []}\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "credit.json"
    config_path.write_text('{"budget_tokens": 64}\n', encoding="utf-8")
    captured = []
    monkeypatch.setattr(train_cli, "run", lambda config: captured.append(config))

    result = CliRunner().invoke(
        train_cli.train_command,
        [
            "--algo",
            "grpo",
            "--ckpt",
            "actor",
            "--dataset-path",
            "dataset",
            "--reward-ckpt",
            "reward-model",
            "--agent-fn",
            "examples/agentic/trainable_turns_ablation/run_agent.py",
            "--turn-credit-fn-path",
            str(hook_path),
            "--turn-credit-config-path",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(captured) == 1
    assert captured[0].turn_credit_fn_path == str(hook_path)
    assert captured[0].turn_credit_config_path == str(config_path)


def test_cli_rejects_turn_credit_for_gspo_before_run(tmp_path):
    hook_path = tmp_path / "credit.py"
    hook_path.write_text(
        "def route_turn_credit(batch, *, step, config):\n"
        "    return {'trajectories': []}\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        train_cli.train_command,
        [
            "--algo",
            "gspo",
            "--ckpt",
            "actor",
            "--dataset-path",
            "dataset",
            "--reward-ckpt",
            "reward-model",
            "--agent-fn",
            "examples/agentic/trainable_turns_ablation/run_agent.py",
            "--turn-credit-fn-path",
            str(hook_path),
        ],
    )

    assert result.exit_code != 0
    assert "--turn-credit-fn-path currently supports only --algo grpo" in result.output


def test_reference_hook_generates_a_valid_artifact():
    hook_path = (
        "examples/agentic/trainable_turns_ablation/"
        "outcome_broadcast_turn_credit.py"
    )
    config_path = (
        "examples/agentic/trainable_turns_ablation/"
        "outcome_broadcast_turn_credit.json"
    )
    spans = [
        agentic.ResponseSpan("assistant_tool_call", 2),
        agentic.ResponseSpan("assistant_text", 2),
    ]
    batch, response_mask, loss_mask = _credit_batch(spans, advantage=-0.5)

    result = load_turn_credit_fn(hook_path)(
        batch,
        step=batch.step,
        config=load_turn_credit_config(config_path),
    )
    routed = route_turn_credit_batch(
        batch=batch,
        result=result,
        token_row_lengths=[5],
        response_masks=[response_mask],
        base_loss_masks=[loss_mask],
    )

    assert routed.selected_tokens == 4
    assert routed.budget_tokens == 64
    assert {row["sign"] for row in routed.diagnostics} == {-1}


def test_hook_loader_rejects_bad_signature_before_rollout(tmp_path):
    hook_path = tmp_path / "bad_credit.py"
    hook_path.write_text(
        "def route_turn_credit(batch):\n"
        "    return {'trajectories': []}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"route_turn_credit\(batch, \*, step, config\)"):
        load_turn_credit_fn(str(hook_path))

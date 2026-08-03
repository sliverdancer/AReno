from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from areno.api.backend.areno.backend import ArenoBackend, _packed_trainable_token_count
from areno.api.models import TrainSequence
from areno.api.seeding import epoch_dataset_view
from areno.api.token_budget import (
    TokenBudgetTracker,
    assert_token_budget_outputs_available,
    checkpoint_identity,
    rng_state_identity,
    validate_terminal_evidence,
    write_terminal_evidence,
)
from areno.api.trainers.policy_only import PolicyOnlyTrainer
from areno.engine.data.batch import TrainStats


def test_packed_count_uses_post_mask_next_token_positions():
    pack = {
        "prompt_mask": torch.tensor([[False, True, False, False, False]]),
        "loss_mask": torch.tensor([[True, True, True, False, True]]),
        "lengths": torch.tensor([4]),
    }
    assert _packed_trainable_token_count(pack) == 1


def test_backend_counts_only_tokens_entering_completed_optimizer_steps():
    class _FakeEngine:
        def step(self, packs, gradient_accumulation_steps=None):
            assert len(packs) == 2
            assert gradient_accumulation_steps is None
            return [
                TrainStats(loss=2.0, stepped=False, global_step=0),
                TrainStats(loss=4.0, stepped=True, global_step=1),
            ]

    backend = ArenoBackend()
    backend._engine = _FakeEngine()
    sequences = [
        TrainSequence(tokens=[1, 2, 3], prompt_mask=[True, False, False], logprobs=[0.0] * 3),
        TrainSequence(
            tokens=[1, 2, 3, 4],
            prompt_mask=[True, False, False, False],
            loss_mask=[False, True, False, True],
            logprobs=[0.0] * 4,
        ),
    ]
    result = backend.train(None, sequences, lambda *_args: None, mini_bs=1)
    assert result["completed_trainable_tokens"] == 4.0
    assert result["optimizer_steps_completed"] == 1.0
    assert result["optimizer_global_step"] == 1.0


def test_backend_exposes_multiple_completed_steps_for_strict_tracker_rejection():
    class _FakeEngine:
        def step(self, packs, gradient_accumulation_steps=None):
            del gradient_accumulation_steps
            return [
                TrainStats(loss=1.0, stepped=True, global_step=index + 1)
                for index, _pack in enumerate(packs)
            ]

    backend = ArenoBackend()
    backend._engine = _FakeEngine()
    sequence = TrainSequence(tokens=[1, 2], prompt_mask=[True, False], logprobs=[0.0, 0.0])
    result = backend.train(None, [sequence, sequence], lambda *_args: None, mini_bs=1)
    assert result["optimizer_steps_completed"] == 2.0
    with pytest.raises(RuntimeError, match="exactly one"):
        TokenBudgetTracker(2).consume(result)


def test_backend_skips_index_zero_only_signal_without_engine_step():
    class _FakeEngine:
        def step(self, *_args, **_kwargs):
            raise AssertionError("index zero is not a causal next-token target")

    backend = ArenoBackend()
    backend._engine = _FakeEngine()
    sequence = TrainSequence(tokens=[1], prompt_mask=[False], loss_mask=[True], logprobs=[0.0])
    result = backend.train(None, [sequence], lambda *_args: None, mini_bs=1)
    assert result["optimizer_steps_completed"] == 0.0
    assert result["completed_trainable_tokens"] == 0.0


def test_tracker_exact_hit_and_overshoot_are_complete_step_semantics():
    exact = TokenBudgetTracker(5)
    result = exact.consume(
        {"completed_trainable_tokens": 5.0, "optimizer_steps_completed": 1.0, "optimizer_global_step": 1.0}
    )
    assert result == {
        "target": 5,
        "tokens_before_step": 0,
        "terminal_step_tokens": 5,
        "tokens_after_step": 5,
        "overshoot": 0,
        "optimizer_global_step": 1,
        "target_reached": True,
        "optimizer_step_skipped": False,
        "optimizer_step_skipped_batches": 0,
    }

    overshoot = TokenBudgetTracker(5)
    first = overshoot.consume(
        {"completed_trainable_tokens": 3, "optimizer_steps_completed": 1, "optimizer_global_step": 1}
    )
    second = overshoot.consume(
        {"completed_trainable_tokens": 4, "optimizer_steps_completed": 1, "optimizer_global_step": 2}
    )
    assert first["target_reached"] is False
    assert second["tokens_before_step"] == 3
    assert second["tokens_after_step"] == 7
    assert second["overshoot"] == 2


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"completed_trainable_tokens": 1, "optimizer_steps_completed": 0, "optimizer_global_step": 0},
        {"completed_trainable_tokens": 1, "optimizer_steps_completed": 2, "optimizer_global_step": 2},
        {"completed_trainable_tokens": 0, "optimizer_steps_completed": 1, "optimizer_global_step": 1},
        {"completed_trainable_tokens": 1.5, "optimizer_steps_completed": 1, "optimizer_global_step": 1},
    ],
)
def test_tracker_rejects_ambiguous_or_incomplete_optimizer_events(result):
    with pytest.raises(RuntimeError):
        TokenBudgetTracker(10).consume(result)


def test_tracker_rejects_non_contiguous_optimizer_global_steps():
    tracker = TokenBudgetTracker(10)
    tracker.consume({"completed_trainable_tokens": 2, "optimizer_steps_completed": 1, "optimizer_global_step": 1})
    with pytest.raises(RuntimeError, match="not contiguous"):
        tracker.consume(
            {"completed_trainable_tokens": 2, "optimizer_steps_completed": 1, "optimizer_global_step": 3}
        )


def test_tracker_accepts_zero_step_batch_without_advancing_budget():
    tracker = TokenBudgetTracker(10)
    result = tracker.consume({"completed_trainable_tokens": 0, "optimizer_steps_completed": 0})
    assert result["optimizer_step_skipped"] is True
    assert result["tokens_after_step"] == 0
    assert tracker.skipped_batches == 1


def test_dataset_and_rng_identities_are_deterministic_and_scoped():
    first = epoch_dataset_view(list(range(12)), seed=42, epoch=3)
    second = epoch_dataset_view(list(range(12)), seed=42, epoch=3)
    assert first.seed == second.seed
    assert first.order_sha256 == second.order_sha256
    identity = rng_state_identity()
    assert identity["scope"] == "parent_process_only"
    assert identity["worker_rng_state_saved"] is False
    assert len(identity["sha256"]) == 64


def test_terminal_evidence_is_atomic_non_overwriting_and_checkpoint_bound(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "weights.bin").write_bytes(b"weights")
    identity = checkpoint_identity(str(checkpoint))
    assert identity["file_count"] == 1
    assert len(identity["manifest_sha256"]) == 64

    evidence_dir = tmp_path / "metrics"
    destination = write_terminal_evidence(str(evidence_dir), {"status": "TARGET_REACHED"})
    assert json.loads(destination.read_text()) == {"status": "TARGET_REACHED"}
    with pytest.raises(RuntimeError, match="already exists"):
        write_terminal_evidence(str(evidence_dir), {"status": "TARGET_NOT_REACHED"})
    with pytest.raises(RuntimeError, match="already exists"):
        assert_token_budget_outputs_available(str(evidence_dir), str(tmp_path / "save"), 10)


def test_preflight_rejects_existing_terminal_checkpoint(tmp_path):
    checkpoint = tmp_path / "save" / "token_budget_target_10_optimizer_step_000001"
    checkpoint.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="checkpoint already exists"):
        assert_token_budget_outputs_available(str(tmp_path / "metrics"), str(tmp_path / "save"), 10)


def test_policy_terminal_checkpoint_precedes_bound_evidence(tmp_path, monkeypatch):
    events = []

    class _FakeAreno:
        def save_checkpoint(self, path):
            events.append("checkpoint")
            checkpoint = type(tmp_path)(path)
            checkpoint.mkdir(parents=True)
            (checkpoint / "weights.bin").write_bytes(b"weights")
            return str(checkpoint)

    trainer = PolicyOnlyTrainer.__new__(PolicyOnlyTrainer)
    trainer.config = SimpleNamespace(
        save_path=str(tmp_path / "save"),
        metrics_log_dir=str(tmp_path / "metrics"),
        seed=42,
        dataset_path="dataset",
    )
    trainer.areno = _FakeAreno()
    trainer.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "areno.api.trainers.policy_only.record_dashboard_state",
        lambda *_args, **_kwargs: None,
    )
    dataset_view = type(
        "DatasetView",
        (),
        {"seed": 123, "order_sha256": "a" * 64, "__len__": lambda self: 4},
    )()
    trainer._complete_token_budget(
        accounting={
            "target": 5,
            "tokens_before_step": 3,
            "terminal_step_tokens": 4,
            "tokens_after_step": 7,
            "overshoot": 2,
            "optimizer_global_step": 2,
            "target_reached": True,
            "optimizer_step_skipped": False,
            "optimizer_step_skipped_batches": 1,
        },
        epoch=0,
        trainer_step=1,
        prompt_batch_index=1,
        rows_before_batch=2,
        rows_after_batch=4,
        accepted_items=2,
        accepted_records=[{"id": 1}, {"id": 2}],
        dataset_view=dataset_view,
        current_rollout_seed=7,
        next_rollout_seed=8,
        rng_before_step={"sha256": "b" * 64},
        rng_after_train={"sha256": "c" * 64},
    )
    events.append("evidence")
    evidence = json.loads((tmp_path / "metrics" / "token_budget_terminal.json").read_text())
    assert events == ["checkpoint", "evidence"]
    assert evidence["status"] == "TARGET_REACHED"
    assert evidence["trainer_iteration"] == 2
    assert evidence["checkpoint"]["file_count"] == 1
    assert validate_terminal_evidence(str(tmp_path / "metrics" / "token_budget_terminal.json")) == evidence

    evidence["overshoot"] = 3
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="overshoot"):
        validate_terminal_evidence(str(tampered), verify_checkpoint=False)


def test_policy_fit_rejects_existing_evidence_before_runtime_init(tmp_path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    (metrics / "token_budget_terminal.json").write_text("{}")
    trainer = PolicyOnlyTrainer.__new__(PolicyOnlyTrainer)
    trainer.config = SimpleNamespace(
        seed=42,
        max_trainable_tokens=5,
        metrics_log_dir=str(metrics),
        save_path=str(tmp_path / "save"),
    )
    trainer.areno = SimpleNamespace(init=lambda: pytest.fail("runtime init must not be called"))
    with pytest.raises(RuntimeError, match="already exists"):
        trainer.fit()


def test_policy_loop_records_empty_skip_then_stops_immediately_at_target(tmp_path, monkeypatch):
    class _FakeAreno:
        def __init__(self):
            self.finish_calls = 0
            self.train_calls = 0
            self.save_calls = []

        def get_tokenizer(self):
            return SimpleNamespace(eos_token_id=2)

        def load_prompt_batches(self, dataset, *, batch_size, max_prompt_tokens):
            del dataset, batch_size, max_prompt_tokens
            item0 = SimpleNamespace(record={"id": 0})
            item1 = SimpleNamespace(record={"id": 1})
            return [
                SimpleNamespace(items=[item0], scanned=1),
                SimpleNamespace(items=[item1], scanned=1),
            ]

        def train(self, *_args, **_kwargs):
            self.train_calls += 1
            return {
                "completed_trainable_tokens": 2,
                "optimizer_steps_completed": 1,
                "optimizer_global_step": 1,
            }

        def finish_step(self):
            self.finish_calls += 1

        def save_checkpoint(self, path):
            self.save_calls.append(path)
            checkpoint = type(tmp_path)(path)
            checkpoint.mkdir(parents=True)
            (checkpoint / "weights.bin").write_bytes(b"weights")
            return str(checkpoint)

    trainer = PolicyOnlyTrainer.__new__(PolicyOnlyTrainer)
    trainer.config = SimpleNamespace(
        seed=42,
        chat_template_enable_thinking=None,
        greedy=False,
        temperature=1.0,
        max_new_tokens=8,
        max_context_len=64,
        max_prompt_tokens=16,
        top_k=-1,
        top_p=1.0,
        epochs=3,
        batch_size=1,
        mini_bs=1,
        gradient_accumulation_steps=None,
        max_steps=None,
        max_trainable_tokens=2,
        save_interval=1,
        save_path=str(tmp_path / "save"),
        metrics_log_dir=str(tmp_path / "metrics"),
        dataset_path="dataset",
        agent_fn="agent.py",
    )
    trainer.areno = _FakeAreno()
    trainer.dataset = [{"id": 0}, {"id": 1}]
    trainer.loss_fn = object()
    trainer._turn_credit_fn = object()
    trainer.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    rollout_calls = []

    async def fake_rollout(sampling_params, prompt_batch):
        rollout_calls.append((sampling_params.seed, prompt_batch.items[0].record["id"]))
        return object()

    trainer._run_agentic_rollout = fake_rollout
    trainer._log_agentic_sample_completions = lambda *_args: None
    batches = iter([([], [0.0], []), ([object()], [1.0], [])])
    trainer._materialize_agentic_train_batch = lambda *_args: next(batches)
    monkeypatch.setattr(
        "areno.api.trainers.policy_only.record_dashboard_state",
        lambda *_args, **_kwargs: None,
    )

    trainer._fit_initialized()

    evidence = json.loads((tmp_path / "metrics" / "token_budget_terminal.json").read_text())
    assert len(rollout_calls) == 2
    assert trainer.areno.finish_calls == 1
    assert trainer.areno.train_calls == 1
    assert len(trainer.areno.save_calls) == 1
    assert evidence["optimizer_step_skipped_batches"] == 1
    assert evidence["tokens_after_step"] == 2

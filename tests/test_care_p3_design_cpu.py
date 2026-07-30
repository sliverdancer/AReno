from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from areno.api.rewards import RewardEvent, RewardRecord
from areno.experimental.care import (
    CalibrationTrajectory,
    CalibrationTurn,
    TurnCreditBatch,
    TurnCreditSpan,
    TurnCreditTrajectory,
    calibrate_block_threshold,
    route_turn_credit_batch,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "agentic" / "care_bifurcation"


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tool_call_json(bit: int) -> str:
    return json.dumps(
        {
            "id": "call",
            "type": "function",
            "function": {
                "name": "choose_bit",
                "arguments": json.dumps({"bit": bit}),
            },
        },
        sort_keys=True,
    )


def _batch(*, duplicate_prompt: bool = False, invalid_prompt: int | None = None):
    generator = _load_module("care_p3_generator", "dataset_generator.py")
    task = _load_module("care_p3_task", "task.py")
    records = generator.generate_records(20, seed=7301)
    trajectories = []
    for prompt_index, record in enumerate(records):
        weights = tuple(record["weights"])
        actions = [int(weight > 0) for weight in weights]
        spans = []
        for turn_index, action in enumerate(actions):
            raw_calls = (
                ("{",)
                if invalid_prompt == prompt_index and turn_index == 0
                else (_tool_call_json(action),)
            )
            spans.append(
                TurnCreditSpan(
                    turn_index=turn_index,
                    kind="assistant_tool_call",
                    response_start=turn_index * 10,
                    response_end=(turn_index + 1) * 10,
                    token_mass=10,
                    eligible_token_mass=10,
                    eligibility_mask=(True,) * 10,
                    rollout_logprobs=(-0.1,) * 10,
                    raw_text="call",
                    raw_tool_calls_json=raw_calls,
                )
            )
        spans.append(
            TurnCreditSpan(
                turn_index=4,
                kind="assistant_text",
                response_start=40,
                response_end=45,
                token_mass=5,
                eligible_token_mass=5,
                eligibility_mask=(True,) * 5,
                rollout_logprobs=(-0.1,) * 5,
                raw_text="summary",
                raw_tool_calls_json=(),
            )
        )
        resolved_prompt = 0 if duplicate_prompt and prompt_index == 1 else prompt_index
        trajectories.append(
            TurnCreditTrajectory(
                trajectory_id=f"p{resolved_prompt}:s0:r{prompt_index}",
                prompt_index=resolved_prompt,
                sample_index=0,
                terminal_reward=float(
                    task.terminal_reward(
                        weights,
                        int(record["threshold"]),
                        actions,
                    )
                ),
                outcome_advantage=0.0,
                spans=tuple(spans),
                source_record_json=json.dumps(record, sort_keys=True),
                messages_json="[]",
                trace_json="[]",
            )
        )
    return TurnCreditBatch(
        schema_version="care.turn_credit.v1",
        seed=3101,
        step=0,
        trajectories=tuple(trajectories),
    )


def _route(router, batch, config):
    result = router.route_turn_credit(batch, step=0, config=config)
    routed = route_turn_credit_batch(
        batch=batch,
        result=result,
        token_row_lengths=[45] * len(batch.trajectories),
        response_masks=[[True] * 45 for _ in batch.trajectories],
        base_loss_masks=[[True] * 45 for _ in batch.trajectories],
        mini_bs=5,
        gradient_accumulation_steps=4,
    )
    return result, routed


def test_block_calibration_rejects_too_few_trajectories_for_alpha():
    trajectories = [
        CalibrationTrajectory(
            trajectory_id=f"t{index}",
            turns=(
                CalibrationTurn(
                    turn_index=0,
                    confidence=0.9,
                    predicted_sign=1,
                    oracle_sign=1,
                    token_mass=1,
                ),
            ),
        )
        for index in range(8)
    ]

    with pytest.raises(ValueError, match="insufficient calibration trajectories"):
        calibrate_block_threshold(
            trajectories,
            alpha=0.10,
            budget_tokens=4,
        )


def test_care_router_separates_calibration_and_update_and_matches_audit_cost():
    router = _load_module("care_p3_router", "care_router.py")
    batch = _batch()
    base_config = {
        "alpha": 0.2,
        "budget_tokens": 256,
        "min_calibration_trajectories": 10,
        "min_update_trajectories": 10,
        "scorer_seed": 9102,
    }
    care_result, care_routed = _route(
        router,
        batch,
        {**base_config, "mode": "care"},
    )
    control_result, control_routed = _route(
        router,
        batch,
        {**base_config, "mode": "uncalibrated"},
    )

    care_rows = care_result["trajectories"]
    control_rows = control_result["trajectories"]
    calibration_rows = care_rows[::2]
    update_rows = care_rows[1::2]
    assert all(row["budget_tokens"] == 0 for row in calibration_rows)
    assert all(row["selected_tokens"] == 0 for row in calibration_rows)
    assert all(row["budget_tokens"] == 256 for row in update_rows)
    assert sum(row["audit_calls"] for row in care_rows) == 300
    assert sum(row["audit_calls"] for row in control_rows) == 300
    assert care_routed.budget_tokens == control_routed.budget_tokens == 2560
    assert care_routed.selected_tokens <= control_routed.selected_tokens
    assert care_routed.selected_tokens > 0


def test_invalid_update_trajectory_is_retained_but_fully_abstained():
    router = _load_module("care_p3_router_invalid", "care_router.py")
    batch = _batch(invalid_prompt=1)
    result, routed = _route(
        router,
        batch,
        {
            "mode": "care",
            "alpha": 0.2,
            "budget_tokens": 256,
            "min_calibration_trajectories": 10,
            "min_update_trajectories": 10,
            "scorer_seed": 9102,
        },
    )

    invalid_row = result["trajectories"][1]
    assert invalid_row["diagnostics"]["valid_trajectory"] is False
    assert invalid_row["selected_tokens"] == 0
    assert routed.loss_masks[1] == [False] * 45


def test_router_rejects_duplicate_prompt_blocks():
    router = _load_module("care_p3_router_duplicates", "care_router.py")
    batch = _batch(duplicate_prompt=True)

    with pytest.raises(ValueError, match="requires n_samples=1"):
        router.route_turn_credit(
            batch,
            step=0,
            config={
                "mode": "care",
                "alpha": 0.2,
                "budget_tokens": 256,
                "min_calibration_trajectories": 10,
                "min_update_trajectories": 10,
                "scorer_seed": 9102,
            },
        )


def test_dataset_and_reward_are_deterministic_and_strict():
    generator = _load_module("care_p3_generator_deterministic", "dataset_generator.py")
    reward = _load_module("care_p3_reward", "reward.py")
    first = generator.generate_records(20, seed=7301)
    second = generator.generate_records(20, seed=7301)
    assert first == second
    source = first[0]
    actions = [int(weight > 0) for weight in source["weights"]]
    valid_record = RewardRecord(
        prompt="p",
        completion="c",
        source_record=source,
        trace=[
            RewardEvent(
                type="assistant_tool_call",
                name="choose_bit",
                arguments={"bit": bit},
            )
            for bit in actions
        ],
    )
    invalid_record = valid_record.model_copy(
        update={"trace": valid_record.trace[:-1]}
    )

    assert reward.reward_fn(valid_record) == 1.0
    assert reward.reward_fn(invalid_record) == 0.0


def test_agent_tool_executor_rejects_multiple_calls():
    agent = _load_module("care_p3_agent", "run_agent.py")
    message = {
        "tool_calls": [
            {
                "id": "a",
                "function": {"name": "choose_bit", "arguments": '{"bit": 1}'},
            },
            {
                "id": "b",
                "function": {"name": "choose_bit", "arguments": '{"bit": 0}'},
            },
        ]
    }

    results, accepted = agent._tool_result_messages(
        message,
        weights=(2, -1, 1, -2),
        turn_index=0,
        valid_actions=[],
    )

    assert accepted is None
    assert len(results) == 2
    assert all(json.loads(result["content"])["ok"] is False for result in results)


def test_p3_prepare_writes_six_commands_and_gpu_block(tmp_path, monkeypatch):
    prepare = _load_module("prepare_p3", "prepare_p3.py")
    monkeypatch.setattr(
        prepare,
        "_git_output",
        lambda _repo_root, *args: (
            "fixture-commit" if args == ("rev-parse", "HEAD") else ""
        ),
    )

    manifest = prepare.prepare(
        repo_root=REPO_ROOT,
        run_root=tmp_path,
        ckpt=prepare.DEFAULT_CKPT,
        dataset_seed=prepare.DEFAULT_DATASET_SEED,
        count=20,
        max_steps=1,
    )

    assert manifest["authorization"] == "PREPARE_ONLY_GPU_NOT_AUTHORIZED"
    assert manifest["protocol"] == "CARE-P3-PILOT-v0.2"
    assert manifest["supersedes_protocol"] == "CARE-P3-PILOT-v0.1"
    assert manifest["dynamic_hook_preflight"]["status"] == "passed"
    assert manifest["dynamic_hook_preflight"]["callable"] == "route_turn_credit"
    assert len(manifest["commands"]) == 6
    assert manifest["train_seeds"] == [3101, 3102, 3103]
    assert manifest["model_asset"]["full_file_hash_verification_required"] is True
    assert manifest["fixed_design"]["maximum_total_gpu_hours"] == 6
    assert manifest["fixed_design"]["maximum_total_spend_cny"] == 60
    assert (tmp_path / "GPU_EXECUTION_NOT_AUTHORIZED").exists()
    for run_id, command in manifest["commands"].items():
        assert "--algo grpo" in command
        assert "--n-samples 1" in command
        assert "--gradient-accumulation-steps 4" in command
        assert "--disable-thinking" in command
        seed = run_id.rsplit("-", 1)[-1]
        assert f"--seed {seed}" in command


def test_modelscope_snapshot_verifier_accepts_exact_files_and_rejects_drift(
    tmp_path,
):
    verifier = _load_module("care_p3_asset_verifier", "fetch_modelscope_snapshot.py")
    content = b"frozen-model-fixture"
    manifest = {
        "hub": "modelscope",
        "model_id": "fixture/model",
        "revision": "master",
        "files": [
            {
                "path": "weights.bin",
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        ],
    }
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "weights.bin").write_bytes(content)

    summary = verifier.verify_snapshot(snapshot, manifest)
    assert summary["verified_file_count"] == 1

    (snapshot / "weights.bin").write_bytes(b"drift")
    with pytest.raises(ValueError, match="content_mismatch"):
        verifier.verify_snapshot(snapshot, manifest)


def test_p3_executor_rejects_dirty_or_remote_checkpoint(tmp_path):
    prepare = _load_module("prepare_p3", "prepare_p3.py")
    executor = _load_module("care_p3_executor", "execute_p3.py")
    manifest = {
        "protocol": prepare.PROTOCOL,
        "authorization": "PREPARE_ONLY_GPU_NOT_AUTHORIZED",
        "commands": {
            f"{arm}-seed-{seed}": "areno train"
            for seed in prepare.TRAIN_SEEDS
            for arm in prepare.ARMS
        },
        "git_commit": "unused",
        "git_status": " M dirty.py",
        "checkpoint": "Qwen/Qwen3-0.6B",
    }

    with pytest.raises(ValueError, match="dirty checkout"):
        executor.validate_preflight(manifest, repo_root=REPO_ROOT)


def test_p3_collector_deduplicates_audit_calls_and_writes_artifacts(tmp_path):
    _load_module("prepare_p3", "prepare_p3.py")
    collector = _load_module("care_p3_collector", "collect_p3.py")
    diagnostics = [
        {
            "step": 0,
            "trajectory_id": "p0:s0:r0",
            "selected_mass": 0,
            "masked_mass": 10,
            "audit_calls": 30,
            "trajectory_diagnostics": {
                "threshold": 0.6,
                "role": "calibration",
                "valid_trajectory": True,
            },
        },
        {
            "step": 0,
            "trajectory_id": "p0:s0:r0",
            "selected_mass": 0,
            "masked_mass": 10,
            "audit_calls": 30,
            "trajectory_diagnostics": {
                "threshold": 0.6,
                "role": "calibration",
                "valid_trajectory": True,
            },
        },
        {
            "step": 0,
            "trajectory_id": "p1:s0:r1",
            "selected_mass": 10,
            "masked_mass": 0,
            "audit_calls": 0,
            "trajectory_diagnostics": {
                "threshold": 0.6,
                "role": "update",
                "valid_trajectory": True,
            },
        },
    ]
    summary = collector.summarize_diagnostics(diagnostics)
    assert summary[0]["audit_calls"] == 30
    assert summary[0]["calibration_blocks"] == 1
    assert summary[0]["update_blocks"] == 1

    series_by_run = {}
    diagnostics_by_run = {}
    for arm in collector.ARMS:
        for seed in collector.TRAIN_SEEDS:
            series_by_run[(arm, seed)] = {
                field: {0: float(index + 1)}
                for index, field in enumerate(collector.METRIC_TAGS)
            }
            diagnostics_by_run[(arm, seed)] = summary
    rows = collector.build_rows(series_by_run, diagnostics_by_run)
    csv_path, json_path = collector.write_artifacts(
        rows,
        tmp_path,
        manifest={"protocol": "fixture"},
    )

    assert len(rows) == 6
    assert csv_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["manifest"]["protocol"] == "fixture"
    assert payload["rows"] == rows

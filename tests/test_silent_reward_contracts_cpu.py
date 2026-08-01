from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from research.silent_reward_contracts.auditor import (
    canonical_action_arguments,
    masks_are_distinguishable,
    reward_is_informative,
)
from research.silent_reward_contracts.generate_p1_artifacts import (
    SCHEMA_VERSION,
    write_evidence,
)
from research.silent_reward_contracts.generate_p2_artifacts import (
    SCHEMA_VERSION as P2_SCHEMA_VERSION,
    write_evidence as write_p2_evidence,
)
from research.silent_reward_contracts.arca import RULE_CODES, audit_case
from research.silent_reward_contracts.build_evaluation_cases import build_dev_cases
from research.silent_reward_contracts.evaluate_auditor import evaluate
from research.silent_reward_contracts.openrlhf_adapter import build_heldout_cases
from research.silent_reward_contracts.summarize_p3 import write_summary
from research.silent_reward_contracts.areal_adapter import build_replication
from research.silent_reward_contracts.p4.collect_p4 import build_rows
from research.silent_reward_contracts.p4.prepare_p4 import build_commands, prepare
from research.silent_reward_contracts.p4.execute_p4 import EXPECTED_ORDER
from research.silent_reward_contracts.p4.reward_canonical import (
    reward_fn as canonical_reward,
)
from research.silent_reward_contracts.p4.reward_strict import reward_fn as strict_reward
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_canonical_action_arguments_accepts_equivalent_object_and_string():
    expected = {"bit": 1}
    assert canonical_action_arguments(expected) == expected
    assert canonical_action_arguments('{"bit": 1}') == expected


@pytest.mark.parametrize("value", ["{", "[]", None, 1])
def test_canonical_action_arguments_rejects_non_objects(value):
    with pytest.raises(ValueError):
        canonical_action_arguments(value)


def test_identifiability_checks_are_explicit():
    assert masks_are_distinguishable([True, False], [False, True])
    assert not masks_are_distinguishable([True, False], [True, False])
    assert reward_is_informative([0.0, 1.0])
    assert not reward_is_informative([0.0, 0.0])


def test_p1_artifact_generation_reproduces_production_mismatch(tmp_path):
    evidence = write_evidence(tmp_path, REPO_ROOT)
    assert evidence["schema_version"] == SCHEMA_VERSION
    assert evidence["gpu_executed"] is False
    assert evidence["production_argument_types"] == ["str"]

    reward_case, alias_case = evidence["cases"]
    assert reward_case["observed_value"] == 0.0
    assert reward_case["oracle_value"] == 1.0
    assert reward_case["detected"] is True
    assert alias_case["contract_status"] == "PASS_DECLARED_EQUIVALENCE_CONTROL"
    assert evidence["masks"]["all_assistant"] == [True, True, True, True]
    assert evidence["masks"]["last_assistant"] == [False, False, True, True]
    assert evidence["masks"]["final_answer"] == [False, False, True, True]

    payload = json.loads(
        (tmp_path / "production_contract_cases.json").read_text(encoding="utf-8")
    )
    with (tmp_path / "production_contract_cases.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert payload == evidence
    assert [row["case_id"] for row in rows] == [
        "ARCA-P1-F1-STRINGIFIED-ARGUMENTS",
        "ARCA-P1-F5-ISSUE199-ALIAS",
    ]


def test_p1_artifacts_are_byte_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_evidence(first, REPO_ROOT)
    write_evidence(second, REPO_ROOT)
    for name in ("production_contract_cases.json", "production_contract_cases.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def _write_minimal_verl_snapshot(root: Path) -> None:
    sources = {
        "verl/experimental/agent_loop/tool_agent_loop.py": """
output.extra_fields.update({"turn_scores": [], "tool_rewards": rewards})
""",
        "verl/experimental/agent_loop/agent_loop.py": """
default_extra_keys = {"turn_scores", "tool_rewards"}
""",
        "verl/workers/reward_manager/naive.py": """
from collections import defaultdict
from typing import Any
import torch
from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", compute_score_timeout=None):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.compute_score_timeout = compute_score_timeout

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["rollout_reward_scores"] = rollout_reward_scores
            try:
                score = self.compute_score(
                    data_source=data_item.non_tensor_batch[self.reward_fn_key],
                    solution_str="decoded",
                    ground_truth=data_item.non_tensor_batch["reward_model"]["ground_truth"],
                    extra_info=extra_info,
                )
            except TimeoutError:
                print("assigning reward 0.0")
                score = 0.0
            reward_tensor[i, valid_response_length - 1] = score
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
""",
        "verl/tools/function_tool.py": 'SHAPES = "(response, reward)"\n',
    }
    for relative, source in sources.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source.strip() + "\n", encoding="utf-8")


def test_p2_artifact_generation_detects_cross_system_contracts(tmp_path):
    verl_root = tmp_path / "verl"
    _write_minimal_verl_snapshot(verl_root)
    output = tmp_path / "output"
    evidence = write_p2_evidence(output, REPO_ROOT, verl_root)
    assert evidence["schema_version"] == P2_SCHEMA_VERSION
    assert evidence["gpu_executed"] is False
    assert evidence["summary"] == {
        "natural_medium_or_high_detected": 3,
        "independent_frameworks": 2,
        "distinct_failure_classes": 3,
        "conclusion_flips": 2,
    }
    assert all(case["detected"] for case in evidence["cases"])
    assert json.loads(
        (output / "cross_system_cases.json").read_text(encoding="utf-8")
    ) == evidence


def test_p2_artifacts_are_byte_deterministic(tmp_path):
    verl_root = tmp_path / "verl"
    _write_minimal_verl_snapshot(verl_root)
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_p2_evidence(first, REPO_ROOT, verl_root)
    write_p2_evidence(second, REPO_ROOT, verl_root)
    for name in ("cross_system_cases.json", "cross_system_cases.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_frozen_dev_evaluator_detects_all_registered_single_faults():
    metrics, rows = evaluate(build_dev_cases())
    assert metrics["macro_recall"] == 1.0
    assert metrics["clean_false_positive_rate"] == 0.0
    assert metrics["high_natural_recall"] == 1.0
    assert metrics["natural_conclusion_flips_detected"] == 2
    assert metrics["wall_time_under_limit"] is True
    assert set(metrics["rule_metrics"]) == set(RULE_CODES)
    assert all(row["exact_match"] for row in rows)


def test_frozen_rules_do_not_exempt_undeclared_aliases():
    case = {
        "payload": {
            "treatment_masks": {"arm-a": [1, 0], "arm-b": [1, 0]},
            "declared_aliases": [],
        }
    }
    assert audit_case(case) == ["ARCA-F5-TREATMENT"]


def _write_minimal_openrlhf_snapshot(root: Path) -> None:
    sources = {
        "openrlhf/utils/agent.py": """
step_reward = step_result["rewards"]
final_response = {"reward": total_reward}
output = {"reward": None}
""",
        "openrlhf/trainer/ppo_utils/samples_generator.py": """
reward_val = response.get("reward", None)
""",
        "openrlhf/cli/train_ppo_ray.py": """
assert n_samples_per_prompt > 1, "requires n_samples_per_prompt > 1"
""",
    }
    for relative, source in sources.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source.strip() + "\n", encoding="utf-8")


def test_heldout_adapter_builds_four_workloads_without_changing_rules(tmp_path):
    root = tmp_path / "openrlhf"
    _write_minimal_openrlhf_snapshot(root)
    payload = build_heldout_cases(root)
    metrics, rows = evaluate(payload)
    assert payload["split"] == "heldout"
    assert len(payload["cases"]) == 80
    assert {case["workload"] for case in payload["cases"]} == {
        "reward_flow",
        "group_advantage",
        "treatment_identity",
        "failure_status",
    }
    assert metrics["macro_recall"] == 1.0
    assert metrics["clean_false_positive_rate"] == 0.0
    assert metrics["high_natural_recall"] is None
    assert all(row["exact_match"] for row in rows)


def test_p3_summary_is_byte_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_summary = write_summary(first, REPO_ROOT)
    second_summary = write_summary(second, REPO_ROOT)
    assert first_summary["outcome"] == (
        "PASS_P3_CPU_AUDITOR_TO_GPU_AUTHORIZATION_REQUEST"
    )
    assert first_summary["conditions"] == second_summary["conditions"]
    assert first_summary["evidence_strength"] == {
        "development_natural_failures": 3,
        "heldout_natural_failures": 0,
        "heldout_source_derived_clean_cases": 40,
        "heldout_mutation_failures": 40,
        "post_freeze_replication_natural_failures": 1,
    }
    assert first_summary["post_freeze_replication"]["exact_match"] is True
    for name in ("gate_summary.json", "baseline_comparison.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def _write_minimal_areal_snapshot(root: Path) -> None:
    types_source = '''
from __future__ import annotations
from dataclasses import dataclass, field
import torch
from openai.types.chat import ChatCompletion
from openai.types.responses.response import Response
from openai.types.responses.response_input_param import ResponseInputParam
from areal.api import ModelResponse
from areal.utils import logging

@dataclass
class InteractionWithTokenLogpReward:
    model_response: ModelResponse | None = None
    reward: float | None = None
    original_reward: float | None = None
    messages: list[dict] = field(default_factory=list)
    _cache: dict[str, torch.Tensor] | None = None

    def to_tensor_dict(self):
        resp = self.model_response
        seq = resp.input_tokens + resp.output_tokens
        reward = self.reward if self.reward is not None else 0.0
        original_reward = self.original_reward if self.original_reward is not None else reward
        return {
            "input_ids": torch.tensor(seq).unsqueeze(0),
            "rewards": torch.tensor([float(reward)]),
            "original_rewards": torch.tensor([float(original_reward)]),
        }
'''
    sources = {
        "areal/experimental/openai/types.py": types_source,
        "areal/experimental/openai/cache.py": 'WARNING = "does not have a reward set"\n',
        "areal/api/cli_args.py": 'WARNING = "singleton group centering erases the task reward"\n',
    }
    for relative, source in sources.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source.strip() + "\n", encoding="utf-8")


def test_areal_optional_replication_detects_missing_reward_conflation(tmp_path):
    root = tmp_path / "areal"
    _write_minimal_areal_snapshot(root)
    result = build_replication(root)
    assert result["dynamic_probe"]["input_reward"] is None
    assert result["dynamic_probe"]["output_reward"] == 0.0
    assert result["dynamic_probe"]["structured_missing_status"] is False
    assert result["predicted_violations"] == ["ARCA-F6-STATUS"]
    assert result["exact_match"] is True


def test_p4_reward_intervention_changes_only_argument_normalization():
    source = {"weights": [1, 1, 1, 1], "threshold": 0}
    events = [
        SimpleNamespace(
            type="assistant_tool_call",
            name="choose_bit",
            arguments='{"bit": 1}',
        )
        for _ in range(4)
    ]
    record = SimpleNamespace(trace=events, source_record=source)
    assert strict_reward(record) == 0.0
    assert canonical_reward(record) == 1.0


def test_p4_command_matrix_is_paired_and_minimal(tmp_path):
    commands = build_commands(REPO_ROOT, tmp_path, tmp_path / "data.jsonl", "/model")
    assert list(commands) == [
        "strict-seed-3101", "canonical-seed-3101",
        "strict-seed-3102", "canonical-seed-3102",
        "strict-seed-3103", "canonical-seed-3103",
    ]
    for seed in (3101, 3102, 3103):
        strict = commands[f"strict-seed-{seed}"]
        canonical = commands[f"canonical-seed-{seed}"]
        differing = [
            (left, right)
            for left, right in zip(strict, canonical, strict=True)
            if left != right
        ]
        assert len(differing) == 2
        assert "reward_strict.py" in differing[0][0]
        assert "reward_canonical.py" in differing[0][1]


def test_p4_manifest_persists_frozen_run_order(tmp_path, monkeypatch):
    def fake_git(command, **_):
        stdout = "b280a85\n" if command[1:3] == ["rev-parse", "HEAD"] else ""
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(
        "research.silent_reward_contracts.p4.prepare_p4.subprocess.run",
        fake_git,
    )
    manifest = prepare(REPO_ROOT, tmp_path, "/model")
    assert manifest["protocol"] == "ARCA-P4-DYNAMIC-v0.2"
    assert manifest["supersedes_protocol"] == "ARCA-P4-DYNAMIC-v0.1"
    assert manifest["run_order"] == EXPECTED_ORDER
    reloaded = json.loads(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert reloaded["run_order"] == EXPECTED_ORDER


def test_p4_collector_aligns_two_arms_by_three_seeds():
    series = {}
    for arm in ("strict", "canonical"):
        for seed in (3101, 3102, 3103):
            series[(arm, seed)] = {
                field: {0: float(index)}
                for index, field in enumerate(
                    ("reward_mean", "reward_std", "trainable_tokens", "masked_response_tokens")
                )
            }
    rows = build_rows(series)
    assert len(rows) == 6
    assert {(row["arm"], row["seed"], row["step"]) for row in rows} == {
        (arm, seed, 0)
        for arm in ("strict", "canonical")
        for seed in (3101, 3102, 3103)
    }

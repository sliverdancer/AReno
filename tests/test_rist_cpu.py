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

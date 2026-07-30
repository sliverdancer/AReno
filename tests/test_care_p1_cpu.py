from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "research" / "care_turn_credit_mainconf" / "p1"
sys.path.insert(0, str(P1))

from bifurcation_oracle import enumerate_oracle_rows, exact_credit
from block_risk import (
    AuditTrajectory,
    AuditTurn,
    assert_nested_monotonicity,
    crc_threshold,
    routed_turns,
    wrong_sign_gradient_mass_loss,
)


def load_runner():
    spec = importlib.util.spec_from_file_location("care_p1_runner", P1 / "run_p1.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_oracle_exposes_both_terminal_credit_conflicts():
    rows = enumerate_oracle_rows()

    assert len(rows) == 64
    assert any(row["reward"] == 1 and row["oracle_sign"] == -1 for row in rows)
    assert any(row["reward"] == 0 and row["oracle_sign"] == 1 for row in rows)
    assert exact_credit((1,), 0) == -exact_credit((1,), 1)


def test_budgeted_block_loss_is_nested_and_bounded():
    trajectory = AuditTrajectory(
        trajectory_id="tau",
        turns=(
            AuditTurn(0, 0.95, 1, -1, 3),
            AuditTurn(1, 0.80, 1, 1, 2),
            AuditTurn(2, 0.60, -1, 1, 2),
        ),
    )

    assert_nested_monotonicity([trajectory], token_budget=5)
    assert [turn.turn_index for turn in routed_turns(trajectory, -1, 5)] == [0, 1]
    assert wrong_sign_gradient_mass_loss(trajectory, -1, 5) == 3 / 5
    assert wrong_sign_gradient_mass_loss(trajectory, 0.95, 5) == 0


def test_crc_abstains_when_calibration_sample_is_too_small():
    trajectories = [
        AuditTrajectory(
            trajectory_id=f"tau-{index}",
            turns=(AuditTurn(0, 0.9, 1, 1, 1),),
        )
        for index in range(2)
    ]

    threshold = crc_threshold(trajectories, alpha=0.10, token_budget=1)

    assert threshold == float("inf")


def test_clustered_turn_iid_calibration_violates_nominal_expected_risk():
    runner = load_runner()
    result = runner.clustered_dependence_stress()

    assert result["independent_turn_expected_risk"] <= result["alpha"]
    assert result["clustered_naive_expected_risk"] > result["alpha"]
    assert result["clustered_block_expected_risk"] <= result["alpha"]


def test_stale_threshold_fails_and_same_policy_refresh_controls():
    runner = load_runner()
    result = runner.policy_shift_stress()

    assert result["stale_threshold_expected_risk"] > result["alpha"]
    assert result["refreshed_threshold_expected_risk"] <= result["alpha"]


def test_artifact_generation_is_deterministic_and_cpu_verifiable(tmp_path):
    runner = load_runner()
    first = tmp_path / "first"
    second = tmp_path / "second"

    summary_first = runner.generate(first)
    summary_second = runner.generate(second)

    assert summary_first == summary_second
    assert (first / "p1_summary.json").read_bytes() == (
        second / "p1_summary.json"
    ).read_bytes()
    assert (first / "bifurcation_trajectories.csv").read_bytes() == (
        second / "bifurcation_trajectories.csv"
    ).read_bytes()

    manifest = json.loads((first / "artifact_manifest.json").read_text())
    with (first / "bifurcation_trajectories.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 64
    assert set(manifest["artifacts"]) == {
        "bifurcation_trajectories.csv",
        "p1_summary.json",
    }

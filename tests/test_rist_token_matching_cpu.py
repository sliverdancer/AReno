"""CPU-only scientific-contract tests for RIST token robustness."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    ROOT
    / "research/reward_identifiability_supervision_topology/successors/rist_v2_1/stages"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_token_auc_integrates_every_observed_knot_not_quartile_approximation():
    matcher = _load(
        "rist_exact_token_auc", STAGES / "P3_DESIGN" / "token_matching.py"
    )
    curves = {
        "AF": [
            {"cumulative_trainable_tokens": 0.0, "strict_success": 0.0},
            {"cumulative_trainable_tokens": 10.0, "strict_success": 1.0},
            {"cumulative_trainable_tokens": 100.0, "strict_success": 0.0},
        ],
        **{
            arm: [
                {"cumulative_trainable_tokens": 0.0, "strict_success": 0.0},
                {"cumulative_trainable_tokens": 100.0, "strict_success": 0.0},
            ]
            for arm in ("LF", "AN", "LN")
        },
    }
    result = matcher.match_curves(curves)
    assert result["grid"] == [0.0, 25.0, 50.0, 75.0, 100.0]
    assert result["integration_grid"] == [0.0, 10.0, 100.0]
    assert result["normalized_auc"]["AF"] == pytest.approx(0.5)
    assert result["integration_uses_all_observed_token_knots"] is True
    assert result["independent_token_budget_schedule"] is False


def test_token_overlap_must_cover_total_exposure_not_only_observed_range():
    matcher = _load(
        "rist_total_token_support", STAGES / "P3_DESIGN" / "token_matching.py"
    )
    curves = {
        arm: [
            {"cumulative_trainable_tokens": 600.0, "strict_success": 0.1},
            {"cumulative_trainable_tokens": 1000.0, "strict_success": 0.2},
        ]
        for arm in ("AF", "LF", "AN", "LN")
    }
    with pytest.raises(ValueError, match="total exposure"):
        matcher.match_curves(curves)


def test_token_robustness_rejects_mean_only_consistency():
    analyzer = _load(
        "rist_token_seed_stability", STAGES / "P4_ANALYSIS" / "analyze_results.py"
    )
    unstable = analyzer._token_robustness(
        [0.3, 0.3, 0.3, 0.3],
        [0.4, 0.4, 0.4, -0.1],
        [0.2, 0.2, 0.2, -0.1],
    )
    assert unstable["mean_sign_consistent"] is True
    assert unstable["paired_estimand_sign_agreement"] == 0.75
    assert unstable["token_auc_stability"]["stability_pass"] is True
    assert unstable["token_endpoint_stability"]["stability_pass"] is True
    assert unstable["pass"] is True

    unstable = analyzer._token_robustness(
        [0.3, 0.3, 0.3, 0.3],
        [0.4, 0.4, -0.1, -0.1],
        [0.2, 0.2, -0.1, -0.1],
    )
    assert unstable["mean_sign_consistent"] is True
    assert unstable["paired_estimand_sign_agreement"] == 0.5
    assert unstable["token_auc_stability"]["stability_pass"] is False
    assert unstable["pass"] is False


def test_token_robustness_requires_complete_paired_seed_estimands():
    analyzer = _load(
        "rist_token_complete_pairs", STAGES / "P4_ANALYSIS" / "analyze_results.py"
    )
    result = analyzer._token_robustness([0.2, 0.2, 0.2], [0.1, 0.1], [0.1, 0.1])
    assert result["pass"] is False
    assert result["paired_estimand_sign_agreement"] == 0.0

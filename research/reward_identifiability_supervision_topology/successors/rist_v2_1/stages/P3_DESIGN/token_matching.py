"""Outcome-blind retrospective token matching for RIST-v2.1 curves."""

from __future__ import annotations

import math
from typing import Any

GRID_FRACTIONS = (0.00, 0.25, 0.50, 0.75, 1.00)
MIN_COMMON_SUPPORT_FRACTION = 0.50


def _validate_curve(curve: list[dict[str, float]]) -> None:
    if len(curve) < 2:
        raise ValueError("each arm requires at least two curve points")
    tokens = [float(point["cumulative_trainable_tokens"]) for point in curve]
    if (
        any(not math.isfinite(value) for value in tokens)
        or tokens[0] < 0.0
        or any(right <= left for left, right in zip(tokens, tokens[1:]))
    ):
        raise ValueError("cumulative trainable tokens must be strictly increasing")
    for point in curve:
        for key, value in point.items():
            if key != "cumulative_trainable_tokens" and (
                not isinstance(value, (int, float)) or not math.isfinite(float(value))
            ):
                raise ValueError(f"curve metric {key} must be numeric")


def token_grid(curves: dict[str, list[dict[str, float]]]) -> list[float]:
    """Choose an outcome-blind grid from the minimum arm token support."""

    if set(curves) != {"AF", "LF", "AN", "LN"}:
        raise ValueError("token matching requires all four supervision arms")
    for curve in curves.values():
        _validate_curve(curve)
    common_max = min(
        float(curve[-1]["cumulative_trainable_tokens"])
        for curve in curves.values()
    )
    common_min = max(
        float(curve[0]["cumulative_trainable_tokens"])
        for curve in curves.values()
    )
    if common_max <= common_min:
        raise ValueError("arms have no common cumulative-token support")
    return [common_min + fraction * (common_max - common_min) for fraction in GRID_FRACTIONS]


def _interpolate(curve: list[dict[str, float]], token_value: float, metric: str) -> float:
    for left, right in zip(curve, curve[1:]):
        left_token = float(left["cumulative_trainable_tokens"])
        right_token = float(right["cumulative_trainable_tokens"])
        if left_token <= token_value <= right_token:
            weight = (token_value - left_token) / (right_token - left_token)
            return float(left[metric]) + weight * (float(right[metric]) - float(left[metric]))
    if token_value == float(curve[-1]["cumulative_trainable_tokens"]):
        return float(curve[-1][metric])
    raise ValueError("token grid point lies outside one arm curve")


def match_curves(
    curves: dict[str, list[dict[str, float]]], metric: str = "strict_success"
) -> dict[str, Any]:
    """Interpolate every arm on the outcome-blind common token grid."""

    grid = token_grid(curves)
    common_width = grid[-1] - grid[0]
    # The observed-window fraction protects against a sliver intersection after
    # the first evaluation. The total-exposure fraction additionally counts the
    # known but outcome-unobserved interval from token zero to that evaluation.
    support_fraction_by_arm = {
        arm: common_width
        / (
            float(curve[-1]["cumulative_trainable_tokens"])
            - float(curve[0]["cumulative_trainable_tokens"])
        )
        for arm, curve in curves.items()
    }
    minimum_support_fraction = min(support_fraction_by_arm.values())
    total_exposure_support_fraction_by_arm = {
        arm: common_width / float(curve[-1]["cumulative_trainable_tokens"])
        for arm, curve in curves.items()
    }
    minimum_total_exposure_support_fraction = min(
        total_exposure_support_fraction_by_arm.values()
    )
    if (
        minimum_support_fraction < MIN_COMMON_SUPPORT_FRACTION
        or minimum_total_exposure_support_fraction < MIN_COMMON_SUPPORT_FRACTION
    ):
        raise ValueError(
            "common cumulative-token support covers less than "
            f"{MIN_COMMON_SUPPORT_FRACTION:.0%} of at least one arm's "
            "observed window or total exposure"
        )
    matched = {
        arm: [_interpolate(curve, token_value, metric) for token_value in grid]
        for arm, curve in curves.items()
    }
    # Integrate on every piecewise-linear knot from every arm. The five-point
    # grid remains a reporting grid only; using it for integration would hide
    # learning-curve bends that fall between its quartiles.
    integration_grid = sorted(
        {
            grid[0],
            grid[-1],
            *(
                float(point["cumulative_trainable_tokens"])
                for curve in curves.values()
                for point in curve
                if grid[0]
                <= float(point["cumulative_trainable_tokens"])
                <= grid[-1]
            ),
        }
    )
    integration_values = {
        arm: [_interpolate(curve, token_value, metric) for token_value in integration_grid]
        for arm, curve in curves.items()
    }
    auc = {}
    width = grid[-1] - grid[0]
    for arm, values in integration_values.items():
        area = sum(
            (right_x - left_x) * (left_y + right_y) / 2.0
            for left_x, right_x, left_y, right_y in zip(
                integration_grid[:-1],
                integration_grid[1:],
                values[:-1],
                values[1:],
                strict=True,
            )
        )
        auc[arm] = area / width
    return {
        "grid": grid,
        "integration_grid": integration_grid,
        "matched": matched,
        "normalized_auc": auc,
        "common_support_endpoint": {
            arm: values[-1] for arm, values in matched.items()
        },
        "support_fraction_by_arm": support_fraction_by_arm,
        "total_exposure_support_fraction_by_arm": (
            total_exposure_support_fraction_by_arm
        ),
        "minimum_common_support_fraction": minimum_support_fraction,
        "minimum_total_exposure_support_fraction": (
            minimum_total_exposure_support_fraction
        ),
        "required_common_support_fraction": MIN_COMMON_SUPPORT_FRACTION,
        "grid_uses_outcomes": False,
        "integration_uses_all_observed_token_knots": True,
        "estimand": "retrospective_observed_window_equal_token_exposure",
        "independent_token_budget_schedule": False,
        "optimizer_step_path_controlled": False,
        "trajectory_count_controlled": False,
    }

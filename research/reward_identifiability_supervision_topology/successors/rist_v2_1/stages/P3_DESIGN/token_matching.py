"""Outcome-blind common-support token matching for RIST-v2.1 training curves."""

from __future__ import annotations

from typing import Any

GRID_FRACTIONS = (0.25, 0.50, 0.75, 1.00)


def _validate_curve(curve: list[dict[str, float]]) -> None:
    if len(curve) < 2:
        raise ValueError("each arm requires at least two curve points")
    tokens = [float(point["cumulative_trainable_tokens"]) for point in curve]
    if tokens[0] < 0.0 or any(right <= left for left, right in zip(tokens, tokens[1:])):
        raise ValueError("cumulative trainable tokens must be strictly increasing")


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
    matched = {
        arm: [_interpolate(curve, token_value, metric) for token_value in grid]
        for arm, curve in curves.items()
    }
    auc = {}
    width = grid[-1] - grid[0]
    for arm, values in matched.items():
        area = sum(
            (right_x - left_x) * (left_y + right_y) / 2.0
            for left_x, right_x, left_y, right_y in zip(
                grid[:-1], grid[1:], values[:-1], values[1:], strict=True
            )
        )
        auc[arm] = area / width
    return {
        "grid": grid,
        "matched": matched,
        "normalized_auc": auc,
        "grid_uses_outcomes": False,
    }

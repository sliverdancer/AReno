"""Deterministic four-turn bifurcation task with exact counterfactual credit."""

from __future__ import annotations

import hashlib
from fractions import Fraction
from itertools import product
from typing import Iterable


HORIZON = 4
WEIGHT_VALUES = (-2, -1, 1, 2)


def validate_task(weights: Iterable[int], threshold: int) -> tuple[int, ...]:
    """Validate and normalize a task definition."""

    normalized = tuple(int(value) for value in weights)
    if len(normalized) != HORIZON:
        raise ValueError(f"expected {HORIZON} weights")
    if any(value not in WEIGHT_VALUES for value in normalized):
        raise ValueError(f"weights must be drawn from {WEIGHT_VALUES}")
    if int(threshold) not in (-2, 0, 2):
        raise ValueError("threshold must be one of: -2, 0, 2")
    return normalized


def terminal_score(weights: Iterable[int], actions: Iterable[int]) -> int:
    """Return the signed terminal score."""

    weight_tuple = tuple(weights)
    action_tuple = tuple(actions)
    if len(action_tuple) != HORIZON:
        raise ValueError(f"expected {HORIZON} actions")
    if any(action not in (0, 1) for action in action_tuple):
        raise ValueError("actions must be binary")
    return sum(
        weight if action == 1 else -weight
        for weight, action in zip(weight_tuple, action_tuple, strict=True)
    )


def terminal_reward(
    weights: Iterable[int],
    threshold: int,
    actions: Iterable[int],
) -> int:
    """Return deterministic binary success."""

    return int(terminal_score(weights, actions) >= int(threshold))


def q_value(
    weights: tuple[int, ...],
    threshold: int,
    prefix: tuple[int, ...],
    action: int,
) -> Fraction:
    """Exact expected reward under a uniform future action policy."""

    if action not in (0, 1):
        raise ValueError("action must be binary")
    remaining = HORIZON - len(prefix) - 1
    if remaining < 0:
        raise ValueError("prefix is longer than the horizon")
    rewards = [
        terminal_reward(weights, threshold, (*prefix, action, *suffix))
        for suffix in product((0, 1), repeat=remaining)
    ]
    return Fraction(sum(rewards), len(rewards))


def exact_credit(
    weights: tuple[int, ...],
    threshold: int,
    prefix: tuple[int, ...],
    action: int,
) -> Fraction:
    """Chosen-action value minus the uniform action baseline."""

    chosen_q = q_value(weights, threshold, prefix, action)
    baseline = (
        q_value(weights, threshold, prefix, 0)
        + q_value(weights, threshold, prefix, 1)
    ) / 2
    return chosen_q - baseline


def sign(value: int | float | Fraction) -> int:
    """Return the three-way sign."""

    return int(value > 0) - int(value < 0)


def cheap_credit(
    *,
    task_id: str,
    scorer_seed: int,
    turn_index: int,
    weight: int,
    action: int,
) -> tuple[float, int, float]:
    """Return a deterministic noisy local credit estimate.

    The controlled scorer intentionally flips a prespecified fraction of local
    signs. It is a mechanism stressor, not a learned or semantically novel
    scorer.
    """

    local_credit = float(weight if action == 1 else -weight)
    predicted_sign = sign(local_credit)
    flip_rate = 0.35 if abs(weight) == 1 else 0.10
    digest = hashlib.sha256(
        f"{task_id}|{scorer_seed}|{turn_index}".encode("utf-8")
    ).digest()
    uniform = int.from_bytes(digest[:8], "big") / 2**64
    if uniform < flip_rate:
        predicted_sign *= -1
    confidence = 0.60 if abs(weight) == 1 else 0.90
    magnitude = abs(weight) / max(abs(value) for value in WEIGHT_VALUES)
    return predicted_sign * magnitude, predicted_sign, confidence


def counterfactual_terminal_calls(turn_index: int) -> int:
    """Count terminal evaluations used by the exact two-action audit."""

    if turn_index < 0 or turn_index >= HORIZON:
        raise ValueError("turn_index is outside the horizon")
    remaining = HORIZON - turn_index - 1
    return 2 * (2**remaining)

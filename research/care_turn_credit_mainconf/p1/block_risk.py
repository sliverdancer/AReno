"""Trajectory-block risk calibration for the CPU-only CARe P1 study.

This module is deliberately independent of AReno's runtime and public API.  It
constructs a nested, budget-feasible selection family and applies the finite
sample conformal-risk correction to whole trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Iterable, Sequence


@dataclass(frozen=True)
class AuditTurn:
    """One audited turn and the mass its routed update would contribute."""

    turn_index: int
    confidence: float
    predicted_sign: int
    oracle_sign: int
    gradient_mass: int

    def __post_init__(self) -> None:
        if self.predicted_sign not in (-1, 1):
            raise ValueError("predicted_sign must be -1 or 1")
        if self.oracle_sign not in (-1, 0, 1):
            raise ValueError("oracle_sign must be -1, 0, or 1")
        if self.gradient_mass <= 0:
            raise ValueError("gradient_mass must be positive")


@dataclass(frozen=True)
class AuditTrajectory:
    """Exchangeable calibration unit containing dependent turns."""

    trajectory_id: str
    turns: tuple[AuditTurn, ...]


def budget_prefix(
    trajectory: AuditTrajectory,
    token_budget: int,
) -> tuple[AuditTurn, ...]:
    """Return a fixed confidence-ranked prefix whose total mass fits the budget.

    The prefix is fixed before threshold calibration.  We intentionally do not
    backfill after the first item that would exceed the budget: backfilling can
    make thresholded selected sets non-nested.
    """

    if token_budget <= 0:
        raise ValueError("token_budget must be positive")
    ordered = sorted(
        trajectory.turns,
        key=lambda turn: (-turn.confidence, turn.turn_index),
    )
    selected: list[AuditTurn] = []
    used = 0
    for turn in ordered:
        if used + turn.gradient_mass > token_budget:
            break
        selected.append(turn)
        used += turn.gradient_mass
    return tuple(selected)


def routed_turns(
    trajectory: AuditTrajectory,
    threshold: float,
    token_budget: int,
) -> tuple[AuditTurn, ...]:
    """Select a nested subset of the fixed budget prefix.

    Larger thresholds are more conservative.  A strict comparison makes every
    observed confidence value a useful breakpoint.
    """

    return tuple(
        turn
        for turn in budget_prefix(trajectory, token_budget)
        if turn.confidence > threshold
    )


def wrong_sign_gradient_mass_loss(
    trajectory: AuditTrajectory,
    threshold: float,
    token_budget: int,
) -> float:
    """Bounded wrong-sign update mass with a fixed token-budget denominator."""

    wrong_mass = sum(
        turn.gradient_mass
        for turn in routed_turns(trajectory, threshold, token_budget)
        if turn.predicted_sign != turn.oracle_sign
    )
    return wrong_mass / token_budget


def candidate_thresholds(trajectories: Sequence[AuditTrajectory]) -> tuple[float, ...]:
    """Return all breakpoints from fully permissive to fully abstaining."""

    scores = sorted(
        {
            turn.confidence
            for trajectory in trajectories
            for turn in trajectory.turns
        }
    )
    return (-inf, *scores, inf)


def crc_threshold(
    trajectories: Sequence[AuditTrajectory],
    alpha: float,
    token_budget: int,
) -> float:
    """Choose the most permissive threshold satisfying block CRC.

    For losses bounded by one, the corrected calibration risk is
    ``(sum_i L_i(lambda) + 1) / (n + 1)``.
    """

    if not trajectories:
        raise ValueError("at least one calibration trajectory is required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    for threshold in candidate_thresholds(trajectories):
        loss_sum = sum(
            wrong_sign_gradient_mass_loss(trajectory, threshold, token_budget)
            for trajectory in trajectories
        )
        corrected_risk = (loss_sum + 1.0) / (len(trajectories) + 1)
        if corrected_risk <= alpha:
            return threshold
    return inf


def empirical_block_risk(
    trajectories: Iterable[AuditTrajectory],
    threshold: float,
    token_budget: int,
) -> float:
    """Mean block loss for a frozen threshold."""

    losses = [
        wrong_sign_gradient_mass_loss(trajectory, threshold, token_budget)
        for trajectory in trajectories
    ]
    if not losses:
        raise ValueError("at least one evaluation trajectory is required")
    return sum(losses) / len(losses)


def assert_nested_monotonicity(
    trajectories: Sequence[AuditTrajectory],
    token_budget: int,
) -> None:
    """Raise when any block loss increases as the threshold becomes stricter."""

    thresholds = candidate_thresholds(trajectories)
    for trajectory in trajectories:
        losses = [
            wrong_sign_gradient_mass_loss(trajectory, threshold, token_budget)
            for threshold in thresholds
        ]
        if any(right > left + 1e-12 for left, right in zip(losses, losses[1:])):
            raise AssertionError(
                f"non-monotone loss for trajectory {trajectory.trajectory_id}"
            )

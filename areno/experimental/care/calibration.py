"""Trajectory-block calibration for experimental CARe routing."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Sequence


@dataclass(frozen=True, slots=True)
class CalibrationTurn:
    """One audited turn used to calibrate wrong-sign gradient-mass risk."""

    turn_index: int
    confidence: float
    predicted_sign: int
    oracle_sign: int
    token_mass: int

    def __post_init__(self) -> None:
        if self.predicted_sign not in (-1, 1):
            raise ValueError("predicted_sign must be -1 or 1")
        if self.oracle_sign not in (-1, 0, 1):
            raise ValueError("oracle_sign must be -1, 0, or 1")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.token_mass <= 0:
            raise ValueError("token_mass must be positive")


@dataclass(frozen=True, slots=True)
class CalibrationTrajectory:
    """One exchangeable calibration block."""

    trajectory_id: str
    turns: tuple[CalibrationTurn, ...]


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Frozen threshold and corrected empirical calibration risk."""

    threshold: float
    corrected_risk: float
    selected_tokens: int
    wrong_sign_tokens: int
    trajectory_count: int


def ranked_budget_prefix(
    turns: Sequence[CalibrationTurn],
    budget_tokens: int,
) -> tuple[CalibrationTurn, ...]:
    """Return a stable confidence-ranked prefix without backfilling."""

    if budget_tokens <= 0:
        raise ValueError("budget_tokens must be positive")
    ordered = sorted(turns, key=lambda turn: (-turn.confidence, turn.turn_index))
    selected: list[CalibrationTurn] = []
    used = 0
    for turn in ordered:
        if used + turn.token_mass > budget_tokens:
            break
        selected.append(turn)
        used += turn.token_mass
    return tuple(selected)


def calibrate_block_threshold(
    trajectories: Sequence[CalibrationTrajectory],
    *,
    alpha: float,
    budget_tokens: int,
) -> CalibrationResult:
    """Choose the most permissive finite-sample block-risk threshold."""

    if not trajectories:
        raise ValueError("at least one calibration trajectory is required")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    minimum_corrected_risk = 1.0 / (len(trajectories) + 1)
    if minimum_corrected_risk > alpha:
        raise ValueError(
            "insufficient calibration trajectories for alpha: "
            f"minimum_corrected_risk={minimum_corrected_risk:.6f} alpha={alpha:.6f}"
        )
    prefixes = {
        trajectory.trajectory_id: ranked_budget_prefix(
            trajectory.turns,
            budget_tokens,
        )
        for trajectory in trajectories
    }
    thresholds = (
        -inf,
        *sorted(
            {
                turn.confidence
                for turns in prefixes.values()
                for turn in turns
            }
        ),
        inf,
    )
    for threshold in thresholds:
        wrong_sign_tokens = 0
        selected_tokens = 0
        block_loss_sum = 0.0
        for trajectory in trajectories:
            selected = tuple(
                turn
                for turn in prefixes[trajectory.trajectory_id]
                if turn.confidence > threshold
            )
            trajectory_wrong = sum(
                turn.token_mass
                for turn in selected
                if turn.predicted_sign != turn.oracle_sign
            )
            wrong_sign_tokens += trajectory_wrong
            selected_tokens += sum(turn.token_mass for turn in selected)
            block_loss_sum += trajectory_wrong / budget_tokens
        corrected_risk = (block_loss_sum + 1.0) / (len(trajectories) + 1)
        if corrected_risk <= alpha:
            return CalibrationResult(
                threshold=float(threshold),
                corrected_risk=corrected_risk,
                selected_tokens=selected_tokens,
                wrong_sign_tokens=wrong_sign_tokens,
                trajectory_count=len(trajectories),
            )
    raise AssertionError("fully abstaining threshold must satisfy bounded block risk")

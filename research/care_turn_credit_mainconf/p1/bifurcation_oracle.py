"""Exact finite-horizon counterfactual oracle for controlled bifurcations."""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from typing import Iterable


WEIGHTS = (2, 1, 2, 1)
SUCCESS_THRESHOLD = 2


def terminal_reward(actions: Iterable[int]) -> int:
    """Return deterministic success for a four-decision trajectory."""

    action_tuple = tuple(actions)
    if len(action_tuple) != len(WEIGHTS):
        raise ValueError(f"expected {len(WEIGHTS)} actions")
    if any(action not in (0, 1) for action in action_tuple):
        raise ValueError("actions must be binary")
    progress = sum(
        weight if action == 1 else -weight
        for weight, action in zip(WEIGHTS, action_tuple)
    )
    return int(progress >= SUCCESS_THRESHOLD)


def q_value(prefix: tuple[int, ...], action: int) -> Fraction:
    """Exact expected terminal reward under a uniform frozen continuation."""

    if action not in (0, 1):
        raise ValueError("action must be binary")
    remaining = len(WEIGHTS) - len(prefix) - 1
    if remaining < 0:
        raise ValueError("prefix is longer than the decision horizon")
    rewards = [
        terminal_reward((*prefix, action, *suffix))
        for suffix in product((0, 1), repeat=remaining)
    ]
    return Fraction(sum(rewards), len(rewards))


def exact_credit(prefix: tuple[int, ...], chosen_action: int) -> Fraction:
    """Chosen-action value minus the uniform action baseline at the same state."""

    chosen_q = q_value(prefix, chosen_action)
    baseline = (q_value(prefix, 0) + q_value(prefix, 1)) / 2
    return chosen_q - baseline


def sign(value: Fraction) -> int:
    """Map an exact rational credit to its three-way sign."""

    return int(value > 0) - int(value < 0)


def enumerate_oracle_rows() -> list[dict[str, object]]:
    """Enumerate every trajectory and every exact turn-credit label."""

    rows: list[dict[str, object]] = []
    for trajectory_index, actions in enumerate(product((0, 1), repeat=len(WEIGHTS))):
        reward = terminal_reward(actions)
        for turn_index, chosen_action in enumerate(actions):
            prefix = actions[:turn_index]
            chosen_q = q_value(prefix, chosen_action)
            baseline = (q_value(prefix, 0) + q_value(prefix, 1)) / 2
            credit = chosen_q - baseline
            rows.append(
                {
                    "trajectory_id": f"tau-{trajectory_index:02d}",
                    "actions": "".join(str(action) for action in actions),
                    "reward": reward,
                    "turn_index": turn_index,
                    "chosen_action": chosen_action,
                    "q_chosen": float(chosen_q),
                    "q_baseline": float(baseline),
                    "credit": float(credit),
                    "oracle_sign": sign(credit),
                }
            )
    return rows

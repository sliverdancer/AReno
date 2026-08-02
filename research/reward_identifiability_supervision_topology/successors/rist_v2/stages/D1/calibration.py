"""CPU-only uncertainty tools for checkpoint-conditional reward resolution."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


def mixed_group_probability(success_probability: float, group_size: int) -> float:
    """Return the probability that a binary-reward group is non-homogeneous."""

    if not 0.0 <= success_probability <= 1.0:
        raise ValueError("success_probability must be in [0, 1]")
    if group_size < 2:
        raise ValueError("group_size must be at least two")
    return (
        1.0
        - success_probability**group_size
        - (1.0 - success_probability) ** group_size
    )


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson interval for a binomial proportion."""

    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("successes and trials define an invalid binomial sample")
    if z <= 0.0:
        raise ValueError("z must be positive")
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z**2 / (4.0 * trials**2)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def mixed_probability_interval(
    successes: int,
    trials: int,
    group_size: int = 8,
) -> tuple[float, float]:
    """Propagate a Wilson success interval through the symmetric M_g function."""

    lower_p, upper_p = wilson_interval(successes, trials)
    endpoint_values = [
        mixed_group_probability(lower_p, group_size),
        mixed_group_probability(upper_p, group_size),
    ]
    lower_mixed = min(endpoint_values)
    if lower_p <= 0.5 <= upper_p:
        upper_mixed = mixed_group_probability(0.5, group_size)
    else:
        upper_mixed = max(endpoint_values)
    return lower_mixed, upper_mixed


def summarize_task(
    successes: int,
    trials: int,
    group_size: int = 8,
) -> dict[str, Any]:
    """Summarize one calibration task without hiding binomial uncertainty."""

    point_success = successes / trials
    lower_success, upper_success = wilson_interval(successes, trials)
    lower_mixed, upper_mixed = mixed_probability_interval(
        successes, trials, group_size
    )
    point_mixed = mixed_group_probability(point_success, group_size)
    if upper_mixed <= 0.25:
        classification = "collapsed"
    elif lower_mixed >= 0.50:
        classification = "resolved"
    else:
        classification = "transition"
    return {
        "successes": successes,
        "trials": trials,
        "group_size": group_size,
        "success_probability": point_success,
        "success_interval": [lower_success, upper_success],
        "mixed_probability": point_mixed,
        "mixed_interval": [lower_mixed, upper_mixed],
        "classification": classification,
    }


def summarize_cell(records: Iterable[tuple[int, int]], group_size: int = 8) -> dict[str, Any]:
    """Return a fail-closed structural-cell summary from independent tasks."""

    tasks = [summarize_task(successes, trials, group_size) for successes, trials in records]
    if len(tasks) < 4:
        raise ValueError("a structural cell requires at least four independent tasks")
    counts = {label: 0 for label in ("collapsed", "transition", "resolved")}
    for task in tasks:
        counts[str(task["classification"])] += 1
    dominant = max(counts, key=counts.get)
    cell_classification = dominant if counts[dominant] / len(tasks) >= 0.75 else "heterogeneous"
    return {
        "task_count": len(tasks),
        "classification_counts": counts,
        "cell_classification": cell_classification,
        "tasks": tasks,
    }

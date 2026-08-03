"""Collect exact token exposure and mixed-reward groups for one training run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

CHECKPOINT_STEPS = (25, 50, 75, 100)
N_SAMPLES = 8
TAGS = {
    "trainable_tokens": "train/trainable_tokens",
    "masked_response_tokens": "train/masked_response_tokens",
    "reward_mean": "rollout/rewards_mean",
    "reward_std": "rollout/rewards_std",
}


def load_tensorboard(log_dir: Path) -> dict[str, list[dict[str, float]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise RuntimeError("TensorBoard is required in the authorized training environment") from exc
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    missing = sorted(set(TAGS.values()) - available)
    if missing:
        raise ValueError(f"training metrics missing tags: {missing}")
    return {
        name: [
            {"step": int(event.step), "value": float(event.value)}
            for event in accumulator.Scalars(tag)
        ]
        for name, tag in TAGS.items()
    }


def summarize_training(
    series: dict[str, list[dict[str, float]]],
    reward_events: list[dict[str, Any]],
) -> dict[str, Any]:
    indexed = {}
    expected_steps = set(range(100))
    for name in TAGS:
        events = series.get(name)
        if events is None:
            raise ValueError(f"missing training series: {name}")
        values = {}
        for event in events:
            step = int(event["step"])
            value = float(event["value"])
            if step in values or not math.isfinite(value):
                raise ValueError(f"invalid or duplicate {name} step {step}")
            values[step] = value
        if set(values) != expected_steps:
            raise ValueError(f"{name} must contain exact metric steps 0 through 99")
        indexed[name] = values
    if len(reward_events) != 100 * N_SAMPLES:
        raise ValueError("reward journal must contain eight outcomes for every step")
    rewards_by_key = {}
    for row in reward_events:
        try:
            key = (int(row["training_step"]), int(row["sample_index"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "reward journal requires integer training_step and sample_index"
            ) from exc
        if key in rewards_by_key:
            raise ValueError(f"duplicate reward event for step/sample {key}")
        rewards_by_key[key] = row
    expected_reward_keys = {
        (step, sample) for step in range(100) for sample in range(N_SAMPLES)
    }
    if set(rewards_by_key) != expected_reward_keys:
        raise ValueError("reward journal does not cover the exact 100 by 8 grid")
    mixed_by_step = []
    for step in range(100):
        group = [rewards_by_key[(step, sample)] for sample in range(N_SAMPLES)]
        if any(type(row.get("reward")) not in (int, float) or row["reward"] not in (0, 0.0, 1, 1.0) for row in group):
            raise ValueError(f"reward journal contains non-binary reward at step {step}")
        mixed_by_step.append(int(len({float(row["reward"]) for row in group}) > 1))
    points = []
    for checkpoint_step in CHECKPOINT_STEPS:
        metric_steps = range(checkpoint_step)
        points.append(
            {
                "checkpoint_step": checkpoint_step,
                "cumulative_trainable_tokens": int(
                    round(sum(indexed["trainable_tokens"][step] for step in metric_steps))
                ),
                "cumulative_masked_response_tokens": int(
                    round(
                        sum(
                            indexed["masked_response_tokens"][step]
                            for step in metric_steps
                        )
                    )
                ),
                "mean_training_reward": sum(
                    indexed["reward_mean"][step] for step in metric_steps
                )
                / checkpoint_step,
                "mean_training_reward_std": sum(
                    indexed["reward_std"][step] for step in metric_steps
                )
                / checkpoint_step,
                "nonzero_advantage_groups": sum(mixed_by_step[:checkpoint_step]),
            }
        )
    if any(
        right["cumulative_trainable_tokens"]
        <= left["cumulative_trainable_tokens"]
        for left, right in zip(points, points[1:])
    ):
        raise ValueError("cumulative trainable-token exposure must increase")
    return {
        "protocol": "RIST-P4-v2.1",
        "metric_steps": 100,
        "n_samples": N_SAMPLES,
        "points": points,
        "total_nonzero_advantage_groups": sum(mixed_by_step),
        "zero_advantage_group_count": len(mixed_by_step) - sum(mixed_by_step),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--reward-journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reward_events = [
        json.loads(line)
        for line in args.reward_journal.read_text().splitlines()
        if line
    ]
    result = summarize_training(load_tensorboard(args.metrics_dir), reward_events)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

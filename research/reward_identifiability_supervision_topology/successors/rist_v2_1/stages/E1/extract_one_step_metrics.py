"""Extract an exact one-step E1 metrics summary from TensorBoard and rewards."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

TAGS = {
    "loss": "train/loss",
    "gradient_norm": "train/grad_norm",
    "trainable_tokens": "train/trainable_tokens",
}


def summarize(
    series: dict[str, list[dict[str, float]]],
    rewards: list[dict[str, Any]],
) -> dict[str, Any]:
    values = {}
    for name in TAGS:
        events = series.get(name)
        if not isinstance(events, list) or len(events) != 1 or int(events[0]["step"]) != 0:
            raise ValueError(f"E1 requires exactly one step-zero {name} metric")
        value = float(events[0]["value"])
        if not math.isfinite(value):
            raise ValueError(f"E1 {name} must be finite")
        values[name] = value
    if int(round(values["trainable_tokens"])) <= 0:
        raise ValueError("E1 requires positive trainable tokens")
    if values["gradient_norm"] <= 0.0:
        raise ValueError("E1 requires a positive gradient norm")
    if len(rewards) != 8:
        raise ValueError("E1 reward journal requires exactly eight samples")
    by_sample = {int(row["sample_index"]): row for row in rewards}
    if set(by_sample) != set(range(8)) or len(by_sample) != len(rewards):
        raise ValueError("E1 reward sample identities must be exactly zero through seven")
    reward_values = [float(by_sample[index]["reward"]) for index in range(8)]
    if any(value not in {0.0, 1.0} for value in reward_values):
        raise ValueError("E1 rewards must be binary")
    mixed = len(set(reward_values)) == 2
    if not mixed:
        raise ValueError("E1 capacity canary requires a mixed-reward group")
    return {
        "protocol": "RIST-E1-ONE-STEP-METRICS-v1",
        "optimizer_step_completed": True,
        "optimizer_step_count": 1,
        "metric_step": 0,
        "trainable_tokens": int(round(values["trainable_tokens"])),
        "loss": values["loss"],
        "gradient_norm": values["gradient_norm"],
        "reward_event_count": len(rewards),
        "mixed_reward_group": mixed,
    }


def load_tensorboard(log_dir: Path) -> dict[str, list[dict[str, float]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise RuntimeError("TensorBoard is required in the E1 environment") from exc
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    missing = sorted(set(TAGS.values()) - available)
    if missing:
        raise ValueError(f"E1 metrics missing tags: {missing}")
    return {
        name: [
            {"step": int(event.step), "value": float(event.value)}
            for event in accumulator.Scalars(tag)
        ]
        for name, tag in TAGS.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--reward-journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rewards = [
        json.loads(line) for line in args.reward_journal.read_text().splitlines() if line
    ]
    result = summarize(load_tensorboard(args.metrics_dir), rewards)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

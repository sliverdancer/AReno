"""Collect P4 TensorBoard reward and token metrics into CSV and JSON."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from research.silent_reward_contracts.p4.prepare_p4 import ARMS, SEEDS


TAGS = {
    "reward_mean": "rollout/rewards_mean",
    "reward_std": "rollout/rewards_std",
    "trainable_tokens": "train/trainable_tokens",
    "masked_response_tokens": "train/masked_response_tokens",
}
FIELDS = ("arm", "seed", "step", *TAGS)


def load_series(log_dir: Path) -> dict[str, dict[int, float]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise RuntimeError("TensorBoard is required: pip install tensorboard") from exc
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    missing = sorted(set(TAGS.values()) - available)
    if missing:
        raise ValueError(f"{log_dir} missing scalar tags: {missing}")
    result = {}
    for field, tag in TAGS.items():
        values = {}
        for event in accumulator.Scalars(tag):
            value = float(event.value)
            if int(event.step) in values or not math.isfinite(value):
                raise ValueError(f"invalid {tag} at step {event.step}")
            values[int(event.step)] = value
        result[field] = values
    return result


def build_rows(series_by_run: dict[tuple[str, int], dict[str, dict[int, float]]]) -> list[dict[str, Any]]:
    expected = {(arm, seed) for arm in ARMS for seed in SEEDS}
    if set(series_by_run) != expected:
        raise ValueError("P4 requires exactly two arms by three seeds")
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            series = series_by_run[(arm, seed)]
            step_sets = [set(values) for values in series.values()]
            if not step_sets[0] or any(steps != step_sets[0] for steps in step_sets[1:]):
                raise ValueError(f"unaligned P4 steps for {arm}-{seed}")
            for step in sorted(step_sets[0]):
                rows.append({"arm": arm, "seed": seed, "step": step, **{field: values[step] for field, values in series.items()}})
    return rows


def write_artifacts(rows: list[dict[str, Any]], run_root: Path, manifest: dict[str, Any]) -> tuple[Path, Path]:
    csv_path = run_root / "steps.csv"
    json_path = run_root / "steps.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"schema_version": 1, "manifest": manifest, "rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path

"""Collect frozen CARe P3 pilot metrics into aligned CSV/JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from prepare_p3 import ARMS, TRAIN_SEEDS


METRIC_TAGS = {
    "reward_mean": "rollout/rewards_mean",
    "reward_std": "rollout/rewards_std",
    "trainable_tokens": "train/trainable_tokens",
    "masked_response_tokens": "train/masked_response_tokens",
}
CSV_FIELDS = (
    "arm",
    "seed",
    "step",
    "reward_mean",
    "reward_std",
    "trainable_tokens",
    "masked_response_tokens",
    "selected_mass",
    "masked_mass",
    "audit_calls",
    "calibration_blocks",
    "update_blocks",
    "valid_trajectories",
    "total_trajectories",
    "threshold",
)


def load_tensorboard_series(log_dir: Path) -> dict[str, dict[int, float]]:
    """Load required scalar series indexed by trainer step."""

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise RuntimeError("TensorBoard is required: pip install tensorboard") from exc
    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    missing = sorted(set(METRIC_TAGS.values()) - available)
    if missing:
        raise ValueError(f"{log_dir} is missing scalar tags: {missing}")
    result = {}
    for field, tag in METRIC_TAGS.items():
        indexed = {}
        for event in accumulator.Scalars(tag):
            step = int(event.step)
            value = float(event.value)
            if step in indexed or not math.isfinite(value):
                raise ValueError(f"invalid {tag} event at step {step}")
            indexed[step] = value
        result[field] = indexed
    return result


def load_diagnostics(path: Path) -> list[dict[str, Any]]:
    """Load turn-credit JSONL diagnostics."""

    if not path.exists():
        raise ValueError(f"missing turn-credit diagnostics: {path}")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"empty turn-credit diagnostics: {path}")
    return records


def summarize_diagnostics(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Aggregate per-turn records without double-counting trajectory audits."""

    by_step: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        by_step.setdefault(int(record["step"]), []).append(record)
    summaries = {}
    for step, step_records in by_step.items():
        trajectory_rows: dict[str, dict[str, Any]] = {}
        thresholds = set()
        for record in step_records:
            trajectory_id = str(record["trajectory_id"])
            diagnostics = record["trajectory_diagnostics"]
            threshold = diagnostics["threshold"]
            thresholds.add(str(threshold))
            trajectory_rows.setdefault(
                trajectory_id,
                {
                    "role": diagnostics["role"],
                    "valid": bool(diagnostics["valid_trajectory"]),
                    "audit_calls": int(record["audit_calls"]),
                },
            )
        if len(thresholds) != 1:
            raise ValueError(f"step {step} has inconsistent thresholds: {thresholds}")
        summaries[step] = {
            "selected_mass": sum(int(row["selected_mass"]) for row in step_records),
            "masked_mass": sum(int(row["masked_mass"]) for row in step_records),
            "audit_calls": sum(row["audit_calls"] for row in trajectory_rows.values()),
            "calibration_blocks": sum(
                row["role"] == "calibration"
                for row in trajectory_rows.values()
            ),
            "update_blocks": sum(
                row["role"] == "update"
                for row in trajectory_rows.values()
            ),
            "valid_trajectories": sum(
                row["valid"]
                for row in trajectory_rows.values()
            ),
            "total_trajectories": len(trajectory_rows),
            "threshold": next(iter(thresholds)),
        }
    return summaries


def build_rows(
    series_by_run: dict[tuple[str, int], dict[str, dict[int, float]]],
    diagnostics_by_run: dict[tuple[str, int], dict[int, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Align all required metrics for exactly two arms and three seeds."""

    expected_runs = {(arm, seed) for arm in ARMS for seed in TRAIN_SEEDS}
    if set(series_by_run) != expected_runs or set(diagnostics_by_run) != expected_runs:
        raise ValueError("P3 run set must contain both arms and all three seeds")
    rows = []
    for arm in ARMS:
        for seed in TRAIN_SEEDS:
            series = series_by_run[(arm, seed)]
            diagnostic_steps = diagnostics_by_run[(arm, seed)]
            step_sets = [set(values) for values in series.values()]
            expected_steps = step_sets[0]
            if (
                not expected_steps
                or any(steps != expected_steps for steps in step_sets[1:])
                or set(diagnostic_steps) != expected_steps
            ):
                raise ValueError(f"unaligned steps for {arm} seed {seed}")
            for step in sorted(expected_steps):
                summary = diagnostic_steps[step]
                rows.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "step": step,
                        **{
                            field: values[step]
                            for field, values in series.items()
                        },
                        **summary,
                    }
                )
    return rows


def write_artifacts(
    rows: list[dict[str, Any]],
    run_root: Path,
    *,
    manifest: dict[str, Any],
) -> tuple[Path, Path]:
    """Write stable CSV and JSON representations."""

    csv_path = run_root / "pilot_steps.csv"
    json_path = run_root / "pilot_steps.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest": manifest,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/care-p3-pilot"),
    )
    args = parser.parse_args()
    run_root = args.run_root.expanduser().resolve()
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    series_by_run = {}
    diagnostics_by_run = {}
    for arm in ARMS:
        for seed in TRAIN_SEEDS:
            metrics_dir = run_root / "runs" / f"{arm}-seed-{seed}" / "metrics"
            series_by_run[(arm, seed)] = load_tensorboard_series(metrics_dir)
            diagnostics_by_run[(arm, seed)] = summarize_diagnostics(
                load_diagnostics(metrics_dir / "turn_credit_diagnostics.jsonl")
            )
    rows = build_rows(series_by_run, diagnostics_by_run)
    csv_path, json_path = write_artifacts(rows, run_root, manifest=manifest)
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

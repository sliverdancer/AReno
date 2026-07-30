"""Export issue #199 TensorBoard scalars to stable CSV and JSON artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ARMS = ("all_assistant", "last_assistant", "final_answer")
METRIC_TAGS = {
    "reward_mean": "rollout/rewards_mean",
    "reward_std": "rollout/rewards_std",
    "trainable_tokens": "train/trainable_tokens",
    "masked_response_tokens": "train/masked_response_tokens",
}
CSV_FIELDS = (
    "arm",
    "step",
    "reward_mean",
    "reward_std",
    "trainable_tokens",
    "masked_response_tokens",
)


def load_tensorboard_series(log_dir: Path) -> dict[str, list[dict[str, float | int]]]:
    """Read every required scalar series from one TensorBoard log directory."""

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise RuntimeError("TensorBoard is required: pip install tensorboard") from exc

    accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    missing = sorted(set(METRIC_TAGS.values()) - available)
    if missing:
        raise ValueError(f"{log_dir} is missing required scalar tags: {', '.join(missing)}")
    return {
        tag: [
            {"step": int(event.step), "value": float(event.value)}
            for event in accumulator.Scalars(tag)
        ]
        for tag in METRIC_TAGS.values()
    }


def build_rows(
    series_by_arm: dict[str, dict[str, list[dict[str, float | int]]]],
) -> list[dict[str, float | int | str]]:
    """Align required scalar series by step and validate all three arms."""

    missing_arms = sorted(set(ARMS) - set(series_by_arm))
    extra_arms = sorted(set(series_by_arm) - set(ARMS))
    if missing_arms or extra_arms:
        raise ValueError(f"arm mismatch: missing={missing_arms}, extra={extra_arms}")

    rows = []
    for arm in ARMS:
        by_metric = {}
        for field, tag in METRIC_TAGS.items():
            events = series_by_arm[arm].get(tag)
            if events is None:
                raise ValueError(f"{arm} is missing metric {tag}")
            indexed = {}
            for event in events:
                step = int(event["step"])
                value = float(event["value"])
                if step in indexed:
                    raise ValueError(f"{arm} metric {tag} has duplicate step {step}")
                if not math.isfinite(value):
                    raise ValueError(f"{arm} metric {tag} step {step} is not finite")
                indexed[step] = value
            by_metric[field] = indexed

        step_sets = {field: set(values) for field, values in by_metric.items()}
        expected_steps = step_sets["reward_mean"]
        if not expected_steps:
            raise ValueError(f"{arm} has no metric steps")
        mismatched = {
            field: sorted(steps)
            for field, steps in step_sets.items()
            if steps != expected_steps
        }
        if mismatched:
            raise ValueError(
                f"{arm} metric steps are not aligned: expected={sorted(expected_steps)}, "
                f"mismatched={mismatched}"
            )

        for step in sorted(expected_steps):
            trainable_tokens = _as_token_count(
                by_metric["trainable_tokens"][step],
                arm=arm,
                step=step,
                field="trainable_tokens",
            )
            masked_response_tokens = _as_token_count(
                by_metric["masked_response_tokens"][step],
                arm=arm,
                step=step,
                field="masked_response_tokens",
            )
            rows.append(
                {
                    "arm": arm,
                    "step": step,
                    "reward_mean": by_metric["reward_mean"][step],
                    "reward_std": by_metric["reward_std"][step],
                    "trainable_tokens": trainable_tokens,
                    "masked_response_tokens": masked_response_tokens,
                }
            )
    return rows


def write_artifacts(
    rows: list[dict[str, float | int | str]],
    output_dir: Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write deterministic column-oriented CSV and schema-versioned JSON."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ablation_steps.csv"
    json_path = output_dir / "ablation_steps.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": 1,
        "metrics": METRIC_TAGS,
        "metadata": metadata or {},
        "rows": rows,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, json_path


def collect_run(run_root: Path) -> tuple[Path, Path]:
    """Load the three arm directories under ``run_root`` and export artifacts."""

    series_by_arm = {
        arm: load_tensorboard_series(run_root / "arms" / arm / "metrics")
        for arm in ARMS
    }
    manifest_path = run_root / "manifest.json"
    metadata = {}
    if manifest_path.exists():
        metadata["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
    return write_artifacts(build_rows(series_by_arm), run_root, metadata=metadata)


def _as_token_count(value: float, *, arm: str, step: int, field: str) -> int:
    rounded = round(value)
    if value < 0 or not math.isclose(value, rounded, abs_tol=1e-6):
        raise ValueError(f"{arm} {field} step {step} must be a non-negative integer, got {value}")
    return int(rounded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    csv_path, json_path = collect_run(args.run_root.expanduser().resolve())
    print(json.dumps({"ok": True, "csv": str(csv_path), "json": str(json_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

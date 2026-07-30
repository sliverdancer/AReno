"""Generate deterministic P1 exact-oracle and risk-calibration artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import comb
from pathlib import Path
from typing import Any

from bifurcation_oracle import enumerate_oracle_rows


def binomial_probability_at_most(n: int, probability: float, maximum: int) -> float:
    """Exact binomial CDF by finite summation."""

    if maximum < 0:
        return 0.0
    return sum(
        comb(n, count)
        * probability**count
        * (1.0 - probability) ** (n - count)
        for count in range(min(maximum, n) + 1)
    )


def maximum_accepted_errors(
    unit_count: int,
    alpha: float,
    duplicated_turns: int = 1,
) -> int:
    """Largest number of bad blocks whose duplicated-unit CRC bound passes."""

    accepted = -1
    for bad_blocks in range(unit_count + 1):
        loss_sum = bad_blocks * duplicated_turns
        effective_units = unit_count * duplicated_turns
        if (loss_sum + 1.0) / (effective_units + 1) <= alpha:
            accepted = bad_blocks
    return accepted


def clustered_dependence_stress() -> dict[str, Any]:
    """Analytic counterexample to treating dependent turns as i.i.d. units."""

    calibration_blocks = 2
    turns_per_block = 10
    alpha = 0.10
    bad_block_probability = 0.33

    naive_maximum = maximum_accepted_errors(
        calibration_blocks,
        alpha,
        duplicated_turns=turns_per_block,
    )
    block_maximum = maximum_accepted_errors(calibration_blocks, alpha)

    naive_selection_probability = binomial_probability_at_most(
        calibration_blocks,
        bad_block_probability,
        naive_maximum,
    )
    block_selection_probability = binomial_probability_at_most(
        calibration_blocks,
        bad_block_probability,
        block_maximum,
    )
    clustered_naive_expected_risk = (
        bad_block_probability * naive_selection_probability
    )
    clustered_block_expected_risk = (
        bad_block_probability * block_selection_probability
    )

    independent_turn_maximum = max(
        errors
        for errors in range(calibration_blocks * turns_per_block + 1)
        if (errors + 1.0) / (calibration_blocks * turns_per_block + 1) <= alpha
    )
    independent_selection_probability = binomial_probability_at_most(
        calibration_blocks * turns_per_block,
        bad_block_probability,
        independent_turn_maximum,
    )
    independent_expected_risk = (
        bad_block_probability * independent_selection_probability
    )

    return {
        "alpha": alpha,
        "calibration_blocks": calibration_blocks,
        "turns_per_block": turns_per_block,
        "bad_block_probability": bad_block_probability,
        "naive_maximum_bad_blocks": naive_maximum,
        "block_maximum_bad_blocks": block_maximum,
        "clustered_naive_expected_risk": clustered_naive_expected_risk,
        "clustered_block_expected_risk": clustered_block_expected_risk,
        "independent_turn_expected_risk": independent_expected_risk,
        "naive_violates_nominal_risk": clustered_naive_expected_risk > alpha,
        "block_controls_nominal_risk": clustered_block_expected_risk <= alpha,
    }


def policy_shift_stress() -> dict[str, Any]:
    """Show why a threshold cannot be carried across policy iterations."""

    calibration_blocks = 20
    alpha = 0.10
    calibration_bad_probability = 0.05
    shifted_bad_probability = 0.25
    maximum_bad_blocks = maximum_accepted_errors(calibration_blocks, alpha)

    stale_selection_probability = binomial_probability_at_most(
        calibration_blocks,
        calibration_bad_probability,
        maximum_bad_blocks,
    )
    refreshed_selection_probability = binomial_probability_at_most(
        calibration_blocks,
        shifted_bad_probability,
        maximum_bad_blocks,
    )
    stale_expected_risk = shifted_bad_probability * stale_selection_probability
    refreshed_expected_risk = (
        shifted_bad_probability * refreshed_selection_probability
    )

    return {
        "alpha": alpha,
        "calibration_blocks": calibration_blocks,
        "calibration_bad_probability": calibration_bad_probability,
        "shifted_bad_probability": shifted_bad_probability,
        "maximum_bad_blocks": maximum_bad_blocks,
        "stale_threshold_expected_risk": stale_expected_risk,
        "refreshed_threshold_expected_risk": refreshed_expected_risk,
        "stale_threshold_violates_nominal_risk": stale_expected_risk > alpha,
        "refreshed_threshold_controls_nominal_risk": (
            refreshed_expected_risk <= alpha
        ),
    }


def oracle_summary(rows: list[dict[str, object]]) -> dict[str, Any]:
    """Summarize sign conflicts hidden by terminal outcome."""

    successful_negative = sum(
        row["reward"] == 1 and row["oracle_sign"] == -1 for row in rows
    )
    failed_positive = sum(
        row["reward"] == 0 and row["oracle_sign"] == 1 for row in rows
    )
    sign_counts = {
        str(sign_value): sum(row["oracle_sign"] == sign_value for row in rows)
        for sign_value in (-1, 0, 1)
    }
    return {
        "trajectory_count": len({row["trajectory_id"] for row in rows}),
        "turn_label_count": len(rows),
        "sign_counts": sign_counts,
        "negative_turns_inside_successes": successful_negative,
        "positive_turns_inside_failures": failed_positive,
        "contains_both_outcome_credit_conflicts": (
            successful_negative > 0 and failed_positive > 0
        ),
    }


def sha256(path: Path) -> str:
    """Hash one generated artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate(output_dir: Path) -> dict[str, Any]:
    """Write deterministic CSV/JSON evidence and a content manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    oracle_rows = enumerate_oracle_rows()
    csv_path = output_dir / "bifurcation_trajectories.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(oracle_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(oracle_rows)

    summary = {
        "protocol_id": "CARE-P1-v0.1",
        "deterministic": True,
        "gpu_training_performed": False,
        "oracle": oracle_summary(oracle_rows),
        "clustered_dependence": clustered_dependence_stress(),
        "policy_shift": policy_shift_stress(),
    }
    summary_path = output_dir / "p1_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "protocol_id": "CARE-P1-v0.1",
        "artifacts": {
            csv_path.name: {"sha256": sha256(csv_path), "rows": len(oracle_rows)},
            summary_path.name: {"sha256": sha256(summary_path)},
        },
    }
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results"),
    )
    args = parser.parse_args()
    summary = generate(args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

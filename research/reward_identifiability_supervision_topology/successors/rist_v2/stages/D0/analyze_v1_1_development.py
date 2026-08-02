"""Diagnose RIST-v1.1 task-resolution miscalibration without held-out access."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = average
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("correlation inputs must be non-empty and equal length")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)
    )
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_scale == 0.0 or right_scale == 0.0:
        return 0.0
    return numerator / (left_scale * right_scale)


def _spearman(left: list[float], right: list[float]) -> float:
    return _pearson(_rank(left), _rank(right))


def analyze(task_path: Path, qwen_path: Path) -> dict[str, Any]:
    """Return the frozen D0 diagnosis from the complete Qwen development cell."""

    tasks = [
        json.loads(line)
        for line in task_path.read_text(encoding="utf-8").splitlines()
    ]
    if len(tasks) != 32 or any(task.get("split") != "qualification" for task in tasks):
        raise ValueError("D0 requires the immutable 32-row v1 qualification file")
    task_by_signature = {str(task["task_signature"]): task for task in tasks}
    if len(task_by_signature) != 32:
        raise ValueError("task signatures must be unique")

    qwen = json.loads(qwen_path.read_text(encoding="utf-8"))
    expected_hash = sha256(task_path)
    if qwen.get("qualification_sha256") != expected_hash:
        raise ValueError("Qwen result does not match the immutable task file")
    if qwen.get("model_cell") != "qwen3_0_6b":
        raise ValueError("D0 accepts only the complete Qwen development cell")
    if qwen.get("infrastructure_error") is not None:
        raise ValueError("Qwen cell must be infrastructure-complete")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in qwen.get("trajectories", []):
        signature = str(trajectory.get("task_signature"))
        if signature not in task_by_signature:
            raise ValueError(f"unknown task signature: {signature}")
        grouped[signature].append(trajectory)
    if set(grouped) != set(task_by_signature):
        raise ValueError("Qwen result does not cover all 32 tasks")
    if any(len(rows) != 8 for rows in grouped.values()):
        raise ValueError("every task must retain exactly eight trajectories")

    task_records = []
    for signature, task in task_by_signature.items():
        rows = grouped[signature]
        successes = sum(int(row["strict_reward"]) for row in rows)
        task_records.append(
            {
                "task_signature": signature,
                "factor_cell": task["factor_cell"],
                "analytic_stratum": task["reward_resolution_stratum"],
                "analytic_success_probability": float(
                    task["uniform_strict_success_probability"]
                ),
                "analytic_mixed_probability_g8": float(
                    task["mixed_group_probability_g8"]
                ),
                "empirical_successes": successes,
                "empirical_trials": len(rows),
                "empirical_success_rate": successes / len(rows),
                "empirical_mixed": 0 < successes < len(rows),
            }
        )
    task_records.sort(key=lambda row: str(row["task_signature"]))

    stratum_summary = {}
    for stratum in ("low", "intermediate", "high"):
        rows = [row for row in task_records if row["analytic_stratum"] == stratum]
        mixed = sum(int(row["empirical_mixed"]) for row in rows)
        stratum_summary[stratum] = {
            "task_count": len(rows),
            "mixed_group_count": mixed,
            "mixed_group_rate": mixed / len(rows),
            "mean_empirical_success_rate": sum(
                float(row["empirical_success_rate"]) for row in rows
            )
            / len(rows),
        }

    analytic_mixed = [
        float(row["analytic_mixed_probability_g8"]) for row in task_records
    ]
    empirical_mixed = [float(row["empirical_mixed"]) for row in task_records]
    analytic_success = [
        float(row["analytic_success_probability"]) for row in task_records
    ]
    empirical_success = [
        float(row["empirical_success_rate"]) for row in task_records
    ]
    raw_response_count = sum(
        int(row.get("raw_response_count", 0))
        for rows in grouped.values()
        for row in rows
    )
    high_not_above_low = (
        stratum_summary["high"]["mixed_group_rate"]
        <= stratum_summary["low"]["mixed_group_rate"]
    )
    gates = {
        "task_hash_matches_qwen_result": qwen["qualification_sha256"]
        == expected_hash,
        "complete_32_by_8_trajectories": len(qwen["trajectories"]) == 256,
        "all_1024_raw_responses_retained": raw_response_count == 1024,
        "zero_retry_repair_or_fabrication": int(qwen.get("retry_count", -1)) == 0
        and int(qwen.get("fabricated_call_count", -1)) == 0,
        "no_training": qwen.get("training_performed") is False,
        "heldout_not_opened_by_d0": True,
        "analytic_high_not_more_mixed_than_low": high_not_above_low,
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "protocol": "RIST-D0-v2.0",
        "stage": "D0",
        "stage_status": "PASS" if passed else "KILL",
        "decision": (
            "PASS_D0_MODEL_CONDITIONAL_MISCALIBRATION_TO_D1_CPU_CALIBRATOR"
            if passed
            else "KILL_RIST_V2_D0_INPUT_OR_DIAGNOSIS_GATE"
        ),
        "scientific_effect_estimated": False,
        "source_use": "DEVELOPMENT_ONLY_NO_FUTURE_CONFIRMATORY_SPLICE",
        "heldout_data_opened": False,
        "input_hashes": {
            "v1_qualification_sha256": expected_hash,
            "qwen_result_sha256": sha256(qwen_path),
        },
        "gates": gates,
        "strata": stratum_summary,
        "diagnostics": {
            "analytic_mixed_brier_score": sum(
                (prediction - outcome) ** 2
                for prediction, outcome in zip(
                    analytic_mixed, empirical_mixed, strict=True
                )
            )
            / len(task_records),
            "spearman_analytic_to_empirical_mixed": _spearman(
                analytic_mixed, empirical_mixed
            ),
            "spearman_analytic_to_empirical_success": _spearman(
                analytic_success, empirical_success
            ),
            "mixed_factor_cells": sorted(
                str(row["factor_cell"])
                for row in task_records
                if row["empirical_mixed"]
            ),
        },
        "task_records": task_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--qwen-result", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.tasks, args.qwen_result)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["stage_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

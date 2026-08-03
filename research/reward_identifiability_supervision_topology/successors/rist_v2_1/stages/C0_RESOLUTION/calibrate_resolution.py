"""Calibrate checkpoint-conditional mixed-group resolution from raw rewards."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

GROUP_SIZE = 8
GROUPS_PER_TASK = 4
TASKS_PER_CELL = 4
TRIALS_PER_TASK = GROUP_SIZE * GROUPS_PER_TASK
D1_PATH = Path(__file__).resolve().parents[3] / "rist_v2/stages/D1/calibration.py"


def _wilson(successes: int, trials: int) -> tuple[float, float]:
    spec = importlib.util.spec_from_file_location("rist_v2_1_c0_d1", D1_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("D1 uncertainty helper unavailable")
    spec.loader.exec_module(module)
    return module.wilson_interval(successes, trials)


def summarize_split(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    """Summarize one pretraining split from exact task/seed strict rewards."""

    by_task: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    seen = set()
    for row in rows:
        if row.get("split") != split:
            raise ValueError("resolution rows contain an unexpected split")
        reward = row.get("strict_success")
        if type(reward) is not int or reward not in (0, 1):
            raise ValueError("strict_success must be integer zero or one")
        key = (str(row["structural_cell"]), str(row["task_id"]))
        identity = (*key, int(row["rollout_seed"]))
        if identity in seen:
            raise ValueError("duplicate task/seed resolution observation")
        seen.add(identity)
        by_task[key].append((int(row["rollout_seed"]), reward))

    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (cell, task_id), observations in sorted(by_task.items()):
        observations.sort()
        if len(observations) != TRIALS_PER_TASK:
            raise ValueError("each calibration task requires exactly 32 rollout seeds")
        groups = []
        for start in range(0, TRIALS_PER_TASK, GROUP_SIZE):
            rewards = [reward for _, reward in observations[start : start + GROUP_SIZE]]
            groups.append(int(len(set(rewards)) > 1))
        cells[cell].append(
            {
                "task_id": task_id,
                "mixed_groups": sum(groups),
                "group_count": len(groups),
                "mixed_rate": sum(groups) / len(groups),
            }
        )
    if len(cells) != 8 or any(len(tasks) != TASKS_PER_CELL for tasks in cells.values()):
        raise ValueError("resolution split requires four tasks in each of eight cells")

    summaries = {}
    for cell, tasks in sorted(cells.items()):
        mixed = sum(task["mixed_groups"] for task in tasks)
        total = sum(task["group_count"] for task in tasks)
        lower, upper = _wilson(mixed, total)
        if upper <= 0.25:
            classification = "collapsed"
            agreeing = sum(task["mixed_rate"] <= 0.25 for task in tasks)
        elif lower >= 0.50:
            classification = "resolved"
            agreeing = sum(task["mixed_rate"] >= 0.50 for task in tasks)
        else:
            classification = "transition"
            agreeing = TASKS_PER_CELL
        if classification != "transition" and agreeing < 3:
            classification = "heterogeneous"
        summaries[cell] = {
            "classification": classification,
            "mixed_groups": mixed,
            "group_count": total,
            "mixed_rate": mixed / total,
            "wilson_interval": [lower, upper],
            "agreeing_task_count": agreeing,
            "tasks": tasks,
        }
    return {"split": split, "cells": summaries, "row_count": len(rows)}


def calibrate_checkpoint(
    calibration_rows: list[dict[str, Any]],
    qualification_rows: list[dict[str, Any]],
    checkpoint: str,
) -> dict[str, Any]:
    calibration = summarize_split(calibration_rows, "calibration")
    qualification = summarize_split(qualification_rows, "qualification")
    resolution_map = {}
    transport = {}
    for cell in sorted(calibration["cells"]):
        development_label = calibration["cells"][cell]["classification"]
        qualification_label = qualification["cells"][cell]["classification"]
        passed = (
            development_label in {"collapsed", "resolved"}
            and qualification_label == development_label
        )
        transport[cell] = {
            "calibration": development_label,
            "qualification": qualification_label,
            "passed": passed,
        }
        if passed:
            resolution_map[cell] = "low" if development_label == "collapsed" else "high"
    counts = {
        band: sum(value == band for value in resolution_map.values())
        for band in ("low", "high")
    }
    passed = counts["low"] >= 2 and counts["high"] >= 2
    return {
        "protocol": "RIST-C0-v2.1",
        "checkpoint": checkpoint,
        "calibration": calibration,
        "qualification": qualification,
        "transport": transport,
        "resolution_map": resolution_map,
        "band_cell_counts": counts,
        "passed": passed,
        "decision": "PASS_C0" if passed else "KILL_C0_TASK_POOL",
    }


def _read_collection_result(path: Path, split: str) -> list[dict[str, Any]]:
    result = json.loads(path.read_text())
    if not isinstance(result, dict) or result.get("protocol") != "RIST-C0-v2.1":
        raise ValueError("calibration input must be a C0 collection result")
    if (
        result.get("split") != split
        or result.get("complete") is not True
        or result.get("infrastructure_error") is not None
        or result.get("retry_count") != 0
    ):
        raise ValueError("C0 collection result is incomplete or belongs to another split")
    rows = result.get("trajectories")
    if not isinstance(rows, list) or len(rows) != int(result["expected_trajectory_count"]):
        raise ValueError("C0 collection result has incomplete trajectories")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calibrate_checkpoint(
        _read_collection_result(args.calibration, "calibration"),
        _read_collection_result(args.qualification, "qualification"),
        args.checkpoint,
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

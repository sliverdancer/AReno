"""Pure C0 v2.3 resolution, transport, and structural filtering logic."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from typing import Any

GROUP_SIZE = 8
GROUPS_PER_TASK = 4
TASKS_PER_CELL = 4
TRIALS_PER_TASK = GROUP_SIZE * GROUPS_PER_TASK


def canonical_sha256(value: Any) -> str:
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError("invalid Wilson interval counts")
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return center - margin, center + margin


def summarize_split(rows: list[dict[str, Any]], split: str) -> dict[str, Any]:
    by_task: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    seen = set()
    for row in rows:
        if row.get("split") != split or type(row.get("strict_success")) is not int:
            raise ValueError("resolution rows have an invalid split or reward")
        reward = row["strict_success"]
        if reward not in (0, 1) or type(row.get("rollout_seed")) is not int:
            raise ValueError("resolution rewards and seeds must be exact integers")
        key = (str(row["structural_cell"]), str(row["task_id"]))
        identity = (*key, row["rollout_seed"])
        if identity in seen:
            raise ValueError("duplicate task/seed resolution observation")
        seen.add(identity)
        by_task[key].append((row["rollout_seed"], reward))

    cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (cell, task_id), observations in sorted(by_task.items()):
        observations.sort()
        if len(observations) != TRIALS_PER_TASK:
            raise ValueError("each task requires exactly 32 rollout seeds")
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
        lower, upper = wilson_interval(mixed, total)
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


def calibrate_family(
    calibration_rows: list[dict[str, Any]],
    qualification_rows: list[dict[str, Any]],
    family: str,
    checkpoint: str,
) -> dict[str, Any]:
    calibration = summarize_split(calibration_rows, "calibration")
    qualification = summarize_split(qualification_rows, "qualification")
    resolution_map = {}
    transport = {}
    for cell in sorted(calibration["cells"]):
        development = calibration["cells"][cell]["classification"]
        confirmation = qualification["cells"][cell]["classification"]
        passed = development in {"collapsed", "resolved"} and confirmation == development
        transport[cell] = {
            "calibration": development,
            "qualification": confirmation,
            "passed": passed,
        }
        if passed:
            resolution_map[cell] = "low" if development == "collapsed" else "high"
    counts = {band: sum(value == band for value in resolution_map.values()) for band in ("low", "high")}
    passed = counts["low"] >= 2 and counts["high"] >= 2
    return {
        "protocol": "RIST-C0-v2.3-RESOLUTION-FAMILY-v1",
        "family": family,
        "checkpoint": checkpoint,
        "calibration": calibration,
        "qualification": qualification,
        "transport": transport,
        "resolution_map": resolution_map,
        "band_cell_counts": counts,
        "passed": passed,
        "decision": "PASS_FAMILY_RESOLUTION" if passed else "KILL_C0_FAMILY_RESOLUTION",
    }


def combine_families(results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(results) != 2 or len({row.get("family") for row in results}) != 2:
        raise ValueError("cross-family resolution requires exactly two families")
    if not all(row.get("passed") is True for row in results):
        return {
            "protocol": "RIST-C0-v2.3-CROSS-FAMILY-v1",
            "passed": False,
            "decision": "KILL_C0_FAMILY_RESOLUTION",
            "common_resolution_map": {},
            "band_cell_counts": {"low": 0, "high": 0},
        }
    maps = [row["resolution_map"] for row in results]
    common = {
        cell: maps[0][cell]
        for cell in sorted(set(maps[0]) & set(maps[1]))
        if maps[0][cell] == maps[1][cell]
    }
    counts = {band: sum(value == band for value in common.values()) for band in ("low", "high")}
    passed = counts["low"] >= 2 and counts["high"] >= 2
    return {
        "protocol": "RIST-C0-v2.3-CROSS-FAMILY-v1",
        "families": [row["family"] for row in results],
        "checkpoints": [row["checkpoint"] for row in results],
        "common_resolution_map": common,
        "band_cell_counts": counts,
        "passed": passed,
        "decision": "PASS_C0_V2_3_TO_FILTERED_TRAIN" if passed else "KILL_C0_NO_COMMON_BANDS",
    }


def filter_train_rows(
    source_rows: list[dict[str, Any]], resolution_map: dict[str, str]
) -> list[dict[str, Any]]:
    counts = Counter(resolution_map.values())
    if set(resolution_map.values()) != {"low", "high"} or counts["low"] < 2 or counts["high"] < 2:
        raise ValueError("resolution map requires at least two cells per band")
    if len(source_rows) != 32 or any(row.get("split") != "train" for row in source_rows):
        raise ValueError("D3 source must contain exactly 32 train rows")
    all_cell_counts = Counter(str(row.get("structural_cell")) for row in source_rows)
    if len(all_cell_counts) != 8 or set(all_cell_counts.values()) != {4}:
        raise ValueError("D3 source must contain four tasks in each of eight cells")
    selected = []
    for source in source_rows:
        cell = str(source["structural_cell"])
        if cell in resolution_map:
            row = dict(source)
            row["resolution_band"] = resolution_map[cell]
            selected.append(row)
    selected_counts = Counter(str(row["structural_cell"]) for row in selected)
    if set(selected_counts) != set(resolution_map) or set(selected_counts.values()) != {4}:
        raise ValueError("filter must retain whole four-task cells")
    return selected


def select_e1_rows(filtered_rows: list[dict[str, Any]], common_map: dict[str, str]) -> tuple[str, list[dict[str, Any]]]:
    high_cells = sorted(cell for cell, band in common_map.items() if band == "high")
    if len(high_cells) < 2:
        raise ValueError("E1 requires at least two common high cells")
    selected_cell = high_cells[0]
    rows = [dict(row) for row in filtered_rows if str(row.get("structural_cell")) == selected_cell]
    if len(rows) != 4 or any(row.get("resolution_band") != "high" for row in rows):
        raise ValueError("E1 must use one whole transported high cell")
    return selected_cell, rows

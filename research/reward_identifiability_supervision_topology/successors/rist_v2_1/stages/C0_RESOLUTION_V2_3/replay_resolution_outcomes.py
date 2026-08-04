"""Recompute C0 v2.3 strict outcomes from frozen tasks and raw responses."""

from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Any

EVALUATOR_PATH = Path(__file__).resolve().parents[1] / "D4_EVAL/evaluate_checkpoint.py"
TRAJECTORY_FIELDS = (
    "task_id",
    "task_signature",
    "structural_cell",
    "rollout_seed",
    "actions",
    "strict_success",
    "completed_turns",
    "invalid_reason",
    "raw_response_count",
)


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("rist_c0_v2_3_resolution_evaluator", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("C0 v2.3 evaluator is unavailable")
    spec.loader.exec_module(module)
    return module


def replay_outcomes(
    *,
    source_rows: list[dict[str, Any]],
    rollout_seeds: list[int],
    split: str,
    trajectories: list[dict[str, Any]],
    journal_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replay raw responses and require exact agreement with submitted rows."""

    source_split = f"c0v2_3_{split}"
    if (
        len(source_rows) != 32
        or any(type(row.get("id")) is not str or not row["id"] for row in source_rows)
        or any(
            type(row.get("task_signature")) is not str or not row["task_signature"]
            for row in source_rows
        )
        or len({row["id"] for row in source_rows}) != 32
        or len({row["task_signature"] for row in source_rows}) != 32
        or any(row.get("split") != source_split for row in source_rows)
    ):
        raise ValueError("resolution replay requires exactly 32 source tasks from one split")
    tasks = {row["id"]: row for row in source_rows}
    if any(type(seed) is not int for seed in rollout_seeds):
        raise ValueError("resolution rollout seeds must be exact integers")
    seeds = list(rollout_seeds)
    if len(seeds) != 32 or len(set(seeds)) != 32:
        raise ValueError("resolution replay requires exactly 32 unique rollout seeds")
    expected_pairs = {(task_id, seed) for task_id in tasks for seed in seeds}

    submitted: dict[tuple[str, int], dict[str, Any]] = {}
    for row in trajectories:
        if type(row.get("task_id")) is not str or not row["task_id"]:
            raise ValueError("submitted task id must be a non-empty string")
        if type(row.get("rollout_seed")) is not int:
            raise ValueError("submitted rollout seed must be an integer")
        pair = (row["task_id"], row["rollout_seed"])
        if pair in submitted:
            raise ValueError("duplicate submitted task/seed trajectory")
        submitted[pair] = row
    if set(submitted) != expected_pairs:
        raise ValueError("submitted trajectories do not cover the frozen task/seed grid")

    journals: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in journal_rows:
        if type(row.get("task_id")) is not str or not row["task_id"]:
            raise ValueError("raw journal task id must be a non-empty string")
        if type(row.get("rollout_seed")) is not int or type(row.get("turn_index")) is not int:
            raise ValueError("raw journal seed and turn index must be integers")
        pair = (row["task_id"], row["rollout_seed"])
        if pair not in expected_pairs:
            raise ValueError("raw journal contains an unexpected task/seed identity")
        journals[pair].append(row)
    if set(journals) != expected_pairs:
        raise ValueError("raw journal does not cover the frozen task/seed grid")

    evaluator = _load_evaluator()
    derived = []
    for task_id, seed in sorted(expected_pairs):
        task = tasks[task_id]
        rows = sorted(journals[(task_id, seed)], key=lambda row: row["turn_index"])
        turn_indices = [row["turn_index"] for row in rows]
        if turn_indices != list(range(len(rows))) or not 1 <= len(rows) <= len(task["turns"]):
            raise ValueError("raw journal turns must be contiguous and bounded")
        actions = []
        invalid_reason = None
        for position, row in enumerate(rows):
            if type(row.get("task_signature")) is not str or row["task_signature"] != task["task_signature"]:
                raise ValueError("raw journal task signature mismatch")
            response = row.get("raw_response")
            try:
                message = response["choices"][0]["message"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError("raw response lacks the frozen message schema") from exc
            parsed = evaluator.parse_call(message, task["turns"][position])
            if not parsed["valid"]:
                invalid_reason = parsed["reason"]
                if position != len(rows) - 1:
                    raise ValueError("raw journal continues after the first invalid call")
                break
            call = parsed["call"]
            actions.append({"name": call["name"], "arguments": call["arguments"]})
        if invalid_reason is None and len(rows) != len(task["turns"]):
            raise ValueError("valid raw journal terminates before the oracle route is complete")
        replayed = {
            "task_id": task_id,
            "task_signature": task["task_signature"],
            "structural_cell": task["structural_cell"],
            "rollout_seed": seed,
            "actions": actions,
            "strict_success": int(actions == task["oracle_actions"]),
            "completed_turns": len(actions),
            "invalid_reason": invalid_reason,
            "raw_response_count": len(rows),
            "split": split,
        }
        submitted_row = submitted[(task_id, seed)]
        required = set(TRAJECTORY_FIELDS) | {"split"}
        if not required <= set(submitted_row):
            raise ValueError("submitted trajectory lacks a required outcome field")
        if submitted_row["split"] != split or any(
            type(submitted_row[field]) is not type(replayed[field])
            or submitted_row[field] != replayed[field]
            for field in TRAJECTORY_FIELDS
        ):
            raise ValueError("submitted trajectory outcome disagrees with raw-response replay")
        derived.append(replayed)
    return derived

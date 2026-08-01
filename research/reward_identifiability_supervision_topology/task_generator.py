"""Generate the deterministic RIST workflow task factorial."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from itertools import product
from pathlib import Path
from typing import Any

GENERATOR_VERSION = "rist-workflow-v1"
DEFAULT_SEED = 20260801
GROUP_SIZE = 8
SPLIT_REPLICATES = {"train": 3, "qualification": 1, "heldout": 1}
EXPECTED_TOOLS = (
    "scan_registry",
    "inspect_candidate",
    "verify_route",
    "submit_route",
)
DISTRACTOR_TOOL_POOL = (
    "estimate_cost",
    "request_hint",
    "reset_route",
    "summarize_registry",
)
FACTOR_LEVELS = {
    "constraint_slack": (0, 2),
    "distractor_tools": (0, 2),
    "dependency_depth": (0, 3),
    "argument_options": (1, 4),
    "tool_choice_mode": ("forced", "free"),
}


def generate_splits(seed: int = DEFAULT_SEED) -> dict[str, list[dict[str, Any]]]:
    """Return balanced, signature-disjoint task splits."""

    splits: dict[str, list[dict[str, Any]]] = {}
    for split, replicates in SPLIT_REPLICATES.items():
        rows = []
        for factors in _factor_cells():
            for replicate in range(replicates):
                rows.append(_build_task(split, factors, replicate, seed))
        rows.sort(key=lambda row: str(row["task_signature"]))
        for split_index, row in enumerate(rows):
            row["id"] = f"{split}-{split_index:03d}"
        splits[split] = rows
    validate_splits(splits)
    return splits


def validate_splits(splits: dict[str, list[dict[str, Any]]]) -> None:
    """Reject schema drift, cell imbalance, or signature leakage."""

    if set(splits) != set(SPLIT_REPLICATES):
        raise ValueError(f"split names must be {sorted(SPLIT_REPLICATES)}")
    all_signatures: dict[str, str] = {}
    expected_cells = {cell_id(factors) for factors in _factor_cells()}
    for split, rows in splits.items():
        expected_count = 32 * SPLIT_REPLICATES[split]
        if len(rows) != expected_count:
            raise ValueError(
                f"{split} must contain {expected_count} tasks, got {len(rows)}"
            )
        cell_counts = {identifier: 0 for identifier in expected_cells}
        for row in rows:
            validate_task(row)
            if row.get("split") != split:
                raise ValueError(f"task {row.get('id')} has wrong split label")
            identifier = str(row["factor_cell"])
            if identifier not in cell_counts:
                raise ValueError(f"unknown factor cell: {identifier}")
            cell_counts[identifier] += 1
            signature = str(row["task_signature"])
            if signature in all_signatures:
                raise ValueError(
                    f"signature shared by {all_signatures[signature]} and {split}"
                )
            all_signatures[signature] = split
        if set(cell_counts.values()) != {SPLIT_REPLICATES[split]}:
            raise ValueError(f"{split} factor cells are not balanced")


def validate_task(task: dict[str, Any]) -> None:
    """Validate one task's oracle and analytic reward-resolution fields."""

    if task.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("generator version mismatch")
    oracle = task.get("oracle_actions")
    turns = task.get("turns")
    if not isinstance(oracle, list) or len(oracle) != 4:
        raise ValueError("every task must have exactly four oracle actions")
    if not isinstance(turns, list) or len(turns) != 4:
        raise ValueError("every task must have exactly four turn definitions")
    recomputed_sequences = 1
    for turn_index, (action, turn) in enumerate(zip(oracle, turns, strict=True)):
        expected_name = EXPECTED_TOOLS[turn_index]
        if action.get("name") != expected_name:
            raise ValueError("oracle tool order mismatch")
        if action.get("arguments") != {"code": turn.get("correct_code")}:
            raise ValueError("oracle argument mismatch")
        tool_options = turn.get("offered_tools")
        argument_candidates = turn.get("argument_candidates")
        if not isinstance(tool_options, list) or expected_name not in tool_options:
            raise ValueError("expected tool absent from offered tools")
        if len(tool_options) != len(set(tool_options)):
            raise ValueError("offered tool names must be unique")
        if not isinstance(argument_candidates, list):
            raise ValueError("argument candidates must be a list")
        if turn["correct_code"] not in argument_candidates:
            raise ValueError("correct code absent from argument candidates")
        if len(argument_candidates) != len(set(argument_candidates)):
            raise ValueError("argument candidates must be unique")
        recomputed_sequences *= len(tool_options) * len(argument_candidates)
    if recomputed_sequences != int(task.get("possible_action_sequences", 0)):
        raise ValueError("possible-action-sequence count mismatch")
    expected_p = 1.0 / recomputed_sequences
    observed_p = float(task.get("uniform_strict_success_probability", -1.0))
    if not math.isclose(expected_p, observed_p, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("uniform strict-success probability mismatch")
    expected_mixed = 1.0 - expected_p**GROUP_SIZE - (1.0 - expected_p) ** GROUP_SIZE
    observed_mixed = float(task.get("mixed_group_probability_g8", -1.0))
    if not math.isclose(expected_mixed, observed_mixed, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("mixed-group probability mismatch")
    if task.get("reward_resolution_stratum") != resolution_stratum(expected_mixed):
        raise ValueError("reward-resolution stratum mismatch")
    expected_signature = _task_signature(task)
    if task.get("task_signature") != expected_signature:
        raise ValueError("task signature mismatch")


def strict_reward(task: dict[str, Any], actions: list[dict[str, Any]]) -> int:
    """Return one only for the exact, complete oracle call sequence."""

    return int(actions == task["oracle_actions"])


def resolution_stratum(mixed_group_probability: float) -> str:
    """Map analytic mixed-group probability to a frozen coarse stratum."""

    if mixed_group_probability < 0.1:
        return "low"
    if mixed_group_probability < 0.8:
        return "intermediate"
    return "high"


def cell_id(factors: dict[str, Any]) -> str:
    """Return a stable identifier for one factorial cell."""

    return "-".join(
        (
            f"s{factors['constraint_slack']}",
            f"z{factors['distractor_tools']}",
            f"d{factors['dependency_depth']}",
            f"a{factors['argument_options']}",
            "f" if factors["tool_choice_mode"] == "forced" else "u",
        )
    )


def write_splits(output_dir: Path, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Write JSONL splits plus a content-addressed manifest."""

    splits = generate_splits(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for split, rows in splits.items():
        path = output_dir / f"{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        files[split] = {
            "file": path.name,
            "count": len(rows),
            "sha256": _sha256(path),
            "task_signatures": [str(row["task_signature"]) for row in rows],
        }
    manifest = {
        "schema_version": 1,
        "protocol": "RIST-P1-v1.0",
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "group_size": GROUP_SIZE,
        "split_replicates": SPLIT_REPLICATES,
        "factor_levels": FACTOR_LEVELS,
        "files": files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _factor_cells() -> list[dict[str, Any]]:
    names = list(FACTOR_LEVELS)
    return [
        dict(zip(names, values, strict=True))
        for values in product(*(FACTOR_LEVELS[name] for name in names))
    ]


def _build_task(
    split: str,
    factors: dict[str, Any],
    replicate: int,
    seed: int,
) -> dict[str, Any]:
    nonce = hashlib.sha256(
        f"{seed}:{split}:{cell_id(factors)}:{replicate}".encode()
    ).hexdigest()[:16]
    dependency_start = 4 - int(factors["dependency_depth"])
    visible_codes = []
    turns = []
    oracle_actions = []
    for turn_index, expected_tool in enumerate(EXPECTED_TOOLS):
        correct_code = _code(nonce, turn_index, 0)
        dependent = turn_index >= dependency_start
        if not dependent:
            visible_codes.append(correct_code)
        base_options = int(factors["argument_options"]) if dependent else 1
        final_turn_slack = (
            int(factors["constraint_slack"]) if turn_index == 3 else 0
        )
        candidate_count = base_options + final_turn_slack
        candidates = [
            _code(nonce, turn_index, candidate_index)
            for candidate_index in range(candidate_count)
        ]
        offered_tools = [expected_tool]
        if factors["tool_choice_mode"] == "free":
            offered_tools.extend(
                DISTRACTOR_TOOL_POOL[
                    (turn_index + offset) % len(DISTRACTOR_TOOL_POOL)
                ]
                for offset in range(int(factors["distractor_tools"]))
            )
        next_code = (
            _code(nonce, turn_index + 1, 0) if turn_index + 1 < 4 else None
        )
        turns.append(
            {
                "turn_index": turn_index,
                "expected_tool": expected_tool,
                "correct_code": correct_code,
                "argument_candidates": candidates,
                "offered_tools": offered_tools,
                "depends_on_previous_observation": dependent,
                "successful_observation": {"next_code": next_code},
            }
        )
        oracle_actions.append(
            {"name": expected_tool, "arguments": {"code": correct_code}}
        )
    possible_sequences = math.prod(
        len(turn["offered_tools"]) * len(turn["argument_candidates"])
        for turn in turns
    )
    success_probability = 1.0 / possible_sequences
    mixed_probability = (
        1.0
        - success_probability**GROUP_SIZE
        - (1.0 - success_probability) ** GROUP_SIZE
    )
    task: dict[str, Any] = {
        "generator_version": GENERATOR_VERSION,
        "dataset_seed": seed,
        "split": split,
        "replicate": replicate,
        "scenario_nonce": nonce,
        "factors": dict(factors),
        "factor_cell": cell_id(factors),
        "prompt": (
            "Complete the four-step registry route. Use exactly one offered tool "
            "per turn and pass its required code. Prompt-visible codes: "
            + (", ".join(visible_codes) if visible_codes else "none")
            + ". Hidden codes are revealed only by successful prior observations."
        ),
        "turns": turns,
        "oracle_actions": oracle_actions,
        "possible_action_sequences": possible_sequences,
        "uniform_strict_success_probability": success_probability,
        "mixed_group_probability_g8": mixed_probability,
        "reward_resolution_stratum": resolution_stratum(mixed_probability),
    }
    task["task_signature"] = _task_signature(task)
    return task


def _task_signature(task: dict[str, Any]) -> str:
    excluded = {
        "id",
        "task_signature",
        "uniform_strict_success_probability",
        "mixed_group_probability_g8",
        "reward_resolution_stratum",
        "possible_action_sequences",
    }
    payload = {key: value for key, value in task.items() if key not in excluded}
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _code(nonce: str, turn_index: int, candidate_index: int) -> str:
    digest = hashlib.sha256(
        f"{nonce}:{turn_index}:{candidate_index}".encode()
    ).hexdigest()
    return f"r{turn_index}-{digest[:8]}"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    print(json.dumps(write_splits(args.output_dir, args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

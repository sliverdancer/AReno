"""Build the outcome-unopened, task- and seed-disjoint RIST v4 C0 pool.

v4 retires the v3 hash-code transport. It uses short semantic codes while
preserving strict tool-call and argument matching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[6]
RESEARCH_ROOT = REPO_ROOT / "research/reward_identifiability_supervision_topology"
GENERATOR_VERSION = "rist-c0-v4-semantic-code-pool"
POOL_SEED = 202608100401
CANARY_TASK_SEED = 202608100402
REPLICATES_PER_CELL = 4
ROLLOUT_SEEDS = {
    "capacity_canary": list(range(41001, 41009)),
    "calibration": list(range(42001, 42033)),
    "qualification": list(range(43001, 43033)),
}
EXPECTED_TOOLS = (
    "scan_registry",
    "inspect_candidate",
    "verify_route",
    "submit_route",
)
DISTRACTOR_TOOLS = (
    "scan_catalog",
    "inspect_catalog",
    "verify_candidate",
    "submit_candidate",
)
LABELS = (
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
)
CODE_PREFIXES = ("A", "B", "C", "D", "E", "F", "G", "H")
CELL_SPECS = (
    {
        "cell": "c00",
        "tool_decoys": 0,
        "argument_decoys": 0,
        "dependency_depth": 0,
        "selector": "direct",
        "semantic_ambiguity": 0,
        "anchor_intent": "high",
    },
    {
        "cell": "c01",
        "tool_decoys": 0,
        "argument_decoys": 1,
        "dependency_depth": 0,
        "selector": "direct",
        "semantic_ambiguity": 0,
        "anchor_intent": "high",
    },
    {
        "cell": "c02",
        "tool_decoys": 1,
        "argument_decoys": 1,
        "dependency_depth": 1,
        "selector": "direct",
        "semantic_ambiguity": 1,
        "anchor_intent": "mixed",
    },
    {
        "cell": "c03",
        "tool_decoys": 1,
        "argument_decoys": 2,
        "dependency_depth": 1,
        "selector": "relational",
        "semantic_ambiguity": 1,
        "anchor_intent": "mixed",
    },
    {
        "cell": "c04",
        "tool_decoys": 2,
        "argument_decoys": 3,
        "dependency_depth": 2,
        "selector": "direct",
        "semantic_ambiguity": 2,
        "anchor_intent": "mixed",
    },
    {
        "cell": "c05",
        "tool_decoys": 2,
        "argument_decoys": 3,
        "dependency_depth": 2,
        "selector": "relational",
        "semantic_ambiguity": 2,
        "anchor_intent": "mixed",
    },
    {
        "cell": "c06",
        "tool_decoys": 3,
        "argument_decoys": 5,
        "dependency_depth": 3,
        "selector": "direct",
        "semantic_ambiguity": 3,
        "anchor_intent": "low",
    },
    {
        "cell": "c07",
        "tool_decoys": 3,
        "argument_decoys": 5,
        "dependency_depth": 3,
        "selector": "relational",
        "semantic_ambiguity": 3,
        "anchor_intent": "low",
    },
)


def _walk_values(value: Any, key: str) -> Iterable[Any]:
    if isinstance(value, dict):
        for current_key, current_value in value.items():
            if current_key == key:
                yield current_value
            yield from _walk_values(current_value, key)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item, key)


def _prior_values() -> tuple[set[str], set[int]]:
    signatures: set[str] = set()
    seeds: set[int] = set()
    v4_root = RESEARCH_ROOT / "successors/rist_v4"
    for path in RESEARCH_ROOT.rglob("*.json"):
        if path.is_relative_to(v4_root):
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        for items in _walk_values(value, "task_signatures"):
            if isinstance(items, list):
                signatures.update(str(item) for item in items)
        for items in _walk_values(value, "rollout_seeds"):
            if isinstance(items, list):
                seeds.update(int(item) for item in items if isinstance(item, int))
        for seed_key in ("pool_seed", "capacity_canary_seed", "canary_seed"):
            for item in _walk_values(value, seed_key):
                if isinstance(item, int):
                    seeds.add(item)
    return signatures, seeds


def _code(nonce: str, turn_index: int, candidate_index: int) -> str:
    digest = int(
        hashlib.sha256(f"{nonce}:{turn_index}:{candidate_index}:code".encode()).hexdigest(),
        16,
    )
    return f"{CODE_PREFIXES[(digest + turn_index) % len(CODE_PREFIXES)]}{1 + (digest % 9)}"


def _label(nonce: str, turn_index: int, candidate_index: int, ambiguity: int) -> str:
    base = LABELS[(turn_index + candidate_index) % len(LABELS)]
    if ambiguity <= 0:
        return base
    suffix = chr(ord("A") + ((candidate_index + ambiguity + turn_index) % 4))
    return f"{base}-{suffix}"


def _build_task(split: str, spec: dict[str, Any], replicate: int, seed: int) -> dict[str, Any]:
    nonce = hashlib.sha256(
        f"{GENERATOR_VERSION}:{seed}:{split}:{spec['cell']}:{replicate}".encode()
    ).hexdigest()[:12]
    dependency_start = 4 - int(spec["dependency_depth"])
    turns = []
    oracle = []
    visible_turns = []
    for index, expected_tool in enumerate(EXPECTED_TOOLS):
        candidate_count = int(spec["argument_decoys"]) + 1
        records = [
            {
                "label": _label(nonce, index, candidate, int(spec["semantic_ambiguity"])),
                "code": _code(nonce, index, candidate),
            }
            for candidate in range(candidate_count)
        ]
        if len({record["code"] for record in records}) != len(records):
            # Deterministically repair rare short-code collisions while keeping codes short.
            seen: set[str] = set()
            for candidate, record in enumerate(records):
                attempt = 0
                while record["code"] in seen:
                    prefix = CODE_PREFIXES[(candidate + attempt) % len(CODE_PREFIXES)]
                    digit = 1 + ((candidate + attempt) % 9)
                    record["code"] = f"{prefix}{digit}"
                    attempt += 1
                    if attempt > 72:
                        raise ValueError("unable to repair short-code collision")
                seen.add(record["code"])
        target_index = (
            0
            if spec["selector"] == "direct"
            else int(hashlib.sha256(f"{nonce}:{index}:target".encode()).hexdigest(), 16)
            % len(records)
        )
        target_label = records[target_index]["label"]
        offered_tools = [expected_tool]
        offered_tools.extend(
            DISTRACTOR_TOOLS[(index + offset) % len(DISTRACTOR_TOOLS)]
            for offset in range(int(spec["tool_decoys"]))
        )
        dependent = index >= dependency_start
        turn = {
            "turn_index": index,
            "expected_tool": expected_tool,
            "offered_tools": offered_tools,
            "candidate_records": records,
            "target_label": target_label,
            "selection_rule": str(spec["selector"]),
            "depends_on_previous_observation": dependent,
        }
        turns.append(turn)
        if not dependent:
            visible_turns.append(turn)
        oracle.append(
            {"name": expected_tool, "arguments": {"code": records[target_index]["code"]}}
        )
    task: dict[str, Any] = {
        "generator_version": GENERATOR_VERSION,
        "dataset_seed": seed,
        "split": split,
        "replicate": replicate,
        "scenario_nonce": nonce,
        "structural_cell": spec["cell"],
        "factors": {
            key: value
            for key, value in spec.items()
            if key not in {"cell", "anchor_intent"}
        },
        "prompt_contract": {
            "instruction": (
                "Complete four ordered registry actions. At each turn choose the "
                "expected tool and pass the short code whose label matches the target label."
            ),
            "initial_visible_turns": visible_turns,
            "hidden_turns_revealed_by_success": int(spec["dependency_depth"]),
        },
        "turns": turns,
        "oracle_actions": oracle,
    }
    task["task_signature"] = _signature(task)
    return task


def validate_task(task: dict[str, Any]) -> None:
    forbidden = {
        "reward_resolution_stratum",
        "mixed_group_probability_g8",
        "strict_success",
        "anchor_intent",
    }
    if forbidden.intersection(task):
        raise ValueError("v4 tasks may not contain outcome or resolution labels")
    if task.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("generator version mismatch")
    turns = task.get("turns")
    oracle = task.get("oracle_actions")
    if not isinstance(turns, list) or len(turns) != 4:
        raise ValueError("every task must contain four turns")
    if not isinstance(oracle, list) or len(oracle) != 4:
        raise ValueError("every task must contain four oracle actions")
    for index, (turn, action) in enumerate(zip(turns, oracle, strict=True)):
        expected_tool = EXPECTED_TOOLS[index]
        if turn["expected_tool"] != expected_tool or action["name"] != expected_tool:
            raise ValueError("oracle tool order mismatch")
        offered = turn["offered_tools"]
        records = turn["candidate_records"]
        if len(offered) != len(set(offered)) or offered.count(expected_tool) != 1:
            raise ValueError("expected tool must be unique among offered tools")
        labels = [record["label"] for record in records]
        codes = [record["code"] for record in records]
        if len(labels) != len(set(labels)) or len(codes) != len(set(codes)):
            raise ValueError("candidate labels and codes must be unique")
        if any(len(code) > 3 or len(code) < 2 for code in codes):
            raise ValueError("v4 codes must be short semantic tokens")
        matches = [record for record in records if record["label"] == turn["target_label"]]
        if len(matches) != 1:
            raise ValueError("target label must identify one unique candidate")
        if action["arguments"] != {"code": matches[0]["code"]}:
            raise ValueError("oracle argument mismatch")
    expected_signature = _signature(task)
    if task.get("task_signature") != expected_signature:
        raise ValueError("task signature mismatch")


def build_rows() -> dict[str, list[dict[str, Any]]]:
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("calibration", "qualification"):
        rows = [
            _build_task(split, spec, replicate, POOL_SEED)
            for spec in CELL_SPECS
            for replicate in range(REPLICATES_PER_CELL)
        ]
        rows.sort(key=lambda row: str(row["task_signature"]))
        for index, row in enumerate(rows):
            row["id"] = f"{split}-v4-{index:03d}"
            validate_task(row)
        counts = Counter(str(row["structural_cell"]) for row in rows)
        if len(rows) != 32 or len(counts) != 8 or set(counts.values()) != {4}:
            raise ValueError(f"{split} is not balanced across eight cells")
        rows_by_split[split] = rows
    hardest = next(spec for spec in CELL_SPECS if spec["cell"] == "c07")
    canary = _build_task("capacity_canary", hardest, 0, CANARY_TASK_SEED)
    canary["id"] = "capacity-canary-v4-c07"
    validate_task(canary)
    rows_by_split["capacity_canary"] = [canary]

    signatures = [
        str(row["task_signature"])
        for rows in rows_by_split.values()
        for row in rows
    ]
    prior_signatures, prior_seeds = _prior_values()
    if len(signatures) != len(set(signatures)):
        raise ValueError("v4 task signatures overlap internally")
    if set(signatures) & prior_signatures:
        raise ValueError("v4 task signatures overlap a prior lineage")
    all_seeds = {POOL_SEED, CANARY_TASK_SEED}.union(
        *(set(values) for values in ROLLOUT_SEEDS.values())
    )
    if len(all_seeds) != 2 + sum(len(values) for values in ROLLOUT_SEEDS.values()):
        raise ValueError("v4 seeds overlap internally")
    if all_seeds & prior_seeds:
        raise ValueError("v4 seeds overlap a prior lineage")
    return rows_by_split


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    code_lengths = [
        len(candidate["code"])
        for row in rows
        for turn in row["turns"]
        for candidate in turn["candidate_records"]
    ]
    return {
        "max_code_length": max(code_lengths),
        "min_code_length": min(code_lengths),
        "candidate_count_by_cell": {
            cell: sorted(
                {
                    len(turn["candidate_records"])
                    for row in rows
                    if row["structural_cell"] == cell
                    for turn in row["turns"]
                }
            )
            for cell in sorted({row["structural_cell"] for row in rows})
        },
        "tool_count_by_cell": {
            cell: sorted(
                {
                    len(turn["offered_tools"])
                    for row in rows
                    if row["structural_cell"] == cell
                    for turn in row["turns"]
                }
            )
            for cell in sorted({row["structural_cell"] for row in rows})
        },
        "dependency_depth_by_cell": {
            cell: sorted(
                {
                    int(row["prompt_contract"]["hidden_turns_revealed_by_success"])
                    for row in rows
                    if row["structural_cell"] == cell
                }
            )
            for cell in sorted({row["structural_cell"] for row in rows})
        },
    }


def write_pool(output_dir: Path) -> dict[str, Any]:
    rows_by_split = build_rows()
    output_dir.mkdir(parents=True, exist_ok=False)
    splits: dict[str, Any] = {}
    for split, rows in rows_by_split.items():
        path = output_dir / f"{split}.jsonl"
        path.write_bytes(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
        )
        splits[split] = {
            "file": path.name,
            "task_count": len(rows),
            "rollout_seeds": ROLLOUT_SEEDS[split],
            "trajectory_count_per_family": len(rows) * len(ROLLOUT_SEEDS[split]),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "task_signatures": [row["task_signature"] for row in rows],
            "structural_stats": _stats(rows),
        }
    manifest = {
        "protocol": "RIST-C0-v4.0-FRESH-POOL-v1",
        "generator_version": GENERATOR_VERSION,
        "pool_seed": POOL_SEED,
        "capacity_canary_seed": CANARY_TASK_SEED,
        "retired_parent_protocols": ["v2.1", "v2.2", "v2.3", "v3.1"],
        "v3_outcomes_used_for_task_selection": False,
        "prior_tasks_or_seeds_used": False,
        "prior_outcomes_used": False,
        "qualification_opened": False,
        "heldout_opened": False,
        "bfcl_opened": False,
        "model_accessed": False,
        "gpu_used": False,
        "training_performed": False,
        "cell_specs": CELL_SPECS,
        "splits": splits,
    }
    (output_dir / "manifest.json").write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return manifest


def _signature(task: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in task.items()
        if key not in {"id", "task_signature"}
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_pool(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

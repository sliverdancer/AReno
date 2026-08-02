"""Generate the deterministic RIST-v2 structural candidate pool."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

GENERATOR_VERSION = "rist-workflow-v2-candidate"
DEFAULT_SEED = 20260802
SPLITS = ("calibration", "qualification", "heldout")
REPLICATES_PER_CELL = 4
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
CELL_SPECS = (
    {"cell": "c00", "tool_decoys": 0, "argument_decoys": 0, "dependency_depth": 0, "selector": "direct"},
    {"cell": "c01", "tool_decoys": 1, "argument_decoys": 0, "dependency_depth": 0, "selector": "direct"},
    {"cell": "c02", "tool_decoys": 0, "argument_decoys": 1, "dependency_depth": 2, "selector": "relational"},
    {"cell": "c03", "tool_decoys": 1, "argument_decoys": 1, "dependency_depth": 2, "selector": "relational"},
    {"cell": "c04", "tool_decoys": 2, "argument_decoys": 3, "dependency_depth": 3, "selector": "direct"},
    {"cell": "c05", "tool_decoys": 2, "argument_decoys": 3, "dependency_depth": 3, "selector": "relational"},
    {"cell": "c06", "tool_decoys": 3, "argument_decoys": 7, "dependency_depth": 3, "selector": "direct"},
    {"cell": "c07", "tool_decoys": 3, "argument_decoys": 7, "dependency_depth": 3, "selector": "relational"},
)


def generate_splits(seed: int = DEFAULT_SEED) -> dict[str, list[dict[str, Any]]]:
    """Return balanced, signature-disjoint structural tasks."""

    splits: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        rows = [
            _build_task(split, spec, replicate, seed)
            for spec in CELL_SPECS
            for replicate in range(REPLICATES_PER_CELL)
        ]
        rows.sort(key=lambda row: str(row["task_signature"]))
        for index, row in enumerate(rows):
            row["id"] = f"{split}-{index:03d}"
        splits[split] = rows
    validate_splits(splits)
    return splits


def validate_splits(splits: dict[str, list[dict[str, Any]]]) -> None:
    """Reject imbalance, leakage, oracle ambiguity, or resolution labels."""

    if set(splits) != set(SPLITS):
        raise ValueError("candidate pool must contain the three frozen splits")
    signatures: set[str] = set()
    expected_cells = {str(spec["cell"]) for spec in CELL_SPECS}
    for split, rows in splits.items():
        if len(rows) != len(CELL_SPECS) * REPLICATES_PER_CELL:
            raise ValueError(f"{split} has the wrong task count")
        counts = {cell: 0 for cell in expected_cells}
        for row in rows:
            validate_task(row)
            if row["split"] != split:
                raise ValueError("split label mismatch")
            counts[str(row["structural_cell"])] += 1
            signature = str(row["task_signature"])
            if signature in signatures:
                raise ValueError("task signatures must be cross-split disjoint")
            signatures.add(signature)
        if set(counts.values()) != {REPLICATES_PER_CELL}:
            raise ValueError("structural cells must be balanced")


def validate_task(task: dict[str, Any]) -> None:
    """Validate one v2 candidate task and its unique strict oracle."""

    forbidden = {"reward_resolution_stratum", "mixed_group_probability_g8"}
    if forbidden.intersection(task):
        raise ValueError("D2 tasks may not contain model-free resolution labels")
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
        matches = [record for record in records if record["label"] == turn["target_label"]]
        if len(matches) != 1:
            raise ValueError("target label must identify one unique candidate")
        if action["arguments"] != {"code": matches[0]["code"]}:
            raise ValueError("oracle argument mismatch")
    expected_signature = _signature(task)
    if task.get("task_signature") != expected_signature:
        raise ValueError("task signature mismatch")


def write_splits(output_dir: Path, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Write the candidate pool and a content-addressed manifest."""

    splits = generate_splits(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, rows in splits.items():
        path = output_dir / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        files[split] = {
            "file": path.name,
            "count": len(rows),
            "sha256": _sha256(path),
            "task_signatures": [row["task_signature"] for row in rows],
        }
    manifest = {
        "schema_version": 1,
        "protocol": "RIST-D2-v2.0",
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "replicates_per_cell": REPLICATES_PER_CELL,
        "cell_specs": CELL_SPECS,
        "files": files,
        "model_accessed": False,
        "gpu_used": False,
        "training_performed": False,
        "heldout_content_used_by_selection": False,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _build_task(split: str, spec: dict[str, Any], replicate: int, seed: int) -> dict[str, Any]:
    nonce = hashlib.sha256(
        f"{seed}:{split}:{spec['cell']}:{replicate}".encode()
    ).hexdigest()[:16]
    dependency_start = 4 - int(spec["dependency_depth"])
    turns = []
    oracle = []
    visible_turns = []
    for index, expected_tool in enumerate(EXPECTED_TOOLS):
        records = [
            {
                "label": _token(nonce, index, candidate, "label"),
                "code": _token(nonce, index, candidate, "code"),
            }
            for candidate in range(int(spec["argument_decoys"]) + 1)
        ]
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
        "factors": {key: value for key, value in spec.items() if key != "cell"},
        "prompt_contract": {
            "instruction": "Complete four ordered registry actions. At each turn choose the expected tool and pass the code whose label matches the target label.",
            "initial_visible_turns": visible_turns,
            "hidden_turns_revealed_by_success": int(spec["dependency_depth"]),
        },
        "turns": turns,
        "oracle_actions": oracle,
    }
    task["task_signature"] = _signature(task)
    return task


def _signature(task: dict[str, Any]) -> str:
    payload = {key: value for key, value in task.items() if key not in {"id", "task_signature"}}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _token(nonce: str, turn: int, candidate: int, kind: str) -> str:
    return hashlib.sha256(
        f"{nonce}:{turn}:{candidate}:{kind}".encode()
    ).hexdigest()[:10]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

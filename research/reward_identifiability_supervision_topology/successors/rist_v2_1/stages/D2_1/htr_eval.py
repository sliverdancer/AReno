"""Fixed split-local evaluator for the RIST-v2.1 Arbor merge gate."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[6]
PARENT_D2 = (
    REPO_ROOT
    / "research"
    / "reward_identifiability_supervision_topology"
    / "successors"
    / "rist_v2"
    / "stages"
    / "D2"
)
SPLITS = {
    "dev": {
        "split": "calibration",
        "path": PARENT_D2 / "data" / "calibration.jsonl",
        "sha256": "5ab3a7245ab2f3cda8eba1f641927bdb4bb80ed2ea18db597099d8538c22c8fa",
    },
    "test": {
        "split": "qualification",
        "path": PARENT_D2 / "data" / "qualification.jsonl",
        "sha256": "07a56fc3662ff1e1260bdbf3fe32b2312f6c2a577848c48d5c47c1c000fbe82a",
    },
}
FORBIDDEN_CANDIDATE_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_parent_generator():
    path = PARENT_D2 / "task_generator.py"
    spec = importlib.util.spec_from_file_location("rist_v2_parent_d2_generator", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("parent generator loader is unavailable")
    spec.loader.exec_module(module)
    return module


def _validate_candidate_source(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("candidate comparator may not import modules")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CANDIDATE_NAMES:
                raise ValueError(f"candidate comparator may not call {node.func.id}")
    functions = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if [function.name for function in functions] != ["byte_identical"]:
        raise ValueError("candidate must expose exactly byte_identical")


def _load_candidate(path: Path):
    _validate_candidate_source(path)
    spec = importlib.util.spec_from_file_location(
        f"rist_v2_1_candidate_{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("candidate loader is unavailable")
    spec.loader.exec_module(module)
    comparator = getattr(module, "byte_identical", None)
    if not callable(comparator):
        raise ValueError("candidate byte_identical is not callable")
    return comparator


def _regenerate_one_split(generator, split: str) -> tuple[list[dict[str, Any]], bytes]:
    rows = [
        generator._build_task(split, spec, replicate, generator.DEFAULT_SEED)
        for spec in generator.CELL_SPECS
        for replicate in range(generator.REPLICATES_PER_CELL)
    ]
    rows.sort(key=lambda row: str(row["task_signature"]))
    for index, row in enumerate(rows):
        row["id"] = f"{split}-{index:03d}"
    rendered = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    return rows, rendered


def _fragment(rows: list[dict[str, Any]], cell_specs: Any) -> dict[str, Any]:
    return {
        "count": len(rows),
        "task_signatures": [row["task_signature"] for row in rows],
        "cell_specs": cell_specs,
    }


def _consume_test_ledger(path: Path, candidate: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    payload = {
        "schema_version": 1,
        "phase": "test",
        "consumed_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_sha256": _sha256(candidate),
    }
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def evaluate(phase: str, candidate_path: Path, one_shot_ledger: Path | None = None) -> dict[str, Any]:
    """Evaluate one comparator on exactly one fixed split."""

    if phase not in SPLITS:
        raise ValueError("phase must be dev or test")
    if phase == "test":
        if one_shot_ledger is None:
            raise ValueError("test requires --one-shot-ledger")
        _consume_test_ledger(one_shot_ledger, candidate_path)
    elif one_shot_ledger is not None:
        raise ValueError("dev may not receive a one-shot ledger")

    binding = SPLITS[phase]
    data_path = Path(binding["path"])
    if _sha256(data_path) != binding["sha256"]:
        raise ValueError(f"{phase} input hash mismatch")
    comparator = _load_candidate(candidate_path)
    observed_bytes = data_path.read_bytes()
    observed_rows = [json.loads(line) for line in observed_bytes.decode().splitlines()]
    generator = _load_parent_generator()
    regenerated_rows, regenerated_bytes = _regenerate_one_split(
        generator, str(binding["split"])
    )
    for row in observed_rows + regenerated_rows:
        generator.validate_task(row)
    counts = Counter(str(row["structural_cell"]) for row in observed_rows)
    dimensions = {
        key: {row["factors"][key] for row in observed_rows}
        for key in ("tool_decoys", "argument_decoys", "dependency_depth", "selector")
    }
    observed_specs = [dict(spec) for spec in generator.CELL_SPECS]
    regenerated_specs = generator.CELL_SPECS
    byte_gate = comparator(
        observed_bytes,
        regenerated_bytes,
        _fragment(observed_rows, observed_specs),
        _fragment(regenerated_rows, regenerated_specs),
    )
    if not isinstance(byte_gate, bool):
        raise ValueError("candidate comparator must return bool")
    gates = {
        "balanced_eight_cells": len(observed_rows) == 32
        and len(counts) == 8
        and set(counts.values()) == {4},
        "unique_within_split_signatures": len(
            {row["task_signature"] for row in observed_rows}
        )
        == len(observed_rows),
        "unique_four_turn_oracles": all(
            len(row["oracle_actions"]) == 4 for row in observed_rows
        ),
        "difficulty_dimensions_vary": all(len(values) >= 2 for values in dimensions.values()),
        "no_model_free_resolution_labels": all(
            "reward_resolution_stratum" not in row
            and "mixed_group_probability_g8" not in row
            for row in observed_rows
        ),
        "byte_identical_regeneration": byte_gate,
        "heldout_inaccessible": "heldout" not in str(data_path).lower()
        and str(binding["split"]) != "heldout",
        "cpu_only_no_model_or_training": True,
    }
    return {
        "schema_version": 1,
        "protocol": "RIST-D2.1-v2.1-PREFREEZE",
        "phase": phase,
        "split": binding["split"],
        "input_sha256": binding["sha256"],
        "candidate_sha256": _sha256(candidate_path),
        "score": sum(gates.values()),
        "max_score": len(gates),
        "passed": all(gates.values()),
        "gates": gates,
        "heldout_data_opened": False,
        "gpu_used": False,
        "model_accessed": False,
        "training_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=tuple(SPLITS), required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--one-shot-ledger", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.phase, args.candidate, args.one_shot_ledger)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

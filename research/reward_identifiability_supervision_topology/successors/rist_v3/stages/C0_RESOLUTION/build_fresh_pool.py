"""Build the outcome-unopened, task- and seed-disjoint RIST v3 C0 pool."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[6]
RESEARCH_ROOT = REPO_ROOT / "research/reward_identifiability_supervision_topology"
GENERATOR_PATH = (
    RESEARCH_ROOT / "successors/rist_v2/stages/D2/task_generator.py"
)
POOL_SEED = 202608050301
CANARY_TASK_SEED = 202608050302
ROLLOUT_SEEDS = {
    "capacity_canary": list(range(31001, 31009)),
    "calibration": list(range(32001, 32033)),
    "qualification": list(range(33001, 33033)),
}


def _load_generator():
    spec = importlib.util.spec_from_file_location("rist_v3_task_generator", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("RIST task generator is unavailable")
    spec.loader.exec_module(module)
    return module


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
    v3_root = RESEARCH_ROOT / "successors/rist_v3"
    for path in RESEARCH_ROOT.rglob("*.json"):
        if path.is_relative_to(v3_root):
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


def build_rows() -> dict[str, list[dict[str, Any]]]:
    generator = _load_generator()
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("calibration", "qualification"):
        rows = [
            generator._build_task(f"rist_v3_{split}", spec, replicate, POOL_SEED)
            for spec in generator.CELL_SPECS
            for replicate in range(generator.REPLICATES_PER_CELL)
        ]
        rows.sort(key=lambda row: str(row["task_signature"]))
        for index, row in enumerate(rows):
            row["id"] = f"{split}-v3-{index:03d}"
            generator.validate_task(row)
        counts = Counter(str(row["structural_cell"]) for row in rows)
        if len(rows) != 32 or len(counts) != 8 or set(counts.values()) != {4}:
            raise ValueError(f"{split} is not balanced across eight cells")
        rows_by_split[split] = rows
    hardest = next(spec for spec in generator.CELL_SPECS if spec["cell"] == "c07")
    canary = generator._build_task(
        "rist_v3_capacity_canary", hardest, 0, CANARY_TASK_SEED
    )
    canary["id"] = "capacity-canary-v3-c07"
    generator.validate_task(canary)
    rows_by_split["capacity_canary"] = [canary]

    signatures = [
        str(row["task_signature"])
        for rows in rows_by_split.values()
        for row in rows
    ]
    prior_signatures, prior_seeds = _prior_values()
    if len(signatures) != len(set(signatures)):
        raise ValueError("v3 task signatures overlap internally")
    if set(signatures) & prior_signatures:
        raise ValueError("v3 task signatures overlap a prior lineage")
    all_seeds = {POOL_SEED, CANARY_TASK_SEED}.union(
        *(set(values) for values in ROLLOUT_SEEDS.values())
    )
    if len(all_seeds) != 2 + sum(len(values) for values in ROLLOUT_SEEDS.values()):
        raise ValueError("v3 seeds overlap internally")
    if all_seeds & prior_seeds:
        raise ValueError("v3 seeds overlap a prior lineage")
    return rows_by_split


def write_pool(output_dir: Path) -> dict[str, Any]:
    rows_by_split = build_rows()
    output_dir.mkdir(parents=True, exist_ok=False)
    splits: dict[str, Any] = {}
    for split, rows in rows_by_split.items():
        path = output_dir / f"{split}.jsonl"
        path.write_bytes("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8"))
        splits[split] = {
            "file": path.name,
            "task_count": len(rows),
            "rollout_seeds": ROLLOUT_SEEDS[split],
            "trajectory_count_per_family": len(rows) * len(ROLLOUT_SEEDS[split]),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "task_signatures": [row["task_signature"] for row in rows],
        }
    manifest = {
        "protocol": "RIST-C0-v3.0-FRESH-POOL-v1",
        "pool_seed": POOL_SEED,
        "capacity_canary_seed": CANARY_TASK_SEED,
        "retired_parent_protocols": ["v2.1", "v2.2", "v2.3"],
        "prior_tasks_or_seeds_used": False,
        "prior_outcomes_used": False,
        "qualification_opened": False,
        "heldout_opened": False,
        "bfcl_opened": False,
        "model_accessed": False,
        "gpu_used": False,
        "training_performed": False,
        "splits": splits,
    }
    (output_dir / "manifest.json").write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_pool(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

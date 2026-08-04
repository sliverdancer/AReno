"""Build the fresh, outcome-unopened C0 v2.3 pool after v2.2 protocol failure."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[6]
GENERATOR_PATH = (
    REPO_ROOT
    / "research/reward_identifiability_supervision_topology/successors/rist_v2"
    / "stages/D2/task_generator.py"
)
POOL_SEED = 2026080417
CANARY_SEED = 2026080418
ROLLOUT_SEEDS = {
    "calibration": list(range(16001, 16033)),
    "qualification": list(range(17001, 17033)),
    "capacity_canary": list(range(18001, 18009)),
}


def _load_generator():
    spec = importlib.util.spec_from_file_location("rist_c0_v2_3_generator", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("RIST task generator is unavailable")
    spec.loader.exec_module(module)
    return module


def _prior_signatures() -> set[str]:
    paths = [
        REPO_ROOT
        / "research/reward_identifiability_supervision_topology/successors/rist_v2/stages/D2/data/manifest.json",
        REPO_ROOT
        / "research/reward_identifiability_supervision_topology/successors/rist_v2_1/stages/D3/data/manifest.json",
        REPO_ROOT
        / "research/reward_identifiability_supervision_topology/successors/rist_v2_1/stages/D4_EVAL/data/manifest.json",
        REPO_ROOT
        / "research/reward_identifiability_supervision_topology/successors/rist_v2_1/stages/C0_RESOLUTION_V2_2/data/manifest.json",
    ]
    signatures = set()
    for path in paths:
        manifest = json.loads(path.read_text())
        if "files" in manifest:
            rows = manifest["files"].values()
        elif "splits" in manifest:
            rows = manifest["splits"].values()
        else:
            rows = [manifest]
        for row in rows:
            signatures.update(str(value) for value in row.get("task_signatures", []))
    return signatures


def build_rows() -> dict[str, list[dict[str, Any]]]:
    generator = _load_generator()
    rows_by_split = {}
    for split in ("calibration", "qualification"):
        source_split = f"c0v2_3_{split}"
        rows = [
            generator._build_task(source_split, spec, replicate, POOL_SEED)
            for spec in generator.CELL_SPECS
            for replicate in range(generator.REPLICATES_PER_CELL)
        ]
        rows.sort(key=lambda row: str(row["task_signature"]))
        for index, row in enumerate(rows):
            row["id"] = f"{split}-v2_3-{index:03d}"
            generator.validate_task(row)
        counts = Counter(str(row["structural_cell"]) for row in rows)
        if len(rows) != 32 or len(counts) != 8 or set(counts.values()) != {4}:
            raise ValueError(f"{split} is not balanced across eight structural cells")
        rows_by_split[split] = rows
    hardest = next(spec for spec in generator.CELL_SPECS if spec["cell"] == "c07")
    canary = generator._build_task(
        "c0v2_3_capacity_canary", hardest, 0, CANARY_SEED
    )
    canary["id"] = "capacity-canary-v2_3-c07"
    generator.validate_task(canary)
    rows_by_split["capacity_canary"] = [canary]
    signatures = [
        str(row["task_signature"])
        for rows in rows_by_split.values()
        for row in rows
    ]
    if len(signatures) != len(set(signatures)):
        raise ValueError("C0 v2.3 task signatures overlap internally")
    if set(signatures) & _prior_signatures():
        raise ValueError("C0 v2.3 task signatures overlap a retired or downstream pool")
    return rows_by_split


def write_pool(output_dir: Path) -> dict[str, Any]:
    rows_by_split = build_rows()
    output_dir.mkdir(parents=True, exist_ok=False)
    splits = {}
    for split, rows in rows_by_split.items():
        path = output_dir / f"{split}.jsonl"
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
        splits[split] = {
            "file": path.name,
            "task_count": len(rows),
            "rollout_seeds": ROLLOUT_SEEDS[split],
            "trajectory_count": len(rows) * len(ROLLOUT_SEEDS[split]),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "task_signatures": [row["task_signature"] for row in rows],
        }
    manifest = {
        "protocol": "RIST-C0-v2.3-FRESH-POOL",
        "pool_seed": POOL_SEED,
        "capacity_canary_seed": CANARY_SEED,
        "previous_c0_task_pools_retired": ["v2.1", "v2.2"],
        "prior_outcomes_used": False,
        "heldout_content_opened": False,
        "model_accessed": False,
        "gpu_used": False,
        "training_performed": False,
        "splits": splits,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = write_pool(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build fresh RIST-v2.1 development-curve and confirmatory splits."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[6]
PARENT_GENERATOR = (
    REPO_ROOT
    / "research/reward_identifiability_supervision_topology/successors/rist_v2"
    / "stages/D2/task_generator.py"
)
SEED = 20260802
SPLITS = {"dev_curve": 2, "confirmatory": 4}
ROLLOUT_SEEDS = {
    "dev_curve": [8101, 8202],
    "confirmatory": [9101, 9202, 9303, 9404],
}


def _load_generator():
    spec = importlib.util.spec_from_file_location("rist_v2_d4_generator", PARENT_GENERATOR)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("parent generator loader unavailable")
    spec.loader.exec_module(module)
    return module


def build_rows(split: str, seed: int = SEED) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError("split must be dev_curve or confirmatory")
    generator = _load_generator()
    rows = [
        generator._build_task(split, spec, replicate, seed)
        for spec in generator.CELL_SPECS
        for replicate in range(SPLITS[split])
    ]
    rows.sort(key=lambda row: str(row["task_signature"]))
    for index, row in enumerate(rows):
        row["id"] = f"{split}-{index:03d}"
        generator.validate_task(row)
    counts = Counter(str(row["structural_cell"]) for row in rows)
    if len(counts) != 8 or set(counts.values()) != {SPLITS[split]}:
        raise ValueError("evaluation split is not balanced across eight cells")
    return rows


def write_splits(output_dir: Path, seed: int = SEED) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "protocol": "RIST-D4-EVAL-v2.1",
        "schema_version": 1,
        "seed": seed,
        "gpu_used": False,
        "model_accessed": False,
        "training_performed": False,
        "parent_heldout_opened": False,
        "sampling": {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 128,
            "retry_count": 0
        },
        "development_checkpoint_steps": [25, 50, 75, 100],
        "confirmatory_checkpoint_step": 100,
        "splits": {},
    }
    all_signatures: set[str] = set()
    for split in SPLITS:
        rows = build_rows(split, seed)
        signatures = {str(row["task_signature"]) for row in rows}
        if all_signatures & signatures:
            raise ValueError("evaluation split signatures overlap")
        all_signatures |= signatures
        encoded = "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in rows
        ).encode("utf-8")
        materialized = split == "dev_curve"
        if materialized:
            (output_dir / f"{split}.jsonl").write_bytes(encoded)
        manifest["splits"][split] = {
            "count": len(rows),
            "tasks_per_cell": SPLITS[split],
            "rollout_seeds": ROLLOUT_SEEDS[split],
            "trajectory_count": len(rows) * len(ROLLOUT_SEEDS[split]),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "task_signatures": sorted(signatures),
            "materialized": materialized,
        }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_splits(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

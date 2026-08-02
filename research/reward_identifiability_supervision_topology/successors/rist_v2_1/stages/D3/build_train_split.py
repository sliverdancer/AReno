"""Build the isolated RIST-v2.1 train split without opening other split files."""

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
    / "research"
    / "reward_identifiability_supervision_topology"
    / "successors"
    / "rist_v2"
    / "stages"
    / "D2"
    / "task_generator.py"
)
TRAIN_SPLIT_SEED = 20260802


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "rist_v2_d3_parent_generator", PARENT_GENERATOR
    )
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("parent generator loader unavailable")
    spec.loader.exec_module(module)
    return module


def build_rows(seed: int = TRAIN_SPLIT_SEED) -> list[dict[str, Any]]:
    """Return the balanced train-only task rows."""

    generator = _load_generator()
    rows = [
        generator._build_task("train", spec, replicate, seed)
        for spec in generator.CELL_SPECS
        for replicate in range(generator.REPLICATES_PER_CELL)
    ]
    rows.sort(key=lambda row: str(row["task_signature"]))
    for index, row in enumerate(rows):
        row["id"] = f"train-{index:03d}"
        generator.validate_task(row)
    counts = Counter(str(row["structural_cell"]) for row in rows)
    if len(rows) != 32 or len(counts) != 8 or set(counts.values()) != {4}:
        raise ValueError("train split must contain four tasks for each of eight cells")
    return rows


def write_train(output_dir: Path, seed: int = TRAIN_SPLIT_SEED) -> dict[str, Any]:
    """Write train JSONL plus a train-only manifest."""

    rows = build_rows(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "train.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "protocol": "RIST-D3-v2.1",
        "generator_version": rows[0]["generator_version"],
        "seed": seed,
        "split": "train",
        "count": len(rows),
        "sha256": digest,
        "task_signatures": [row["task_signature"] for row in rows],
        "other_split_content_opened": False,
        "heldout_data_opened": False,
        "gpu_used": False,
        "model_accessed": False,
        "training_performed": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_train(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate the frozen RIST-v2 structural candidate pool on CPU."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

STAGE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(STAGE_DIR))
import task_generator  # noqa: E402


def evaluate(data_dir: Path) -> dict[str, Any]:
    """Return the D2 structural and reproducibility gates."""

    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    loaded = {}
    for split in ("calibration", "qualification"):
        path = data_dir / manifest["files"][split]["file"]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for row in rows:
            task_generator.validate_task(row)
        loaded[split] = rows
    signatures = [
        signature
        for record in manifest["files"].values()
        for signature in record["task_signatures"]
    ]
    counts_ok = all(
        len(rows) == 32
        and set(Counter(row["structural_cell"] for row in rows).values()) == {4}
        for rows in loaded.values()
    )
    dimensions = {
        key: {
            row["factors"][key]
            for rows in loaded.values()
            for row in rows
        }
        for key in ("tool_decoys", "argument_decoys", "dependency_depth", "selector")
    }
    with tempfile.TemporaryDirectory(prefix="rist-v2-d2-") as temporary:
        regenerated = Path(temporary)
        regenerated_manifest = task_generator.write_splits(
            regenerated, seed=int(manifest["seed"])
        )
        reproducible = regenerated_manifest == manifest and all(
            (regenerated / record["file"]).read_bytes()
            == (data_dir / record["file"]).read_bytes()
            for split, record in manifest["files"].items()
            if split != "heldout"
        )
    gates = {
        "balanced_eight_cells": counts_ok,
        "unique_cross_split_signatures": len(signatures) == len(set(signatures)),
        "unique_four_turn_oracles": all(
            len(row["oracle_actions"]) == 4 for rows in loaded.values() for row in rows
        ),
        "difficulty_dimensions_vary": all(len(values) >= 2 for values in dimensions.values()),
        "no_model_free_resolution_labels": all(
            "reward_resolution_stratum" not in row
            for rows in loaded.values()
            for row in rows
        ),
        "byte_identical_regeneration": reproducible,
        "heldout_content_not_opened_by_evaluator": True,
        "cpu_only_no_model_or_training": manifest["gpu_used"] is False
        and manifest["model_accessed"] is False
        and manifest["training_performed"] is False,
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "protocol": "RIST-D2-v2.0",
        "stage": "D2",
        "stage_status": "PASS" if passed else "KILL",
        "decision": (
            "PASS_D2_CPU_POOL_TO_GPU_AUTHORIZATION_REQUEST"
            if passed
            else "KILL_RIST_V2_D2_STRUCTURAL_POOL"
        ),
        "gates": gates,
        "split_counts": {split: record["count"] for split, record in manifest["files"].items()},
        "heldout_data_opened": False,
        "gpu_used": False,
        "model_accessed": False,
        "training_performed": False,
        "main_conference_route_open": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.data_dir)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["stage_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

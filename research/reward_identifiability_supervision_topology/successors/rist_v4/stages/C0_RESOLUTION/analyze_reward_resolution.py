"""Analyze RIST C0 v4 calibration reward resolution after finalization.

This analyzer is intentionally split from the collection finalizer: the finalizer
only validates evidence integrity, while this file inspects strict-success
outcomes to decide whether calibration can admit a sealed qualification run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PROTOCOL = "RIST-C0-v4.0-REWARD-RESOLUTION-ANALYSIS-v1"
ADMISSION_PROTOCOL = "RIST-C0-v4.0-CALIBRATION-ADMISSION-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def classify_cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task_id"])].append(int(row["strict_success"]))
    if len(by_task) != 4:
        raise ValueError("each v4 calibration cell must contain exactly four task groups")
    group_summaries = []
    mixed = 0
    successes = 0
    total = 0
    for task_id in sorted(by_task):
        values = by_task[task_id]
        if len(values) != 32:
            raise ValueError("each v4 calibration task group must contain 32 rollout seeds")
        success_count = sum(values)
        failures = len(values) - success_count
        is_mixed = 0 < success_count < len(values)
        mixed += int(is_mixed)
        successes += success_count
        total += len(values)
        group_summaries.append(
            {
                "task_id": task_id,
                "success_count": success_count,
                "failure_count": failures,
                "mixed_reward_group": is_mixed,
            }
        )
    if mixed >= 2:
        label = "high"
    elif mixed == 0:
        label = "low"
    else:
        label = "ambiguous"
    return {
        "resolution_label": label,
        "mixed_group_count": mixed,
        "task_group_count": len(by_task),
        "success_count": successes,
        "trajectory_count": total,
        "strict_success_rate": successes / total,
        "groups": group_summaries,
    }


def analyze(
    *,
    manifest_path: Path,
    final_path: Path,
    qwen_trajectories: Path,
    gemma_trajectories: Path,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    final = _read_json(final_path)
    if manifest.get("protocol") != "RIST-C0-v4.0-STAGE-MANIFEST-v1":
        raise ValueError("unexpected v4 stage manifest")
    if manifest.get("split") != "calibration":
        raise PermissionError("reward-resolution analysis is calibration-only")
    if manifest.get("qualification_permitted") is not False:
        raise PermissionError("qualification must still be sealed during calibration analysis")
    if final.get("protocol") != "RIST-C0-v4.0-SCIENTIFIC-COLLECTION-FINAL-v1":
        raise ValueError("unexpected v4 collection final")
    if final.get("split") != "calibration" or final.get("passed") is not True:
        raise PermissionError("calibration collection must pass before outcome analysis")
    if final.get("outcomes_inspected") is not False:
        raise PermissionError("collection finalizer must remain outcome-free")

    rows_by_family = {
        "qwen3": _read_jsonl(qwen_trajectories),
        "gemma4": _read_jsonl(gemma_trajectories),
    }
    cells: dict[str, dict[str, Any]] = {}
    for family, rows in rows_by_family.items():
        by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_cell[str(row["structural_cell"])].append(row)
        if set(by_cell) != {f"c{index:02d}" for index in range(8)}:
            raise ValueError(f"{family} must contain exactly the eight frozen cells")
        cells[family] = {
            cell: classify_cell(by_cell[cell])
            for cell in sorted(by_cell)
        }

    common_low = [
        cell for cell in sorted(cells["qwen3"])
        if cells["qwen3"][cell]["resolution_label"] == "low"
        and cells["gemma4"][cell]["resolution_label"] == "low"
    ]
    common_high = [
        cell for cell in sorted(cells["qwen3"])
        if cells["qwen3"][cell]["resolution_label"] == "high"
        and cells["gemma4"][cell]["resolution_label"] == "high"
    ]
    passed = len(common_low) >= 2 and len(common_high) >= 2
    frozen_map = {
        **{cell: "low" for cell in common_low},
        **{cell: "high" for cell in common_high},
    }
    decision = (
        "PASS_CALIBRATION_TO_QUALIFICATION"
        if passed
        else "KILL_C0_V4_CALIBRATION_NO_COMMON_RESOLUTION_CONTRAST"
    )
    return {
        "protocol": PROTOCOL,
        "decision": decision,
        "passed": passed,
        "manifest_sha256": _sha256(manifest_path),
        "pool_manifest_sha256": manifest["pool_manifest_sha256"],
        "calibration_result_sha256": _sha256(final_path),
        "trajectory_sha256": {
            "qwen3": _sha256(qwen_trajectories),
            "gemma4": _sha256(gemma_trajectories),
        },
        "thresholds": {
            "groups_per_cell": 4,
            "rollout_seeds_per_group": 32,
            "high": "mixed_group_count >= 2",
            "low": "mixed_group_count == 0",
            "ambiguous": "mixed_group_count == 1",
        },
        "common_low_cells": common_low,
        "common_high_cells": common_high,
        "frozen_whole_cell_map": frozen_map,
        "cells": cells,
        "qualification_accessed": False,
        "heldout_accessed": False,
        "bfcl_accessed": False,
        "training_performed": False,
    }


def admission_from_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    if analysis.get("protocol") != PROTOCOL:
        raise ValueError("unexpected v4 reward-resolution analysis")
    return {
        "protocol": ADMISSION_PROTOCOL,
        "decision": analysis["decision"],
        "passed": analysis["passed"],
        "pool_manifest_sha256": analysis["pool_manifest_sha256"],
        "calibration_result_sha256": analysis["calibration_result_sha256"],
        "frozen_whole_cell_map": analysis["frozen_whole_cell_map"],
        "qualification_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--final", type=Path, required=True)
    parser.add_argument("--qwen-trajectories", type=Path, required=True)
    parser.add_argument("--gemma-trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admission-output", type=Path)
    args = parser.parse_args()
    result = analyze(
        manifest_path=args.manifest,
        final_path=args.final,
        qwen_trajectories=args.qwen_trajectories,
        gemma_trajectories=args.gemma_trajectories,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.admission_output is not None:
        args.admission_output.write_text(
            json.dumps(admission_from_analysis(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

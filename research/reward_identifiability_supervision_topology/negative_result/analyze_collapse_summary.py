"""Build reusable evidence tables for the RIST reward-resolution collapse result."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _v3_rows(value: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, cells in sorted(value["families"].items()):
        for cell, summary in sorted(cells.items()):
            tasks = summary["tasks"]
            success_count = sum(int(task["k"]) for task in tasks)
            trajectory_count = sum(int(task["n"]) for task in tasks)
            rows.append(
                {
                    "lineage": "RIST-C0-v3.1",
                    "family": family,
                    "cell": cell,
                    "resolution_label": summary["label"],
                    "mixed_group_count": int(summary["mixed_task_count"]),
                    "collapsed_group_count": int(summary["collapsed_task_count"]),
                    "success_count": success_count,
                    "trajectory_count": trajectory_count,
                    "strict_success_rate": success_count / trajectory_count,
                }
            )
    return rows


def _v4_rows(value: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, cells in sorted(value["cells"].items()):
        for cell, summary in sorted(cells.items()):
            rows.append(
                {
                    "lineage": "RIST-C0-v4.0",
                    "family": family,
                    "cell": cell,
                    "resolution_label": summary["resolution_label"],
                    "mixed_group_count": int(summary["mixed_group_count"]),
                    "collapsed_group_count": int(summary["task_group_count"])
                    - int(summary["mixed_group_count"]),
                    "success_count": int(summary["success_count"]),
                    "trajectory_count": int(summary["trajectory_count"]),
                    "strict_success_rate": float(summary["strict_success_rate"]),
                }
            )
    return rows


def _lineage_summary(
    lineage: str,
    rows: list[dict[str, Any]],
    common_low: list[str],
    common_high: list[str],
    decision: str,
) -> dict[str, Any]:
    families = sorted({row["family"] for row in rows})
    labels = {
        family: {
            row["cell"]: row["resolution_label"]
            for row in rows
            if row["family"] == family
        }
        for family in families
    }
    common_ambiguous = [
        cell
        for cell in sorted({row["cell"] for row in rows})
        if all(labels[family].get(cell) == "ambiguous" for family in families)
    ]
    return {
        "lineage": lineage,
        "decision": decision,
        "families": families,
        "cell_count": len({row["cell"] for row in rows}),
        "trajectory_count": sum(int(row["trajectory_count"]) for row in rows),
        "common_low_cells": common_low,
        "common_high_cells": common_high,
        "common_ambiguous_cells": common_ambiguous,
        "go_gate_passed": len(common_low) >= 2 and len(common_high) >= 2,
    }


def build(v3_path: Path, v4_path: Path) -> dict[str, Any]:
    v3 = _read_json(v3_path)
    v4 = _read_json(v4_path)
    rows = [*_v3_rows(v3), *_v4_rows(v4)]
    return {
        "protocol": "RIST-NEGATIVE-RESULT-COLLAPSE-SUMMARY-v1",
        "source_paths": {
            "v3_resolution_analysis": str(v3_path),
            "v4_reward_resolution_analysis": str(v4_path),
        },
        "lineages": [
            _lineage_summary(
                "RIST-C0-v3.1",
                [row for row in rows if row["lineage"] == "RIST-C0-v3.1"],
                list(v3["common_low_cells"]),
                list(v3["common_high_cells"]),
                "KILL_C0_V3_1_CALIBRATION_NO_COMMON_HIGH_CELLS",
            ),
            _lineage_summary(
                "RIST-C0-v4.0",
                [row for row in rows if row["lineage"] == "RIST-C0-v4.0"],
                list(v4["common_low_cells"]),
                list(v4["common_high_cells"]),
                str(v4["decision"]),
            ),
        ],
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3-analysis", type=Path, required=True)
    parser.add_argument("--v4-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.v3_analysis, args.v4_analysis)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "EVIDENCE_SUMMARY.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "EVIDENCE_SUMMARY.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "lineage",
                "family",
                "cell",
                "resolution_label",
                "mixed_group_count",
                "collapsed_group_count",
                "success_count",
                "trajectory_count",
                "strict_success_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(result["rows"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

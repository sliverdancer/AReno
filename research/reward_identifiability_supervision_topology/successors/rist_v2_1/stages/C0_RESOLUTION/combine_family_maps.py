"""Intersect transported resolution cells across two checkpoint families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def combine_maps(family_results: list[dict[str, Any]]) -> dict[str, Any]:
    if len(family_results) != 2 or len({row["checkpoint"] for row in family_results}) != 2:
        raise ValueError("cross-family resolution requires exactly two checkpoints")
    if not all(row.get("passed") is True for row in family_results):
        return {
            "protocol": "RIST-C0-v2.1",
            "passed": False,
            "decision": "KILL_C0_FAMILY_CALIBRATION",
            "common_resolution_map": {},
        }
    maps = [row["resolution_map"] for row in family_results]
    common = {
        cell: maps[0][cell]
        for cell in sorted(set(maps[0]) & set(maps[1]))
        if maps[0][cell] == maps[1][cell]
    }
    counts = {band: sum(value == band for value in common.values()) for band in ("low", "high")}
    passed = counts["low"] >= 2 and counts["high"] >= 2
    return {
        "protocol": "RIST-C0-v2.1",
        "checkpoints": [row["checkpoint"] for row in family_results],
        "common_resolution_map": common,
        "band_cell_counts": counts,
        "passed": passed,
        "decision": "PASS_C0_TO_FILTERED_TRAIN" if passed else "KILL_C0_NO_COMMON_BANDS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = combine_maps([json.loads(path.read_text()) for path in args.family_result])
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

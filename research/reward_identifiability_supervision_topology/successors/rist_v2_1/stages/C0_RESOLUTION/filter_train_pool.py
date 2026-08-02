"""Filter D3 train tasks by a transported cross-family resolution map."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def filter_train(
    source_path: Path, map_result: dict[str, Any], output_dir: Path
) -> dict[str, Any]:
    if map_result.get("passed") is not True:
        raise ValueError("cannot filter train pool from a failed resolution map")
    resolution_map = {
        str(cell): str(band)
        for cell, band in map_result["common_resolution_map"].items()
    }
    if set(resolution_map.values()) != {"low", "high"}:
        raise ValueError("resolution map must contain low and high bands")
    counts = Counter(resolution_map.values())
    if counts["low"] < 2 or counts["high"] < 2:
        raise ValueError("resolution map requires at least two cells per band")
    source_bytes = source_path.read_bytes()
    rows = [json.loads(line) for line in source_bytes.decode().splitlines() if line]
    selected = [row for row in rows if row["structural_cell"] in resolution_map]
    if not selected or any(row.get("split") != "train" for row in selected):
        raise ValueError("filtered source must contain train rows")
    cell_counts = Counter(str(row["structural_cell"]) for row in selected)
    if set(cell_counts) != set(resolution_map) or set(cell_counts.values()) != {4}:
        raise ValueError("filter must retain all four tasks from every selected cell")
    for row in selected:
        row["resolution_band"] = resolution_map[str(row["structural_cell"])]
    encoded = "".join(
        json.dumps(row, sort_keys=True) + "\n" for row in selected
    ).encode()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train.jsonl"
    output_path.write_bytes(encoded)
    result = {
        "protocol": "RIST-C0-v2.1",
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "resolution_map_sha256": hashlib.sha256(
            (json.dumps(map_result, sort_keys=True) + "\n").encode()
        ).hexdigest(),
        "train_sha256": hashlib.sha256(encoded).hexdigest(),
        "selected_cells": sorted(resolution_map),
        "band_cell_counts": dict(counts),
        "task_count": len(selected),
        "individual_outcome_selection": False,
        "output": str(output_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--resolution-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = filter_train(
        args.source,
        json.loads(args.resolution_map.read_text()),
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

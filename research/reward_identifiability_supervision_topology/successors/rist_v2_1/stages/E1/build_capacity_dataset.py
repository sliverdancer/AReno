"""Build a structural, outcome-blind E1 canary dataset after C0 passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def build_capacity_dataset(
    filtered_train_path: Path,
    resolution_result: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    if resolution_result.get("passed") is not True:
        raise ValueError("E1 dataset requires a passing C0 resolution result")
    resolution_map = resolution_result.get("common_resolution_map", {})
    high_cells = sorted(
        str(cell) for cell, band in resolution_map.items() if band == "high"
    )
    if len(high_cells) < 2:
        raise ValueError("E1 dataset requires at least two transported high cells")
    selected_cell = high_cells[0]
    source_bytes = filtered_train_path.read_bytes()
    rows = [json.loads(line) for line in source_bytes.decode().splitlines() if line]
    selected = [row for row in rows if str(row.get("structural_cell")) == selected_cell]
    if len(selected) != 4 or any(row.get("resolution_band") != "high" for row in selected):
        raise ValueError("E1 must retain all four tasks from one transported high cell")
    encoded = "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected).encode()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(encoded)
    result = {
        "protocol": "RIST-E1-CAPACITY-DATA-v2.1",
        "selection_rule": "LEXICOGRAPHIC_FIRST_TRANSPORTED_HIGH_CELL",
        "selection_uses_individual_outcomes": False,
        "selected_cell": selected_cell,
        "task_count": len(selected),
        "source_train_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "resolution_result_sha256": hashlib.sha256(
            (json.dumps(resolution_result, sort_keys=True) + "\n").encode()
        ).hexdigest(),
        "capacity_train_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-train", type=Path, required=True)
    parser.add_argument("--resolution-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_capacity_dataset(
        args.filtered_train,
        json.loads(args.resolution_result.read_text()),
        args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

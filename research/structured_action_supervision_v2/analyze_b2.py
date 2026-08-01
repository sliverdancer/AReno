"""Aggregate SAS-TR-v2.0 B2 evidence and apply the frozen decision rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import b2_inference as core
from prepare_protocol import select_eligible_cell


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(paths: list[Path], expected_split: str) -> dict[str, Any]:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if any(payload.get("protocol_id") != core.PROTOCOL_ID for payload in payloads):
        raise ValueError("protocol mismatch in B2 evidence")
    if any(payload.get("split") != expected_split for payload in payloads):
        raise ValueError("split mismatch in B2 evidence")
    cells: dict[str, list[dict[str, Any]]] = {}
    files: list[dict[str, Any]] = []
    for path, payload in zip(paths, payloads, strict=True):
        if len(payload.get("trajectories", [])) != 16:
            raise ValueError(f"each B2 evidence file must contain 16 trajectories: {path}")
        cells.setdefault(payload["cell_id"], []).extend(payload["trajectories"])
        files.append({"path": str(path), "sha256": _sha256(path)})
    expected_cells = {"D128", "D512", "N128", "N512"}
    if expected_split == "calibration" and set(cells) != expected_cells:
        raise ValueError(f"calibration requires exactly {sorted(expected_cells)}")
    if expected_split == "validation" and len(cells) != 1:
        raise ValueError("validation requires exactly one selected cell")
    for cell, items in cells.items():
        seeds = {int(item["sampling_seed"]) for item in items}
        if len(items) != 32 or seeds != {3101, 3202}:
            raise ValueError(f"{cell} must contain 32 trajectories crossed with both frozen seeds")
    cell_results = {cell: core.summarize(items) for cell, items in sorted(cells.items())}
    for result in cell_results.values():
        result["passes_frozen_gates"] = (
            result["raw_evidence_complete"]
            and result["first_turn_executable_rate"] >= 0.95
            and result["four_turn_completion_rate"] >= 0.75
            and 0.05 <= result["positive_reward_rate"] <= 0.95
            and result["fabricated_call_count"] == 0
        )
    selection_input = {
        cell: result
        for cell, result in cell_results.items()
        if result["passes_frozen_gates"]
    }
    selected = select_eligible_cell(selection_input) if expected_split == "calibration" else None
    return {
        "schema_version": 1,
        "protocol_id": core.PROTOCOL_ID,
        "split": expected_split,
        "files": files,
        "cell_results": cell_results,
        "selected_cell": selected,
        "decision": (
            "PASS_CALIBRATION_SELECT_" + selected
            if selected
            else "KILL_QWEN3_0_6B_INSTRUMENT"
        ) if expected_split == "calibration" else "VALIDATION_AGGREGATED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("calibration", "validation"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("evidence", type=Path, nargs="+")
    args = parser.parse_args()
    result = aggregate([path.resolve() for path in args.evidence], args.split)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "selected_cell": result["selected_cell"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

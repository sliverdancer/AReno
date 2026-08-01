"""Evaluate one RIST P1 split against the frozen CPU gates."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

RESEARCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RESEARCH_DIR))
import task_generator  # noqa: E402


def evaluate(data_dir: Path, split: str) -> dict[str, Any]:
    """Return the eight frozen gate results for train or qualification."""

    if split not in {"train", "qualification"}:
        raise ValueError("P1 evaluator may score only train or qualification")
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    split_path = data_dir / manifest["files"][split]["file"]
    rows = [json.loads(line) for line in split_path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        task_generator.validate_task(row)
    expected_count = 32 * task_generator.SPLIT_REPLICATES[split]
    cell_counts = Counter(str(row["factor_cell"]) for row in rows)
    signatures = [str(row["task_signature"]) for row in rows]
    oracle_valid = all(
        len(row["oracle_actions"]) == 4
        and task_generator.strict_reward(row, row["oracle_actions"]) == 1
        and task_generator.strict_reward(row, row["oracle_actions"][:-1]) == 0
        for row in rows
    )
    analytic_valid = all(
        int(row["possible_action_sequences"]) >= 1
        and 0.0 < float(row["uniform_strict_success_probability"]) <= 1.0
        and 0.0 <= float(row["mixed_group_probability_g8"]) <= 1.0
        for row in rows
    )
    strata = {str(row["reward_resolution_stratum"]) for row in rows}
    action_reward_separation = any(
        int(row["possible_action_sequences"]) >= 256
        and float(row["mixed_group_probability_g8"]) < 0.2
        for row in rows
    )
    manifest_signatures = [
        signature
        for split_record in manifest["files"].values()
        for signature in split_record["task_signatures"]
    ]
    cross_split_disjoint = len(manifest_signatures) == len(set(manifest_signatures))
    with tempfile.TemporaryDirectory(prefix="rist-p1-regenerate-") as tmp:
        regenerated_dir = Path(tmp)
        task_generator.write_splits(regenerated_dir, seed=int(manifest["seed"]))
        reproducible = (
            (regenerated_dir / split_path.name).read_bytes() == split_path.read_bytes()
            and (regenerated_dir / "manifest.json").read_bytes()
            == (data_dir / "manifest.json").read_bytes()
        )
    gates = {
        "expected_task_count": len(rows) == expected_count,
        "balanced_32_cell_factorial": len(cell_counts) == 32
        and set(cell_counts.values()) == {task_generator.SPLIT_REPLICATES[split]},
        "unique_cross_split_signatures": len(signatures) == len(set(signatures))
        and cross_split_disjoint,
        "unique_four_action_strict_oracle": oracle_valid,
        "finite_exact_analytic_measurements": analytic_valid,
        "low_intermediate_high_resolution": strata == {"low", "intermediate", "high"},
        "action_reward_diversity_separated": action_reward_separation,
        "byte_identical_regeneration": reproducible,
    }
    return {
        "schema_version": 1,
        "protocol": "RIST-P1-v1.0",
        "split": split,
        "score": sum(gates.values()),
        "max_score": len(gates),
        "passed": all(gates.values()),
        "task_count": len(rows),
        "factor_cell_count": len(cell_counts),
        "resolution_strata": sorted(strata),
        "possible_action_sequences": {
            "min": min(int(row["possible_action_sequences"]) for row in rows),
            "max": max(int(row["possible_action_sequences"]) for row in rows),
        },
        "mixed_group_probability_g8": {
            "min": min(float(row["mixed_group_probability_g8"]) for row in rows),
            "max": max(float(row["mixed_group_probability_g8"]) for row in rows),
        },
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "qualification"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.data_dir, args.split)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


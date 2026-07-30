"""Generate disjoint deterministic task splits for the SAS research instrument."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations, product
from pathlib import Path
from typing import Any

SHOPPING_DIR = Path(__file__).resolve().parents[1] / "shopping"
sys.path.insert(0, str(SHOPPING_DIR))
from game import CATALOG, best_bundle  # noqa: E402

GENERATOR_VERSION = "sas-shopping-v1"
DEFAULT_SEED = 20260729
SPLIT_COUNTS = {"train": 48, "qualification": 16, "heldout": 16}
BUDGET_SLACKS = (0, 15, 30, 45)
PROFILE_ITEM_IDS = {
    "jacket": ("packable-rain-shell", "trail-softshell"),
    "shoes": ("ultralight-hiker", "city-commuter"),
    "bottle": ("insulated-bottle-750", "collapsible-bottle"),
}


def generate_splits(seed: int = DEFAULT_SEED) -> dict[str, list[dict[str, Any]]]:
    """Return deterministic, signature-disjoint train/qualification/heldout rows."""

    candidates = _candidate_records()
    ranked = sorted(
        candidates,
        key=lambda record: hashlib.sha256(
            f"{seed}:{record['constraint_signature']}".encode()
        ).hexdigest(),
    )
    expected = sum(SPLIT_COUNTS.values())
    if len(ranked) != expected:
        raise RuntimeError(f"expected {expected} candidate tasks, got {len(ranked)}")

    result: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for split, count in SPLIT_COUNTS.items():
        rows = []
        for split_index, record in enumerate(ranked[cursor : cursor + count]):
            row = dict(record)
            row["id"] = f"{split}-{split_index:03d}"
            row["split"] = split
            row["dataset_seed"] = seed
            row["generator_version"] = GENERATOR_VERSION
            rows.append(row)
        result[split] = rows
        cursor += count
    validate_splits(result)
    return result


def validate_splits(splits: dict[str, list[dict[str, Any]]]) -> None:
    """Reject duplicate signatures, invalid bundles, and malformed split labels."""

    if set(splits) != set(SPLIT_COUNTS):
        raise ValueError(f"split names must be {sorted(SPLIT_COUNTS)}")
    seen: dict[str, str] = {}
    for split, rows in splits.items():
        if len(rows) != SPLIT_COUNTS[split]:
            raise ValueError(
                f"{split} must contain {SPLIT_COUNTS[split]} records, got {len(rows)}"
            )
        for record in rows:
            if record.get("split") != split:
                raise ValueError(f"record {record.get('id')} has wrong split label")
            signature = str(record.get("constraint_signature", ""))
            if not signature:
                raise ValueError(f"record {record.get('id')} has no constraint signature")
            if signature in seen:
                raise ValueError(
                    f"constraint signature appears in both {seen[signature]} and {split}"
                )
            seen[signature] = split
            if not best_bundle(record):
                raise ValueError(f"record {record.get('id')} has no valid bundle")


def write_splits(output_dir: Path, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    """Write JSONL splits and a content-addressed manifest."""

    splits = generate_splits(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, rows in splits.items():
        path = output_dir / f"{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        files[split] = {
            "path": str(path),
            "count": len(rows),
            "sha256": _sha256(path),
            "constraint_signatures": [
                row["constraint_signature"] for row in rows
            ],
        }
    manifest = {
        "schema_version": 1,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "split_counts": SPLIT_COUNTS,
        "files": files,
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _candidate_records() -> list[dict[str, Any]]:
    by_id = {str(item["id"]): item for item in CATALOG}
    category_sets = [
        *combinations(PROFILE_ITEM_IDS, 2),
        tuple(PROFILE_ITEM_IDS),
    ]
    records = []
    for categories in category_sets:
        for selected_ids in product(*(PROFILE_ITEM_IDS[category] for category in categories)):
            selected_items = [by_id[item_id] for item_id in selected_ids]
            required = {
                str(item["category"]): [str(feature) for feature in item["features"][:2]]
                for item in selected_items
            }
            base_total = sum(int(item["price"]) for item in selected_items)
            for slack in BUDGET_SLACKS:
                constraints = {
                    "categories": list(categories),
                    "budget": base_total + slack,
                    "required_features_by_category": required,
                }
                signature = hashlib.sha256(
                    json.dumps(
                        constraints,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                records.append(
                    {
                        "kit_name": f"sas-{signature[:12]}",
                        **constraints,
                        "constraint_signature": signature,
                    }
                )
    return records


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    manifest = write_splits(args.output_dir, seed=args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


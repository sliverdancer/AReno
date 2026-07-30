"""Normalize SAS shopping JSONL rows into prompt-bearing training records."""

from __future__ import annotations

import sys
from pathlib import Path

SHOPPING_DIR = Path(__file__).resolve().parents[1] / "shopping"
sys.path.insert(0, str(SHOPPING_DIR))
from game import make_prompt  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Load already-normalized SAS records without weakening their constraints."""

    records = []
    for source in default_loader(dataset_path):
        record = dict(source)
        categories = record.get("categories")
        required = record.get("required_features_by_category")
        if not isinstance(categories, list) or not categories:
            raise ValueError("SAS record categories must be a non-empty list")
        if not isinstance(required, dict):
            raise ValueError("SAS record required_features_by_category must be an object")
        if not isinstance(record.get("constraint_signature"), str):
            raise ValueError("SAS record must contain constraint_signature")
        record["categories"] = [str(category) for category in categories]
        record["required_features_by_category"] = {
            category: [str(feature) for feature in required.get(category, [])]
            for category in record["categories"]
        }
        record["prompt"] = make_prompt(record)
        records.append(record)
    return records


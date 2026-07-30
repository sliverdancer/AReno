"""Dataset loader for the controlled CARe bifurcation pilot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset_generator  # noqa: E402
import task  # noqa: E402


def load_training_dataset(
    dataset_path: str,
    *,
    default_loader=None,
    **_: object,
) -> list[dict]:
    """Load JSONL and add the exact model-visible prompt."""

    del default_loader
    path = Path(dataset_path).expanduser()
    if path.exists():
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        records = dataset_generator.generate_records()
    normalized = []
    for index, record in enumerate(records):
        weights = task.validate_task(record["weights"], record["threshold"])
        threshold = int(record["threshold"])
        normalized.append(
            {
                "id": str(record.get("id", f"care-bifurcation-{index:05d}")),
                "prompt": (
                    "Complete four binary decisions. At turn t, choosing bit 1 "
                    "adds weight[t] and bit 0 subtracts weight[t]. "
                    f"Weights: {list(weights)}. Target final score: at least {threshold}. "
                    "Call choose_bit exactly once at each turn."
                ),
                "weights": list(weights),
                "threshold": threshold,
            }
        )
    return normalized

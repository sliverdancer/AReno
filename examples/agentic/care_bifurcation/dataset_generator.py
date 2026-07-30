"""Generate deterministic CARe bifurcation tasks."""

from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import product
from pathlib import Path
from typing import TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import task  # noqa: E402


DEFAULT_COUNT = 64
DEFAULT_SEED = 7301


def generate_records(
    count: int = DEFAULT_COUNT,
    *,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Generate unique signed-weight tasks in a frozen random order."""

    if count <= 0:
        raise ValueError("count must be positive")
    candidates = [
        {"weights": list(weights), "threshold": threshold}
        for weights in product(task.WEIGHT_VALUES, repeat=task.HORIZON)
        if any(value > 0 for value in weights)
        and any(value < 0 for value in weights)
        for threshold in (-2, 0, 2)
    ]
    if count > len(candidates):
        raise ValueError(f"count exceeds {len(candidates)} unique tasks")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return [
        {
            "id": f"care-bifurcation-{index:05d}",
            "weights": candidate["weights"],
            "threshold": candidate["threshold"],
        }
        for index, candidate in enumerate(candidates[:count])
    ]


def write_jsonl(records: list[dict], output: TextIO) -> None:
    """Write records as canonical JSONL."""

    for record in records:
        output.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    path = Path(args.output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        write_jsonl(generate_records(args.count, seed=args.seed), handle)


if __name__ == "__main__":
    main()

"""Deterministic clean-reset replay gate for future Tau3 adapters."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


def _load_transcript_validator():
    path = Path(__file__).with_name("external_env_contract.py")
    spec = importlib.util.spec_from_file_location("rist_v2_1_external_contract_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load external transcript contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_transcript


validate_transcript = _load_transcript_validator()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_partitions(partitions: dict[str, list[str]]) -> dict[str, set[str]]:
    required = {"development", "training", "confirmatory"}
    if set(partitions) != required:
        raise ValueError("partitions must contain development, training, and confirmatory")
    normalized: dict[str, set[str]] = {}
    for split, values in partitions.items():
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise ValueError(f"{split} IDs must be non-empty strings")
        if len(values) != len(set(values)):
            raise ValueError(f"{split} contains duplicate IDs")
        normalized[split] = set(values)
    if any(
        normalized[left] & normalized[right]
        for index, left in enumerate(sorted(required))
        for right in sorted(required)[index + 1 :]
    ):
        raise ValueError("partition IDs must be pairwise disjoint")
    return normalized


def qualify_replays(
    replay_a: dict[str, list[dict[str, Any]]],
    replay_b: dict[str, list[dict[str, Any]]],
    partitions: dict[str, list[str]],
) -> dict[str, Any]:
    """Require two byte-canonical-equivalent development replays."""

    split_ids = _validate_partitions(partitions)
    if not replay_a or set(replay_a) != set(replay_b):
        raise ValueError("two replays must contain the same non-empty episode set")
    if not set(replay_a) <= split_ids["development"]:
        raise ValueError("qualification may use development episode IDs only")

    rows = []
    for episode_id in sorted(replay_a):
        first = replay_a[episode_id]
        second = replay_b[episode_id]
        first_result = validate_transcript(first)
        second_result = validate_transcript(second)
        if first_result["episode_id"] != episode_id or second_result["episode_id"] != episode_id:
            raise ValueError("transcript episode_id does not match replay key")
        first_hash = _canonical_sha256(first)
        second_hash = _canonical_sha256(second)
        if first_hash != second_hash:
            raise ValueError(f"nondeterministic replay for {episode_id}")
        rows.append(
            {
                "episode_id": episode_id,
                "transcript_sha256": first_hash,
                "step_count": first_result["step_count"],
                "final_state_hash": first_result["final_state_hash"],
            }
        )
    return {
        "protocol": "RIST-X0-TAU3-REPLAY-v1",
        "environment_qualification_pass": True,
        "clean_reset_replay_count": 2,
        "episode_count": len(rows),
        "qualification_split": "development",
        "partition_disjointness_pass": True,
        "llm_judge_used": False,
        "episodes": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-a", type=Path, required=True)
    parser.add_argument("--replay-b", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = qualify_replays(
        json.loads(args.replay_a.read_text(encoding="utf-8")),
        json.loads(args.replay_b.read_text(encoding="utf-8")),
        json.loads(args.partitions.read_text(encoding="utf-8")),
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

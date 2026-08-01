"""Analyze the frozen SAS-B3-v3.0 within-group reward preflight."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def analyze(paths: list[Path], manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate complete per-seed evidence and apply the frozen signal gate."""

    expected_seeds = [int(seed) for seed in manifest["sampling_seeds"]]
    if len(paths) != len(expected_seeds):
        raise ValueError(
            f"expected {len(expected_seeds)} seed files, found {len(paths)}"
        )
    by_seed: dict[int, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol_id") != manifest["protocol_id"]:
            raise ValueError(f"protocol mismatch in {path}")
        seed = int(payload["sampling_seed"])
        if seed in by_seed:
            raise ValueError(f"duplicate sampling seed {seed}")
        trajectories = payload.get("trajectories")
        if not isinstance(trajectories, list) or len(trajectories) != 16:
            raise ValueError(f"{path} must contain exactly 16 trajectories")
        by_seed[seed] = payload
    if sorted(by_seed) != sorted(expected_seeds):
        raise ValueError(
            f"sampling seed mismatch: expected={sorted(expected_seeds)} "
            f"found={sorted(by_seed)}"
        )

    all_trajectories: list[dict[str, Any]] = []
    rewards_by_row: dict[str, list[float]] = {}
    for seed in expected_seeds:
        for trajectory in by_seed[seed]["trajectories"]:
            if trajectory.get("sampling_seed") != seed:
                raise ValueError("trajectory sampling seed mismatch")
            row_id = str(trajectory["row_id"])
            reward = float(trajectory["strict_reward"])
            if not math.isfinite(reward):
                raise ValueError(f"non-finite strict reward for {row_id}")
            rewards_by_row.setdefault(row_id, []).append(reward)
            all_trajectories.append(trajectory)
    if len(rewards_by_row) != 16 or any(
        len(rewards) != len(expected_seeds) for rewards in rewards_by_row.values()
    ):
        raise ValueError("preflight must form 16 complete eight-sample groups")

    count = len(all_trajectories)
    first = sum(bool(item.get("first_turn_executable")) for item in all_trajectories)
    complete = sum(bool(item.get("four_turn_complete")) for item in all_trajectories)
    positive = sum(float(item["strict_reward"]) > 0 for item in all_trajectories)
    fabricated = sum(int(item.get("fabricated_call_count", 0)) for item in all_trajectories)
    raw_complete = all(
        isinstance(item.get("turns"), list)
        and bool(item["turns"])
        and all(isinstance(turn.get("raw_response"), dict) for turn in item["turns"])
        and item.get("terminal_reason") not in {"REQUEST_ERROR", "GPU_DEADLINE_EXCEEDED"}
        for item in all_trajectories
    )
    mixed_groups = sum(
        any(reward > 0 for reward in rewards)
        and any(reward <= 0 for reward in rewards)
        for rewards in rewards_by_row.values()
    )
    nonzero_advantage = sum(
        len(rewards)
        for rewards in rewards_by_row.values()
        if max(rewards) > min(rewards)
    )
    metrics = {
        "trajectory_count": count,
        "first_turn_executable_rate": first / count,
        "four_turn_completion_rate": complete / count,
        "positive_reward_rate": positive / count,
        "mixed_strict_success_groups": mixed_groups,
        "nonzero_advantage_trajectories": nonzero_advantage,
        "fabricated_call_count": fabricated,
        "raw_evidence_complete": raw_complete,
        "group_rewards": dict(sorted(rewards_by_row.items())),
    }
    gates = manifest["gates"]
    checks = {
        "first_turn_executable": metrics["first_turn_executable_rate"]
        >= gates["first_turn_executable_rate_min"],
        "four_turn_completion": metrics["four_turn_completion_rate"]
        >= gates["four_turn_completion_rate_min"],
        "positive_reward_support": gates["positive_reward_rate_min"]
        <= metrics["positive_reward_rate"]
        <= gates["positive_reward_rate_max"],
        "mixed_strict_success_groups": mixed_groups
        >= gates["mixed_strict_success_groups_min"],
        "nonzero_advantage_trajectories": nonzero_advantage
        >= gates["nonzero_advantage_trajectories_min"],
        "no_fabricated_calls": fabricated <= gates["fabricated_call_count_max"],
        "raw_evidence_complete": raw_complete is gates["raw_evidence_complete"],
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "protocol_id": manifest["protocol_id"],
        "stage": manifest["stage"],
        "passed": passed,
        "decision": manifest["pass_decision"] if passed else manifest["fail_decision"],
        "checks": checks,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = analyze(sorted(args.input_dir.glob("signal_seed_*.json")), manifest)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": result["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Analyze frozen RIST P2 cross-family qualification evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def analyze_model(payload: dict[str, Any]) -> dict[str, Any]:
    """Compute per-model interface, reward-resolution, and action-diversity gates."""

    trajectories = payload.get("trajectories") or []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        groups[str(trajectory["task_signature"])].append(trajectory)
    first_rate = _mean(bool(row["first_turn_executable"]) for row in trajectories)
    completion_rate = _mean(bool(row["four_turn_complete"]) for row in trajectories)
    success_rate = _mean(int(row["strict_reward"]) for row in trajectories)
    mixed_groups = 0
    nonzero_advantage_trajectories = 0
    stratum_mixed: Counter[str] = Counter()
    multi_action_homogeneous = 0
    reward_entropies = []
    unique_sequences_by_task = {}
    for signature, rows in groups.items():
        reward_values = [int(row["strict_reward"]) for row in rows]
        rewards = set(reward_values)
        reward_entropies.append(_binary_entropy(_mean(reward_values)))
        sequences = {
            json.dumps(row["actions"], sort_keys=True, separators=(",", ":"))
            for row in rows
        }
        unique_sequences_by_task[signature] = len(sequences)
        if len(rewards) > 1:
            mixed_groups += 1
            nonzero_advantage_trajectories += len(rows)
            stratum_mixed[str(rows[0]["analytic_stratum"])] += 1
        elif len(sequences) > 1:
            multi_action_homogeneous += 1
    evidence_complete = (
        len(trajectories) == int(payload.get("expected_trajectories", -1))
        and len(groups) == 32
        and all(len(rows) == 8 for rows in groups.values())
        and all(int(row.get("raw_response_count", 0)) == 4 for row in trajectories)
    )
    fabricated = sum(int(row.get("fabricated_call_count", 0)) for row in trajectories)
    invalid_reason_counts: Counter[str] = Counter()
    parsed_action_count = 0
    raw_response_count = 0
    for row in trajectories:
        invalid_reason_counts.update(str(reason) for reason in row.get("invalid_reasons", []))
        parsed_action_count += sum(action is not None for action in row.get("actions", []))
        raw_response_count += int(row.get("raw_response_count", 0))
    gates = {
        "first_turn_executable": first_rate >= 0.90,
        "four_turn_complete": completion_rate >= 0.75,
        "nondegenerate_strict_success": 0.05 <= success_rate <= 0.95,
        "mixed_groups": mixed_groups >= 6,
        "nonzero_advantage_trajectories": nonzero_advantage_trajectories >= 48,
        "raw_evidence_and_no_fabrication": evidence_complete
        and fabricated == 0
        and payload.get("infrastructure_error") is None,
    }
    return {
        "model_cell": payload.get("model_cell"),
        "trajectory_count": len(trajectories),
        "task_group_count": len(groups),
        "first_turn_executable_rate": first_rate,
        "four_turn_completion_rate": completion_rate,
        "strict_success_rate": success_rate,
        "mixed_groups": mixed_groups,
        "advantage_collapse_rate": 1.0 - mixed_groups / 32.0,
        "nonzero_advantage_trajectories": nonzero_advantage_trajectories,
        "stratum_mixed_groups": dict(sorted(stratum_mixed.items())),
        "multi_action_homogeneous_groups": multi_action_homogeneous,
        "mean_empirical_reward_entropy_bits": _mean(reward_entropies),
        "mean_unique_action_sequences": _mean(unique_sequences_by_task.values()),
        "invalid_reason_counts": dict(sorted(invalid_reason_counts.items())),
        "raw_parsed_agreement_rate": (
            parsed_action_count / raw_response_count if raw_response_count else math.nan
        ),
        "fabricated_call_count": fabricated,
        "evidence_complete": evidence_complete,
        "gates": gates,
        "passed": all(gates.values()),
    }


def analyze_cross_family(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply all per-model and cross-family P2 gates."""

    models = [analyze_model(payload) for payload in payloads]
    cross_gates = {
        "two_frozen_model_families": len(models) == 2
        and {model["model_cell"] for model in models}
        == {"qwen3_0_6b", "gemma4_e2b_it"},
        "both_models_pass": len(models) == 2 and all(model["passed"] for model in models),
        "two_strata_mixed_per_model": len(models) == 2
        and all(
            sum(count >= 2 for count in model["stratum_mixed_groups"].values()) >= 2
            for model in models
        ),
        "action_reward_distinction_per_model": len(models) == 2
        and all(model["multi_action_homogeneous_groups"] >= 1 for model in models),
    }
    passed = all(cross_gates.values())
    if any(payload.get("infrastructure_error") is not None for payload in payloads):
        decision = "INVALID_P2_INFRASTRUCTURE"
        status = "INVALID"
    elif not all(model["gates"]["first_turn_executable"] and model["gates"]["four_turn_complete"] for model in models):
        decision = "KILL_CHECKPOINT_INTERFACE"
        status = "KILL"
    elif not passed:
        decision = "KILL_TASK_INSTRUMENT_NO_MODEL_RESOLUTION"
        status = "KILL"
    else:
        decision = "PASS_P2_CROSS_FAMILY_RESOLUTION_TO_P3_PILOT"
        status = "PASS"
    return {
        "schema_version": 1,
        "protocol": "RIST-P2-v1.0",
        "stage": "P2",
        "stage_status": status,
        "decision": decision,
        "models": models,
        "cross_family_gates": cross_gates,
    }


def _mean(values) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        return math.nan
    return sum(materialized) / len(materialized)


def _binary_entropy(success_probability: float) -> float:
    if success_probability in (0.0, 1.0):
        return 0.0
    return -success_probability * math.log2(success_probability) - (
        1.0 - success_probability
    ) * math.log2(1.0 - success_probability)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.model_result]
    result = analyze_cross_family(payloads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["stage_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

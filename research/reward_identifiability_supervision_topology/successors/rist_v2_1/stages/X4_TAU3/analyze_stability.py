"""Analyze validated X4 outcomes at the paired-training-seed level."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from statistics import NormalDist
from typing import Any

FAMILIES = ("qwen3", "gemma4")
ALGORITHMS = ("gspo", "grpo")
ARMS = ("AF", "LF", "AN", "LN")
METRICS = ("strict_success", "token_auc", "token_endpoint")
MIN_SIGN_AGREEMENT = 0.75


def _sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


def _interaction(values: dict[str, float]) -> float:
    if set(values) != set(ARMS):
        raise ValueError("interaction requires AF/LF/AN/LN")
    return values["AF"] - values["LF"] - values["AN"] + values["LN"]


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_records(
    records: list[dict[str, Any]], seed_bank: dict[str, Any]
) -> tuple[list[int], dict[tuple[int, str, str, str], dict[str, Any]]]:
    expected_order = [int(seed) for seed in seed_bank["seeds"]]
    observed_seeds = sorted({int(row["seed"]) for row in records}, key=expected_order.index)
    if observed_seeds != expected_order[: len(observed_seeds)] or len(observed_seeds) < 12:
        raise ValueError("results must be a prefix of the frozen seed bank with >=12 seeds")
    expected = {
        (seed, family, algorithm, arm)
        for seed in observed_seeds
        for family in FAMILIES
        for algorithm in ALGORITHMS
        for arm in ARMS
    }
    indexed = {
        (int(row["seed"]), row["family"], row["algorithm"], row["arm"]): row
        for row in records
    }
    if len(indexed) != len(records) or set(indexed) != expected:
        raise ValueError("results do not cover the exact paired-seed factorial prefix")
    for row in records:
        if row.get("catastrophic") is not False:
            raise ValueError("catastrophic or missing run invalidates X4")
        for metric in METRICS:
            value = row.get(metric)
            if (
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"invalid {metric}")
        support = row.get("token_support_fraction")
        if type(support) not in (int, float) or not 0.0 <= float(support) <= 1.0:
            raise ValueError("invalid token support")
    return observed_seeds, indexed


def _metric_interactions(
    seeds: list[int],
    indexed: dict[tuple[int, str, str, str], dict[str, Any]],
    metric: str,
) -> dict[str, dict[int, float]]:
    result = {}
    for family in FAMILIES:
        for algorithm in ALGORITHMS:
            block = f"{family}:{algorithm}"
            result[block] = {
                seed: _interaction(
                    {
                        arm: float(indexed[(seed, family, algorithm, arm)][metric])
                        for arm in ARMS
                    }
                )
                for seed in seeds
            }
    return result


def _required_n(sd: float, plan: dict[str, Any]) -> int:
    if sd < 0.0 or not math.isfinite(sd):
        raise ValueError("planning SD must be finite and nonnegative")
    z_alpha = NormalDist().inv_cdf(1.0 - float(plan["alpha"]) / 2.0)
    z_power = NormalDist().inv_cdf(float(plan["target_power"]))
    raw = math.ceil(
        ((z_alpha + z_power) * sd / float(plan["minimum_practical_interaction"]))
        ** 2
    )
    return max(int(plan["planning_seed_floor"]), raw)


def _stability(values: list[float]) -> dict[str, Any]:
    mean = statistics.fmean(values)
    sign = _sign(mean)
    agreement = sum(_sign(value) == sign for value in values) / len(values)
    loo = [
        statistics.fmean(values[:index] + values[index + 1 :])
        for index in range(len(values))
    ]
    return {
        "mean": mean,
        "sign": sign,
        "seed_sign_agreement": agreement,
        "leave_one_seed_out_means": loo,
        "pass": (
            sign != 0
            and agreement >= MIN_SIGN_AGREEMENT
            and all(_sign(value) == sign for value in loo)
        ),
    }


def analyze(
    records: list[dict[str, Any]],
    seed_bank: dict[str, Any],
    evidence_validation: dict[str, Any],
) -> dict[str, Any]:
    if evidence_validation.get("passed") is not True:
        raise ValueError("X4 analysis requires passed artifact evidence validation")
    if evidence_validation.get("records_sha256") != _canonical_sha256(records):
        raise ValueError("analysis records are not bound to evidence validation")
    seeds, indexed = _validate_records(records, seed_bank)
    interactions = {
        metric: _metric_interactions(seeds, indexed, metric) for metric in METRICS
    }
    pooled = {
        metric: [
            statistics.fmean(block[seed] for block in interactions[metric].values())
            for seed in seeds
        ]
        for metric in METRICS
    }
    planning_seeds = [int(seed) for seed in seed_bank["planning_seeds"]]
    planning_values = pooled["strict_success"][: len(planning_seeds)]
    planning_sd = statistics.stdev(planning_values)
    plan = seed_bank["power_plan"]
    required_n = _required_n(planning_sd, plan)
    maximum_n = int(plan["maximum_seed_count"])
    if required_n > maximum_n:
        return {
            "protocol": seed_bank["protocol"],
            "decision": "KILL_X4_VARIANCE_EXCEEDS_FROZEN_32_SEED_BANK",
            "powered": False,
            "actual_n": len(seeds),
            "required_n": required_n,
            "planning_seed_sd": planning_sd,
        }
    if len(seeds) < required_n:
        return {
            "protocol": seed_bank["protocol"],
            "decision": "EXPAND_X4_IN_FROZEN_SEED_ORDER",
            "powered": False,
            "actual_n": len(seeds),
            "required_n": required_n,
            "next_seeds": seed_bank["seeds"][len(seeds) : required_n],
            "planning_seed_sd": planning_sd,
        }

    strict_stability = _stability(pooled["strict_success"])
    primary_sign = strict_stability["sign"]
    block_means = {
        metric: {
            block: statistics.fmean(values.values())
            for block, values in interactions[metric].items()
        }
        for metric in METRICS
    }
    cross_block = all(
        _sign(value) == primary_sign and primary_sign != 0
        for value in block_means["strict_success"].values()
    )
    token_direction = all(
        _sign(value) == primary_sign
        for metric in ("token_auc", "token_endpoint")
        for value in block_means[metric].values()
    )
    support_pass = all(float(row["token_support_fraction"]) >= 0.50 for row in records)
    mean = strict_stability["mean"]
    actual_sd = statistics.stdev(pooled["strict_success"])
    z_alpha = NormalDist().inv_cdf(1.0 - float(plan["alpha"]) / 2.0)
    z_power = NormalDist().inv_cdf(float(plan["target_power"]))
    sensitivity_mde = (z_alpha + z_power) * actual_sd / math.sqrt(len(seeds))
    practical = abs(mean) >= float(plan["minimum_practical_interaction"])
    sensitivity_pass = sensitivity_mde <= abs(mean)
    passed = all(
        (
            strict_stability["pass"],
            cross_block,
            token_direction,
            support_pass,
            practical,
            sensitivity_pass,
        )
    )
    return {
        "protocol": seed_bank["protocol"],
        "decision": "PASS_X4_POWERED_STABLE_INTERACTION" if passed else "FAIL_X4_STABILITY_GATE",
        "powered": True,
        "actual_n": len(seeds),
        "required_n": required_n,
        "planning_seed_sd": planning_sd,
        "observed_seed_sd": actual_sd,
        "sensitivity_mde": sensitivity_mde,
        "primary": strict_stability,
        "block_means": block_means,
        "cross_block_direction_pass": cross_block,
        "token_direction_pass": token_direction,
        "token_support_pass": support_pass,
        "minimum_practical_effect_pass": practical,
        "sensitivity_pass": sensitivity_pass,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--seed-bank", type=Path, required=True)
    parser.add_argument("--evidence-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.records.read_text())
    result = analyze(
        records,
        json.loads(args.seed_bank.read_text()),
        json.loads(args.evidence_validation.read_text()),
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result.get("passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())

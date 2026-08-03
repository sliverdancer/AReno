"""Freeze the outcome-blind paired-seed bank and prospective X4 power plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

PROTOCOL = "RIST-X4-TAU3-POWERED-v2.1"
SEED_DOMAIN = "rist-x4-tau3-outcome-blind-seed-bank-v1"
ALPHA = 0.05
TARGET_POWER = 0.80
MIN_PRACTICAL_INTERACTION = 0.05
ASSUMED_SEED_INTERACTION_SD = 0.06
PLANNING_SEED_FLOOR = 12
MAX_SEED_COUNT = 32
FORBIDDEN_PILOT_SEEDS = {7101, 7202, 7303}


def required_seed_count() -> int:
    z_alpha = NormalDist().inv_cdf(1.0 - ALPHA / 2.0)
    z_power = NormalDist().inv_cdf(TARGET_POWER)
    return math.ceil(
        ((z_alpha + z_power) * ASSUMED_SEED_INTERACTION_SD / MIN_PRACTICAL_INTERACTION)
        ** 2
    )


def build_seed_bank() -> dict:
    planning_count = required_seed_count()
    if planning_count != PLANNING_SEED_FLOOR:
        raise RuntimeError("planning constants no longer produce the frozen 12-seed floor")
    count = MAX_SEED_COUNT
    seeds = []
    index = 0
    while len(seeds) < count:
        digest = hashlib.sha256(f"{SEED_DOMAIN}:{index}".encode()).digest()
        seed = 100_000 + int.from_bytes(digest[:8], "big") % 900_000
        index += 1
        if seed in FORBIDDEN_PILOT_SEEDS or seed in seeds:
            continue
        seeds.append(seed)
    canonical = json.dumps(seeds, separators=(",", ":")).encode()
    return {
        "protocol": PROTOCOL,
        "seed_unit": "independent_paired_training_seed",
        "selection": "sha256_domain_separated_before_any_x3_or_x4_outcome",
        "seed_domain_sha256": hashlib.sha256(SEED_DOMAIN.encode()).hexdigest(),
        "seed_count": count,
        "seeds": seeds,
        "planning_seeds": seeds[:PLANNING_SEED_FLOOR],
        "expansion_seeds": seeds[PLANNING_SEED_FLOOR:],
        "seeds_sha256": hashlib.sha256(canonical).hexdigest(),
        "pilot_seeds_excluded": sorted(FORBIDDEN_PILOT_SEEDS),
        "power_plan": {
            "test": "two_sided_paired_seed_mean_normal_approximation",
            "alpha": ALPHA,
            "target_power": TARGET_POWER,
            "minimum_practical_interaction": MIN_PRACTICAL_INTERACTION,
            "planning_only_assumed_seed_interaction_sd": ASSUMED_SEED_INTERACTION_SD,
            "planning_seed_floor": PLANNING_SEED_FLOOR,
            "maximum_seed_count": MAX_SEED_COUNT,
            "actual_required_n_rule": (
                "ceil(((z_(1-alpha/2)+z_power)*planning_seed_sd/min_effect)^2); "
                "clip upward to planning floor; KILL if above maximum"
            ),
            "multiplicity": "one_pooled_primary_interaction; block signs are gates",
        },
        "outcomes_read": False,
        "model_accessed": False,
        "gpu_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_seed_bank(), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

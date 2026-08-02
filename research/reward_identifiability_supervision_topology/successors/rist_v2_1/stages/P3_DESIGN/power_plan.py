"""Prospective paired-seed normal-approximation power planning for RIST."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist


def required_paired_seeds(
    minimum_effect: float,
    contrast_sd: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Return a conservative normal-approximation paired-seed count."""

    if minimum_effect <= 0.0 or contrast_sd <= 0.0:
        raise ValueError("minimum_effect and contrast_sd must be positive")
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha and power must lie strictly between zero and one")
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1.0 - alpha / 2.0)
    z_power = normal.inv_cdf(power)
    estimate = ((z_alpha + z_power) * contrast_sd / minimum_effect) ** 2
    return max(3, math.ceil(estimate))


def sensitivity_plan(
    minimum_effect: float = 0.10,
    standard_deviations: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20),
) -> dict:
    return {
        "minimum_effect": minimum_effect,
        "alpha": 0.05,
        "power": 0.80,
        "statistical_unit": "paired_training_seed",
        "three_seed_status": "pilot_variance_only",
        "rows": [
            {
                "paired_contrast_sd": sd,
                "required_paired_seeds": required_paired_seeds(minimum_effect, sd),
            }
            for sd in standard_deviations
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-effect", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(
            sensitivity_plan(minimum_effect=args.minimum_effect),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

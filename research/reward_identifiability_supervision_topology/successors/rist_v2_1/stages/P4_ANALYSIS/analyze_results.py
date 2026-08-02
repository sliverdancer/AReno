"""Analyze complete RIST run bundles at the paired-training-seed level."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ARMS = ("AF", "LF", "AN", "LN")
MIN_CONFIRMATORY_SEEDS = 8
MIN_INTERACTION = 0.10
MAX_FAILURE_RATE = 0.10
BOOTSTRAP_REPLICATES = 10_000
P3_ROOT = Path(__file__).resolve().parents[1] / "P3_DESIGN"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"module loader unavailable: {path}")
    spec.loader.exec_module(module)
    return module


def _is_probability(value: Any) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def validate_bundle(
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
    evidence_root: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Validate an exact result for every frozen manifest run."""

    expected = {str(row["run_id"]): row for row in manifest["runs"]}
    observed = {str(row["run_id"]): row for row in results}
    if len(observed) != len(results):
        raise ValueError("run results contain duplicate run_id values")
    if set(observed) != set(expected):
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise ValueError(f"run bundle mismatch: missing={missing}, extra={extra}")

    for run_id, result in observed.items():
        design = expected[run_id]
        for field in ("family", "algorithm", "arm", "seed"):
            if result.get(field) != design[field]:
                raise ValueError(f"{run_id}: {field} does not match frozen manifest")
        digest = result.get("raw_evidence_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{run_id}: missing raw evidence SHA-256")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(f"{run_id}: invalid raw evidence SHA-256") from exc
        if evidence_root is not None:
            relative = Path(str(result.get("raw_evidence_path", "")))
            if not relative.parts or relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"{run_id}: unsafe raw evidence path")
            evidence_path = evidence_root / relative
            if not evidence_path.is_file():
                raise ValueError(f"{run_id}: raw evidence file is missing")
            actual_digest = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
            if actual_digest != digest:
                raise ValueError(f"{run_id}: raw evidence hash mismatch")
        if type(result.get("catastrophic")) is not bool:
            raise ValueError(f"{run_id}: catastrophic must be boolean")
        if not _is_probability(result.get("confirmatory_strict_success")):
            raise ValueError(f"{run_id}: invalid confirmatory strict success")
        confirmatory_bands = result.get("confirmatory_strict_success_by_resolution")
        if not isinstance(confirmatory_bands, dict) or set(confirmatory_bands) != {"low", "high"} or not all(
            _is_probability(value) for value in confirmatory_bands.values()
        ):
            raise ValueError(f"{run_id}: invalid confirmatory resolution endpoints")
        if not isinstance(result.get("nonzero_advantage_groups"), int) or result["nonzero_advantage_groups"] < 0:
            raise ValueError(f"{run_id}: invalid nonzero_advantage_groups")
        curve = result.get("curve")
        if not isinstance(curve, list) or len(curve) < 2:
            raise ValueError(f"{run_id}: token analysis requires at least two curve points")
        steps = []
        tokens = []
        for point in curve:
            step = int(point["step"])
            token_count = float(point["cumulative_trainable_tokens"])
            success = point["strict_success"]
            bands = point["strict_success_by_resolution"]
            if not _is_probability(success):
                raise ValueError(f"{run_id}: invalid strict_success")
            if set(bands) != {"low", "high"} or not all(
                _is_probability(value) for value in bands.values()
            ):
                raise ValueError(f"{run_id}: invalid resolution-band endpoints")
            steps.append(step)
            tokens.append(token_count)
        if any(right <= left for left, right in zip(steps, steps[1:])):
            raise ValueError(f"{run_id}: steps must be strictly increasing")
        if any(right <= left for left, right in zip(tokens, tokens[1:])):
            raise ValueError(f"{run_id}: cumulative tokens must be strictly increasing")
        if not result["catastrophic"] and steps[-1] != int(manifest["max_steps"]):
            raise ValueError(f"{run_id}: completed run misses frozen max_steps")
    return observed, evidence_root is not None


def _contrasts(values: dict[str, float]) -> dict[str, float]:
    if set(values) != set(ARMS):
        raise ValueError("contrasts require AF, LF, AN, and LN")
    return {
        "temporal_main": (values["AF"] + values["AN"] - values["LF"] - values["LN"]) / 2.0,
        "content_main": (values["AN"] + values["LN"] - values["AF"] - values["LF"]) / 2.0,
        "interaction": (values["AN"] - values["AF"]) - (values["LN"] - values["LF"]),
    }


def _bootstrap_mean(values: list[float], key: str) -> dict[str, Any]:
    if not values:
        raise ValueError("bootstrap requires values")
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    draws = sorted(
        statistics.fmean(rng.choice(values) for _ in values)
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    lower = draws[int(0.025 * (len(draws) - 1))]
    upper = draws[int(0.975 * (len(draws) - 1))]
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "ci95": [lower, upper],
    }


def _sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


def analyze_bundle(
    manifest: dict[str, Any],
    results: list[dict[str, Any]],
    evidence_root: Path | None = None,
) -> dict[str, Any]:
    """Compute block-specific seed contrasts and fail-closed route gates."""

    observed, raw_evidence_verified = validate_bundle(
        manifest, results, evidence_root=evidence_root
    )
    matching = _load_module(
        "rist_v2_1_p4_token_matching", P3_ROOT / "token_matching.py"
    )
    power = _load_module("rist_v2_1_p4_power", P3_ROOT / "power_plan.py")
    groups: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in observed.values():
        key = (str(result["family"]), str(result["algorithm"]), int(result["seed"]))
        groups[key][str(result["arm"])] = result
    if any(set(arms) != set(ARMS) for arms in groups.values()):
        raise ValueError("every family/algorithm/seed group requires all four arms")

    seed_rows = []
    token_not_estimable = []
    for (family, algorithm, seed), arms in sorted(groups.items()):
        endpoints = {
            arm: 0.0
            if row["catastrophic"]
            else float(row["confirmatory_strict_success"])
            for arm, row in arms.items()
        }
        resolution = {
            band: _contrasts(
                {
                    arm: 0.0
                    if row["catastrophic"]
                    else float(row["confirmatory_strict_success_by_resolution"][band])
                    for arm, row in arms.items()
                }
            )
            for band in ("low", "high")
        }
        curves = {
            arm: [
                {
                    "cumulative_trainable_tokens": float(point["cumulative_trainable_tokens"]),
                    "strict_success": float(point["strict_success"]),
                }
                for point in row["curve"]
            ]
            for arm, row in arms.items()
        }
        try:
            token_result = matching.match_curves(curves)
            token_contrasts = _contrasts(token_result["normalized_auc"])
        except ValueError as exc:
            token_result = None
            token_contrasts = None
            token_not_estimable.append(
                {"family": family, "algorithm": algorithm, "seed": seed, "reason": str(exc)}
            )
        seed_rows.append(
            {
                "family": family,
                "algorithm": algorithm,
                "seed": seed,
                "step": _contrasts(endpoints),
                "token_auc": token_contrasts,
                "resolution": resolution,
                "catastrophic_arms": sorted(
                    arm for arm, row in arms.items() if row["catastrophic"]
                ),
                "zero_advantage_arms": sorted(
                    arm for arm, row in arms.items() if row["nonzero_advantage_groups"] == 0
                ),
                "token_grid": None if token_result is None else token_result["grid"],
            }
        )

    block_rows = []
    for family in sorted({row["family"] for row in seed_rows}):
        for algorithm in sorted({row["algorithm"] for row in seed_rows}):
            rows = [
                row
                for row in seed_rows
                if row["family"] == family and row["algorithm"] == algorithm
            ]
            step_values = [row["step"]["interaction"] for row in rows]
            token_values = [
                row["token_auc"]["interaction"]
                for row in rows
                if row["token_auc"] is not None
            ]
            step_summary = _bootstrap_mean(step_values, f"{family}:{algorithm}:step")
            token_summary = (
                _bootstrap_mean(token_values, f"{family}:{algorithm}:token")
                if len(token_values) == len(rows)
                else None
            )
            resolution_summary = {
                band: _bootstrap_mean(
                    [row["resolution"][band]["interaction"] for row in rows],
                    f"{family}:{algorithm}:resolution:{band}",
                )
                for band in ("low", "high")
            }
            resolution_moderation = _bootstrap_mean(
                [
                    row["resolution"]["high"]["interaction"]
                    - row["resolution"]["low"]["interaction"]
                    for row in rows
                ],
                f"{family}:{algorithm}:resolution-moderation",
            )
            planning_sd = max(0.05, float(step_summary["sample_sd"]))
            required_seeds = max(
                MIN_CONFIRMATORY_SEEDS,
                power.required_paired_seeds(MIN_INTERACTION, planning_sd),
            )
            step_sign = _sign(float(step_summary["mean"]))
            token_sign = 0 if token_summary is None else _sign(float(token_summary["mean"]))
            block_rows.append(
                {
                    "family": family,
                    "algorithm": algorithm,
                    "seed_count": len(rows),
                    "step_interaction": step_summary,
                    "token_auc_interaction": token_summary,
                    "resolution_interaction": resolution_summary,
                    "resolution_moderation_high_minus_low": resolution_moderation,
                    "required_paired_seeds": required_seeds,
                    "effect_threshold_pass": abs(float(step_summary["mean"])) >= MIN_INTERACTION,
                    "step_ci_excludes_zero": step_summary["ci95"][0] > 0.0 or step_summary["ci95"][1] < 0.0,
                    "token_sign_consistent": token_summary is not None and step_sign != 0 and step_sign == token_sign,
                    "power_pass": len(rows) >= required_seeds,
                }
            )

    total_runs = len(results)
    catastrophic_runs = sum(bool(row["catastrophic"]) for row in results)
    zero_advantage_runs = sum(row["nonzero_advantage_groups"] == 0 for row in results)
    signs = {_sign(float(row["step_interaction"]["mean"])) for row in block_rows}
    cross_block_sign_consistent = len(signs) == 1 and 0 not in signs
    failure_gates = {
        "catastrophic_run_rate": catastrophic_runs / total_runs,
        "zero_advantage_run_rate": zero_advantage_runs / total_runs,
        "catastrophic_rate_pass": catastrophic_runs / total_runs <= MAX_FAILURE_RATE,
        "zero_advantage_rate_pass": zero_advantage_runs / total_runs <= MAX_FAILURE_RATE,
    }
    treatment_ready = all(
        row.get("scientific_treatment_ready") is True for row in manifest["runs"]
    )
    execution_authority_recorded = (
        manifest.get("execution_authorized") is True
        and manifest.get("commands_are_templates_only") is False
    )
    main_track_eligible = (
        not token_not_estimable
        and raw_evidence_verified
        and treatment_ready
        and execution_authority_recorded
        and cross_block_sign_consistent
        and all(
            row["effect_threshold_pass"]
            and row["step_ci_excludes_zero"]
            and row["token_sign_consistent"]
            and row["power_pass"]
            for row in block_rows
        )
        and failure_gates["catastrophic_rate_pass"]
        and failure_gates["zero_advantage_rate_pass"]
    )
    return {
        "protocol": "RIST-P4-v2.1",
        "statistical_unit": "paired_training_seed_within_family_algorithm",
        "run_count": total_runs,
        "seed_rows": seed_rows,
        "blocks": block_rows,
        "cross_block_step_sign_consistent": cross_block_sign_consistent,
        "token_not_estimable": token_not_estimable,
        "failure_gates": failure_gates,
        "raw_evidence_files_verified": raw_evidence_verified,
        "scientific_treatment_ready": treatment_ready,
        "execution_authority_recorded": execution_authority_recorded,
        "main_track_eligible": main_track_eligible,
        "three_seed_pilot_only": all(row["seed_count"] == 3 for row in block_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = [
        json.loads(line)
        for line in args.results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    analysis = analyze_bundle(manifest, results, evidence_root=args.evidence_root)
    args.output.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if analysis["main_track_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

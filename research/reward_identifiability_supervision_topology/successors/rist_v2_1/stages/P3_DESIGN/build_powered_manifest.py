"""Convert a passed three-seed pilot into a frozen powered expansion template."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PILOT_SEEDS = (7101, 7202, 7303)
PAIRED_SEED_BANK = tuple(7101 + 101 * index for index in range(32))
MIN_POWERED_SEEDS = 8
MAX_POWERED_SEEDS = len(PAIRED_SEED_BANK)


def _replace_flag(command: list[str], flag: str, value: str) -> list[str]:
    result = list(command)
    index = result.index(flag)
    result[index + 1] = value
    return result


def _expanded_command(template: list[str], old_run_id: str, run_id: str, seed: int) -> list[str]:
    command = _replace_flag(template, "--seed", str(seed))
    return [value.replace(old_run_id, run_id) for value in command]


def _pilot_gate(analysis: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    if analysis.get("protocol") != "RIST-P4-v2.1":
        reasons.append("UNEXPECTED_PILOT_ANALYSIS")
    if analysis.get("three_seed_pilot_only") is not True:
        reasons.append("NOT_EXACT_THREE_SEED_PILOT")
    if analysis.get("raw_evidence_files_verified") is not True:
        reasons.append("RAW_EVIDENCE_NOT_VERIFIED")
    if analysis.get("scientific_treatment_ready") is not True:
        reasons.append("TREATMENT_NOT_READY")
    if analysis.get("execution_authority_recorded") is not True:
        reasons.append("PILOT_EXECUTION_NOT_AUTHORIZED")
    if analysis.get("token_not_estimable"):
        reasons.append("TOKEN_MATCHED_NOT_ESTIMABLE")
    if analysis.get("stable_interaction_across_seeds_and_blocks") is not True:
        reasons.append("INTERACTION_NOT_STABLE_IN_PILOT")
    blocks = analysis.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 4:
        reasons.append("FOUR_BLOCK_ANALYSIS_MISSING")
    else:
        if any(block.get("effect_threshold_pass") is not True for block in blocks):
            reasons.append("MINIMUM_EFFECT_NOT_MET")
        if any(block.get("token_sign_consistent") is not True for block in blocks):
            reasons.append("TOKEN_SIGN_REVERSAL")
    failure = analysis.get("failure_gates") or {}
    if failure.get("catastrophic_rate_pass") is not True:
        reasons.append("CATASTROPHIC_RATE_TOO_HIGH")
    if failure.get("zero_advantage_rate_pass") is not True:
        reasons.append("ZERO_ADVANTAGE_RATE_TOO_HIGH")
    return not reasons, reasons


def build_powered_manifest(
    pilot_manifest: dict[str, Any],
    pilot_analysis: dict[str, Any],
    pilot_analysis_sha256: str,
) -> dict[str, Any]:
    if pilot_manifest.get("protocol") != "RIST-P3-DIAGNOSTIC-v2.1":
        raise ValueError("unexpected pilot execution manifest")
    pilot_runs = pilot_manifest.get("runs")
    if not isinstance(pilot_runs, list) or len(pilot_runs) != 48:
        raise ValueError("powered expansion requires the exact 48-run pilot")
    if {int(row["seed"]) for row in pilot_runs} != set(PILOT_SEEDS):
        raise ValueError("pilot seed set differs from the frozen first three seeds")
    if len(pilot_analysis_sha256) != 64:
        raise ValueError("pilot analysis SHA256 is required")
    try:
        int(pilot_analysis_sha256, 16)
    except ValueError as exc:
        raise ValueError("pilot analysis SHA256 is invalid") from exc
    gate_pass, reasons = _pilot_gate(pilot_analysis)
    if not gate_pass:
        return {
            "protocol": "RIST-P3-POWERED-EXPANSION-v1",
            "decision": "KILL_POWERED_EXPANSION",
            "reasons": reasons,
            "pilot_analysis_sha256": pilot_analysis_sha256,
            "execution_authorized": False,
            "run_count": 0,
            "runs": [],
        }
    required = max(
        MIN_POWERED_SEEDS,
        max(int(block["required_paired_seeds"]) for block in pilot_analysis["blocks"]),
    )
    if required > MAX_POWERED_SEEDS:
        return {
            "protocol": "RIST-P3-POWERED-EXPANSION-v1",
            "decision": "KILL_POWER_EXCEEDS_FROZEN_SEED_BANK",
            "required_paired_seeds": required,
            "maximum_paired_seeds": MAX_POWERED_SEEDS,
            "pilot_analysis_sha256": pilot_analysis_sha256,
            "execution_authorized": False,
            "run_count": 0,
            "runs": [],
        }
    templates = {
        (row["family"], row["algorithm"], row["arm"]): row
        for row in pilot_runs
        if int(row["seed"]) == PILOT_SEEDS[0]
    }
    if len(templates) != 16:
        raise ValueError("pilot manifest lacks one template per factorial cell")
    selected_seeds = PAIRED_SEED_BANK[:required]
    runs = []
    for key in sorted(templates):
        template = templates[key]
        for seed in selected_seeds:
            run_id = f"{template['family']}-{template['algorithm']}-{template['arm']}-{seed}"
            runs.append(
                {
                    **{field: value for field, value in template.items() if field not in {"run_id", "seed", "command", "required_environment"}},
                    "run_id": run_id,
                    "seed": seed,
                    "command": _expanded_command(
                        template["command"], template["run_id"], run_id, seed
                    ),
                    "required_environment": {
                        name: value.replace(template["run_id"], run_id)
                        for name, value in template["required_environment"].items()
                    },
                    "pilot_run_already_observed": seed in PILOT_SEEDS,
                    "execution_authorized": False,
                }
            )
    return {
        "protocol": "RIST-P3-POWERED-EXPANSION-v1",
        "decision": "GO_POWERED_EXPANSION_AFTER_SEPARATE_AUTHORIZATION",
        "pilot_analysis_sha256": pilot_analysis_sha256,
        "seed_bank": list(PAIRED_SEED_BANK),
        "selected_seeds": list(selected_seeds),
        "required_paired_seeds": required,
        "additional_paired_seeds": required - len(PILOT_SEEDS),
        "run_count": len(runs),
        "additional_run_count": 16 * (required - len(PILOT_SEEDS)),
        "execution_authorized": False,
        "commands_are_templates_only": True,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--pilot-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis_bytes = args.pilot_analysis.read_bytes()
    result = build_powered_manifest(
        json.loads(args.pilot_manifest.read_text()),
        json.loads(analysis_bytes),
        hashlib.sha256(analysis_bytes).hexdigest(),
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["decision"].startswith("GO_") else 2


if __name__ == "__main__":
    raise SystemExit(main())

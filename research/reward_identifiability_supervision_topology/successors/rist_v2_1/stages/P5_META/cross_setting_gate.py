"""Fail-closed cross-setting gate for synthetic, Tau3, and sealed BFCL results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

SETTINGS = ("rist_synthetic", "tau3", "bfcl_sealed")


def _sign(value: float) -> int:
    return (value > 0.0) - (value < 0.0)


def evaluate_cross_setting(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Evaluate the frozen paper-level transport requirements."""

    if set(summaries) != set(SETTINGS):
        raise ValueError("cross-setting gate requires synthetic, Tau3, and sealed BFCL")
    rows = []
    for setting in SETTINGS:
        summary = summaries[setting]
        estimate = float(summary["interaction_estimate"])
        interval = [float(value) for value in summary["ci95"]]
        if not math.isfinite(estimate) or len(interval) != 2 or not all(
            math.isfinite(value) for value in interval
        ):
            raise ValueError(f"{setting}: non-finite estimate or interval")
        if interval[0] > interval[1]:
            raise ValueError(f"{setting}: reversed interval")
        families = sorted({str(value) for value in summary["families"]})
        algorithms = sorted({str(value) for value in summary["algorithms"]})
        token_required = setting != "bfcl_sealed"
        row = {
            "setting": setting,
            "completed": summary.get("completed") is True,
            "interaction_estimate": estimate,
            "ci95": interval,
            "ci_excludes_zero": interval[0] > 0.0 or interval[1] < 0.0,
            "sign": _sign(estimate),
            "families": families,
            "algorithms": algorithms,
            "two_families_pass": len(families) >= 2,
            "two_algorithms_pass": len(algorithms) >= 2,
            "token_required": token_required,
            "token_robustness_pass": (
                summary.get("token_sign_consistent") is True if token_required else True
            ),
            "catastrophic_rate_pass": float(summary["catastrophic_run_rate"]) <= 0.10,
            "raw_evidence_complete": summary.get("raw_evidence_complete") is True,
            "powered": summary.get("powered") is True,
            "frozen_analysis_hash_present": isinstance(
                summary.get("analysis_protocol_sha256"), str
            )
            and len(summary["analysis_protocol_sha256"]) == 64,
            "environment_qualification_pass": (
                summary.get("environment_qualification_pass") is True
                if setting == "tau3"
                else True
            ),
            "sealed_once_pass": (
                summary.get("sealed_once") is True if setting == "bfcl_sealed" else True
            ),
        }
        rows.append(row)
    signs = {row["sign"] for row in rows}
    sign_consistent = len(signs) == 1 and 0 not in signs
    passed = sign_consistent and all(
        row["completed"]
        and row["ci_excludes_zero"]
        and row["two_families_pass"]
        and row["two_algorithms_pass"]
        and row["token_robustness_pass"]
        and row["catastrophic_rate_pass"]
        and row["raw_evidence_complete"]
        and row["powered"]
        and row["frozen_analysis_hash_present"]
        and row["environment_qualification_pass"]
        and row["sealed_once_pass"]
        for row in rows
    )
    return {
        "protocol": "RIST-P5-v2.1",
        "settings": rows,
        "interaction_sign_consistent": sign_consistent,
        "main_track_eligible": passed,
        "decision": "GO_MAIN_TRACK" if passed else "DO_NOT_UPGRADE_MAIN_TRACK",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = json.loads(args.summaries.read_text(encoding="utf-8"))
    result = evaluate_cross_setting(summaries)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if result["main_track_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

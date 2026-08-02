"""Reconcile central P6 manuscript numbers against shipped JSON evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_claims(repo_root: Path, output: Path | None = None) -> dict[str, Any]:
    base = repo_root / "research/silent_reward_contracts"
    tex = (base / "p6/paper/main.tex").read_text(encoding="utf-8")
    tex_plain = tex.replace("\\_", "_")
    dev = _load(base / "p3/artifacts/dev/metrics.json")
    dev_cases = _load(base / "p3/dev_cases.json")["cases"]
    heldout = _load(base / "p3/artifacts/heldout/metrics.json")
    heldout_cases = _load(base / "p3/heldout_cases.json")["cases"]
    p5 = _load(base / "p5/artifacts/external_metrics.json")
    p5_cases = _load(base / "p5/artifacts/external_cases.json")["cases"]
    p4 = _load(base / "p4/gpu_gate_decision.json")

    heldout_clean = sum(not case["expected_violations"] for case in heldout_cases)
    heldout_mutation = len(heldout_cases) - heldout_clean
    dev_clean = sum(not case["expected_violations"] for case in dev_cases)
    p5_clean = sum(not case["expected_violations"] for case in p5_cases)
    p5_natural = sum(bool(case["natural_case"]) for case in p5_cases)
    number_words = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")

    expected = {
        "development_case_count": str(dev["case_count"]),
        "development_clean_fpr": f"0/{dev_clean}",
        "heldout_partition": f"{heldout_clean} source-derived clean controls and {heldout_mutation}",
        "heldout_macro_recall": f"macro recall {heldout['macro_recall']:.2f}",
        "heldout_clean_fpr": f"{int(heldout['clean_false_positive_rate'] * heldout_clean)}/{heldout_clean}",
        "heldout_upper_wilson": f"{heldout['clean_false_positive_wilson_95'][1]:.4f}",
        "p5_partition": f"{number_words[p5_natural]} natural cases and {number_words[p5_clean]} clean",
        "p5_exact": f"all {p5['case_count']} cases",
        "p5_clean_fpr": f"0/{p5_clean}",
        "p5_upper_wilson": f"{p5['clean_false_positive_wilson_95'][1]:.4f}",
        "p4_status": str(p4["decision"]),
        "p4_scientific_status": str(p4["scientific_outcome"]),
    }
    missing = {name: value for name, value in expected.items() if value not in tex_plain}
    result = {
        "schema_version": "arca.paper-claim-audit.v1",
        "protocol_id": "ARCA-P6-PAPER-v0.1",
        "expected_claims": expected,
        "missing_claims": missing,
        "status": "PASS_PAPER_CLAIM_AUDIT" if not missing else "FAIL_PAPER_CLAIM_AUDIT",
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if missing:
        raise ValueError(f"manuscript claim mismatch: {missing}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_claims(args.repo_root.resolve(), args.output), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluate frozen ARCA rules on development or held-out case files."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import time
from typing import Any

from research.silent_reward_contracts.arca import RULE_CODES, audit_case


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 1.0]
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def evaluate(payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    rows = []
    by_rule = defaultdict(lambda: {"tp": 0, "fn": 0})
    clean_total = 0
    clean_false_positive = 0
    high_natural_total = 0
    high_natural_detected = 0
    natural_flips = 0
    for case in payload["cases"]:
        expected = set(case["expected_violations"])
        predicted = set(audit_case(case))
        for rule in expected:
            by_rule[rule]["tp" if rule in predicted else "fn"] += 1
        if not expected:
            clean_total += 1
            clean_false_positive += bool(predicted)
        natural = bool(case.get("natural_case"))
        if natural and case.get("severity") == "high":
            high_natural_total += 1
            high_natural_detected += expected.issubset(predicted)
        if natural and case.get("conclusion_flip") and expected.issubset(predicted):
            natural_flips += 1
        rows.append(
            {
                "case_id": case["case_id"],
                "framework": case["framework"],
                "workload": case["workload"],
                "seed": case["seed"],
                "natural_case": natural,
                "expected": ";".join(sorted(expected)),
                "predicted": ";".join(sorted(predicted)),
                "exact_match": expected == predicted,
            }
        )

    recalls = []
    rule_metrics = {}
    for rule in RULE_CODES:
        counts = by_rule[rule]
        total = counts["tp"] + counts["fn"]
        if total:
            recall = counts["tp"] / total
            recalls.append(recall)
            rule_metrics[rule] = {
                **counts,
                "recall": recall,
                "wilson_95": _wilson(counts["tp"], total),
            }
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    fpr = clean_false_positive / clean_total if clean_total else 0.0
    elapsed = time.perf_counter() - started
    metrics = {
        "schema_version": "arca.evaluation.v1",
        "split": payload["split"],
        "case_count": len(rows),
        "macro_recall": macro_recall,
        "clean_false_positive_rate": fpr,
        "clean_false_positive_wilson_95": _wilson(clean_false_positive, clean_total),
        "high_natural_recall": (
            high_natural_detected / high_natural_total
            if high_natural_total
            else None
        ),
        "natural_conclusion_flips_detected": natural_flips,
        "wall_time_limit_seconds": 60.0,
        "wall_time_under_limit": elapsed <= 60.0,
        "rule_metrics": rule_metrics,
    }
    return metrics, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "heldout"), required=True)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cases_path = args.cases or root / "p3" / f"{args.split}_cases.json"
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    if payload["split"] != args.split:
        raise ValueError("case-file split does not match --split")
    metrics, rows = evaluate(payload)
    if args.score_only:
        score = metrics["macro_recall"] if metrics["clean_false_positive_rate"] <= 0.05 else 0.0
        print(f"{score:.12f}")
        return 0
    output_dir = args.output_dir or root / "p3" / "artifacts" / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "cases.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

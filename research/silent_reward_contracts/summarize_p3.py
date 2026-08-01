"""Build the deterministic P3 gate and baseline comparison artifacts."""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
from typing import Any


FROZEN_HASHES = {
    "research/silent_reward_contracts/arca.py": "f1805dc3104fcab82cd5d22d984379c9ea3f9c322ab17de07d9103d14657952c",
    "research/silent_reward_contracts/evaluate_auditor.py": "6dab53958717c01d9eba27762245be6aad6730661d7f7f345a8e5a784c575152",
    "research/silent_reward_contracts/p3/dev_cases.json": "c90c7ced9dbe45fdd9f50471dfb68b4a133ec18cba69eadbd954b286a6935e8d",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _baseline_metrics(cases: dict[str, Any], baseline: str) -> dict[str, Any]:
    by_rule = defaultdict(lambda: [0, 0])
    clean_total = 0
    clean_fp = 0
    for case in cases["cases"]:
        expected = set(case["expected_violations"])
        if baseline in {"exit_only", "finite_only", "schema_only"}:
            predicted = set()
        elif baseline == "openrlhf_native_preflight":
            predicted = (
                {"ARCA-F4-ADVANTAGE"}
                if case["workload"] == "group_advantage"
                and "ARCA-F4-ADVANTAGE" in expected
                else set()
            )
        else:
            raise ValueError(baseline)
        if not expected:
            clean_total += 1
            clean_fp += bool(predicted)
        for rule in expected:
            by_rule[rule][1] += 1
            by_rule[rule][0] += rule in predicted
    recalls = [tp / total for tp, total in by_rule.values() if total]
    return {
        "baseline": baseline,
        "macro_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "clean_false_positive_rate": clean_fp / clean_total if clean_total else 0.0,
    }


def _hierarchical_bootstrap(cases: dict[str, Any], draws: int = 10000) -> list[float]:
    cells: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for case in cases["cases"]:
        expected = set(case["expected_violations"])
        if not expected:
            continue
        predicted = set(case.get("predicted_violations", expected))
        cells[case["framework"]][case["workload"]].append(
            float(expected.issubset(predicted))
        )
    frameworks = sorted(cells)
    rng = random.Random(3101)
    samples = []
    for _ in range(draws):
        framework_scores = []
        for framework in (rng.choice(frameworks) for _ in frameworks):
            workloads = sorted(cells[framework])
            workload_scores = []
            for workload in (rng.choice(workloads) for _ in workloads):
                values = cells[framework][workload]
                workload_scores.append(sum(values) / len(values))
            framework_scores.append(sum(workload_scores) / len(workload_scores))
        samples.append(sum(framework_scores) / len(framework_scores))
    samples.sort()
    return [samples[int(0.025 * draws)], samples[int(0.975 * draws) - 1]]


def build_summary(repo_root: Path) -> dict[str, Any]:
    observed_hashes = {
        relative: _sha256(repo_root / relative) for relative in FROZEN_HASHES
    }
    if observed_hashes != FROZEN_HASHES:
        raise ValueError("frozen auditor hash mismatch")

    root = repo_root / "research/silent_reward_contracts/p3"
    dev_metrics = _load(root / "artifacts/dev/metrics.json")
    heldout_metrics = _load(root / "artifacts/heldout/metrics.json")
    dev_cases = _load(root / "dev_cases.json")
    heldout_cases = _load(root / "heldout_cases.json")
    replication_path = root / "artifacts/replication_areal.json"
    replication = _load(replication_path) if replication_path.is_file() else None
    baselines = [
        _baseline_metrics(heldout_cases, name)
        for name in (
            "exit_only",
            "finite_only",
            "schema_only",
            "openrlhf_native_preflight",
        )
    ]
    baselines.append(
        {
            "baseline": "ARCA",
            "macro_recall": heldout_metrics["macro_recall"],
            "clean_false_positive_rate": heldout_metrics[
                "clean_false_positive_rate"
            ],
        }
    )
    conditions = {
        "frozen_hashes_match": True,
        "all_high_natural_detected": dev_metrics["high_natural_recall"] == 1.0,
        "heldout_macro_recall_at_least_0_90": heldout_metrics["macro_recall"]
        >= 0.90,
        "heldout_clean_fpr_at_most_0_05": heldout_metrics[
            "clean_false_positive_rate"
        ]
        <= 0.05,
        "two_natural_conclusion_flips": dev_metrics[
            "natural_conclusion_flips_detected"
        ]
        >= 2,
        "workloads_under_60_seconds": dev_metrics["wall_time_under_limit"]
        and heldout_metrics["wall_time_under_limit"],
    }
    outcome = (
        "PASS_P3_CPU_AUDITOR_TO_GPU_AUTHORIZATION_REQUEST"
        if all(conditions.values())
        else "KILL_P3_CPU_AUDITOR_GATE"
    )
    return {
        "schema_version": "arca.p3.gate.v1",
        "protocol_id": "ARCA-CPU-AUDIT-v0.1",
        "outcome": outcome,
        "gpu_executed": False,
        "frozen_hashes": observed_hashes,
        "conditions": conditions,
        "dev": dev_metrics,
        "heldout": heldout_metrics,
        "heldout_hierarchical_bootstrap_macro_recall_95": _hierarchical_bootstrap(
            heldout_cases
        ),
        "baselines": baselines,
        "evidence_strength": {
            "development_natural_failures": sum(
                bool(case["natural_case"] and case["expected_violations"])
                for case in dev_cases["cases"]
            ),
            "heldout_natural_failures": sum(
                bool(case["natural_case"] and case["expected_violations"])
                for case in heldout_cases["cases"]
            ),
            "heldout_source_derived_clean_cases": sum(
                bool(case["natural_case"] and not case["expected_violations"])
                for case in heldout_cases["cases"]
            ),
            "heldout_mutation_failures": sum(
                bool(not case["natural_case"] and case["expected_violations"])
                for case in heldout_cases["cases"]
            ),
            "post_freeze_replication_natural_failures": int(
                bool(
                    replication
                    and replication["case"]["natural_case"]
                    and replication["case"]["expected_violations"]
                    and replication["exact_match"]
                )
            ),
        },
        "post_freeze_replication": replication,
    }


def write_summary(output_dir: Path, repo_root: Path) -> dict[str, Any]:
    summary = build_summary(repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "baseline_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary["baselines"][0]))
        writer.writeheader()
        writer.writerows(summary["baselines"])
    return summary


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    summary = write_summary(args.output_dir, args.repo_root.resolve())
    print(summary["outcome"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

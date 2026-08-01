"""Build the frozen development cases and registered single-fault mutations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "arca.cases.v1"


def _case(
    case_id: str,
    framework: str,
    workload: str,
    seed: int,
    payload: dict[str, Any],
    expected: list[str],
    *,
    natural: bool = False,
    severity: str = "synthetic",
    conclusion_flip: bool = False,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "framework": framework,
        "workload": workload,
        "seed": seed,
        "payload": payload,
        "expected_violations": expected,
        "natural_case": natural,
        "severity": severity,
        "conclusion_flip": conclusion_flip,
    }


def build_dev_cases() -> dict[str, Any]:
    cases = [
        _case(
            "natural-areno-argument-type",
            "AReno",
            "typed_arguments",
            0,
            {"argument_variants": [{"bit": 1}, '"not-an-object"']},
            ["ARCA-F1-TYPE"],
            natural=True,
            severity="high",
            conclusion_flip=True,
        ),
        _case(
            "natural-verl-tool-reward-key",
            "veRL",
            "reward_flow",
            0,
            {
                "intended_reward_keys": ["tool_rewards"],
                "produced_reward_keys": ["tool_rewards"],
                "consumed_reward_keys": ["reward_scores"],
            },
            ["ARCA-F2-REWARD-FLOW"],
            natural=True,
            severity="high",
            conclusion_flip=True,
        ),
        _case(
            "natural-verl-timeout-zero",
            "veRL",
            "failure_status",
            0,
            {
                "execution_failures": [
                    {
                        "kind": "TimeoutError",
                        "numeric_reward": 0.0,
                        "structured_status": None,
                    }
                ]
            },
            ["ARCA-F6-STATUS"],
            natural=True,
            severity="medium",
        ),
    ]

    clean_payloads = {
        "typed_arguments": {
            "argument_variants": [{"bit": 1}, '{"bit": 1}']
        },
        "trace_pairing": {"call_ids": ["c1"], "result_call_ids": ["c1"]},
        "reward_flow": {
            "intended_reward_keys": ["reward"],
            "produced_reward_keys": ["reward"],
            "consumed_reward_keys": ["reward"],
        },
        "reward_signal": {"reward_controls": [0.0, 1.0]},
        "group_advantage": {"group_rewards": [[0.0, 1.0], [1.0, 0.0]]},
        "treatments": {
            "treatment_masks": {"a": [1, 0], "b": [1, 0]},
            "declared_aliases": [["a", "b"]],
        },
        "failure_status": {
            "execution_failures": [
                {
                    "kind": "TimeoutError",
                    "numeric_reward": 0.0,
                    "structured_status": "timeout",
                }
            ]
        },
        "provenance": {
            "provenance": {
                "required": {"commit": "abc"},
                "observed": {"commit": "abc"},
            }
        },
    }
    fault_payloads = {
        "typed_arguments": (
            {"argument_variants": ["{"]},
            "ARCA-F1-TYPE",
        ),
        "trace_pairing": (
            {"call_ids": ["c1"], "result_call_ids": []},
            "ARCA-F2-PAIRING",
        ),
        "reward_flow": (
            {
                "intended_reward_keys": ["reward"],
                "produced_reward_keys": ["reward"],
                "consumed_reward_keys": ["score"],
            },
            "ARCA-F2-REWARD-FLOW",
        ),
        "reward_signal": (
            {"reward_controls": [0.0, 0.0]},
            "ARCA-F3-INFORMATIVE",
        ),
        "group_advantage": (
            {"group_rewards": [[1.0], [0.0]]},
            "ARCA-F4-ADVANTAGE",
        ),
        "treatments": (
            {
                "treatment_masks": {"a": [1, 0], "b": [1, 0]},
                "declared_aliases": [],
            },
            "ARCA-F5-TREATMENT",
        ),
        "failure_status": (
            {
                "execution_failures": [
                    {
                        "kind": "TimeoutError",
                        "numeric_reward": 0.0,
                        "structured_status": None,
                    }
                ]
            },
            "ARCA-F6-STATUS",
        ),
        "provenance": (
            {
                "provenance": {
                    "required": {"commit": "abc"},
                    "observed": {"commit": "def"},
                }
            },
            "ARCA-F6-PROVENANCE",
        ),
    }
    for seed in range(10):
        for workload, payload in clean_payloads.items():
            cases.append(
                _case(
                    f"clean-{workload}-{seed}",
                    "registered-fixture",
                    workload,
                    seed,
                    payload,
                    [],
                )
            )
        for workload, (payload, code) in fault_payloads.items():
            cases.append(
                _case(
                    f"mutation-{workload}-{seed}",
                    "registered-fixture",
                    workload,
                    seed,
                    payload,
                    [code],
                )
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "split": "dev",
        "mutation_policy": "single_fault_only",
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(build_dev_cases(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Held-out OpenRLHF source adapter; does not alter frozen ARCA rules."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any


OPENRLHF_COMMIT = "bc71bb19464aca306b33080b2d2bb45d154e2f49"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _keys(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    keys = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            value = node.slice
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                keys.add(value.value)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "get" and node.args:
                value = node.args[0]
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    keys.add(value.value)
        elif isinstance(node, ast.Dict):
            for value in node.keys:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    keys.add(value.value)
    return keys


def inspect_openrlhf(root: Path) -> dict[str, Any]:
    files = {
        "agent": root / "openrlhf/utils/agent.py",
        "sample_generator": root
        / "openrlhf/trainer/ppo_utils/samples_generator.py",
        "cli": root / "openrlhf/cli/train_ppo_ray.py",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing held-out source files: {missing}")
    agent_keys = _keys(files["agent"])
    sample_keys = _keys(files["sample_generator"])
    cli_text = files["cli"].read_text(encoding="utf-8")
    return {
        "commit": OPENRLHF_COMMIT,
        "source_hashes": {
            name: _sha256(path) for name, path in sorted(files.items())
        },
        "facts": {
            "step_reward_key": "rewards" if "rewards" in agent_keys else None,
            "executor_reward_key": "reward" if "reward" in agent_keys else None,
            "sample_consumer_reward_key": (
                "reward" if "reward" in sample_keys else None
            ),
            "group_estimators_require_multiple_samples": (
                "requires n_samples_per_prompt > 1" in cli_text
            ),
            "remote_reward_failure_stays_none": (
                '"reward": None' in files["agent"].read_text(encoding="utf-8")
            ),
        },
    }


def _case(
    case_id: str,
    workload: str,
    seed: int,
    payload: dict[str, Any],
    expected: list[str],
    *,
    natural: bool,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "framework": "OpenRLHF",
        "workload": workload,
        "seed": seed,
        "payload": payload,
        "expected_violations": expected,
        "natural_case": natural,
        "severity": "clean_source" if natural else "synthetic",
        "conclusion_flip": False,
    }


def build_heldout_cases(root: Path) -> dict[str, Any]:
    inspection = inspect_openrlhf(root)
    facts = inspection["facts"]
    if facts != {
        "step_reward_key": "rewards",
        "executor_reward_key": "reward",
        "sample_consumer_reward_key": "reward",
        "group_estimators_require_multiple_samples": True,
        "remote_reward_failure_stays_none": True,
    }:
        raise ValueError(f"held-out source adapter assumptions failed: {facts}")

    cases = []
    for seed in range(10):
        cases.extend(
            [
                _case(
                    f"heldout-clean-reward-flow-{seed}",
                    "reward_flow",
                    seed,
                    {
                        "intended_reward_keys": ["reward"],
                        "produced_reward_keys": ["reward"],
                        "consumed_reward_keys": ["reward"],
                    },
                    [],
                    natural=True,
                ),
                _case(
                    f"heldout-mutation-reward-flow-{seed}",
                    "reward_flow",
                    seed,
                    {
                        "intended_reward_keys": ["reward"],
                        "produced_reward_keys": ["reward"],
                        "consumed_reward_keys": ["score"],
                    },
                    ["ARCA-F2-REWARD-FLOW"],
                    natural=False,
                ),
                _case(
                    f"heldout-clean-group-advantage-{seed}",
                    "group_advantage",
                    seed,
                    {"group_rewards": [[0.0, 1.0]]},
                    [],
                    natural=True,
                ),
                _case(
                    f"heldout-mutation-group-advantage-{seed}",
                    "group_advantage",
                    seed,
                    {"group_rewards": [[1.0]]},
                    ["ARCA-F4-ADVANTAGE"],
                    natural=False,
                ),
                _case(
                    f"heldout-clean-treatment-{seed}",
                    "treatment_identity",
                    seed,
                    {
                        "treatment_masks": {"arm-a": [1, 0], "arm-b": [0, 1]},
                        "declared_aliases": [],
                    },
                    [],
                    natural=True,
                ),
                _case(
                    f"heldout-mutation-treatment-{seed}",
                    "treatment_identity",
                    seed,
                    {
                        "treatment_masks": {"arm-a": [1, 0], "arm-b": [1, 0]},
                        "declared_aliases": [],
                    },
                    ["ARCA-F5-TREATMENT"],
                    natural=False,
                ),
                _case(
                    f"heldout-clean-failure-status-{seed}",
                    "failure_status",
                    seed,
                    {
                        "execution_failures": [
                            {
                                "kind": "remote_reward_failure",
                                "numeric_reward": None,
                                "structured_status": None,
                            }
                        ]
                    },
                    [],
                    natural=True,
                ),
                _case(
                    f"heldout-mutation-failure-status-{seed}",
                    "failure_status",
                    seed,
                    {
                        "execution_failures": [
                            {
                                "kind": "remote_reward_failure",
                                "numeric_reward": 0.0,
                                "structured_status": None,
                            }
                        ]
                    },
                    ["ARCA-F6-STATUS"],
                    natural=False,
                ),
            ]
        )
    return {
        "schema_version": "arca.cases.v1",
        "split": "heldout",
        "framework_inspection": inspection,
        "mutation_policy": "single_fault_only",
        "cases": cases,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--openrlhf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_heldout_cases(args.openrlhf_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

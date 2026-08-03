"""Build the unauthorized one-seed Tau3 airline factorial pilot manifest."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

DESIGN_PATH = Path(__file__).parents[1] / "P3_DESIGN" / "design_matrix.py"
TRAIN_PATH = (
    "research/reward_identifiability_supervision_topology/successors/rist_v2_1/"
    "stages/X3_TAU3/data/train.jsonl"
)
TRAIN_SHA256 = "1a3f762af0cff8a3920dd5c80204bf79e433e6442321699244ea24d5ba5815bf"
PILOT_SEED = 7101
PILOT_STEPS = 25
REPO_ROOT = Path(__file__).resolve().parents[6]


def _load_design():
    spec = importlib.util.spec_from_file_location("rist_x3_design", DESIGN_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("P3 design module unavailable")
    spec.loader.exec_module(module)
    return module


def _command(row: dict[str, Any], run_root: Path) -> list[str]:
    run_id = f"tau3-{row['family']}-{row['algorithm']}-{row['arm']}-{PILOT_SEED}"
    return [
        "areno",
        "train",
        "--ckpt",
        "{QWEN3_MODEL_PATH}" if row["family"] == "qwen3" else "{GEMMA4_MODEL_PATH}",
        "--dataset-path",
        TRAIN_PATH,
        "--dataset-loader-fn",
        "examples/agentic/rist_v2_1_tau3/dataset_loader.py",
        "--reward-fn-path",
        "examples/agentic/rist_v2_1_tau3/reward.py",
        "--agent-fn",
        "examples/agentic/rist_v2_1_tau3/run_agent.py",
        "--algo",
        str(row["algorithm"]),
        "--seed",
        str(PILOT_SEED),
        "--tp-size",
        "1",
        "--world-size",
        "1",
        "--batch-size",
        "1",
        "--n-samples",
        "8",
        "--mini-bs",
        "8",
        "--max-running-prompts",
        "8",
        "--max-prompt-tokens",
        "4096",
        "--max-new-tokens",
        "512",
        "--max-context-len",
        "16384",
        "--agent-timeout-s",
        "1800",
        "--attn-backend",
        "native",
        "--drop-rollout-state",
        "--save-path",
        str(run_root / run_id / "checkpoints"),
        "--save-interval",
        str(PILOT_STEPS),
        "--max-steps",
        str(PILOT_STEPS),
        "--metrics-log-dir",
        str(run_root / run_id / "metrics"),
        "--trainable-turns",
        str(row["trainable_turns"]),
        "--tool-call-supervision",
        str(row["tool_call_supervision"]),
    ]


def build_manifest(run_root: Path) -> dict[str, Any]:
    design = _load_design()
    rows = [row for row in design.build_matrix() if row["seed"] == PILOT_SEED]
    jobs = []
    for row in rows:
        run_id = f"tau3-{row['family']}-{row['algorithm']}-{row['arm']}-{PILOT_SEED}"
        jobs.append(
            {
                **row,
                "run_id": run_id,
                "dataset": TRAIN_PATH,
                "dataset_sha256": TRAIN_SHA256,
                "max_steps": PILOT_STEPS,
                "expected_episode_count": PILOT_STEPS * 8,
                "command_template": _command(row, run_root),
                "required_environment": {
                    "RIST_REQUIRE_EVIDENCE_JOURNALS": "1",
                    "RIST_TAU3_USER_LLM": "{FROZEN_USER_SIMULATOR_REVISION}",
                    "RIST_RAW_JOURNAL_PATH": str(run_root / run_id / "raw_events.jsonl"),
                    "RIST_REWARD_JOURNAL_PATH": str(run_root / run_id / "reward_events.jsonl"),
                },
                "required_artifacts": [
                    "raw_events.jsonl",
                    "reward_events.jsonl",
                    "metrics_manifest.json",
                    "checkpoint_manifest.json",
                    "run_evidence.json",
                ],
                "execution_authorized": False,
            }
        )
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "protocol": "RIST-X3-TAU3-PILOT-v2.1",
        "source_commit": source_commit,
        "scope": "AIRLINE_ONLY_ONE_SEED_FACTORIAL_DEVELOPMENT",
        "tau3_tag": "v1.0.1",
        "tau3_commit": "fc0055dc4e0a316c3f83133267fbd6faaa770992",
        "user_simulator_revision": None,
        "user_simulator_identity_schema": {
            "required": [
                "provider",
                "model",
                "revision",
                "runtime_value",
                "temperature",
                "seed_derivation",
                "num_retries",
            ],
            "temperature": 0.0,
            "num_retries": 0,
        },
        "user_simulator_authorized": False,
        "policy_retry_count": 0,
        "user_retry_count": 0,
        "pilot_seed": PILOT_SEED,
        "pilot_steps": PILOT_STEPS,
        "job_count": len(jobs),
        "execution_authorized": False,
        "commands_are_templates_only": True,
        "heldout_permitted": False,
        "bfcl_permitted": False,
        "prerequisites": [
            "PASS_C0_TO_FILTERED_TRAIN",
            "PASS_E1_BOTH_FAMILIES_BOTH_ALGORITHMS",
            "FROZEN_AND_AUTHORIZED_USER_SIMULATOR_REVISION",
        ],
        "jobs": jobs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(args.run_root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

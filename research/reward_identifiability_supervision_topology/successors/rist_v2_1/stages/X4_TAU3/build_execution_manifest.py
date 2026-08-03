"""Build the unauthorized 32-seed-ceiling X4 Tau3 execution manifest."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path
from typing import Any

STAGE = Path(__file__).resolve().parent
REPO = STAGE.parents[5]
X3 = STAGE.parent / "X3_TAU3"
TRAIN = X3 / "data/train.jsonl"
TRAIN_REL = str(TRAIN.relative_to(REPO))
FAMILIES = {
    "qwen3": ("Qwen/Qwen3-0.6B", "{QWEN3_MODEL_PATH}"),
    "gemma4": ("google/gemma-4-E2B-it", "{GEMMA4_MODEL_PATH}"),
}
ALGORITHMS = ("gspo", "grpo")
ARMS = {
    "AF": ("all_assistant", "full"),
    "LF": ("last_assistant", "full"),
    "AN": ("all_assistant", "name_only"),
    "LN": ("last_assistant", "name_only"),
}
STEPS = 25
GROUP_SIZE = 8
SOURCE_FILES = (
    "PROTOCOL.md",
    "freeze_seed_bank.py",
    "build_execution_manifest.py",
    "validate_evidence.py",
    "analyze_stability.py",
    "SEED_BANK.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command(job: dict[str, Any], run_root: Path) -> list[str]:
    return [
        "areno",
        "train",
        "--ckpt",
        FAMILIES[job["family"]][1],
        "--dataset-path",
        TRAIN_REL,
        "--dataset-loader-fn",
        "examples/agentic/rist_v2_1_tau3/dataset_loader.py",
        "--reward-fn-path",
        "examples/agentic/rist_v2_1_tau3/reward.py",
        "--agent-fn",
        "examples/agentic/rist_v2_1_tau3/run_agent.py",
        "--algo",
        job["algorithm"],
        "--seed",
        str(job["seed"]),
        "--batch-size",
        "1",
        "--n-samples",
        str(GROUP_SIZE),
        "--mini-bs",
        str(GROUP_SIZE),
        "--max-running-prompts",
        str(GROUP_SIZE),
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
        str(run_root / job["run_id"] / "checkpoints"),
        "--save-interval",
        str(STEPS),
        "--max-steps",
        str(STEPS),
        "--metrics-log-dir",
        str(run_root / job["run_id"] / "metrics"),
        "--trainable-turns",
        job["trainable_turns"],
        "--tool-call-supervision",
        job["tool_call_supervision"],
    ]


def build_manifest(run_root: Path, seed_bank: dict[str, Any] | None = None) -> dict:
    seed_bank = seed_bank or json.loads((STAGE / "SEED_BANK.json").read_text())
    seeds = [int(seed) for seed in seed_bank["seeds"]]
    planning = set(int(seed) for seed in seed_bank["planning_seeds"])
    train_sha = _sha256(TRAIN)
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    jobs = []
    for seed, family, algorithm, arm in itertools.product(
        seeds, FAMILIES, ALGORITHMS, ARMS
    ):
        trainable_turns, supervision = ARMS[arm]
        run_id = f"tau3-x4-{family}-{algorithm}-{arm}-{seed}"
        job = {
            "run_id": run_id,
            "seed": seed,
            "seed_tranche": "planning" if seed in planning else "expansion",
            "family": family,
            "checkpoint": FAMILIES[family][0],
            "algorithm": algorithm,
            "arm": arm,
            "trainable_turns": trainable_turns,
            "tool_call_supervision": supervision,
            "max_steps": STEPS,
            "group_size": GROUP_SIZE,
            "expected_episode_count": STEPS * GROUP_SIZE,
            "dataset": TRAIN_REL,
            "dataset_sha256": train_sha,
            "token_common_support_required": True,
            "execution_authorized": False,
        }
        job["command_template"] = _command(job, run_root)
        job["required_environment"] = {
            "RIST_X3_RUN_ID": run_id,
            "RIST_X3_SOURCE_COMMIT": "{FROZEN_X4_SOURCE_COMMIT}",
            "RIST_TAU3_USER_LLM": "{FROZEN_USER_SIMULATOR_MODEL}",
            "RIST_TAU3_USER_LLM_REVISION": "{FROZEN_USER_SIMULATOR_REVISION}",
            "RIST_TAU3_USER_LLM_AUTHORIZATION_SHA256": "{AUTHORIZATION_SHA256}",
            "RIST_RAW_JOURNAL_PATH": str(run_root / run_id / "raw_events.jsonl"),
            "RIST_REWARD_JOURNAL_PATH": str(run_root / run_id / "reward_events.jsonl"),
        }
        jobs.append(job)
    required_ids = sorted(
        json.loads(line)["id"] for line in TRAIN.read_text().splitlines()
    )
    return {
        "protocol": "RIST-X4-TAU3-POWERED-v2.1",
        "source_commit": source_commit,
        "source_file_sha256": {
            name: _sha256(STAGE / name)
            for name in SOURCE_FILES
            if (STAGE / name).is_file()
        },
        "x3_pilot_prerequisite": {
            "required_status": "PASS",
            "required_job_count": 16,
            "required_episode_count": 3200,
            "validation_sha256": None,
        },
        "seed_bank_sha256": _sha256(STAGE / "SEED_BANK.json"),
        "seed_bank": seeds,
        "planning_seeds": seed_bank["planning_seeds"],
        "maximum_seed_count": len(seeds),
        "power_plan": seed_bank["power_plan"],
        "family_bindings": {
            family: {
                "revision": None,
                "tokenizer_revision": None,
                "weights_manifest_sha256": None,
            }
            for family in FAMILIES
        },
        "user_simulator_binding": None,
        "required_training_task_ids": required_ids,
        "job_count": len(jobs),
        "planning_job_count": len(planning) * 16,
        "maximum_episode_count": len(jobs) * STEPS * GROUP_SIZE,
        "execution_authorized": False,
        "commands_are_templates_only": True,
        "heldout_permitted": False,
        "bfcl_permitted": False,
        "jobs": jobs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.run_root)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

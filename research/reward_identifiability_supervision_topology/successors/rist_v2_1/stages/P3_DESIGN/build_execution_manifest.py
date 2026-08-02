"""Build a non-executable, fail-closed RIST-v2.1 pilot run manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[6]
DESIGN_PATH = Path(__file__).with_name("design_matrix.py")
TRAIN_PATH = (
    REPO_ROOT
    / "research/reward_identifiability_supervision_topology/successors/rist_v2_1"
    / "stages/D3/data/train.jsonl"
)
DEFAULT_MAX_STEPS = 100
DEFAULT_SAVE_INTERVAL = 25


def _load_design():
    spec = importlib.util.spec_from_file_location("rist_v2_1_design_manifest", DESIGN_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("design module loader unavailable")
    spec.loader.exec_module(module)
    return module


def _command(
    row: dict[str, Any],
    run_root: Path,
    max_steps: int,
    train_path: str,
) -> list[str]:
    command = [
        "areno",
        "train",
        "--ckpt",
        str(row["checkpoint"]),
        "--dataset-path",
        train_path,
        "--dataset-loader-fn",
        "examples/agentic/rist_v2_1/dataset_loader.py",
        "--reward-fn-path",
        "examples/agentic/rist_v2_1/reward.py",
        "--agent-fn",
        "examples/agentic/rist_v2_1/run_agent.py",
        "--algo",
        str(row["algorithm"]),
        "--seed",
        str(row["seed"]),
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
        "1024",
        "--max-new-tokens",
        "128",
        "--max-context-len",
        "4096",
        "--agent-timeout-s",
        "900",
        "--attn-backend",
        "native",
        "--drop-rollout-state",
        "--save-path",
        str(run_root / "runs" / row["run_id"] / "checkpoints"),
        "--save-interval",
        str(DEFAULT_SAVE_INTERVAL),
        "--max-steps",
        str(max_steps),
        "--metrics-log-dir",
        str(run_root / "runs" / row["run_id"] / "metrics"),
        "--trainable-turns",
        str(row["trainable_turns"]),
        "--tool-call-supervision",
        str(row["tool_call_supervision"]),
    ]
    return command


def build_manifest(
    run_root: Path,
    max_steps: int = DEFAULT_MAX_STEPS,
    filtered_train_path: Path | None = None,
    resolution_map_path: Path | None = None,
) -> dict[str, Any]:
    """Return the auditable pilot matrix without granting execution authority."""

    if max_steps != DEFAULT_MAX_STEPS:
        raise ValueError("RIST-v2.1 diagnostic pilot is frozen at 100 steps")
    design = _load_design()
    rows = design.build_matrix()
    design.validate_matrix(rows)
    if (filtered_train_path is None) != (resolution_map_path is None):
        raise ValueError("filtered train and resolution map must be supplied together")
    dataset_ready = filtered_train_path is not None
    if dataset_ready:
        if not filtered_train_path.is_file() or not resolution_map_path.is_file():
            raise FileNotFoundError("filtered train or resolution map is missing")
        train_path = str(filtered_train_path)
        train_hash = hashlib.sha256(filtered_train_path.read_bytes()).hexdigest()
        resolution_map_hash = hashlib.sha256(resolution_map_path.read_bytes()).hexdigest()
    else:
        train_path = "{FILTERED_TRAIN_JSONL}"
        train_hash = None
        resolution_map_hash = None
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runs = []
    for row in rows:
        runs.append(
            {
                **row,
                "command": _command(row, run_root, max_steps, train_path),
                "scientific_treatment_ready": False,
                "required_environment": {
                    "RIST_RAW_JOURNAL_PATH": str(
                        run_root / "runs" / row["run_id"] / "raw_responses.jsonl"
                    ),
                    "RIST_REWARD_JOURNAL_PATH": str(
                        run_root / "runs" / row["run_id"] / "reward_events.jsonl"
                    ),
                },
            }
        )
    return {
        "protocol": "RIST-P3-DIAGNOSTIC-v2.1",
        "source_commit": head,
        "train_sha256": train_hash,
        "resolution_map_sha256": resolution_map_hash,
        "resolution_filtered_dataset_ready": dataset_ready,
        "max_steps": max_steps,
        "save_interval": DEFAULT_SAVE_INTERVAL,
        "saved_checkpoint_steps": list(
            range(DEFAULT_SAVE_INTERVAL, max_steps + 1, DEFAULT_SAVE_INTERVAL)
        ),
        "run_count": len(runs),
        "execution_authorized": False,
        "commands_are_templates_only": True,
        "blocked_by": [
            "T0_REAL_QWEN_GEMMA_TOKENIZER_FIXTURES",
            "C0_COMMON_TRANSPORTED_RESOLUTION_BANDS",
            "E1_PER_CHECKPOINT_TRAINING_CAPACITY",
            "EXPLICIT_GPU_TRAINING_AUTHORIZATION",
        ],
        "pilot_scope": "variance_and_power_only_three_seeds",
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(args.run_root, args.max_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

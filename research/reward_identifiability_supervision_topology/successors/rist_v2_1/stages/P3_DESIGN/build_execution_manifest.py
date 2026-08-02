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


def _load_design():
    spec = importlib.util.spec_from_file_location("rist_v2_1_design_manifest", DESIGN_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("design module loader unavailable")
    spec.loader.exec_module(module)
    return module


def _command(row: dict[str, Any], run_root: Path, max_steps: int) -> list[str]:
    command = [
        "areno",
        "train",
        "--ckpt",
        str(row["checkpoint"]),
        "--dataset-path",
        str(TRAIN_PATH),
        "--dataset-loader-fn",
        str(REPO_ROOT / "examples/agentic/rist_v2_1/dataset_loader.py"),
        "--reward-fn-path",
        str(REPO_ROOT / "examples/agentic/rist_v2_1/reward.py"),
        "--agent-fn",
        str(REPO_ROOT / "examples/agentic/rist_v2_1/run_agent.py"),
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
        "--max-steps",
        str(max_steps),
        "--metrics-log-dir",
        str(run_root / "runs" / row["run_id"] / "metrics"),
        "--trainable-turns",
        str(row["trainable_turns"]),
    ]
    if row["mask_tool_call_args"]:
        command.append("--mask-tool-call-args")
    return command


def build_manifest(run_root: Path, max_steps: int = DEFAULT_MAX_STEPS) -> dict[str, Any]:
    """Return the auditable pilot matrix without granting execution authority."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    design = _load_design()
    rows = design.build_matrix()
    design.validate_matrix(rows)
    train_hash = hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest()
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
                "command": _command(row, run_root, max_steps),
                "scientific_treatment_ready": row["content_claim"] == "full_call",
            }
        )
    return {
        "protocol": "RIST-P3-DIAGNOSTIC-v2.1",
        "source_commit": head,
        "train_sha256": train_hash,
        "max_steps": max_steps,
        "run_count": len(runs),
        "execution_authorized": False,
        "commands_are_templates_only": True,
        "blocked_by": [
            "T0_EXACT_NAME_ONLY_TREATMENT",
            "T0_REAL_QWEN_GEMMA_TOKENIZER_FIXTURES",
            "E1_PER_CHECKPOINT_TRAINING_CAPACITY",
            "X1_EXTERNAL_ENVIRONMENT_QUALIFICATION",
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

"""Prepare or explicitly execute the bounded SAS Q1 GPU pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_CHECKPOINT = "Qwen/Qwen3-0.6B"
DEFAULT_PROTOCOL_ID = "SAS-P0-v1.0"
DEFAULT_SEEDS = (1101, 2202)
DEFAULT_MAX_STEPS = 8
ARMS = {
    "AF": {"trainable_turns": "all_assistant", "mask_tool_call_args": False},
    "LF": {"trainable_turns": "last_assistant", "mask_tool_call_args": False},
}


def build_commands(
    *,
    repo_root: Path,
    run_root: Path,
    dataset_path: Path,
    checkpoint: str = DEFAULT_CHECKPOINT,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict[str, list[str]]:
    """Build isolated arm/seed commands using qualification data only."""

    shared = [
        "areno",
        "train",
        "--ckpt",
        checkpoint,
        "--model-hub",
        "modelscope",
        "--dataset-path",
        str(dataset_path),
        "--dataset-loader-fn",
        str(
            repo_root
            / "examples/agentic/structured_action_supervision/dataset_loader.py"
        ),
        "--reward-fn-path",
        str(repo_root / "examples/agentic/structured_action_supervision/reward.py"),
        "--agent-fn",
        str(repo_root / "examples/agentic/structured_action_supervision/run_agent.py"),
        "--algo",
        "gspo",
        "--tp-size",
        "1",
        "--world-size",
        "1",
        "--batch-size",
        "1",
        "--n-samples",
        "4",
        "--mini-bs",
        "4",
        "--max-running-prompts",
        "4",
        "--max-prompt-tokens",
        "512",
        "--max-new-tokens",
        "128",
        "--max-context-len",
        "2048",
        "--agent-timeout-s",
        "900",
        "--attn-backend",
        "native",
        "--adam-8bit",
        "--drop-rollout-state",
        "--max-steps",
        str(max_steps),
    ]
    commands = {}
    for seed in seeds:
        for arm, settings in ARMS.items():
            key = f"{arm}-seed-{seed}"
            command = [
                *shared,
                "--seed",
                str(seed),
                "--metrics-log-dir",
                str(run_root / "runs" / key / "metrics"),
                "--trainable-turns",
                settings["trainable_turns"],
            ]
            if settings["mask_tool_call_args"]:
                command.append("--mask-tool-call-args")
            commands[key] = command
    return commands


def prepare(
    *,
    repo_root: Path,
    run_root: Path,
    checkpoint: str = DEFAULT_CHECKPOINT,
    protocol_id: str = DEFAULT_PROTOCOL_ID,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict[str, Any]:
    """Validate Q0 and write an idempotent Q1 pilot manifest."""

    research_root = repo_root / "research/structured_action_supervision_v1"
    q0_result_path = research_root / "stages/Q0/stage_result.json"
    q0_assessment_path = research_root / "stages/Q0/main_track_assessment.json"
    q0_result = json.loads(q0_result_path.read_text(encoding="utf-8"))
    if q0_result.get("stage_status") != "PASS":
        raise RuntimeError("Q1 cannot open until Q0 stage_status is PASS")
    if not q0_assessment_path.is_file():
        raise RuntimeError("Q1 cannot open without the Q0 main-track assessment")

    dataset_path = research_root / "stages/Q0/dataset/qualification.jsonl"
    split_manifest = json.loads(
        (research_root / "stages/Q0/dataset/split_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    expected_hash = split_manifest["files"]["qualification"]["sha256"]
    actual_hash = _sha256(dataset_path)
    if actual_hash != expected_hash:
        raise RuntimeError("qualification dataset hash does not match the Q0 manifest")
    if not seeds or any(seed < 0 for seed in seeds):
        raise ValueError("Q1 seeds must be non-empty and non-negative")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Q1 seeds must be unique")
    if max_steps <= 0:
        raise ValueError("Q1 max_steps must be positive")

    commands = build_commands(
        repo_root=repo_root,
        run_root=run_root,
        dataset_path=dataset_path,
        checkpoint=checkpoint,
        seeds=seeds,
        max_steps=max_steps,
    )
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "stage": "Q1",
        "status": "PREPARED_GPU_UNAUTHORIZED",
        "checkpoint": checkpoint,
        "model_hub": "modelscope",
        "dataset": {
            "split": "qualification",
            "path": str(dataset_path),
            "sha256": actual_hash,
            "count": split_manifest["files"]["qualification"]["count"],
        },
        "heldout_consumed": False,
        "arms": ARMS,
        "seeds": list(seeds),
        "max_steps": max_steps,
        "commands": {
            key: shlex.join(command) for key, command in commands.items()
        },
        "controlled_difference": [
            "--seed",
            "--trainable-turns",
            "--metrics-log-dir",
        ],
        "claim_boundary": (
            "Q1 is an instrument and variance pilot. It cannot establish efficacy "
            "or trigger GO_MAIN_TRACK."
        ),
    }
    run_root.mkdir(parents=True, exist_ok=True)
    destination = run_root / "pilot_manifest.json"
    serialized = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(f"refusing to overwrite a different Q1 manifest: {destination}")
    destination.write_text(serialized, encoding="utf-8")
    return manifest


def execute_gpu_pilot(
    *,
    repo_root: Path,
    run_root: Path,
    manifest: dict[str, Any],
) -> None:
    """Execute the already-frozen pilot after explicit external authorization."""

    _run(
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "assert torch.cuda.is_available(), 'CUDA GPU is required'; "
                "print(torch.cuda.get_device_name(0)); "
                "print(torch.cuda.get_device_properties(0).total_memory)"
            ),
        ],
        cwd=repo_root,
    )
    commands = build_commands(
        repo_root=repo_root,
        run_root=run_root,
        dataset_path=Path(manifest["dataset"]["path"]),
        checkpoint=str(manifest["checkpoint"]),
        seeds=tuple(int(seed) for seed in manifest["seeds"]),
        max_steps=int(manifest["max_steps"]),
    )
    for key, command in commands.items():
        metrics_dir = run_root / "runs" / key / "metrics"
        if metrics_dir.exists() and any(metrics_dir.iterdir()):
            raise RuntimeError(f"refusing to mix events in {metrics_dir}")
        _run(command, cwd=repo_root)


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("research/structured_action_supervision_v1/stages/Q1"),
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--protocol-id",
        choices=("SAS-P0-v1.0", "SAS-P0-v1.1"),
        default=DEFAULT_PROTOCOL_ID,
    )
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--execute-gpu-pilot", action="store_true")
    args = parser.parse_args()
    if args.prepare and args.execute_gpu_pilot:
        parser.error("--prepare and --execute-gpu-pilot are mutually exclusive")

    repo_root = Path(__file__).resolve().parents[3]
    run_root = args.run_root.expanduser().resolve()
    seeds = tuple(args.seeds) if args.seeds else DEFAULT_SEEDS
    commands = build_commands(
        repo_root=repo_root,
        run_root=run_root,
        dataset_path=(
            repo_root
            / "research/structured_action_supervision_v1/stages/Q0/dataset/qualification.jsonl"
        ),
        checkpoint=args.checkpoint,
        seeds=seeds,
        max_steps=args.max_steps,
    )
    if not args.prepare and not args.execute_gpu_pilot:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "commands": {
                        key: shlex.join(command)
                        for key, command in commands.items()
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    manifest = prepare(
        repo_root=repo_root,
        run_root=run_root,
        checkpoint=args.checkpoint,
        protocol_id=args.protocol_id,
        seeds=seeds,
        max_steps=args.max_steps,
    )
    if args.execute_gpu_pilot:
        execute_gpu_pilot(
            repo_root=repo_root,
            run_root=run_root,
            manifest=manifest,
        )
    else:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

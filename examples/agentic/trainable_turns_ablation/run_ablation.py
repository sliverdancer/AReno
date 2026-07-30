"""Prepare or execute the reproducible three-arm issue #199 GPU ablation.

The default mode is dry-run and does not write files or start training.
``--prepare`` generates the deterministic dataset and manifest on CPU.
``--execute-gpu-training`` performs the same preparation, verifies CUDA, runs
the three arms from the same checkpoint, then exports CSV/JSON metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from collect_metrics import ARMS, collect_run

DEFAULT_CKPT = "Qwen/Qwen3-0.6B"
DEFAULT_COUNT = 32
DEFAULT_DATASET_SEED = 2026
DEFAULT_TRAIN_SEED = 2026
DEFAULT_MAX_STEPS = 10


def build_arm_commands(
    *,
    repo_root: Path,
    run_root: Path,
    dataset_path: Path,
    ckpt: str = DEFAULT_CKPT,
    max_steps: int = DEFAULT_MAX_STEPS,
    train_seed: int = DEFAULT_TRAIN_SEED,
) -> dict[str, list[str]]:
    """Build the three commands with only ``trainable_turns`` and log path varied."""

    shared = [
        "areno",
        "train",
        "--ckpt",
        ckpt,
        "--model-hub",
        "modelscope",
        "--dataset-path",
        str(dataset_path),
        "--dataset-loader-fn",
        str(repo_root / "examples/agentic/tictactoe/dataset_loader.py"),
        "--reward-fn-path",
        str(repo_root / "examples/agentic/tictactoe/reward.py"),
        "--agent-fn",
        str(repo_root / "examples/agentic/trainable_turns_ablation/run_agent.py"),
        "--algo",
        "gspo",
        "--seed",
        str(train_seed),
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
        "64",
        "--max-context-len",
        "2048",
        "--agent-timeout-s",
        "900",
        "--attn-backend",
        "native",
        "--max-steps",
        str(max_steps),
    ]
    return {
        arm: [
            *shared,
            "--metrics-log-dir",
            str(run_root / "arms" / arm / "metrics"),
            "--trainable-turns",
            arm,
        ]
        for arm in ARMS
    }


def prepare(
    *,
    repo_root: Path,
    run_root: Path,
    ckpt: str,
    count: int,
    dataset_seed: int,
    train_seed: int,
    max_steps: int,
) -> dict[str, Any]:
    """Generate deterministic input data, validate its normalized schema, and write a manifest."""

    dataset_path = run_root / "dataset" / "tictactoe.jsonl"
    _generate_dataset(repo_root, dataset_path, count=count, seed=dataset_seed)
    normalized = _inspect_dataset(repo_root, dataset_path)
    commands = build_arm_commands(
        repo_root=repo_root,
        run_root=run_root,
        dataset_path=dataset_path,
        ckpt=ckpt,
        max_steps=max_steps,
        train_seed=train_seed,
    )
    manifest = {
        "schema_version": 1,
        "git_commit": _git_output(repo_root, "rev-parse", "HEAD"),
        "checkpoint": ckpt,
        "model_hub": "modelscope",
        "train_seed": train_seed,
        "dataset": {
            "path": str(dataset_path),
            "sha256": _sha256(dataset_path),
            "count": len(normalized),
            "seed": dataset_seed,
        },
        "arms": list(ARMS),
        "max_steps": max_steps,
        "commands": {arm: shlex.join(command) for arm, command in commands.items()},
        "controlled_difference": ["--trainable-turns", "--metrics-log-dir"],
        "semantic_note": (
            "The agent always requests a final text response after the tool result. "
            "Therefore last_assistant and final_answer should select the same span on "
            "well-formed trajectories and serve as an equivalence-control pair."
        ),
        "reproducibility_limit": (
            "Dataset generation, parent-process seeding, epoch order, and rollout request "
            "seeds are deterministic. CUDA kernels may still prevent bitwise reproducibility."
        ),
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def execute_gpu_training(
    *,
    repo_root: Path,
    run_root: Path,
    manifest: dict[str, Any],
) -> tuple[Path, Path]:
    """Run the prepared commands after CUDA and clean-output checks."""

    environment_dir = run_root / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    _run_and_record(
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "assert torch.cuda.is_available(), 'CUDA GPU is required'; "
                "print(torch.cuda.get_device_name(0))"
            ),
        ],
        cwd=repo_root,
        output_path=environment_dir / "cuda.json",
    )
    _run_and_record(
        ["areno", "env", "--json"],
        cwd=repo_root,
        output_path=environment_dir / "areno-env.json",
    )
    _run_and_record(
        ["areno", "check"],
        cwd=repo_root,
        output_path=environment_dir / "areno-check.json",
    )

    commands = build_arm_commands(
        repo_root=repo_root,
        run_root=run_root,
        dataset_path=Path(manifest["dataset"]["path"]),
        ckpt=str(manifest["checkpoint"]),
        max_steps=int(manifest["max_steps"]),
        train_seed=int(manifest["train_seed"]),
    )
    for arm, command in commands.items():
        metrics_dir = run_root / "arms" / arm / "metrics"
        if metrics_dir.exists() and any(metrics_dir.iterdir()):
            raise RuntimeError(f"refusing to mix events in non-empty directory: {metrics_dir}")
        _run(command, cwd=repo_root)
    return collect_run(run_root)


def _generate_dataset(repo_root: Path, path: Path, *, count: int, seed: int) -> None:
    generator = _load_module(
        "issue199_tictactoe_dataset_generator",
        repo_root / "examples/agentic/tictactoe/dataset_generator.py",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        generator.write_jsonl(generator.generate_records(count, seed=seed), handle)


def _inspect_dataset(repo_root: Path, path: Path) -> list[dict[str, Any]]:
    loader = _load_module(
        "issue199_tictactoe_dataset_loader",
        repo_root / "examples/agentic/tictactoe/dataset_loader.py",
    )
    records = loader.load_training_dataset(str(path))
    if not records:
        raise ValueError("dataset normalization returned no records")
    for index, record in enumerate(records):
        if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
            raise ValueError(f"normalized record {index} has no prompt")
        if "board" not in record or "best_moves" not in record:
            raise ValueError(f"normalized record {index} is missing Tic-Tac-Toe fields")
    return records


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_output(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def _run_and_record(command: list[str], *, cwd: Path, output_path: Path) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    payload = {
        "command": shlex.join(command),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    result.check_returncode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/issue-199-ablation"),
    )
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--dataset-seed", type=int, default=DEFAULT_DATASET_SEED)
    parser.add_argument("--train-seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="Generate CPU-only inputs and manifest.")
    mode.add_argument(
        "--execute-gpu-training",
        action="store_true",
        help="Prepare, verify CUDA, and execute all three GPU arms.",
    )
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.train_seed < 0:
        parser.error("--train-seed must be non-negative")

    repo_root = Path(__file__).resolve().parents[3]
    run_root = args.run_root.expanduser().resolve()
    dataset_path = run_root / "dataset" / "tictactoe.jsonl"
    commands = build_arm_commands(
        repo_root=repo_root,
        run_root=run_root,
        dataset_path=dataset_path,
        ckpt=args.ckpt,
        max_steps=args.max_steps,
        train_seed=args.train_seed,
    )
    if not args.prepare and not args.execute_gpu_training:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "commands": {
                        arm: shlex.join(command)
                        for arm, command in commands.items()
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
        ckpt=args.ckpt,
        count=args.count,
        dataset_seed=args.dataset_seed,
        train_seed=args.train_seed,
        max_steps=args.max_steps,
    )
    print(f"Prepared manifest: {run_root / 'manifest.json'}")
    if args.execute_gpu_training:
        csv_path, json_path = execute_gpu_training(
            repo_root=repo_root,
            run_root=run_root,
            manifest=manifest,
        )
        print(f"CSV: {csv_path}")
        print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

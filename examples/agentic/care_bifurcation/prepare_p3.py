"""Prepare, but never execute, the frozen CARe P3 remote-GPU pilot."""

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


ARMS = ("care", "uncalibrated")
TRAIN_SEEDS = (3101, 3102, 3103)
DEFAULT_CKPT = "Qwen/Qwen3-0.6B"
DEFAULT_DATASET_SEED = 7301
DEFAULT_COUNT = 64
DEFAULT_MAX_STEPS = 1
MODEL_ASSET_PATH = Path(__file__).with_name("modelscope_asset.json")


def build_commands(
    *,
    repo_root: Path,
    run_root: Path,
    dataset_path: Path,
    ckpt: str = DEFAULT_CKPT,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict[str, list[str]]:
    """Build two matched arms for each of three qualification seeds."""

    commands = {}
    for seed in TRAIN_SEEDS:
        for arm in ARMS:
            run_id = f"{arm}-seed-{seed}"
            commands[run_id] = [
                "areno",
                "train",
                "--ckpt",
                ckpt,
                "--model-hub",
                "modelscope",
                "--dataset-path",
                str(dataset_path),
                "--dataset-loader-fn",
                str(repo_root / "examples/agentic/care_bifurcation/dataset_loader.py"),
                "--reward-fn-path",
                str(repo_root / "examples/agentic/care_bifurcation/reward.py"),
                "--agent-fn",
                str(repo_root / "examples/agentic/care_bifurcation/run_agent.py"),
                "--turn-credit-fn-path",
                str(repo_root / "examples/agentic/care_bifurcation/care_router.py"),
                "--turn-credit-config-path",
                str(repo_root / f"examples/agentic/care_bifurcation/{arm}_config.json"),
                "--algo",
                "grpo",
                "--seed",
                str(seed),
                "--tp-size",
                "1",
                "--world-size",
                "1",
                "--batch-size",
                "20",
                "--n-samples",
                "1",
                "--mini-bs",
                "5",
                "--gradient-accumulation-steps",
                "4",
                "--max-running-prompts",
                "20",
                "--max-prompt-tokens",
                "512",
                "--max-new-tokens",
                "64",
                "--max-context-len",
                "4096",
                "--agent-timeout-s",
                "900",
                "--attn-backend",
                "native",
                "--disable-thinking",
                "--max-steps",
                str(max_steps),
                "--metrics-log-dir",
                str(run_root / "runs" / run_id / "metrics"),
                "--trainable-turns",
                "all_assistant",
            ]
    return commands


def prepare(
    *,
    repo_root: Path,
    run_root: Path,
    ckpt: str,
    dataset_seed: int,
    count: int,
    max_steps: int,
) -> dict[str, Any]:
    """Generate deterministic inputs and a command/source manifest."""

    generator = _load_module(
        "care_p3_dataset_generator",
        repo_root / "examples/agentic/care_bifurcation/dataset_generator.py",
    )
    dataset_path = run_root / "dataset" / "bifurcation.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as handle:
        generator.write_jsonl(
            generator.generate_records(count, seed=dataset_seed),
            handle,
        )
    commands = build_commands(
        repo_root=repo_root,
        run_root=run_root,
        dataset_path=dataset_path,
        ckpt=ckpt,
        max_steps=max_steps,
    )
    source_paths = [
        repo_root / "areno/experimental/care/calibration.py",
        repo_root / "areno/experimental/care/turn_credit.py",
        repo_root / "examples/agentic/care_bifurcation/task.py",
        repo_root / "examples/agentic/care_bifurcation/dataset_generator.py",
        repo_root / "examples/agentic/care_bifurcation/dataset_loader.py",
        repo_root / "examples/agentic/care_bifurcation/reward.py",
        repo_root / "examples/agentic/care_bifurcation/run_agent.py",
        repo_root / "examples/agentic/care_bifurcation/care_router.py",
        repo_root / "examples/agentic/care_bifurcation/collect_p3.py",
        repo_root / "examples/agentic/care_bifurcation/execute_p3.py",
        repo_root / "examples/agentic/care_bifurcation/fetch_modelscope_snapshot.py",
        repo_root / "examples/agentic/care_bifurcation/modelscope_asset.json",
        repo_root / "examples/agentic/care_bifurcation/prepare_p3.py",
        repo_root / "examples/agentic/care_bifurcation/care_config.json",
        repo_root / "examples/agentic/care_bifurcation/uncalibrated_config.json",
    ]
    model_asset = json.loads(MODEL_ASSET_PATH.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "protocol": "CARE-P3-PILOT-v0.1",
        "authorization": "PREPARE_ONLY_GPU_NOT_AUTHORIZED",
        "git_commit": _git_output(repo_root, "rev-parse", "HEAD"),
        "git_status": _git_output(repo_root, "status", "--short"),
        "checkpoint": ckpt,
        "model_hub": "modelscope",
        "model_asset": {
            "manifest_path": str(MODEL_ASSET_PATH.relative_to(repo_root)),
            "manifest_sha256": _sha256(MODEL_ASSET_PATH),
            "model_id": model_asset["model_id"],
            "revision": model_asset["revision"],
            "latest_commit_short_id": model_asset["revision_state"][
                "latest_commit_short_id"
            ],
            "full_file_hash_verification_required": True,
        },
        "dataset": {
            "path": str(dataset_path),
            "seed": dataset_seed,
            "count": count,
            "sha256": _sha256(dataset_path),
        },
        "arms": list(ARMS),
        "train_seeds": list(TRAIN_SEEDS),
        "max_steps": max_steps,
        "commands": {
            run_id: shlex.join(command)
            for run_id, command in commands.items()
        },
        "source_sha256": {
            str(path.relative_to(repo_root)): _sha256(path)
            for path in source_paths
        },
        "controlled_difference_within_seed": [
            "--turn-credit-config-path",
            "--metrics-log-dir",
        ],
        "fixed_design": {
            "calibration_split": "even prompt indices, 10 unique blocks",
            "update_split": "odd prompt indices, 10 unique blocks",
            "matched_audits": True,
            "alpha": 0.2,
            "budget_tokens_per_update_trajectory": 256,
            "model_calls_per_trajectory": 5,
            "expected_model_calls_total": 600,
            "counterfactual_terminal_evaluations_per_run": 300,
            "maximum_single_gpu_wall_clock_hours": 1,
            "maximum_total_gpu_hours": 6,
            "proposed_gpu": "AutoDL A800-80GB",
            "observed_gpu_price_cny_per_hour": 4.98,
            "maximum_instance_lifetime_hours": 8,
            "maximum_total_spend_cny": 60,
            "kill_on_first_nonfinite_update": True,
        },
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_root / "GPU_EXECUTION_NOT_AUTHORIZED").write_text(
        "This directory is CPU-prepared only. Obtain explicit user approval before any areno train command.\n",
        encoding="utf-8",
    )
    return manifest


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("artifacts/care-p3-pilot"),
    )
    parser.add_argument("--ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--dataset-seed", type=int, default=DEFAULT_DATASET_SEED)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Write the dataset and manifest. No GPU command is available.",
    )
    args = parser.parse_args()
    if args.count < 20:
        parser.error("--count must be at least 20")
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    repo_root = Path(__file__).resolve().parents[3]
    run_root = args.run_root.expanduser().resolve()
    dataset_path = run_root / "dataset" / "bifurcation.jsonl"
    if not args.prepare:
        commands = build_commands(
            repo_root=repo_root,
            run_root=run_root,
            dataset_path=dataset_path,
            ckpt=args.ckpt,
            max_steps=args.max_steps,
        )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "gpu_execution_authorized": False,
                    "commands": {
                        run_id: shlex.join(command)
                        for run_id, command in commands.items()
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
        dataset_seed=args.dataset_seed,
        count=args.count,
        max_steps=args.max_steps,
    )
    print(f"Prepared manifest: {run_root / 'manifest.json'}")
    print(f"Dataset SHA-256: {manifest['dataset']['sha256']}")
    print("GPU execution remains unauthorized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

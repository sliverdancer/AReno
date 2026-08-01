"""CPU-only preparation for the frozen ARCA P4 paired GPU validation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any


PROTOCOL = "ARCA-P4-DYNAMIC-v0.1"
ARMS = ("strict", "canonical")
SEEDS = (3101, 3102, 3103)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_commands(
    repo_root: Path, run_root: Path, dataset_path: Path, ckpt: str
) -> dict[str, list[str]]:
    commands = {}
    for seed in SEEDS:
        for arm in ARMS:
            run_id = f"{arm}-seed-{seed}"
            commands[run_id] = [
                "areno", "train", "--ckpt", ckpt,
                "--model-hub", "modelscope",
                "--dataset-path", str(dataset_path),
                "--dataset-loader-fn", str(repo_root / "examples/agentic/care_bifurcation/dataset_loader.py"),
                "--reward-fn-path", str(repo_root / f"research/silent_reward_contracts/p4/reward_{arm}.py"),
                "--agent-fn", str(repo_root / "examples/agentic/care_bifurcation/run_agent.py"),
                "--algo", "grpo", "--seed", str(seed),
                "--tp-size", "1", "--world-size", "1",
                "--batch-size", "20", "--n-samples", "1", "--mini-bs", "5",
                "--gradient-accumulation-steps", "4",
                "--max-running-prompts", "20", "--max-prompt-tokens", "512",
                "--max-new-tokens", "64", "--max-context-len", "4096",
                "--agent-timeout-s", "900", "--attn-backend", "native",
                "--disable-thinking", "--max-steps", "1",
                "--metrics-log-dir", str(run_root / "runs" / run_id / "metrics"),
                "--trainable-turns", "all_assistant",
            ]
    return commands


def prepare(repo_root: Path, run_root: Path, ckpt: str) -> dict[str, Any]:
    generator = _load_module(
        "arca_p4_dataset_generator",
        repo_root / "examples/agentic/care_bifurcation/dataset_generator.py",
    )
    dataset_path = run_root / "dataset/bifurcation.jsonl"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as handle:
        generator.write_jsonl(generator.generate_records(64, seed=7301), handle)
    commands = build_commands(repo_root, run_root, dataset_path, ckpt)
    source_paths = [
        repo_root / "examples/agentic/care_bifurcation/dataset_generator.py",
        repo_root / "examples/agentic/care_bifurcation/dataset_loader.py",
        repo_root / "examples/agentic/care_bifurcation/run_agent.py",
        repo_root / "examples/agentic/care_bifurcation/task.py",
        repo_root / "research/silent_reward_contracts/p4/reward_strict.py",
        repo_root / "research/silent_reward_contracts/p4/reward_canonical.py",
    ]
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "authorization": "PREPARE_ONLY_GPU_NOT_AUTHORIZED",
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "git_status": subprocess.run(
            ["git", "status", "--short"], cwd=repo_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "checkpoint": ckpt,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "max_steps": 1,
        "dataset": {"path": str(dataset_path), "seed": 7301, "count": 64, "sha256": _sha256(dataset_path)},
        "commands": {key: shlex.join(value) for key, value in commands.items()},
        "controlled_difference": ["--reward-fn-path", "--metrics-log-dir"],
        "source_sha256": {str(path.relative_to(repo_root)): _sha256(path) for path in source_paths},
        "ceilings": {"single_run_minutes": 60, "total_gpu_hours": 6, "instance_hours": 8, "cny": 60},
        "claim_boundary": "dynamic contract consequence only; not a learning-efficacy comparison",
    }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_root / "GPU_EXECUTION_NOT_AUTHORIZED").write_text(
        "Obtain explicit owner authorization before model download, serving, or training.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    manifest = prepare(repo_root, args.run_root.expanduser().resolve(), args.ckpt)
    print(json.dumps({"gpu_executed": False, "manifest": str(args.run_root / "manifest.json"), "runs": len(manifest["commands"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

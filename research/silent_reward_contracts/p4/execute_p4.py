"""Fail-closed executor for the frozen ARCA P4 paired GPU validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any

from research.silent_reward_contracts.p4.collect_p4 import (
    build_rows,
    load_series,
    write_artifacts,
)
from research.silent_reward_contracts.p4.prepare_p4 import ARMS, PROTOCOL, SEEDS


EXPECTED_ORDER = [f"{arm}-seed-{seed}" for seed in SEEDS for arm in ARMS]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_preflight(
    manifest: dict[str, Any], repo_root: Path, run_root: Path
) -> dict[str, list[str]]:
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError(f"expected {PROTOCOL}")
    if manifest.get("authorization") != "PREPARE_ONLY_GPU_NOT_AUTHORIZED":
        raise ValueError("unexpected manifest authorization state")
    if manifest.get("run_order") != EXPECTED_ORDER:
        raise ValueError("manifest run order does not match the frozen paired order")
    command_strings = manifest.get("commands")
    if (
        not isinstance(command_strings, dict)
        or set(command_strings) != set(EXPECTED_ORDER)
    ):
        raise ValueError("manifest must contain exactly the frozen six commands")
    if manifest.get("git_commit") != _git(repo_root, "rev-parse", "HEAD"):
        raise ValueError("checkout commit differs from manifest")
    if manifest.get("git_status") or _git(repo_root, "status", "--short"):
        raise ValueError("P4 requires a clean checkout")
    dataset = Path(manifest["dataset"]["path"])
    if not dataset.is_file() or _sha256(dataset) != manifest["dataset"]["sha256"]:
        raise ValueError("P4 dataset is missing or changed")
    checkpoint = Path(manifest["checkpoint"])
    if not checkpoint.is_absolute() or not checkpoint.is_dir():
        raise ValueError("P4 checkpoint must be an existing absolute directory")
    if not (run_root / "model_sha256.txt").is_file():
        raise ValueError("P4 model hash manifest is missing")
    for relative, expected_hash in manifest["source_sha256"].items():
        path = repo_root / relative
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"P4 source changed: {relative}")
    return {
        run_id: shlex.split(command_strings[run_id]) for run_id in EXPECTED_ORDER
    }


def _gpu_used_mib() -> int:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return int(output)


def execute(
    commands: dict[str, list[str]], repo_root: Path, run_root: Path
) -> list[dict[str, Any]]:
    if _gpu_used_mib() > 64:
        raise RuntimeError("GPU is not idle before P4")
    results = []
    total_started = time.monotonic()
    for run_id in EXPECTED_ORDER:
        run_dir = run_root / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = run_dir / "stdout.log"
        stderr_path = run_dir / "stderr.log"
        started = time.monotonic()
        status = "completed"
        returncode = None
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            try:
                completed = subprocess.run(
                    commands[run_id],
                    cwd=repo_root,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                    timeout=3600,
                    text=True,
                )
                returncode = int(completed.returncode)
                if returncode != 0:
                    status = "failed"
            except subprocess.TimeoutExpired:
                status = "timeout"
        wall_seconds = time.monotonic() - started
        results.append(
            {
                "run_id": run_id,
                "status": status,
                "returncode": returncode,
                "wall_seconds": wall_seconds,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }
        )
        payload = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "results": results,
            "total_single_gpu_hours": sum(
                row["wall_seconds"] for row in results
            )
            / 3600,
        }
        (run_root / "run_results.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if status != "completed" or not math.isfinite(wall_seconds):
            raise RuntimeError(f"P4 stopped after {run_id}: {status} {returncode}")
        if time.monotonic() - total_started > 6 * 3600:
            raise RuntimeError("P4 exceeded the six GPU-hour ceiling")
    return results


def collect(run_root: Path, manifest: dict[str, Any]) -> tuple[Path, Path]:
    series = {
        (arm, seed): load_series(
            run_root / "runs" / f"{arm}-seed-{seed}" / "metrics"
        )
        for arm in ARMS
        for seed in SEEDS
    }
    return write_artifacts(build_rows(series), run_root, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--execute-gpu-training", action="store_true")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    run_root = args.run_root.expanduser().resolve()
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    commands = validate_preflight(manifest, repo_root, run_root)
    if not args.execute_gpu_training:
        print(json.dumps({"dry_run": True, "run_order": list(commands)}, indent=2))
        return 0
    execute(commands, repo_root, run_root)
    csv_path, json_path = collect(run_root, manifest)
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

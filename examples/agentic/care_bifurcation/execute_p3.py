"""Execute the frozen CARe P3 manifest only after explicit GPU approval."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from areno.experimental.care import load_turn_credit_fn

from prepare_p3 import ARMS, PROTOCOL, TRAIN_SEEDS
from fetch_modelscope_snapshot import load_manifest, verify_snapshot


def validate_preflight(
    manifest: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, list[str]]:
    """Fail closed unless the manifest and checkout match the frozen contract."""

    if manifest.get("authorization") != "PREPARE_ONLY_GPU_NOT_AUTHORIZED":
        raise ValueError("unexpected manifest authorization state")
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError(f"expected protocol {PROTOCOL}")
    expected_run_ids = {
        f"{arm}-seed-{seed}"
        for seed in TRAIN_SEEDS
        for arm in ARMS
    }
    command_strings = manifest.get("commands")
    if not isinstance(command_strings, dict) or set(command_strings) != expected_run_ids:
        raise ValueError("manifest must contain exactly the frozen six runs")
    if manifest.get("git_status"):
        raise ValueError("manifest was prepared from a dirty checkout")
    current_commit = _git_output(repo_root, "rev-parse", "HEAD")
    current_status = _git_output(repo_root, "status", "--short")
    if current_commit != manifest.get("git_commit") or current_status:
        raise ValueError("current checkout does not match the clean frozen commit")
    commands = {
        run_id: shlex.split(command)
        for run_id, command in command_strings.items()
    }
    hook_paths = {
        _option_value(command, "--turn-credit-fn-path")
        for command in commands.values()
    }
    if len(hook_paths) != 1:
        raise ValueError("all frozen commands must use one turn-credit hook")
    hook_path = Path(hook_paths.pop())
    if not hook_path.is_absolute():
        raise ValueError("turn-credit hook must be an absolute path")
    load_turn_credit_fn(str(hook_path))
    checkpoint = Path(str(manifest.get("checkpoint", "")))
    if not checkpoint.is_absolute() or not checkpoint.is_dir():
        raise ValueError(
            "P3 execution requires --ckpt to be a verified absolute local snapshot"
        )
    asset_record = manifest.get("model_asset") or {}
    asset_manifest_path = repo_root / str(asset_record.get("manifest_path", ""))
    if (
        not asset_manifest_path.is_file()
        or _sha256(asset_manifest_path) != asset_record.get("manifest_sha256")
    ):
        raise ValueError("frozen ModelScope asset manifest is missing or changed")
    verify_snapshot(checkpoint, load_manifest(asset_manifest_path))
    return commands


def _option_value(command: list[str], option: str) -> str:
    try:
        index = command.index(option)
        value = command[index + 1]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"frozen command is missing {option}") from exc
    if value.startswith("--"):
        raise ValueError(f"frozen command is missing a value for {option}")
    return value


def execute(
    *,
    repo_root: Path,
    run_root: Path,
    commands: dict[str, list[str]],
    timeout_seconds: int,
) -> list[dict[str, Any]]:
    """Run each declared command sequentially and retain every outcome."""

    results = []
    for seed in TRAIN_SEEDS:
        for arm in ARMS:
            run_id = f"{arm}-seed-{seed}"
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
                        timeout=timeout_seconds,
                        text=True,
                    )
                    returncode = int(completed.returncode)
                    if returncode != 0:
                        status = "failed"
                except subprocess.TimeoutExpired:
                    status = "timeout"
            row = {
                "run_id": run_id,
                "arm": arm,
                "seed": seed,
                "status": status,
                "returncode": returncode,
                "wall_seconds": time.monotonic() - started,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }
            results.append(row)
            _write_run_results(run_root, results)
            if status != "completed":
                raise RuntimeError(
                    f"P3 stopped after {run_id}: status={status} "
                    f"returncode={returncode}"
                )
    return results


def _write_run_results(
    run_root: Path,
    results: list[dict[str, Any]],
) -> None:
    (run_root / "run_results.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "results": results,
                "total_single_gpu_hours": (
                    sum(float(row["wall_seconds"]) for row in results) / 3600.0
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


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
        default=Path("artifacts/care-p3-pilot-v02"),
    )
    parser.add_argument(
        "--execute-gpu-training",
        action="store_true",
        help="Run the six frozen commands. Requires prior explicit user approval.",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    run_root = args.run_root.expanduser().resolve()
    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    commands = validate_preflight(manifest, repo_root=repo_root)
    if not args.execute_gpu_training:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "gpu_execution_authorized": False,
                    "validated_run_ids": list(commands),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    timeout_seconds = int(
        float(
            manifest["fixed_design"]["maximum_single_gpu_wall_clock_hours"]
        )
        * 3600
    )
    execute(
        repo_root=repo_root,
        run_root=run_root,
        commands=commands,
        timeout_seconds=timeout_seconds,
    )
    print(f"Run results: {run_root / 'run_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

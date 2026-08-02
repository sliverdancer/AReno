"""Collect fail-closed pretraining rewards for C0 resolution calibration."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

EVALUATOR_PATH = Path(__file__).resolve().parents[1] / "D4_EVAL/evaluate_checkpoint.py"


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("rist_v2_1_c0_strict_runner", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("strict evaluator loader unavailable")
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _consume_ledger(path: Path, family: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        payload = {
            "protocol": "RIST-C0-v2.1",
            "family": family,
            "split": "qualification",
            "consumed": True,
            "consumed_at_unix_ns": time.time_ns(),
            "result_complete": False,
        }
        os.write(descriptor, (json.dumps(payload, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def collect(
    manifest: dict[str, Any],
    family: str,
    split: str,
    output_path: Path,
    journal_path: Path,
    post_json: Callable[[dict[str, Any]], dict[str, Any]],
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    if manifest.get("protocol") != "RIST-C0-v2.1":
        raise ValueError("unexpected C0 manifest")
    if family not in manifest["models"] or split not in {"calibration", "qualification"}:
        raise ValueError("unknown C0 family or split")
    if output_path.exists() or journal_path.exists():
        raise FileExistsError("C0 output and journal must be fresh paths")
    if split == "qualification":
        if ledger_path is None:
            raise ValueError("C0 qualification requires a one-shot ledger")
        _consume_ledger(ledger_path, family)
    elif ledger_path is not None:
        raise ValueError("C0 calibration must not use a qualification ledger")

    source_dir = Path(manifest["source_data_dir"])
    split_spec = manifest["splits"][split]
    data_path = source_dir / split_spec["file"]
    encoded = data_path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != split_spec["sha256"]:
        raise ValueError("C0 source split hash mismatch")
    tasks = [json.loads(line) for line in encoded.decode().splitlines() if line]
    evaluator = _load_evaluator()
    trajectories = []
    infrastructure_error = None
    for task in tasks:
        for rollout_seed in split_spec["rollout_seeds"]:
            try:
                row = evaluator.run_trajectory(
                    task,
                    int(rollout_seed),
                    manifest["sampling"],
                    post_json,
                    journal_path,
                )
                row["split"] = split
                trajectories.append(row)
            except Exception as exc:
                infrastructure_error = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "task_id": task["id"],
                    "rollout_seed": int(rollout_seed),
                }
                break
        if infrastructure_error is not None:
            break
    expected_count = int(split_spec["trajectory_count"])
    complete = infrastructure_error is None and len(trajectories) == expected_count
    result = {
        "protocol": "RIST-C0-v2.1",
        "family": family,
        "checkpoint": manifest["models"][family],
        "split": split,
        "split_sha256": split_spec["sha256"],
        "expected_trajectory_count": expected_count,
        "trajectory_count": len(trajectories),
        "complete": complete,
        "infrastructure_error": infrastructure_error,
        "retry_count": 0,
        "raw_journal_sha256": (
            hashlib.sha256(journal_path.read_bytes()).hexdigest()
            if journal_path.is_file()
            else None
        ),
        "trajectories": trajectories,
    }
    _write_json(output_path, result)
    if ledger_path is not None:
        ledger = json.loads(ledger_path.read_text())
        ledger.update(
            {
                "result_complete": complete,
                "result_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                "journal_sha256": result["raw_journal_sha256"],
            }
        )
        _write_json(ledger_path, ledger)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--family", choices=("qwen3", "gemma4"), required=True)
    parser.add_argument("--split", choices=("calibration", "qualification"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()
    evaluator = _load_evaluator()
    manifest = json.loads(args.manifest.read_text())
    result = collect(
        manifest,
        args.family,
        args.split,
        args.output,
        args.journal,
        lambda payload: evaluator._post_json(
            args.base_url, args.api_key, args.timeout_seconds, payload
        ),
        args.ledger,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "trajectories"}, indent=2))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

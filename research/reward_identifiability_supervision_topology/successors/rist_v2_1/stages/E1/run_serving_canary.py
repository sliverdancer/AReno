"""Run the frozen 32-task E1 serving or one-task checkpoint reload canary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[6]
EVALUATOR_PATH = Path(__file__).parents[1] / "D4_EVAL" / "evaluate_checkpoint.py"
ROLLOUT_SEED = 12101


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("rist_e1_serving_runner", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("E1 evaluator unavailable")
    spec.loader.exec_module(module)
    return module


def run_canary(
    manifest: dict[str, Any],
    family: str,
    runtime_identity: dict[str, Any],
    task_limit: int,
    journal_path: Path,
    post_json,
    protocol: str,
) -> dict[str, Any]:
    if manifest.get("protocol") != "RIST-E1-CAPACITY-v2.1":
        raise ValueError("unexpected E1 manifest")
    model = manifest["models"][family]
    if any(
        (
            runtime_identity.get("family") != family,
            runtime_identity.get("checkpoint") != model["checkpoint"],
            runtime_identity.get("model_revision") != model["revision"],
            runtime_identity.get("tokenizer_snapshot_sha256")
            != model["tokenizer_snapshot_sha256"],
            runtime_identity.get("source_commit") != manifest["source_commit"],
        )
    ):
        raise ValueError("E1 runtime identity does not match manifest")
    if journal_path.exists():
        raise FileExistsError("E1 serving journal path must be fresh")
    c0_manifest_path = Path(__file__).parents[1] / "C0_RESOLUTION" / "collection_manifest.json"
    c0 = json.loads(c0_manifest_path.read_text())
    split = c0["splits"]["calibration"]
    source = REPO_ROOT / c0["source_data_dir"] / split["file"]
    encoded = source.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != split["sha256"]:
        raise ValueError("E1 serving source hash mismatch")
    tasks = [json.loads(line) for line in encoded.decode().splitlines() if line]
    if task_limit not in {1, 32} or len(tasks) < task_limit:
        raise ValueError("E1 serving canary task limit must be one or 32")
    evaluator = _load_evaluator()
    trajectories = []
    infrastructure_error = None
    for task in tasks[:task_limit]:
        try:
            trajectories.append(
                evaluator.run_trajectory(
                    task,
                    ROLLOUT_SEED,
                    {"temperature": 0.0, "top_p": 1.0, "max_tokens": 128},
                    post_json,
                    journal_path,
                )
            )
        except Exception as exc:
            infrastructure_error = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "task_id": task["id"],
            }
            break
    complete_four_turn_count = sum(row["completed_turns"] == 4 for row in trajectories)
    raw_response_count = sum(row["raw_response_count"] for row in trajectories)
    complete = (
        infrastructure_error is None
        and len(trajectories) == task_limit
        and complete_four_turn_count == task_limit
        and raw_response_count == task_limit * 4
    )
    return {
        "protocol": protocol,
        "family": family,
        "runtime_identity": runtime_identity,
        "rollout_seed": ROLLOUT_SEED,
        "task_count": len(trajectories),
        "expected_task_count": task_limit,
        "complete_four_turn_count": complete_four_turn_count,
        "raw_response_count": raw_response_count,
        "retry_count": 0,
        "infrastructure_error": infrastructure_error,
        "complete": complete,
        "raw_journal_sha256": (
            hashlib.sha256(journal_path.read_bytes()).hexdigest()
            if journal_path.is_file()
            else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--family", choices=("qwen3", "gemma4"), required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--task-limit", type=int, choices=(1, 32), required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluator = _load_evaluator()
    protocol = (
        "RIST-E1-SERVING-CANARY-v1"
        if args.task_limit == 32
        else "RIST-E1-RELOAD-CANARY-v1"
    )
    result = run_canary(
        json.loads(args.manifest.read_text()),
        args.family,
        json.loads(args.runtime_identity.read_text()),
        args.task_limit,
        args.journal,
        lambda payload: evaluator._post_json(
            args.base_url, args.api_key, args.timeout_seconds, payload
        ),
        protocol,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Collect one frozen C0 v2.2 scientific job after a verified canary gate."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Callable


EVALUATOR_PATH = Path(__file__).resolve().parents[1] / "D4_EVAL/evaluate_checkpoint.py"
STAGE_ROOT = Path(__file__).resolve().parent


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("rist_c0_v2_2_scientific_http", EVALUATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("strict C0 v2.2 evaluator is unavailable")
    spec.loader.exec_module(module)
    return module


def _load_stage_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, STAGE_ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"C0 v2.2 stage module is unavailable: {filename}")
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        for row in rows:
            os.write(descriptor, (json.dumps(row, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    temporary.replace(path)


def _job(manifest: dict[str, Any], job_id: str) -> dict[str, Any]:
    matches = [row for row in manifest.get("jobs", []) if row.get("job_id") == job_id]
    if len(matches) != 1:
        raise ValueError("C0 v2.2 job id must resolve exactly once")
    return matches[0]


def _validate_gate(manifest: dict[str, Any], job: dict[str, Any], identity: dict[str, Any]) -> None:
    if manifest.get("protocol") != "RIST-C0-v2.2-SCIENTIFIC-COLLECTION-MANIFEST-v1":
        raise ValueError("unexpected C0 v2.2 scientific manifest")
    gate = manifest.get("canary_gate")
    if not isinstance(gate, dict) or not (
        gate.get("passed") is True
        and gate.get("status") == "PASS"
        and gate.get("outcomes_inspected") is False
        and gate.get("scientific_result") is False
    ):
        raise PermissionError("C0 v2.2 scientific collection is blocked until canary PASS")
    expected = {
        "family": job["family"],
        "checkpoint": job["checkpoint"],
        "model_revision": job["model_revision"],
        "tokenizer_snapshot_sha256": job["tokenizer_snapshot_sha256"],
        "source_commit": job["source_commit"],
        "gpu_uuid": job["gpu_uuid"],
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("C0 v2.2 runtime identity does not match the frozen job")
    canary_gpu = gate.get("gpu_identity")
    if not isinstance(canary_gpu, dict) or identity.get("gpu_uuid") != canary_gpu.get("gpu_uuid"):
        raise ValueError("C0 v2.2 scientific job must use the canary-qualified GPU")
    if job.get("retry_count") != 0 or manifest.get("sampling", {}).get("retry_count") != 0:
        raise ValueError("C0 v2.2 scientific collection forbids retries")

    evidence = gate.get("evidence_files")
    required = {
        "canary_manifest",
        "canary_final",
        "pool_manifest",
        "qwen3_canary_result",
        "qwen3_canary_journal",
        "gemma4_canary_result",
        "gemma4_canary_journal",
    }
    if not isinstance(evidence, dict) or set(evidence) != required:
        raise ValueError("C0 v2.2 canary evidence bindings are incomplete")
    paths = {}
    for name, binding in evidence.items():
        if not isinstance(binding, dict):
            raise ValueError("C0 v2.2 canary evidence binding is malformed")
        path = Path(binding.get("path", ""))
        if not path.is_file() or _sha256(path) != binding.get("sha256"):
            raise ValueError(f"C0 v2.2 canary evidence hash mismatch: {name}")
        paths[name] = path
    canary_manifest = json.loads(paths["canary_manifest"].read_text(encoding="utf-8"))
    recorded_final = json.loads(paths["canary_final"].read_text(encoding="utf-8"))
    bindings = {
        family: (
            json.loads(paths[f"{family}_canary_result"].read_text(encoding="utf-8")),
            paths[f"{family}_canary_journal"],
        )
        for family in ("qwen3", "gemma4")
    }
    recomputed = _load_stage_module(
        "rist_c0_v2_2_runtime_finalizer", "finalize_capacity_canary.py"
    ).finalize(canary_manifest, bindings)
    if recomputed != recorded_final or recomputed.get("passed") is not True:
        raise ValueError("C0 v2.2 runtime canary evidence does not reproduce PASS")
    pool_manifest = json.loads(paths["pool_manifest"].read_text(encoding="utf-8"))
    expected_seeds = [
        int(seed)
        for seed in pool_manifest["splits"]["capacity_canary"]["rollout_seeds"]
    ]
    builder = _load_stage_module(
        "rist_c0_v2_2_runtime_builder", "build_scientific_collection_manifest.py"
    )
    for family in ("qwen3", "gemma4"):
        builder._validate_canary_journal(
            bindings[family][0], bindings[family][1], expected_seeds
        )


def collect_job(
    *,
    manifest: dict[str, Any],
    manifest_sha256: str,
    job_id: str,
    runtime_identity: dict[str, Any],
    pool_manifest: dict[str, Any],
    data_dir: Path,
    result_path: Path,
    trajectory_path: Path,
    journal_path: Path,
    post_json: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Run 1,024 attempts while treating scientific outcomes as opaque payloads."""

    job = _job(manifest, job_id)
    _validate_gate(manifest, job, runtime_identity)
    if pool_manifest.get("protocol") != "RIST-C0-v2.2-FRESH-POOL":
        raise ValueError("unexpected C0 v2.2 pool manifest")
    if any(path.exists() for path in (result_path, trajectory_path, journal_path)):
        raise FileExistsError("C0 v2.2 scientific evidence paths must be fresh")
    split = str(job["split"])
    split_spec = pool_manifest["splits"][split]
    source_path = data_dir / split_spec["file"]
    if _sha256(source_path) != split_spec["sha256"]:
        raise ValueError("C0 v2.2 scientific split hash mismatch")
    tasks = [json.loads(line) for line in source_path.read_text().splitlines() if line]
    seeds = [int(seed) for seed in split_spec["rollout_seeds"]]
    expected_count = len(tasks) * len(seeds)
    if expected_count != 1024 or int(job["trajectory_count"]) != expected_count:
        raise ValueError("C0 v2.2 scientific job must contain exactly 1,024 trajectories")
    concurrency = int(job["request_concurrency"])
    if concurrency != int(manifest["request_concurrency"]) or concurrency <= 0:
        raise ValueError("C0 v2.2 request concurrency mismatch")

    evaluator = _load_evaluator()
    trajectories: list[dict[str, Any]] = []
    infrastructure_error = None
    for task in tasks:
        def run_one(seed: int) -> dict[str, Any]:
            return evaluator.run_trajectory(
                task,
                seed,
                manifest["sampling"],
                post_json,
                journal_path,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {seed: pool.submit(run_one, seed) for seed in seeds}
            for seed in seeds:
                if infrastructure_error is not None:
                    futures[seed].cancel()
                    continue
                try:
                    row = futures[seed].result()
                    # Identity and split are metadata. Scientific reward/action fields
                    # remain opaque here and are not used to decide collection flow.
                    row["split"] = split
                    trajectories.append(row)
                except Exception as exc:
                    infrastructure_error = {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "task_id": task["id"],
                        "rollout_seed": seed,
                    }
                    for future in futures.values():
                        future.cancel()
        if infrastructure_error is not None:
            break

    complete = infrastructure_error is None and len(trajectories) == expected_count
    _write_jsonl(trajectory_path, trajectories)
    result = {
        "protocol": "RIST-C0-v2.2-SCIENTIFIC-JOB-EVIDENCE-v1",
        "job_id": job_id,
        "family": job["family"],
        "split": split,
        "runtime_identity": runtime_identity,
        "collection_manifest_sha256": manifest_sha256,
        "expected_trajectory_count": expected_count,
        "trajectory_count": len(trajectories),
        "request_concurrency": concurrency,
        "retry_count": 0,
        "infrastructure_error": infrastructure_error,
        "complete": complete,
        "outcomes_inspected_by_evidence_chain": False,
        "trajectory_artifact_sha256": _sha256(trajectory_path),
        "raw_journal_sha256": _sha256(journal_path) if journal_path.is_file() else None,
    }
    _write_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    job = _job(manifest, args.job_id)
    pool_path = Path(manifest["pool_manifest"])
    if _sha256(pool_path) != manifest["pool_manifest_sha256"]:
        raise ValueError("C0 v2.2 pool manifest hash mismatch")
    evaluator = _load_evaluator()
    result = collect_job(
        manifest=manifest,
        manifest_sha256=_sha256(args.manifest),
        job_id=args.job_id,
        runtime_identity=json.loads(args.runtime_identity.read_text(encoding="utf-8")),
        pool_manifest=json.loads(pool_path.read_text(encoding="utf-8")),
        data_dir=pool_path.parent,
        result_path=Path(job["result"]),
        trajectory_path=Path(job["trajectories"]),
        journal_path=Path(job["raw_journal"]),
        post_json=lambda payload: evaluator._post_json(
            args.base_url, args.api_key, args.timeout_seconds, payload
        ),
    )
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

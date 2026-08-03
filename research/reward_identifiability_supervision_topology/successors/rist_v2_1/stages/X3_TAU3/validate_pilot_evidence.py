"""Validate artifact-backed evidence for the frozen X3 Tau3 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        rows.append(row)
    return rows


def _require_sha256(value: Any, field: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a SHA256 hex digest") from exc


def _validate_identity(
    evidence: dict[str, Any], job: dict[str, Any], manifest: dict[str, Any]
) -> None:
    exact = {
        "protocol": manifest["protocol"],
        "run_id": job["run_id"],
        "family": job["family"],
        "algorithm": job["algorithm"],
        "arm": job["arm"],
        "seed": job["seed"],
        "max_steps": job["max_steps"],
        "group_size": job["group_size"],
        "source_commit": manifest["source_commit"],
        "tau3_commit": manifest["tau3_commit"],
        "policy_retry_count": 0,
        "user_retry_count": 0,
        "completed": True,
    }
    for field, expected in exact.items():
        if evidence.get(field) != expected:
            raise ValueError(
                f"{job['run_id']} evidence {field} != frozen value {expected!r}"
            )
    simulator = evidence.get("user_simulator")
    if not isinstance(simulator, dict):
        raise ValueError(f"{job['run_id']} lacks structured user_simulator identity")
    for field in ("provider", "model", "revision", "runtime_value", "seed_derivation"):
        if not isinstance(simulator.get(field), str) or not simulator[field]:
            raise ValueError(f"{job['run_id']} user_simulator.{field} is not pinned")
    if simulator.get("temperature") != 0.0 or simulator.get("num_retries") != 0:
        raise ValueError(f"{job['run_id']} user simulator is not zero-temperature/zero-retry")
    model = evidence.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{job['run_id']} lacks model identity")
    for field in ("revision", "tokenizer_revision", "weights_manifest_sha256"):
        if not isinstance(model.get(field), str) or not model[field]:
            raise ValueError(f"{job['run_id']} model.{field} is not pinned")
    _require_sha256(model["weights_manifest_sha256"], "model.weights_manifest_sha256")
    gpu = evidence.get("gpu")
    if not isinstance(gpu, dict) or not all(gpu.get(field) for field in ("name", "uuid")):
        raise ValueError(f"{job['run_id']} lacks GPU name/UUID identity")
    if type(gpu.get("memory_total_bytes")) is not int or gpu["memory_total_bytes"] <= 0:
        raise ValueError(f"{job['run_id']} lacks GPU capacity evidence")
    for field in (
        "source_archive_sha256",
        "raw_events_sha256",
        "reward_events_sha256",
        "metrics_manifest_sha256",
        "checkpoint_manifest_sha256",
    ):
        _require_sha256(evidence.get(field), field)


def _episode_key(row: dict[str, Any]) -> tuple[int, int]:
    try:
        return int(row["training_step"]), int(row["sample_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("journal row lacks integer training_step/sample_index") from exc


def _validate_journals(
    raw_rows: list[dict[str, Any]],
    reward_rows: list[dict[str, Any]],
    job: dict[str, Any],
    allowed_task_ids: set[str],
    evidence_user_simulator: str,
) -> set[str]:
    expected_keys = {
        (step, sample)
        for step in range(job["max_steps"])
        for sample in range(job["group_size"])
    }
    if len(reward_rows) != job["expected_episode_count"]:
        raise ValueError(f"{job['run_id']} has wrong reward episode count")
    reward_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for row in reward_rows:
        key = _episode_key(row)
        if key in reward_by_key:
            raise ValueError(f"{job['run_id']} has duplicate reward episode {key}")
        if int(row.get("prompt_index", -1)) != 0:
            raise ValueError(f"{job['run_id']} expected batch-size-one prompt_index=0")
        if row.get("domain") != "airline" or row.get("task_id") not in allowed_task_ids:
            raise ValueError(f"{job['run_id']} reward journal contains non-training task")
        if type(row.get("reward")) not in (int, float) or float(row["reward"]) not in {
            0.0,
            1.0,
        }:
            raise ValueError(f"{job['run_id']} contains a non-binary reward")
        if row.get("evaluator") != "tau2.EvaluationType.ALL":
            raise ValueError(f"{job['run_id']} contains a non-frozen evaluator")
        if row.get("policy_retry_count") != 0 or row.get("user_retry_count") != 0:
            raise ValueError(f"{job['run_id']} contains a retried episode")
        if row.get("user_simulator") != evidence_user_simulator:
            raise ValueError(f"{job['run_id']} episode user simulator mismatch")
        if type(row.get("user_seed")) is not int:
            raise ValueError(f"{job['run_id']} episode lacks a deterministic user seed")
        _require_sha256(row.get("runtime_evidence_sha256"), "runtime_evidence_sha256")
        reward_by_key[key] = row
    if set(reward_by_key) != expected_keys:
        raise ValueError(f"{job['run_id']} reward journal does not cover the frozen grid")

    raw_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        key = _episode_key(row)
        if key not in expected_keys:
            raise ValueError(f"{job['run_id']} raw journal has an out-of-grid episode")
        if int(row.get("prompt_index", -1)) != 0:
            raise ValueError(f"{job['run_id']} raw prompt_index is not zero")
        if row.get("task_id") != reward_by_key[key]["task_id"]:
            raise ValueError(f"{job['run_id']} raw/reward task identity mismatch")
        raw_by_key[key].append(row)
    if set(raw_by_key) != expected_keys:
        raise ValueError(f"{job['run_id']} raw journal does not cover the frozen grid")

    identity_fields = {"task_id", "training_step", "prompt_index", "sample_index"}
    for key, rows in raw_by_key.items():
        if not any(row.get("phase") == "policy_response" for row in rows):
            raise ValueError(f"{job['run_id']} episode {key} lacks a policy response")
        terminals = [
            row
            for row in rows
            if row.get("phase") in {"environment_step", "cleanup_not_policy"}
            and row.get("simulation_run")
        ]
        if len(terminals) != 1:
            raise ValueError(f"{job['run_id']} episode {key} lacks one terminal SimulationRun")
        evidence = [
            {field: value for field, value in row.items() if field not in identity_fields}
            for row in rows
        ]
        digest = hashlib.sha256(
            json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if digest != reward_by_key[key]["runtime_evidence_sha256"]:
            raise ValueError(f"{job['run_id']} episode {key} evidence hash mismatch")
    return {row["task_id"] for row in reward_rows}


def validate_pilot(
    manifest_path: Path, artifact_root: Path, dataset_path: Path
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    jobs = manifest.get("jobs")
    if manifest.get("job_count") != 16 or not isinstance(jobs, list) or len(jobs) != 16:
        raise ValueError("X3 manifest must contain the frozen 16-cell pilot")
    if manifest.get("pilot_steps") != 25 or manifest.get("execution_authorized") is not False:
        raise ValueError("X3 manifest is not the frozen CPU-only execution template")
    dataset = _read_jsonl(dataset_path)
    allowed_task_ids = {row.get("id") for row in dataset}
    if len(dataset) != 22 or None in allowed_task_ids:
        raise ValueError("X3 dataset must contain exactly 22 frozen airline task IDs")

    simulator_identity = None
    family_model_identity: dict[str, dict[str, Any]] = {}
    covered_tasks: set[str] = set()
    run_summaries = []
    for job in jobs:
        run_root = artifact_root / job["run_id"]
        raw_path = run_root / "raw_events.jsonl"
        reward_path = run_root / "reward_events.jsonl"
        evidence = _read_json(run_root / "run_evidence.json")
        _validate_identity(evidence, job, manifest)
        if evidence["raw_events_sha256"] != _sha256(raw_path):
            raise ValueError(f"{job['run_id']} raw journal file hash mismatch")
        if evidence["reward_events_sha256"] != _sha256(reward_path):
            raise ValueError(f"{job['run_id']} reward journal file hash mismatch")
        if simulator_identity is None:
            simulator_identity = evidence["user_simulator"]
        elif evidence["user_simulator"] != simulator_identity:
            raise ValueError("user simulator identity differs across X3 cells")
        family = job["family"]
        if family in family_model_identity and evidence["model"] != family_model_identity[family]:
            raise ValueError(f"model identity differs within family {family}")
        family_model_identity.setdefault(family, evidence["model"])
        run_tasks = _validate_journals(
            _read_jsonl(raw_path),
            _read_jsonl(reward_path),
            job,
            allowed_task_ids,
            evidence["user_simulator"]["runtime_value"],
        )
        covered_tasks.update(run_tasks)
        run_summaries.append(
            {
                "run_id": job["run_id"],
                "episode_count": job["expected_episode_count"],
                "task_count": len(run_tasks),
            }
        )
    if covered_tasks != allowed_task_ids:
        raise ValueError("X3 pilot does not cover every frozen airline training task")
    return {
        "protocol": manifest["protocol"],
        "status": "PASS",
        "scientific_result": False,
        "job_count": len(jobs),
        "episode_count": sum(row["episode_count"] for row in run_summaries),
        "covered_task_count": len(covered_tasks),
        "policy_retry_count": 0,
        "user_retry_count": 0,
        "runs": run_summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_pilot(args.manifest, args.artifact_root, args.dataset)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

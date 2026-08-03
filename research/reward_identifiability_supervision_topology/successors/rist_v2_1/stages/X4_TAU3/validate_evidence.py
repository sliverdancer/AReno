"""Validate an authorized prefix of the frozen X4 Tau3 evidence archive."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any

STAGE = Path(__file__).resolve().parent
X3_VALIDATOR = STAGE.parent / "X3_TAU3/validate_pilot_evidence.py"
HEX64 = re.compile(r"[0-9a-f]{64}")


def _load_x3_validator():
    spec = importlib.util.spec_from_file_location("rist_x4_x3_validator", X3_VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("X3 evidence validator unavailable")
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number} is not an object")
        rows.append(value)
    return rows


def _digest(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def validate(
    manifest: dict[str, Any],
    evidence_root: Path,
    x3_validation_path: Path,
    executed_seed_count: int,
) -> dict[str, Any]:
    x3 = json.loads(x3_validation_path.read_text())
    prerequisite = manifest["x3_pilot_prerequisite"]
    if prerequisite.get("validation_sha256") != _sha256(x3_validation_path):
        raise ValueError("X3 prerequisite artifact hash mismatch")
    if (
        x3.get("status") != prerequisite["required_status"]
        or x3.get("job_count") != prerequisite["required_job_count"]
        or x3.get("episode_count") != prerequisite["required_episode_count"]
    ):
        raise ValueError("X3 pilot prerequisite did not pass exactly")
    if manifest.get("execution_authorized") is not True:
        raise ValueError("X4 evidence requires a separately authorized resolved manifest")
    simulator = manifest.get("user_simulator_binding")
    if not isinstance(simulator, dict):
        raise ValueError("X4 user simulator binding is unresolved")
    for field in ("model", "revision", "runtime_value", "authorization_sha256"):
        if not simulator.get(field):
            raise ValueError(f"X4 simulator {field} is unresolved")
    if not _digest(simulator["authorization_sha256"]):
        raise ValueError("X4 simulator authorization hash is invalid")
    families = manifest.get("family_bindings", {})
    for family in ("qwen3", "gemma4"):
        binding = families.get(family)
        if not isinstance(binding, dict) or not all(binding.values()):
            raise ValueError(f"X4 model binding is unresolved for {family}")
        if not _digest(binding.get("weights_manifest_sha256")):
            raise ValueError(f"X4 model weight hash is invalid for {family}")
    seeds = [int(seed) for seed in manifest["seed_bank"]]
    if not 12 <= executed_seed_count <= len(seeds) <= 32:
        raise ValueError("executed seed count is outside the frozen 12..32 range")
    selected = set(seeds[:executed_seed_count])
    jobs = [job for job in manifest["jobs"] if int(job["seed"]) in selected]
    if len(jobs) != executed_seed_count * 16:
        raise ValueError("X4 selected jobs do not form complete paired factorial blocks")

    x3_validator = _load_x3_validator()
    allowed_tasks = set(manifest["required_training_task_ids"])
    records = []
    for job in jobs:
        run_id = job["run_id"]
        run_root = evidence_root / run_id
        raw_path = run_root / "raw_events.jsonl"
        reward_path = run_root / "reward_events.jsonl"
        evidence = json.loads((run_root / "run_evidence.json").read_text())
        exact = {
            "protocol": manifest["protocol"],
            "run_id": run_id,
            "source_commit": manifest["source_commit"],
            "family": job["family"],
            "algorithm": job["algorithm"],
            "arm": job["arm"],
            "seed": job["seed"],
            "completed": True,
            "policy_retry_count": 0,
            "user_retry_count": 0,
            "dataset_sha256": job["dataset_sha256"],
            "source_file_sha256": manifest["source_file_sha256"],
            "model": families[job["family"]],
            "user_simulator": simulator,
        }
        for field, expected in exact.items():
            if evidence.get(field) != expected:
                raise ValueError(f"{run_id} evidence {field} mismatch")
        for field in (
            "metrics_manifest_sha256",
            "checkpoint_manifest_sha256",
            "source_archive_sha256",
        ):
            if not _digest(evidence.get(field)):
                raise ValueError(f"{run_id} lacks {field}")
        if evidence.get("raw_events_sha256") != _sha256(raw_path):
            raise ValueError(f"{run_id} raw journal hash mismatch")
        if evidence.get("reward_events_sha256") != _sha256(reward_path):
            raise ValueError(f"{run_id} reward journal hash mismatch")
        run_tasks = x3_validator._validate_journals(
            _read_jsonl(raw_path),
            _read_jsonl(reward_path),
            job,
            allowed_tasks,
            simulator["runtime_value"],
            simulator["revision"],
            simulator["authorization_sha256"],
            manifest["source_commit"],
        )
        if run_tasks != allowed_tasks:
            raise ValueError(f"{run_id} does not exercise all frozen airline tasks")
        record = {
            "run_id": run_id,
            "seed": job["seed"],
            "family": job["family"],
            "algorithm": job["algorithm"],
            "arm": job["arm"],
            "catastrophic": evidence.get("catastrophic"),
            "strict_success": evidence.get("strict_success"),
            "token_auc": evidence.get("token_auc"),
            "token_endpoint": evidence.get("token_endpoint"),
            "token_support_fraction": evidence.get("token_support_fraction"),
        }
        if record["catastrophic"] is not False:
            raise ValueError(f"{run_id} is catastrophic")
        for field in ("strict_success", "token_auc", "token_endpoint"):
            value = record[field]
            if (
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{run_id} has invalid {field}")
        support = record["token_support_fraction"]
        if type(support) not in (int, float) or not 0.0 <= float(support) <= 1.0:
            raise ValueError(f"{run_id} has invalid token support")
        records.append(record)
    records.sort(
        key=lambda row: (
            seeds.index(int(row["seed"])),
            row["family"],
            row["algorithm"],
            row["arm"],
        )
    )
    return {
        "protocol": "RIST-X4-TAU3-EVIDENCE-v2.1",
        "passed": True,
        "executed_seed_count": executed_seed_count,
        "job_count": len(jobs),
        "episode_count": len(jobs) * 200,
        "records_sha256": _canonical_sha256(records),
        "records": records,
        "scientific_result": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--x3-validation", type=Path, required=True)
    parser.add_argument("--executed-seed-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        json.loads(args.manifest.read_text()),
        args.evidence_root,
        args.x3_validation,
        args.executed_seed_count,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

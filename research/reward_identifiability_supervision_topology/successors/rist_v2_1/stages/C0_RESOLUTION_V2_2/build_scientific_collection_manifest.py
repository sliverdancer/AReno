"""Build C0 v2.2 scientific jobs only from a verified capacity-canary PASS."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any


STAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[6]
FINALIZER_PATH = STAGE_ROOT / "finalize_capacity_canary.py"
FAMILIES = ("qwen3", "gemma4")
SCIENTIFIC_SPLITS = ("calibration", "qualification")
EXPECTED_TRAJECTORIES_PER_JOB = 1024
EXPECTED_TRAJECTORIES_TOTAL = 4096


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_finalizer():
    spec = importlib.util.spec_from_file_location("rist_c0_v2_2_canary_finalizer", FINALIZER_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("C0 v2.2 canary finalizer is unavailable")
    spec.loader.exec_module(module)
    return module


def _require_commit(value: Any) -> str:
    text = str(value)
    if len(text) != 40 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError("collection source commit must be a full lowercase Git commit")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if text != head:
        raise ValueError("collection source commit must equal the checked-out Git HEAD")
    return text


def _validate_canary_journal(
    result: dict[str, Any], journal_path: Path, expected_seeds: list[int]
) -> None:
    rows = [json.loads(line) for line in journal_path.read_text().splitlines() if line]
    observed_seeds = [int(row.get("request_seed", -1)) for row in rows]
    if (
        len(rows) != len(expected_seeds)
        or sorted(observed_seeds) != sorted(expected_seeds)
        or len(observed_seeds) != len(set(observed_seeds))
        or [int(seed) for seed in result.get("request_seeds", [])] != expected_seeds
    ):
        raise ValueError("capacity-canary journal does not contain the exact frozen requests")
    for row in rows:
        response = row.get("raw_response")
        if not isinstance(response, dict):
            raise ValueError("capacity-canary journal lacks a raw response")
        digest = hashlib.sha256(
            json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if row.get("response_sha256") != digest:
            raise ValueError("capacity-canary response hash mismatch")


def build_manifest(
    *,
    pool_manifest_path: Path,
    canary_manifest_path: Path,
    canary_final_path: Path,
    canary_bindings: dict[str, tuple[Path, Path]],
    collection_source_commit: str,
    output_root: Path,
) -> dict[str, Any]:
    """Recompute the canary gate and freeze four content-blind collection jobs."""

    pool = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
    canary_manifest = json.loads(canary_manifest_path.read_text(encoding="utf-8"))
    recorded_final = json.loads(canary_final_path.read_text(encoding="utf-8"))
    if pool.get("protocol") != "RIST-C0-v2.2-FRESH-POOL":
        raise ValueError("unexpected C0 v2.2 pool manifest")
    if canary_manifest.get("protocol") != "RIST-C0-v2.2-CANARY-MANIFEST":
        raise ValueError("unexpected C0 v2.2 canary manifest")
    if _sha256(pool_manifest_path) != canary_manifest.get("pool_manifest_sha256"):
        raise ValueError("capacity canary is not bound to this fresh pool")
    if set(canary_bindings) != set(FAMILIES):
        raise ValueError("capacity gate requires both frozen family bindings")

    raw_bindings = {
        family: (
            json.loads(canary_bindings[family][0].read_text(encoding="utf-8")),
            canary_bindings[family][1],
        )
        for family in FAMILIES
    }
    expected_canary_seeds = [
        int(seed) for seed in pool["splits"]["capacity_canary"]["rollout_seeds"]
    ]
    for family in FAMILIES:
        _validate_canary_journal(
            raw_bindings[family][0], raw_bindings[family][1], expected_canary_seeds
        )
    recomputed_final = _load_finalizer().finalize(canary_manifest, raw_bindings)
    if recomputed_final != recorded_final:
        raise ValueError("recorded capacity-canary final does not match raw evidence")
    if not (
        recomputed_final.get("passed") is True
        and recomputed_final.get("status") == "PASS"
        and recomputed_final.get("decision") == "PASS_CANARY_TO_SEPARATE_C0_AUTHORIZATION"
        and recomputed_final.get("outcomes_inspected") is False
        and recomputed_final.get("scientific_result") is False
    ):
        raise ValueError("scientific collection requires an outcome-free capacity-canary PASS")

    source_commit = _require_commit(collection_source_commit)
    gpu_identity = recomputed_final.get("gpu_identity")
    if not isinstance(gpu_identity, dict) or not str(gpu_identity.get("gpu_uuid", "")):
        raise ValueError("capacity canary lacks a bound GPU UUID")
    if float(gpu_identity.get("gpu_total_memory_gib", 0.0)) < float(
        canary_manifest["minimum_gpu_memory_gib"]
    ):
        raise ValueError("capacity-canary GPU is below the frozen memory floor")

    jobs = []
    for family in FAMILIES:
        for split in SCIENTIFIC_SPLITS:
            split_spec = pool["splits"][split]
            trajectory_count = int(split_spec["task_count"]) * len(split_spec["rollout_seeds"])
            if trajectory_count != EXPECTED_TRAJECTORIES_PER_JOB:
                raise ValueError("each C0 v2.2 scientific job must contain 1,024 trajectories")
            job_id = f"{family}-{split}"
            job_root = output_root / job_id
            jobs.append(
                {
                    "job_id": job_id,
                    "family": family,
                    "split": split,
                    "checkpoint": canary_manifest["models"][family],
                    "model_revision": canary_manifest["model_revisions"][family],
                    "tokenizer_snapshot_sha256": canary_manifest["tokenizer_snapshot_sha256"][family],
                    "source_commit": source_commit,
                    "gpu_uuid": gpu_identity["gpu_uuid"],
                    "trajectory_count": trajectory_count,
                    "request_concurrency": int(canary_manifest["request_concurrency"]),
                    "retry_count": 0,
                    "result": str(job_root / "result.json"),
                    "trajectories": str(job_root / "trajectories.jsonl"),
                    "raw_journal": str(job_root / "raw_responses.jsonl"),
                }
            )
    if sum(job["trajectory_count"] for job in jobs) != EXPECTED_TRAJECTORIES_TOTAL:
        raise ValueError("C0 v2.2 collection must freeze exactly 4,096 trajectories")

    evidence_files = {
        "canary_manifest": {
            "path": str(canary_manifest_path),
            "sha256": _sha256(canary_manifest_path),
        },
        "canary_final": {
            "path": str(canary_final_path),
            "sha256": _sha256(canary_final_path),
        },
        "pool_manifest": {
            "path": str(pool_manifest_path),
            "sha256": _sha256(pool_manifest_path),
        },
    }
    for family in FAMILIES:
        evidence_files[f"{family}_canary_result"] = {
            "path": str(canary_bindings[family][0]),
            "sha256": _sha256(canary_bindings[family][0]),
        }
        evidence_files[f"{family}_canary_journal"] = {
            "path": str(canary_bindings[family][1]),
            "sha256": _sha256(canary_bindings[family][1]),
        }
    return {
        "protocol": "RIST-C0-v2.2-SCIENTIFIC-COLLECTION-MANIFEST-v1",
        "collection_source_commit": source_commit,
        "pool_manifest": str(pool_manifest_path),
        "pool_manifest_sha256": evidence_files["pool_manifest"]["sha256"],
        "canary_gate": {
            "passed": True,
            "status": "PASS",
            "outcomes_inspected": False,
            "scientific_result": False,
            "gpu_identity": gpu_identity,
            "evidence_files": evidence_files,
        },
        "models": canary_manifest["models"],
        "model_revisions": canary_manifest["model_revisions"],
        "tokenizer_snapshot_sha256": canary_manifest["tokenizer_snapshot_sha256"],
        "sampling": {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 128,
            "retry_count": 0,
        },
        "request_concurrency": int(canary_manifest["request_concurrency"]),
        "job_count": len(jobs),
        "trajectory_count": sum(job["trajectory_count"] for job in jobs),
        "jobs": jobs,
        "training_permitted": False,
        "heldout_permitted": False,
        "outcomes_inspected_by_gate": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--canary-manifest", type=Path, required=True)
    parser.add_argument("--canary-final", type=Path, required=True)
    for family in ("qwen", "gemma"):
        parser.add_argument(f"--{family}-canary-result", type=Path, required=True)
        parser.add_argument(f"--{family}-canary-journal", type=Path, required=True)
    parser.add_argument("--collection-source-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(
        pool_manifest_path=args.pool_manifest,
        canary_manifest_path=args.canary_manifest,
        canary_final_path=args.canary_final,
        canary_bindings={
            "qwen3": (args.qwen_canary_result, args.qwen_canary_journal),
            "gemma4": (args.gemma_canary_result, args.gemma_canary_journal),
        },
        collection_source_commit=args.collection_source_commit,
        output_root=args.output_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

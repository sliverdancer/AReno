from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    REPO_ROOT
    / "research/reward_identifiability_supervision_topology/successors/rist_v2_1/stages/C0_RESOLUTION_V2_2"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path):
    return json.loads(path.read_text())


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _canary_evidence(tmp_path: Path):
    finalizer = _load("rist_c0_v2_2_finalizer_test", STAGE / "finalize_capacity_canary.py")
    manifest_path = STAGE / "CANARY_EXECUTION_MANIFEST.json"
    manifest = _json(manifest_path)
    canary_seeds = _json(STAGE / "data/manifest.json")["splits"]["capacity_canary"][
        "rollout_seeds"
    ]
    gpu = {
        "gpu_name": "Mock 80GB GPU",
        "gpu_uuid": "GPU-c0-v2-2-test",
        "gpu_total_memory_gib": 80.0,
        "driver_version": "test",
        "cuda_version": "test",
        "torch_version": "test",
    }
    bindings = {}
    path_bindings = {}
    for family in ("qwen3", "gemma4"):
        journal_path = tmp_path / "canary" / family / "raw.jsonl"
        journal_rows = []
        for seed in canary_seeds:
            response = {"choices": [], "seed": seed}
            journal_rows.append(
                {
                    "request_seed": seed,
                    "response_sha256": hashlib.sha256(
                        json.dumps(response, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "raw_response": response,
                }
            )
        _write_jsonl(journal_path, journal_rows)
        identity = {
            "family": family,
            "checkpoint": manifest["models"][family],
            "model_revision": manifest["model_revisions"][family],
            "tokenizer_snapshot_sha256": manifest["tokenizer_snapshot_sha256"][family],
            "source_commit": manifest["source_commit"],
            **gpu,
        }
        result = {
            "protocol": "RIST-C0-v2.2-CAPACITY-CANARY",
            "family": family,
            "runtime_identity": identity,
            "request_concurrency": 8,
            "expected_response_count": 8,
            "response_count": 8,
            "request_seeds": list(canary_seeds),
            "retry_count": 0,
            "infrastructure_error": None,
            "complete": True,
            "outcomes_inspected": False,
            "scientific_result": False,
            "peak_memory_mib": 1000.0,
            "raw_journal_sha256": _sha(journal_path),
        }
        result_path = tmp_path / "canary" / family / "result.json"
        _write_json(result_path, result)
        bindings[family] = (result, journal_path)
        path_bindings[family] = (result_path, journal_path)
    final = finalizer.finalize(manifest, bindings)
    assert final["passed"] is True
    final_path = tmp_path / "canary" / "final.json"
    _write_json(final_path, final)
    return manifest_path, final_path, path_bindings


def _scientific_manifest(tmp_path: Path):
    builder = _load(
        "rist_c0_v2_2_manifest_builder_test",
        STAGE / "build_scientific_collection_manifest.py",
    )
    canary_manifest, canary_final, bindings = _canary_evidence(tmp_path)
    manifest = builder.build_manifest(
        pool_manifest_path=STAGE / "data/manifest.json",
        canary_manifest_path=canary_manifest,
        canary_final_path=canary_final,
        canary_bindings=bindings,
        collection_source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        output_root=tmp_path / "collection",
    )
    manifest["pool_manifest"] = str(STAGE / "data/manifest.json")
    manifest_path = tmp_path / "collection_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, manifest


def test_v2_2_scientific_manifest_requires_recomputed_canary_pass(tmp_path):
    builder = _load(
        "rist_c0_v2_2_manifest_builder_reject_test",
        STAGE / "build_scientific_collection_manifest.py",
    )
    canary_manifest, canary_final, bindings = _canary_evidence(tmp_path)
    qwen_result = _json(bindings["qwen3"][0])
    qwen_result["runtime_identity"]["model_revision"] = "wrong"
    _write_json(bindings["qwen3"][0], qwen_result)
    with pytest.raises(ValueError, match="recorded capacity-canary final"):
        builder.build_manifest(
            pool_manifest_path=STAGE / "data/manifest.json",
            canary_manifest_path=canary_manifest,
            canary_final_path=canary_final,
            canary_bindings=bindings,
            collection_source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
            ).strip(),
            output_root=tmp_path / "collection",
        )


def test_v2_2_scientific_manifest_binds_exact_4096_and_runtime_identity(tmp_path):
    _, manifest = _scientific_manifest(tmp_path)
    assert manifest["canary_gate"]["passed"] is True
    assert manifest["trajectory_count"] == 4096
    assert manifest["job_count"] == 4
    assert sum(job["trajectory_count"] for job in manifest["jobs"]) == 4096
    assert {job["family"] for job in manifest["jobs"]} == {"qwen3", "gemma4"}
    assert {job["split"] for job in manifest["jobs"]} == {"calibration", "qualification"}
    assert all(job["retry_count"] == 0 for job in manifest["jobs"])
    expected_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    assert all(job["source_commit"] == expected_commit for job in manifest["jobs"])
    assert all(job["gpu_uuid"] == "GPU-c0-v2-2-test" for job in manifest["jobs"])


def test_v2_2_collector_rejects_missing_canary_and_wrong_runtime_identity(tmp_path):
    collector = _load(
        "rist_c0_v2_2_collector_gate_test",
        STAGE / "collect_scientific_job.py",
    )
    _, manifest = _scientific_manifest(tmp_path)
    job = manifest["jobs"][0]
    identity = {
        "family": job["family"],
        "checkpoint": job["checkpoint"],
        "model_revision": job["model_revision"],
        "tokenizer_snapshot_sha256": job["tokenizer_snapshot_sha256"],
        "source_commit": job["source_commit"],
        "gpu_uuid": job["gpu_uuid"],
    }
    blocked = json.loads(json.dumps(manifest))
    blocked["canary_gate"]["passed"] = False
    with pytest.raises(PermissionError, match="blocked until canary PASS"):
        collector._validate_gate(blocked, job, identity)
    wrong = dict(identity, gpu_uuid="GPU-wrong")
    with pytest.raises(ValueError, match="runtime identity"):
        collector._validate_gate(manifest, job, wrong)
    collector._validate_gate(manifest, job, identity)
    journal_binding = manifest["canary_gate"]["evidence_files"][
        "qwen3_canary_journal"
    ]
    with Path(journal_binding["path"]).open("a") as stream:
        stream.write('{"tampered": true}\n')
    with pytest.raises(ValueError, match="evidence hash mismatch"):
        collector._validate_gate(manifest, job, identity)


def _materialize_opaque_job(manifest_path: Path, manifest: dict, job: dict) -> None:
    pool_path = Path(manifest["pool_manifest"])
    pool = _json(pool_path)
    split_spec = pool["splits"][job["split"]]
    tasks = [json.loads(line) for line in (pool_path.parent / split_spec["file"]).read_text().splitlines()]
    trajectories = []
    journal = []
    for task in tasks:
        for seed in split_spec["rollout_seeds"]:
            # Deliberately contains identity metadata only: the evidence verifier
            # must not require, inspect, or derive any scientific outcome.
            trajectories.append(
                {
                    "task_id": task["id"],
                    "task_signature": task["task_signature"],
                    "rollout_seed": seed,
                    "split": job["split"],
                    "raw_response_count": 1,
                }
            )
            journal.append(
                {
                    "task_id": task["id"],
                    "task_signature": task["task_signature"],
                    "rollout_seed": seed,
                    "turn_index": 0,
                    "raw_response": {"opaque": True},
                }
            )
    trajectory_path = Path(job["trajectories"])
    journal_path = Path(job["raw_journal"])
    _write_jsonl(trajectory_path, trajectories)
    _write_jsonl(journal_path, journal)
    identity = {
        "family": job["family"],
        "checkpoint": job["checkpoint"],
        "model_revision": job["model_revision"],
        "tokenizer_snapshot_sha256": job["tokenizer_snapshot_sha256"],
        "source_commit": job["source_commit"],
        "gpu_uuid": job["gpu_uuid"],
    }
    _write_json(
        Path(job["result"]),
        {
            "protocol": "RIST-C0-v2.2-SCIENTIFIC-JOB-EVIDENCE-v1",
            "job_id": job["job_id"],
            "family": job["family"],
            "split": job["split"],
            "runtime_identity": identity,
            "collection_manifest_sha256": _sha(manifest_path),
            "expected_trajectory_count": 1024,
            "trajectory_count": 1024,
            "request_concurrency": job["request_concurrency"],
            "retry_count": 0,
            "infrastructure_error": None,
            "complete": True,
            "outcomes_inspected_by_evidence_chain": False,
            "trajectory_artifact_sha256": _sha(trajectory_path),
            "raw_journal_sha256": _sha(journal_path),
        },
    )


def test_v2_2_collector_executes_exact_1024_zero_retry_attempts(tmp_path):
    collector = _load(
        "rist_c0_v2_2_collector_execution_test",
        STAGE / "collect_scientific_job.py",
    )
    validator = _load(
        "rist_c0_v2_2_collector_validator_test",
        STAGE / "validate_scientific_job.py",
    )
    manifest_path, manifest = _scientific_manifest(tmp_path)
    job = manifest["jobs"][0]
    identity = {
        "family": job["family"],
        "checkpoint": job["checkpoint"],
        "model_revision": job["model_revision"],
        "tokenizer_snapshot_sha256": job["tokenizer_snapshot_sha256"],
        "source_commit": job["source_commit"],
        "gpu_uuid": job["gpu_uuid"],
    }
    pool_path = Path(manifest["pool_manifest"])
    pool = _json(pool_path)
    result = collector.collect_job(
        manifest=manifest,
        manifest_sha256=_sha(manifest_path),
        job_id=job["job_id"],
        runtime_identity=identity,
        pool_manifest=pool,
        data_dir=pool_path.parent,
        result_path=Path(job["result"]),
        trajectory_path=Path(job["trajectories"]),
        journal_path=Path(job["raw_journal"]),
        post_json=lambda _: {
            "choices": [{"message": {"role": "assistant", "tool_calls": []}}]
        },
    )
    assert result["complete"] is True
    assert result["trajectory_count"] == 1024
    assert result["retry_count"] == 0
    split_spec = pool["splits"][job["split"]]
    source_path = pool_path.parent / split_spec["file"]
    validation = validator.validate_job(
        manifest=manifest,
        manifest_sha256=_sha(manifest_path),
        pool_manifest=pool,
        source_rows=[json.loads(line) for line in source_path.read_text().splitlines()],
        job_id=job["job_id"],
        result=result,
        trajectories=[
            json.loads(line) for line in Path(job["trajectories"]).read_text().splitlines()
        ],
        trajectory_sha256=_sha(Path(job["trajectories"])),
        journal_rows=[
            json.loads(line) for line in Path(job["raw_journal"]).read_text().splitlines()
        ],
        journal_sha256=_sha(Path(job["raw_journal"])),
    )
    assert validation["passed"] is True
    assert validation["outcomes_inspected"] is False


def test_v2_2_finalizer_revalidates_4096_opaque_trajectories_and_hashes(tmp_path):
    finalizer = _load(
        "rist_c0_v2_2_collection_finalizer_test",
        STAGE / "finalize_scientific_collection.py",
    )
    manifest_path, manifest = _scientific_manifest(tmp_path)
    for job in manifest["jobs"]:
        _materialize_opaque_job(manifest_path, manifest, job)
    result = finalizer.finalize(manifest_path)
    assert result["passed"] is True
    assert result["trajectory_count"] == 4096
    assert result["retry_count"] == 0
    assert result["outcomes_inspected"] is False
    assert all(len(job["raw_journal_sha256"]) == 64 for job in result["jobs"])

    first = manifest["jobs"][0]
    with Path(first["raw_journal"]).open("a") as stream:
        stream.write('{"tampered": true}\n')
    tampered = finalizer.finalize(manifest_path)
    assert tampered["passed"] is False
    assert next(job for job in tampered["jobs"] if job["job_id"] == first["job_id"])["passed"] is False

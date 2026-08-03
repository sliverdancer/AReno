"""CPU-only adversarial tests for the artifact-backed RIST E1 gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
E1 = (
    REPO_ROOT
    / "research/reward_identifiability_supervision_topology/successors/rist_v2_1/stages/E1"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _json(value) -> str:
    return json.dumps(value, sort_keys=True) + "\n"


def _jsonl(rows) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)


def _write(root: Path, relative: str, content: str | bytes) -> tuple[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = content if isinstance(content, bytes) else content.encode()
    path.write_bytes(encoded)
    return relative, hashlib.sha256(encoded).hexdigest()


def _monitor(peak_gib: float, *, exit_code: int = 0) -> dict:
    peak_mib = peak_gib * 1024.0
    return {
        "protocol": "RIST-E1-GPU-MONITOR-v1",
        "gpu_uuid": "GPU-bound",
        "gpu_name": "Bound GPU",
        "total_memory_mib": 81920.0,
        "used_memory_mib": 0.0,
        "driver_version": "1.0",
        "sample_count": 2,
        "peak_memory_mib": peak_mib,
        "peak_memory_gib": peak_gib,
        "command_exit_code": exit_code,
        "samples": [
            {"elapsed_seconds": 0.0, "used_memory_mib": 0.0},
            {"elapsed_seconds": 1.0, "used_memory_mib": peak_mib},
        ],
    }


def _canary_rows(task_count: int) -> list[dict]:
    return [
        {
            "task_id": f"task-{task}",
            "task_signature": f"signature-{task}",
            "rollout_seed": 12101,
            "turn_index": turn,
            "raw_response": {"task": task, "turn": turn},
        }
        for task in range(task_count)
        for turn in range(4)
    ]


def _canary_result(
    protocol: str,
    identity: dict,
    task_count: int,
    journal_sha256: str,
) -> dict:
    return {
        "protocol": protocol,
        "family": identity["family"],
        "runtime_identity": identity,
        "complete": True,
        "infrastructure_error": None,
        "task_count": task_count,
        "expected_task_count": task_count,
        "complete_four_turn_count": task_count,
        "raw_response_count": task_count * 4,
        "retry_count": 0,
        "raw_journal_sha256": journal_sha256,
    }


def _fixture(root: Path):
    validator = _load("e1_validator_fixture", E1 / "validate_capacity_evidence.py")
    builder = _load("e1_manifest_fixture", E1 / "build_capacity_manifest.py")
    metrics = _load("e1_metrics_fixture", E1 / "extract_one_step_metrics.py")
    manifest = builder.build_manifest(root / "runs")
    model = manifest["models"]["qwen3"]
    model["gpu_pairing"] = "GPU-bound"
    for job in manifest["jobs"]:
        if job["family"] == "qwen3":
            job["gpu_pairing"] = "GPU-bound"
    identity = {
        "family": "qwen3",
        "checkpoint": model["checkpoint"],
        "model_revision": model["revision"],
        "tokenizer_snapshot_sha256": model["tokenizer_snapshot_sha256"],
        "model_weights_sha256": "a" * 64,
        "gpu_name": "Bound GPU",
        "gpu_uuid": "GPU-bound",
        "gpu_total_memory_gib": 80.0,
        "driver_version": "1.0",
        "cuda_version": "1.0",
        "torch_version": "1.0",
        "source_commit": manifest["source_commit"],
    }

    resolution = {
        "passed": True,
        "common_resolution_map": {"c06": "high", "c07": "high"},
    }
    resolution_text = _json(resolution)
    filtered = _jsonl(
        {"id": f"f-{index}", "structural_cell": "c06", "resolution_band": "high"}
        for index in range(8)
    )
    capacity = _jsonl(
        {"id": f"c-{index}", "structural_cell": "c06", "resolution_band": "high"}
        for index in range(4)
    )
    capacity_manifest = {
        "protocol": "RIST-E1-CAPACITY-DATA-v2.1",
        "selection_rule": "LEXICOGRAPHIC_FIRST_TRANSPORTED_HIGH_CELL",
        "selection_uses_individual_outcomes": False,
        "selected_cell": "c06",
        "task_count": 4,
        "source_train_sha256": hashlib.sha256(filtered.encode()).hexdigest(),
        "resolution_result_sha256": hashlib.sha256(resolution_text.encode()).hexdigest(),
        "capacity_train_sha256": hashlib.sha256(capacity.encode()).hexdigest(),
    }
    top_content = {
        "runtime_identity": _json(identity),
        "filtered_train": filtered,
        "capacity_train": capacity,
        "resolution_map": resolution_text,
        "capacity_data_manifest": _json(capacity_manifest),
    }
    top_artifacts, top_hashes = {}, {}
    for name, field in validator.TOP_ARTIFACTS.items():
        relative, digest = _write(root, f"top/{name}", top_content[name])
        top_artifacts[name], top_hashes[field] = relative, digest

    serving_rows = _canary_rows(32)
    serving_journal, serving_journal_sha = _write(
        root, "serving/raw.jsonl", _jsonl(serving_rows)
    )
    serving_files = {
        "result": _write(
            root,
            "serving/result.json",
            _json(
                _canary_result(
                    "RIST-E1-SERVING-CANARY-v1",
                    identity,
                    32,
                    serving_journal_sha,
                )
            ),
        )[0],
        "raw_journal": serving_journal,
        "serve_log": _write(root, "serving/serve.log", "clean\n")[0],
        "gpu_monitor": _write(root, "serving/gpu.json", _json(_monitor(10.0)))[0],
    }
    serving_artifacts, serving_hashes = {}, {}
    for name, field in validator.SERVING_ARTIFACTS.items():
        path = root / serving_files[name]
        serving_artifacts[name] = serving_files[name]
        serving_hashes[field] = hashlib.sha256(path.read_bytes()).hexdigest()

    series = {
        "loss": [{"step": 0, "value": 0.5}],
        "gradient_norm": [{"step": 0, "value": 1.0}],
        "trainable_tokens": [{"step": 0, "value": 100.0}],
    }
    training = []
    tracked = {}
    for algorithm, peak in (("gspo", 60.0), ("grpo", 64.0)):
        metrics_dir = root / algorithm / "metrics"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "events.out.tfevents.raw").write_bytes(b"raw-events")
        checkpoint_dir = root / algorithm / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "model.safetensors").write_bytes(b"saved-weights")
        rewards = [
            {
                "task_id": "c-0",
                "training_step": 0,
                "prompt_index": 0,
                "sample_index": sample,
                "resolution_band": "high",
                "reward": float(sample % 2),
            }
            for sample in range(8)
        ]
        raw_rows = [
            {
                "task_id": "c-0",
                "training_step": 0,
                "prompt_index": 0,
                "sample_index": sample,
                "turn_index": turn,
                "raw_response": {"sample": sample, "turn": turn},
            }
            for sample in range(8)
            for turn in range(4)
        ]
        metrics_manifest = validator._build_directory_manifest(
            metrics_dir, "RIST-DIRECTORY-MANIFEST-v1"
        )
        checkpoint_manifest = validator._build_directory_manifest(
            checkpoint_dir, "RIST-E1-CHECKPOINT-MANIFEST-v1"
        )
        checkpoint_manifest_path, checkpoint_manifest_sha = _write(
            root, f"{algorithm}/checkpoint_manifest.json", _json(checkpoint_manifest)
        )
        reload_identity = {
            **identity,
            "parent_run_id": f"qwen3-{algorithm}-AF-8101",
            "checkpoint_manifest_sha256": checkpoint_manifest_sha,
            "loaded_checkpoint_path": next(
                job["artifact_binding_contract"]["checkpoint_dir"]
                for job in manifest["jobs"]
                if job["run_id"] == f"qwen3-{algorithm}-AF-8101"
            ),
        }
        reload_rows = _canary_rows(1)
        reload_journal, reload_journal_sha = _write(
            root, f"{algorithm}/reload.jsonl", _jsonl(reload_rows)
        )
        files = {
            "raw_journal": _write(root, f"{algorithm}/raw.jsonl", _jsonl(raw_rows))[0],
            "reward_journal": _write(root, f"{algorithm}/reward.jsonl", _jsonl(rewards))[0],
            "metrics_manifest": _write(
                root, f"{algorithm}/metrics_manifest.json", _json(metrics_manifest)
            )[0],
            "metrics_summary": _write(
                root,
                f"{algorithm}/metrics_summary.json",
                _json(metrics.summarize(series, rewards)),
            )[0],
            "checkpoint_manifest": checkpoint_manifest_path,
            "reload_journal": reload_journal,
            "reload_result": _write(
                root,
                f"{algorithm}/reload_result.json",
                _json(
                    _canary_result(
                        "RIST-E1-RELOAD-CANARY-v1",
                        reload_identity,
                        1,
                        reload_journal_sha,
                    )
                ),
            )[0],
            "gpu_monitor": _write(
                root, f"{algorithm}/gpu.json", _json(_monitor(peak))
            )[0],
            "train_log": _write(root, f"{algorithm}/train.log", "complete\n")[0],
            "reload_gpu_monitor": _write(
                root, f"{algorithm}/reload_gpu.json", _json(_monitor(12.0))
            )[0],
            "reload_serve_log": _write(
                root, f"{algorithm}/reload_serve.log", "complete\n"
            )[0],
        }
        artifacts, hashes = {}, {}
        for name, field in validator.TRAINING_FILE_ARTIFACTS.items():
            path = root / files[name]
            artifacts[name] = files[name]
            hashes[field] = hashlib.sha256(path.read_bytes()).hexdigest()
        row = {
            "run_id": f"qwen3-{algorithm}-AF-8101",
            "algorithm": algorithm,
            "filtered_train_sha256": top_hashes["filtered_train_sha256"],
            "capacity_train_sha256": top_hashes["capacity_train_sha256"],
            "metrics_dir": metrics_dir.relative_to(root).as_posix(),
            "checkpoint_dir": checkpoint_dir.relative_to(root).as_posix(),
            "optimizer_step_completed": False,
            "gradient_norm": 0.0,
            "checkpoint_roundtrip": False,
            "peak_memory_gib": 0.0,
            "artifacts": artifacts,
            **hashes,
        }
        training.append(row)
        tracked[algorithm] = {
            "row": row,
            "rewards": rewards,
            "summary": metrics.summarize(series, rewards),
            "checkpoint": checkpoint_dir / "model.safetensors",
        }
    evidence = {
        "protocol": "RIST-E1-CAPACITY-EVIDENCE-v2.1",
        **identity,
        "training_seed": 8101,
        "filtered_train_sha256": top_hashes["filtered_train_sha256"],
        "capacity_train_sha256": top_hashes["capacity_train_sha256"],
        "resolution_map_sha256": top_hashes["resolution_map_sha256"],
        "capacity_data_manifest_sha256": top_hashes["capacity_data_manifest_sha256"],
        "runtime_identity_sha256": top_hashes["runtime_identity_sha256"],
        "artifacts": top_artifacts,
        "serving_canary": {"artifacts": serving_artifacts, **serving_hashes},
        "training_canaries": training,
    }
    return validator, manifest, evidence, series, tracked


def test_e1_recomputes_metrics_and_rejects_rehashed_fake_summary(tmp_path):
    validator, manifest, evidence, series, tracked = _fixture(tmp_path)
    loader = lambda _directory: series
    result = validator.validate_capacity(
        evidence, manifest, tmp_path, metrics_loader=loader
    )
    assert result["passed"] is True
    assert result["submitted_summary_is_authoritative"] is False
    assert result["peak_memory_gib"] == 64.0

    row = tracked["grpo"]["row"]
    summary_path = tmp_path / row["artifacts"]["metrics_summary"]
    forged = json.loads(summary_path.read_text())
    forged["gradient_norm"] = 999.0
    summary_path.write_text(_json(forged))
    row["metrics_summary_sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    rejected = validator.validate_capacity(
        evidence, manifest, tmp_path, metrics_loader=loader
    )
    assert rejected["raw_evidence_pass"]["grpo"] is False
    assert rejected["passed"] is False


def test_e1_rejects_checkpoint_substitution_and_recomputed_high_peak(tmp_path):
    validator, manifest, evidence, series, tracked = _fixture(tmp_path)
    loader = lambda _directory: series
    tracked["grpo"]["checkpoint"].write_bytes(b"substituted")
    result = validator.validate_capacity(
        evidence, manifest, tmp_path, metrics_loader=loader
    )
    assert result["raw_evidence_pass"]["grpo"] is False
    assert result["passed"] is False

    tracked["grpo"]["checkpoint"].write_bytes(b"saved-weights")
    row = tracked["grpo"]["row"]
    monitor_path = tmp_path / row["artifacts"]["gpu_monitor"]
    monitor_path.write_text(_json(_monitor(70.0)))
    row["gpu_monitor_sha256"] = hashlib.sha256(monitor_path.read_bytes()).hexdigest()
    result = validator.validate_capacity(
        evidence, manifest, tmp_path, metrics_loader=loader
    )
    assert result["memory_headroom_pass"] is False
    assert result["passed"] is False


def test_e1_binds_gpu_uuid_and_hard_family_memory_floors(tmp_path):
    validator, manifest, evidence, series, _tracked = _fixture(tmp_path)
    loader = lambda _directory: series
    manifest["models"]["qwen3"]["gpu_pairing"] = "GPU-other"
    result = validator.validate_capacity(
        evidence, manifest, tmp_path, metrics_loader=loader
    )
    assert result["source_identity_pass"] is False
    assert result["passed"] is False

    manifest["models"]["qwen3"]["minimum_gpu_memory_gib"] = 1
    with pytest.raises(ValueError, match="weakens"):
        validator.validate_capacity(
            evidence, manifest, tmp_path, metrics_loader=loader
        )
    assert validator.MINIMUM_GPU_MEMORY_GIB == {"qwen3": 24.0, "gemma4": 48.0}
    qwen_job = next(job for job in manifest["jobs"] if job["family"] == "qwen3")
    assert "--log" in qwen_job["monitored_command_template"]
    assert "--log" in qwen_job["monitored_reload_serve_command_template"]
    assert qwen_job["artifact_binding_contract"]["checkpoint_dir"].endswith(
        "step_000001"
    )

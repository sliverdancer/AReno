"""Recompute E1 capacity qualification from frozen, original artifacts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

REQUIRED_ALGORITHMS = {"gspo", "grpo"}
MINIMUM_GPU_MEMORY_GIB = {"qwen3": 24.0, "gemma4": 48.0}
MAXIMUM_PEAK_MEMORY_FRACTION = 0.85
TRAINING_FILE_ARTIFACTS = {
    "raw_journal": "raw_journal_sha256",
    "reward_journal": "reward_journal_sha256",
    "metrics_manifest": "metrics_manifest_sha256",
    "metrics_summary": "metrics_summary_sha256",
    "checkpoint_manifest": "checkpoint_manifest_sha256",
    "reload_journal": "reload_journal_sha256",
    "reload_result": "reload_result_sha256",
    "gpu_monitor": "gpu_monitor_sha256",
    "train_log": "train_log_sha256",
    "reload_gpu_monitor": "reload_gpu_monitor_sha256",
    "reload_serve_log": "reload_serve_log_sha256",
}
SERVING_ARTIFACTS = {
    "result": "result_sha256",
    "raw_journal": "raw_journal_sha256",
    "serve_log": "serve_log_sha256",
    "gpu_monitor": "gpu_monitor_sha256",
}
TOP_ARTIFACTS = {
    "runtime_identity": "runtime_identity_sha256",
    "filtered_train": "filtered_train_sha256",
    "capacity_train": "capacity_train_sha256",
    "resolution_map": "resolution_map_sha256",
    "capacity_data_manifest": "capacity_data_manifest_sha256",
}
IDENTITY_FIELDS = (
    "family",
    "checkpoint",
    "model_revision",
    "tokenizer_snapshot_sha256",
    "model_weights_sha256",
    "gpu_name",
    "gpu_uuid",
    "gpu_total_memory_gib",
    "driver_version",
    "cuda_version",
    "torch_version",
    "source_commit",
)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _is_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(
        character in "0123456789abcdef" for character in value
    )


def _file_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("E1 artifact paths must be non-empty and relative")
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"E1 artifact is missing or outside root: {relative}")
    return path


def _directory_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("E1 artifact directory paths must be non-empty and relative")
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_dir():
        raise ValueError(f"E1 artifact directory is missing or outside root: {relative}")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"E1 JSON artifact must be an object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"E1 JSONL row {line_number} must be an object")
        rows.append(value)
    return rows


def _verify_file_artifacts(
    row: dict[str, Any], root: Path, contract: dict[str, str]
) -> tuple[bool, dict[str, Path]]:
    artifacts = row.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(contract):
        return False, {}
    resolved = {}
    try:
        for name, hash_field in contract.items():
            path = _file_path(root, artifacts[name])
            if row.get(hash_field) != _sha256(path):
                return False, {}
            resolved[name] = path
    except (OSError, ValueError):
        return False, {}
    return True, resolved


def _build_directory_manifest(directory: Path, protocol: str) -> dict[str, Any]:
    files = [
        {
            "path": path.relative_to(directory).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]
    return {"protocol": protocol, "file_count": len(files), "files": files}


def _directory_manifest_matches(
    directory: Path, manifest_path: Path, protocol: str, *, require_weights: bool = False
) -> bool:
    try:
        expected = _build_directory_manifest(directory, protocol)
        observed = _read_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if expected["file_count"] == 0 or observed != expected:
        return False
    return not require_weights or any(
        row["path"].endswith(".safetensors") for row in expected["files"]
    )


def _load_metrics_module():
    path = Path(__file__).with_name("extract_one_step_metrics.py")
    spec = importlib.util.spec_from_file_location("rist_e1_metrics_recompute", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("E1 metrics recomputation module is unavailable")
    spec.loader.exec_module(module)
    return module


def _recompute_metrics(
    metrics_dir: Path,
    reward_journal: Path,
    metrics_loader: Callable[[Path], dict[str, list[dict[str, float]]]] | None,
) -> dict[str, Any]:
    module = _load_metrics_module()
    loader = metrics_loader or module.load_tensorboard
    return module.summarize(loader(metrics_dir), _read_jsonl(reward_journal))


def _canonical_resolution_sha256(value: dict[str, Any]) -> str:
    encoded = (json.dumps(value, sort_keys=True) + "\n").encode()
    return hashlib.sha256(encoded).hexdigest()


def _capacity_data_matches(
    filtered_path: Path,
    capacity_path: Path,
    resolution_path: Path,
    manifest_path: Path,
) -> bool:
    try:
        manifest = _read_json(manifest_path)
        resolution = _read_json(resolution_path)
        rows = _read_jsonl(capacity_path)
        selected_cell = str(manifest["selected_cell"])
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False
    return all(
        (
            manifest.get("protocol") == "RIST-E1-CAPACITY-DATA-v2.1",
            manifest.get("selection_rule")
            == "LEXICOGRAPHIC_FIRST_TRANSPORTED_HIGH_CELL",
            manifest.get("selection_uses_individual_outcomes") is False,
            int(manifest.get("task_count", 0)) == len(rows) == 4,
            manifest.get("source_train_sha256") == _sha256(filtered_path),
            manifest.get("capacity_train_sha256") == _sha256(capacity_path),
            manifest.get("resolution_result_sha256")
            == _canonical_resolution_sha256(resolution),
            resolution.get("passed") is True,
            resolution.get("common_resolution_map", {}).get(selected_cell) == "high",
            all(
                str(row.get("structural_cell")) == selected_cell
                and row.get("resolution_band") == "high"
                for row in rows
            ),
        )
    )


def _identity_matches(identity: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(identity.get(field) == expected.get(field) for field in IDENTITY_FIELDS)


def _monitor_result(
    path: Path,
    expected_identity: dict[str, Any],
    minimum_memory_gib: float,
    *,
    require_zero_exit: bool,
) -> dict[str, Any] | None:
    try:
        monitor = _read_json(path)
        samples = monitor["samples"]
        total_mib = float(monitor["total_memory_mib"])
        claimed_peak_mib = float(monitor["peak_memory_mib"])
        claimed_peak_gib = float(monitor["peak_memory_gib"])
        used = [float(row["used_memory_mib"]) for row in samples]
        elapsed = [float(row["elapsed_seconds"]) for row in samples]
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None
    if (
        monitor.get("protocol") != "RIST-E1-GPU-MONITOR-v1"
        or not samples
        or int(monitor.get("sample_count", -1)) != len(samples)
        or any(not math.isfinite(value) or value < 0.0 for value in used + elapsed)
        or elapsed != sorted(elapsed)
        or not math.isfinite(total_mib)
        or total_mib < minimum_memory_gib * 1000.0
        or any(value > total_mib for value in used)
        or claimed_peak_mib != max(used)
        or not math.isclose(claimed_peak_gib, claimed_peak_mib / 1024.0, abs_tol=1e-12)
        or monitor.get("gpu_uuid") != expected_identity.get("gpu_uuid")
        or monitor.get("gpu_name") != expected_identity.get("gpu_name")
        or monitor.get("driver_version") != expected_identity.get("driver_version")
        or not math.isclose(
            float(expected_identity.get("gpu_total_memory_gib", float("nan"))),
            total_mib / 1024.0,
            abs_tol=1e-9,
        )
        or (require_zero_exit and int(monitor.get("command_exit_code", -1)) != 0)
    ):
        return None
    return {
        "peak_memory_gib": claimed_peak_gib,
        "total_memory_gib": total_mib / 1024.0,
        "command_exit_code": int(monitor.get("command_exit_code", -1)),
    }


def _canary_matches(
    result_path: Path,
    journal_path: Path,
    expected_identity: dict[str, Any],
    *,
    expected_protocol: str,
    expected_task_count: int,
    checkpoint_manifest_sha256: str | None = None,
    run_id: str | None = None,
    checkpoint_path: str | None = None,
) -> bool:
    try:
        result = _read_json(result_path)
        rows = _read_jsonl(journal_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for row in rows:
        try:
            key = (str(row["task_id"]), int(row["rollout_seed"]))
            turn = int(row["turn_index"])
        except (KeyError, TypeError, ValueError):
            return False
        if not isinstance(row.get("task_signature"), str) or not isinstance(
            row.get("raw_response"), dict
        ):
            return False
        grouped[key].append(turn)
    runtime_identity = result.get("runtime_identity")
    if not isinstance(runtime_identity, dict) or not _identity_matches(
        runtime_identity, expected_identity
    ):
        return False
    if checkpoint_manifest_sha256 is not None and (
        runtime_identity.get("checkpoint_manifest_sha256")
        != checkpoint_manifest_sha256
        or runtime_identity.get("parent_run_id") != run_id
        or runtime_identity.get("loaded_checkpoint_path") != checkpoint_path
    ):
        return False
    return all(
        (
            result.get("protocol") == expected_protocol,
            result.get("family") == expected_identity.get("family"),
            result.get("complete") is True,
            result.get("infrastructure_error") is None,
            int(result.get("task_count", 0)) == expected_task_count,
            int(result.get("expected_task_count", 0)) == expected_task_count,
            int(result.get("complete_four_turn_count", 0)) == expected_task_count,
            int(result.get("raw_response_count", 0)) == expected_task_count * 4,
            result.get("retry_count") == 0,
            result.get("raw_journal_sha256") == _sha256(journal_path),
            len(rows) == expected_task_count * 4,
            len(grouped) == expected_task_count,
            all(sorted(turns) == [0, 1, 2, 3] for turns in grouped.values()),
        )
    )


def _training_journals_match(raw_path: Path, reward_path: Path) -> bool:
    try:
        raw_rows = _read_jsonl(raw_path)
        reward_rows = _read_jsonl(reward_path)
        raw_keys = [
            (
                int(row["training_step"]),
                int(row["prompt_index"]),
                int(row["sample_index"]),
                int(row["turn_index"]),
            )
            for row in raw_rows
        ]
        reward_keys = [
            (int(row["training_step"]), int(row["prompt_index"]), int(row["sample_index"]))
            for row in reward_rows
        ]
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False
    expected_reward = {(0, 0, sample) for sample in range(8)}
    turns_by_sample: dict[int, list[int]] = defaultdict(list)
    for training_step, prompt_index, sample_index, turn_index in raw_keys:
        if training_step != 0 or prompt_index != 0 or sample_index not in range(8):
            return False
        turns_by_sample[sample_index].append(turn_index)
    return all(
        (
            len(raw_keys) == len(set(raw_keys)),
            set(turns_by_sample) == set(range(8)),
            all(
                sorted(turns) == list(range(len(turns))) and 1 <= len(turns) <= 4
                for turns in turns_by_sample.values()
            ),
            len(reward_keys) == len(set(reward_keys)) == 8,
            set(reward_keys) == expected_reward,
            all(isinstance(row.get("raw_response"), dict) for row in raw_rows),
            all(
                row.get("resolution_band") == "high"
                and type(row.get("reward")) in (int, float)
                and float(row["reward"]) in {0.0, 1.0}
                for row in reward_rows
            ),
        )
    )


def validate_capacity(
    evidence: dict[str, Any],
    manifest: dict[str, Any],
    artifact_root: Path,
    *,
    metrics_loader: Callable[[Path], dict[str, list[dict[str, float]]]] | None = None,
) -> dict[str, Any]:
    """Recompute all scientific gates; submitted summary booleans are ignored."""

    if evidence.get("protocol") != "RIST-E1-CAPACITY-EVIDENCE-v2.1":
        raise ValueError("unexpected E1 evidence protocol")
    if manifest.get("protocol") != "RIST-E1-CAPACITY-v2.1":
        raise ValueError("unexpected E1 execution manifest")
    family = str(evidence.get("family"))
    if family not in MINIMUM_GPU_MEMORY_GIB or family not in manifest.get("models", {}):
        raise ValueError("E1 evidence family is not in the frozen contract")
    model = manifest["models"][family]
    minimum_memory = MINIMUM_GPU_MEMORY_GIB[family]
    if float(model.get("minimum_gpu_memory_gib", -1)) != minimum_memory:
        raise ValueError("E1 manifest weakens the family GPU-memory floor")

    top_ok, top_paths = _verify_file_artifacts(evidence, artifact_root, TOP_ARTIFACTS)
    identity = _read_json(top_paths["runtime_identity"]) if top_ok else {}
    expected_identity = {field: evidence.get(field) for field in IDENTITY_FIELDS}
    jobs = {
        str(row["algorithm"]): row
        for row in manifest.get("jobs", [])
        if row.get("family") == family
    }
    training = evidence.get("training_canaries")
    if (
        not isinstance(training, list)
        or len(training) != 2
        or {str(row.get("algorithm")) for row in training} != REQUIRED_ALGORITHMS
        or set(jobs) != REQUIRED_ALGORITHMS
    ):
        raise ValueError("capacity evidence requires the exact GSPO and GRPO block")

    concrete_gpu_binding = (
        isinstance(model.get("gpu_pairing"), str)
        and model["gpu_pairing"] == evidence.get("gpu_uuid")
        and "{" not in model["gpu_pairing"]
        and all(job.get("gpu_pairing") == model["gpu_pairing"] for job in jobs.values())
    )
    source_identity_pass = all(
        (
            top_ok,
            _is_commit(manifest.get("source_commit")),
            evidence.get("source_commit") == manifest.get("source_commit"),
            evidence.get("checkpoint") == model.get("checkpoint"),
            evidence.get("model_revision") == model.get("revision"),
            evidence.get("tokenizer_snapshot_sha256")
            == model.get("tokenizer_snapshot_sha256"),
            int(evidence.get("training_seed", -1)) == int(manifest.get("training_seed", -2)),
            _identity_matches(identity, expected_identity),
            concrete_gpu_binding,
            _is_sha256(evidence.get("model_weights_sha256")),
        )
    )
    dataset_pass = top_ok and _capacity_data_matches(
        top_paths["filtered_train"],
        top_paths["capacity_train"],
        top_paths["resolution_map"],
        top_paths["capacity_data_manifest"],
    )

    serving = evidence.get("serving_canary")
    serving_files_ok, serving_paths = (
        _verify_file_artifacts(serving, artifact_root, SERVING_ARTIFACTS)
        if isinstance(serving, dict)
        else (False, {})
    )
    serving_monitor = (
        _monitor_result(
            serving_paths["gpu_monitor"], identity, minimum_memory, require_zero_exit=False
        )
        if serving_files_ok and source_identity_pass
        else None
    )
    serving_log_clean = serving_files_ok and not any(
        marker in serving_paths["serve_log"].read_text(errors="replace").lower()
        for marker in ("out of memory", "cuda oom", "traceback")
    )
    serving_pass = all(
        (
            serving_files_ok,
            serving_monitor is not None,
            serving_log_clean,
            _canary_matches(
                serving_paths["result"],
                serving_paths["raw_journal"],
                identity,
                expected_protocol="RIST-E1-SERVING-CANARY-v1",
                expected_task_count=32,
            )
            if serving_files_ok and source_identity_pass
            else False,
        )
    )

    algorithm_pass: dict[str, bool] = {}
    raw_evidence_pass: dict[str, bool] = {}
    peaks = [serving_monitor["peak_memory_gib"]] if serving_monitor else []
    recomputed_metrics: dict[str, dict[str, Any] | None] = {}
    for row in training:
        algorithm = str(row["algorithm"])
        job = jobs[algorithm]
        files_ok, paths = _verify_file_artifacts(
            row, artifact_root, TRAINING_FILE_ARTIFACTS
        )
        try:
            metrics_dir = _directory_path(artifact_root, row.get("metrics_dir"))
            checkpoint_dir = _directory_path(artifact_root, row.get("checkpoint_dir"))
        except ValueError:
            metrics_dir = checkpoint_dir = artifact_root / "__missing__"
            files_ok = False
        metrics_manifest_ok = files_ok and _directory_manifest_matches(
            metrics_dir, paths["metrics_manifest"], "RIST-DIRECTORY-MANIFEST-v1"
        )
        checkpoint_manifest_ok = files_ok and _directory_manifest_matches(
            checkpoint_dir,
            paths["checkpoint_manifest"],
            "RIST-E1-CHECKPOINT-MANIFEST-v1",
            require_weights=True,
        )
        try:
            recomputed = (
                _recompute_metrics(metrics_dir, paths["reward_journal"], metrics_loader)
                if files_ok and metrics_manifest_ok
                else None
            )
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError):
            recomputed = None
        recomputed_metrics[algorithm] = recomputed
        submitted_summary = _read_json(paths["metrics_summary"]) if files_ok else {}
        summary_exact = recomputed is not None and submitted_summary == recomputed
        monitor = (
            _monitor_result(paths["gpu_monitor"], identity, minimum_memory, require_zero_exit=True)
            if files_ok and source_identity_pass
            else None
        )
        reload_monitor = (
            _monitor_result(
                paths["reload_gpu_monitor"], identity, minimum_memory, require_zero_exit=False
            )
            if files_ok and source_identity_pass
            else None
        )
        if monitor:
            peaks.append(monitor["peak_memory_gib"])
        if reload_monitor:
            peaks.append(reload_monitor["peak_memory_gib"])
        log_clean = files_ok and not any(
            marker in (
                paths["train_log"].read_text(errors="replace")
                + paths["reload_serve_log"].read_text(errors="replace")
            ).lower()
            for marker in ("out of memory", "cuda oom", "traceback")
        )
        checkpoint_manifest_sha = _sha256(paths["checkpoint_manifest"]) if files_ok else ""
        reload_ok = (
            _canary_matches(
                paths["reload_result"],
                paths["reload_journal"],
                identity,
                expected_protocol="RIST-E1-RELOAD-CANARY-v1",
                expected_task_count=1,
                checkpoint_manifest_sha256=checkpoint_manifest_sha,
                run_id=job["run_id"],
                checkpoint_path=job["artifact_binding_contract"]["checkpoint_dir"],
            )
            if files_ok and source_identity_pass
            else False
        )
        journals_ok = files_ok and _training_journals_match(
            paths["raw_journal"], paths["reward_journal"]
        )
        raw_evidence_pass[algorithm] = all(
            (
                files_ok,
                metrics_manifest_ok,
                checkpoint_manifest_ok,
                summary_exact,
                monitor is not None,
                reload_monitor is not None,
                log_clean,
                reload_ok,
                journals_ok,
            )
        )
        algorithm_pass[algorithm] = all(
            (
                raw_evidence_pass[algorithm],
                row.get("run_id") == job.get("run_id"),
                int(job.get("max_steps", 0)) == 1,
                int(job.get("seed", -1)) == int(manifest["training_seed"]),
                job.get("arm") == "AF",
                job.get("trainable_turns") == "all_assistant",
                job.get("tool_call_supervision") == "full",
                row.get("filtered_train_sha256") == evidence.get("filtered_train_sha256"),
                row.get("capacity_train_sha256") == evidence.get("capacity_train_sha256"),
                recomputed is not None
                and int(recomputed["optimizer_step_count"]) == 1
                and int(recomputed["trainable_tokens"]) > 0
                and math.isfinite(float(recomputed["loss"]))
                and math.isfinite(float(recomputed["gradient_norm"]))
                and float(recomputed["gradient_norm"]) > 0.0,
            )
        )

    total_memory = float(identity.get("gpu_total_memory_gib", float("nan")))
    peak_memory = max(peaks) if peaks else float("nan")
    memory_floor_pass = math.isfinite(total_memory) and total_memory * 1024.0 >= minimum_memory * 1000.0
    memory_headroom_pass = (
        math.isfinite(peak_memory)
        and memory_floor_pass
        and peak_memory / total_memory <= MAXIMUM_PEAK_MEMORY_FRACTION
    )
    identity_pass = source_identity_pass and dataset_pass and memory_floor_pass
    passed = all(
        (
            identity_pass,
            serving_pass,
            all(algorithm_pass.values()),
            memory_headroom_pass,
        )
    )
    return {
        "checkpoint": evidence.get("checkpoint"),
        "gpu_name": evidence.get("gpu_name"),
        "gpu_uuid": evidence.get("gpu_uuid"),
        "identity_pass": identity_pass,
        "source_identity_pass": source_identity_pass,
        "dataset_pass": dataset_pass,
        "serving_pass": serving_pass,
        "raw_evidence_pass": raw_evidence_pass,
        "algorithm_pass": algorithm_pass,
        "recomputed_metrics": recomputed_metrics,
        "minimum_gpu_memory_gib": minimum_memory,
        "gpu_total_memory_gib": total_memory,
        "memory_floor_pass": memory_floor_pass,
        "peak_memory_gib": peak_memory,
        "memory_headroom_fraction": (
            1.0 - peak_memory / total_memory
            if math.isfinite(peak_memory) and math.isfinite(total_memory) and total_memory > 0
            else None
        ),
        "memory_headroom_pass": memory_headroom_pass,
        "submitted_summary_is_authoritative": False,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate_capacity(
        json.loads(args.evidence.read_text()),
        json.loads(args.manifest.read_text()),
        args.artifact_root,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

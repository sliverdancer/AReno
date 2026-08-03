"""Assemble E1 evidence only from bound artifact paths and recomputed hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _path(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("E1 binding paths must be relative")
    root = root.resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"E1 bound artifact is missing: {value}")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_row(root: Path, paths: dict[str, Any]) -> tuple[dict, dict]:
    artifacts = {}
    hashes = {}
    field_names = {
        "raw_journal": "raw_journal_sha256",
        "reward_journal": "reward_journal_sha256",
        "metrics_manifest": "metrics_manifest_sha256",
        "metrics_summary": "metrics_summary_sha256",
        "checkpoint_manifest": "checkpoint_manifest_sha256",
        "reload_journal": "reload_journal_sha256",
        "reload_result": "reload_result_sha256",
        "gpu_monitor": "gpu_monitor_sha256",
        "serve_log": "serve_log_sha256",
        "result": "result_sha256",
        "runtime_identity": "runtime_identity_sha256",
        "filtered_train": "filtered_train_sha256",
        "capacity_train": "capacity_train_sha256",
        "resolution_map": "resolution_map_sha256",
        "capacity_data_manifest": "capacity_data_manifest_sha256",
    }
    for name, value in paths.items():
        path = _path(root, value)
        artifacts[name] = _relative(root, path)
        hashes[field_names[name]] = _sha(path)
    return artifacts, hashes


def assemble(
    manifest: dict[str, Any], binding: dict[str, Any], artifact_root: Path
) -> dict[str, Any]:
    if binding.get("protocol") != "RIST-E1-ARTIFACT-BINDING-v1":
        raise ValueError("unexpected E1 artifact binding")
    family = str(binding["family"])
    model = manifest["models"][family]
    identity_path = _path(artifact_root, binding["runtime_identity"])
    identity = json.loads(identity_path.read_text())
    for key, expected in (
        ("family", family),
        ("checkpoint", model["checkpoint"]),
        ("model_revision", model["revision"]),
        ("tokenizer_snapshot_sha256", model["tokenizer_snapshot_sha256"]),
        ("source_commit", manifest["source_commit"]),
    ):
        if identity.get(key) != expected:
            raise ValueError(f"E1 runtime identity mismatch: {key}")
    filtered = _path(artifact_root, binding["filtered_train"])
    capacity = _path(artifact_root, binding["capacity_train"])
    resolution = _path(artifact_root, binding["resolution_map"])
    capacity_data_manifest = json.loads(
        _path(artifact_root, binding["capacity_data_manifest"]).read_text()
    )
    if capacity_data_manifest.get("capacity_train_sha256") != _sha(capacity):
        raise ValueError("E1 capacity dataset manifest hash mismatch")

    serving_result = json.loads(
        _path(artifact_root, binding["serving"]["result"]).read_text()
    )
    serving_monitor = json.loads(
        _path(artifact_root, binding["serving"]["gpu_monitor"]).read_text()
    )
    serving_paths = {
        key: binding["serving"][key]
        for key in ("result", "raw_journal", "serve_log", "gpu_monitor")
    }
    serving_artifacts, serving_hashes = _artifact_row(artifact_root, serving_paths)
    serve_log = _path(artifact_root, binding["serving"]["serve_log"]).read_text(
        errors="replace"
    ).lower()
    serving = {
        "health_pass": serving_result.get("complete") is True,
        "task_count": serving_result.get("task_count"),
        "complete_four_turn_count": serving_result.get("complete_four_turn_count"),
        "raw_response_count": serving_result.get("raw_response_count"),
        "peak_memory_gib": serving_monitor.get("peak_memory_gib"),
        "oom": "out of memory" in serve_log or "cuda oom" in serve_log,
        "retry_count": serving_result.get("retry_count"),
        "artifacts": serving_artifacts,
        **serving_hashes,
    }

    jobs = {
        row["algorithm"]: row
        for row in manifest["jobs"]
        if row["family"] == family
    }
    training_rows = []
    for algorithm in ("gspo", "grpo"):
        spec = binding["training"][algorithm]
        summary = json.loads(_path(artifact_root, spec["metrics_summary"]).read_text())
        monitor = json.loads(_path(artifact_root, spec["gpu_monitor"]).read_text())
        reload_result = json.loads(_path(artifact_root, spec["reload_result"]).read_text())
        paths = {
            key: spec[key]
            for key in (
                "raw_journal",
                "reward_journal",
                "metrics_manifest",
                "metrics_summary",
                "checkpoint_manifest",
                "reload_journal",
                "reload_result",
                "gpu_monitor",
            )
        }
        artifacts, hashes = _artifact_row(artifact_root, paths)
        job = jobs[algorithm]
        training_rows.append(
            {
                "run_id": job["run_id"],
                "algorithm": algorithm,
                "optimizer_step_completed": summary.get("optimizer_step_completed"),
                "optimizer_step_count": summary.get("optimizer_step_count"),
                "max_steps": job["max_steps"],
                "seed": job["seed"],
                "arm": job["arm"],
                "trainable_turns": job["trainable_turns"],
                "tool_call_supervision": job["tool_call_supervision"],
                "filtered_train_sha256": _sha(filtered),
                "capacity_train_sha256": _sha(capacity),
                "resolution_band": "high",
                "selected_structural_cell": capacity_data_manifest.get("selected_cell"),
                "trainable_tokens": summary.get("trainable_tokens"),
                "loss": summary.get("loss"),
                "gradient_norm": summary.get("gradient_norm"),
                "peak_memory_gib": monitor.get("peak_memory_gib"),
                "oom": monitor.get("command_exit_code") != 0,
                "checkpoint_roundtrip": reload_result.get("complete") is True,
                "retry_count": reload_result.get("retry_count"),
                "artifacts": artifacts,
                **hashes,
            }
        )
    capacity_data_manifest_path = _path(
        artifact_root, binding["capacity_data_manifest"]
    )
    top_artifacts, top_hashes = _artifact_row(
        artifact_root,
        {
            "runtime_identity": binding["runtime_identity"],
            "filtered_train": binding["filtered_train"],
            "capacity_train": binding["capacity_train"],
            "resolution_map": binding["resolution_map"],
            "capacity_data_manifest": binding["capacity_data_manifest"],
        },
    )
    return {
        "protocol": "RIST-E1-CAPACITY-EVIDENCE-v2.1",
        "family": family,
        "checkpoint": identity["checkpoint"],
        "model_revision": identity["model_revision"],
        "tokenizer_snapshot_sha256": identity["tokenizer_snapshot_sha256"],
        "model_weights_sha256": identity["model_weights_sha256"],
        "gpu_name": identity["gpu_name"],
        "gpu_uuid": identity["gpu_uuid"],
        "gpu_total_memory_gib": identity["gpu_total_memory_gib"],
        "driver_version": identity["driver_version"],
        "cuda_version": identity["cuda_version"],
        "torch_version": identity["torch_version"],
        "source_commit": identity["source_commit"],
        "training_seed": manifest["training_seed"],
        "filtered_train_sha256": _sha(filtered),
        "capacity_train_sha256": _sha(capacity),
        "resolution_map_sha256": _sha(resolution),
        "capacity_data_manifest_sha256": _sha(capacity_data_manifest_path),
        "runtime_identity_sha256": _sha(identity_path),
        "artifacts": top_artifacts,
        **top_hashes,
        "serving_canary": serving,
        "training_canaries": training_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assemble(
        json.loads(args.manifest.read_text()),
        json.loads(args.binding.read_text()),
        args.artifact_root,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

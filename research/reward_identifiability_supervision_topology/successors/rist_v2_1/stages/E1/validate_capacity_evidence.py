"""Validate artifact-backed per-checkpoint E1 serving and training evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

REQUIRED_ALGORITHMS = {"gspo", "grpo"}
TRAINING_ARTIFACTS = {
    "raw_journal": "raw_journal_sha256",
    "reward_journal": "reward_journal_sha256",
    "metrics_manifest": "metrics_manifest_sha256",
    "metrics_summary": "metrics_summary_sha256",
    "checkpoint_manifest": "checkpoint_manifest_sha256",
    "reload_journal": "reload_journal_sha256",
    "reload_result": "reload_result_sha256",
    "gpu_monitor": "gpu_monitor_sha256",
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


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _artifact_path(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("E1 artifact paths must be non-empty and relative")
    root = root.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"E1 artifact is missing or outside root: {relative}")
    return path


def _verify_artifacts(
    row: dict[str, Any], root: Path, contract: dict[str, str]
) -> tuple[bool, dict[str, Path]]:
    artifacts = row.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(contract):
        return False, {}
    resolved = {}
    for name, hash_field in contract.items():
        path = _artifact_path(root, artifacts[name])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if row.get(hash_field) != digest:
            return False, {}
        resolved[name] = path
    return True, resolved


def _metrics_match(row: dict[str, Any], path: Path) -> bool:
    summary = json.loads(path.read_text())
    return all(
        (
            summary.get("protocol") == "RIST-E1-ONE-STEP-METRICS-v1",
            summary.get("optimizer_step_completed") is True,
            int(summary.get("optimizer_step_count", 0)) == 1,
            int(summary.get("trainable_tokens", 0)) == int(row.get("trainable_tokens", -1)),
            float(summary.get("loss", float("nan"))) == float(row.get("loss", float("nan"))),
            float(summary.get("gradient_norm", float("nan")))
            == float(row.get("gradient_norm", float("nan"))),
            summary.get("mixed_reward_group") is True,
            int(summary.get("reward_event_count", 0)) == 8,
        )
    )


def _reload_matches(row: dict[str, Any], path: Path) -> bool:
    result = json.loads(path.read_text())
    return all(
        (
            result.get("protocol") == "RIST-E1-RELOAD-CANARY-v1",
            result.get("complete") is True,
            int(result.get("task_count", 0)) == 1,
            int(result.get("complete_four_turn_count", 0)) == 1,
            int(result.get("raw_response_count", 0)) == 4,
            result.get("retry_count") == 0,
        )
    )


def validate_capacity(
    evidence: dict[str, Any],
    manifest: dict[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    """Return fail-closed gates bound to a manifest and real artifact bytes."""

    if evidence.get("protocol") != "RIST-E1-CAPACITY-EVIDENCE-v2.1":
        raise ValueError("unexpected E1 evidence protocol")
    if manifest.get("protocol") != "RIST-E1-CAPACITY-v2.1":
        raise ValueError("unexpected E1 execution manifest")
    family = str(evidence.get("family"))
    if family not in manifest.get("models", {}):
        raise ValueError("E1 evidence family is not in the manifest")
    model = manifest["models"][family]
    top_artifacts_pass, top_paths = _verify_artifacts(
        evidence, artifact_root, TOP_ARTIFACTS
    )
    runtime_identity = (
        json.loads(top_paths["runtime_identity"].read_text())
        if top_artifacts_pass
        else {}
    )
    serving = evidence["serving_canary"]
    training = evidence["training_canaries"]
    if (
        len(training) != len(REQUIRED_ALGORITHMS)
        or {str(row["algorithm"]) for row in training} != REQUIRED_ALGORITHMS
    ):
        raise ValueError("capacity evidence requires exactly GSPO and GRPO canaries")
    jobs = {
        str(row["algorithm"]): row
        for row in manifest["jobs"]
        if row["family"] == family
    }
    if set(jobs) != REQUIRED_ALGORITHMS:
        raise ValueError("execution manifest lacks the complete family algorithm block")

    total_memory = float(evidence["gpu_total_memory_gib"])
    peak_memory = max(
        [float(serving["peak_memory_gib"])]
        + [float(row["peak_memory_gib"]) for row in training]
    )
    if not math.isfinite(total_memory) or total_memory <= 0.0:
        raise ValueError("gpu_total_memory_gib must be positive and finite")
    if not math.isfinite(peak_memory) or peak_memory < 0.0:
        raise ValueError("peak memory must be non-negative and finite")

    serving_artifacts_pass, _ = _verify_artifacts(
        serving, artifact_root, SERVING_ARTIFACTS
    )
    serving_pass = all(
        (
            serving.get("health_pass") is True,
            int(serving.get("task_count", 0)) == 32,
            int(serving.get("complete_four_turn_count", 0)) == 32,
            int(serving.get("raw_response_count", 0)) == 128,
            serving.get("oom") is False,
            serving.get("retry_count") == 0,
            serving_artifacts_pass,
        )
    )
    frozen_seed = int(evidence.get("training_seed", -1))
    filtered_train_sha256 = evidence.get("filtered_train_sha256")
    capacity_train_sha256 = evidence.get("capacity_train_sha256")
    artifact_pass = {}
    algorithm_pass = {}
    for row in training:
        algorithm = str(row["algorithm"])
        job = jobs[algorithm]
        artifacts_ok, paths = _verify_artifacts(
            row, artifact_root, TRAINING_ARTIFACTS
        )
        artifact_pass[algorithm] = artifacts_ok
        metrics_ok = artifacts_ok and _metrics_match(row, paths["metrics_summary"])
        reload_ok = artifacts_ok and _reload_matches(row, paths["reload_result"])
        algorithm_pass[algorithm] = all(
            (
                row.get("run_id") == job["run_id"],
                row.get("optimizer_step_completed") is True,
                int(row.get("optimizer_step_count", 0)) == 1,
                int(row.get("max_steps", 0)) == int(job["max_steps"]) == 1,
                int(row.get("seed", -1)) == frozen_seed == int(job["seed"]),
                row.get("arm") == job["arm"] == "AF",
                row.get("trainable_turns") == job["trainable_turns"] == "all_assistant",
                row.get("tool_call_supervision")
                == job["tool_call_supervision"]
                == "full",
                row.get("filtered_train_sha256") == filtered_train_sha256,
                row.get("capacity_train_sha256") == capacity_train_sha256,
                row.get("resolution_band") == "high",
                int(row.get("trainable_tokens", 0)) > 0,
                math.isfinite(float(row.get("loss", float("nan")))),
                math.isfinite(float(row.get("gradient_norm", float("nan"))))
                and float(row.get("gradient_norm", 0.0)) > 0.0,
                row.get("oom") is False,
                row.get("checkpoint_roundtrip") is True,
                row.get("retry_count") == 0,
                artifacts_ok,
                metrics_ok,
                reload_ok,
            )
        )

    identity_pass = all(
        (
            top_artifacts_pass,
            evidence.get("source_commit") == manifest.get("source_commit"),
            evidence.get("checkpoint") == model["checkpoint"],
            evidence.get("model_revision") == model["revision"],
            evidence.get("tokenizer_snapshot_sha256")
            == model["tokenizer_snapshot_sha256"],
            frozen_seed == int(manifest["training_seed"]),
            total_memory >= float(model["minimum_gpu_memory_gib"]),
            all(
                runtime_identity.get(field) == evidence.get(field)
                for field in (
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
            ),
            all(
                isinstance(evidence.get(field), str) and bool(evidence[field])
                for field in (
                    "gpu_name",
                    "gpu_uuid",
                    "driver_version",
                    "cuda_version",
                    "torch_version",
                )
            ),
            all(
                _is_sha256(evidence.get(field))
                for field in (
                    "model_weights_sha256",
                    "filtered_train_sha256",
                    "capacity_train_sha256",
                    "resolution_map_sha256",
                )
            ),
        )
    )
    memory_headroom_pass = peak_memory / total_memory <= 0.85
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
        "identity_pass": identity_pass,
        "top_artifacts_pass": top_artifacts_pass,
        "serving_pass": serving_pass,
        "artifact_pass": artifact_pass,
        "algorithm_pass": algorithm_pass,
        "frozen_training_seed": frozen_seed,
        "peak_memory_gib": peak_memory,
        "memory_headroom_fraction": 1.0 - peak_memory / total_memory,
        "memory_headroom_pass": memory_headroom_pass,
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

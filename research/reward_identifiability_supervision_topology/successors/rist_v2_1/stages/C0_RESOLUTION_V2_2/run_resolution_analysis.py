"""Consume a passed C0 v2.2 collection and freeze transported E1 inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from replay_resolution_outcomes import replay_outcomes
from resolution_analysis import (
    calibrate_family,
    combine_families,
    filter_train_rows,
    select_e1_rows,
)


STAGE_ROOT = Path(__file__).resolve().parent
FINALIZER_PATH = STAGE_ROOT / "finalize_scientific_collection.py"
EXPECTED_JOBS = {
    "qwen3-calibration",
    "qwen3-qualification",
    "gemma4-calibration",
    "gemma4-qualification",
}
ANALYSIS_FREEZE_PATH = STAGE_ROOT / "RESOLUTION_CPU_FREEZE.json"
EXPECTED_ANALYSIS_FILES = {
    "RESOLUTION_ANALYSIS_PROTOCOL.md",
    "replay_resolution_outcomes.py",
    "resolution_analysis.py",
    "run_resolution_analysis.py",
    "validate_resolution_analysis.py",
    "verify_resolution_cpu_freeze.py",
    "../../../../../../tests/test_rist_c0_v2_2_resolution_cpu.py",
    "../E1/validate_c0_admission.py",
    "../E1/C0_GATE_PROTOCOL.md",
    "../D4_EVAL/evaluate_checkpoint.py",
    "finalize_scientific_collection.py",
    "validate_scientific_job.py",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return _parse_jsonl_bytes(path.read_bytes(), path)


def _parse_jsonl_bytes(payload: bytes, source: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in payload.decode().splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects: {source}")
    return rows


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()


def _exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _load_finalizer():
    spec = importlib.util.spec_from_file_location("rist_c0_v2_2_collection_finalizer", FINALIZER_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("C0 v2.2 collection finalizer is unavailable")
    spec.loader.exec_module(module)
    return module


def _load_freeze_verifier():
    path = STAGE_ROOT / "verify_resolution_cpu_freeze.py"
    spec = importlib.util.spec_from_file_location("rist_c0_v2_2_resolution_freeze_verifier", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("C0 v2.2 resolution freeze verifier is unavailable")
    spec.loader.exec_module(module)
    return module


def _validate_d3(
    source_path: Path,
    source_bytes: bytes,
    manifest_path: Path,
    manifest_bytes: bytes,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_bytes)
    rows = _parse_jsonl_bytes(source_bytes, source_path)
    signatures = sorted(str(row.get("task_signature")) for row in rows)
    if not (
        manifest.get("protocol") == "RIST-D3-v2.1"
        and manifest.get("split") == "train"
        and manifest.get("count") == 32
        and manifest.get("heldout_data_opened") is False
        and manifest.get("other_split_content_opened") is False
        and manifest.get("training_performed") is False
        and manifest.get("sha256") == hashlib.sha256(source_bytes).hexdigest()
        and manifest.get("task_signatures") == signatures
        and len(rows) == 32
        and len(set(signatures)) == 32
    ):
        raise ValueError("D3 source or manifest violates the frozen train-only contract")
    return rows, manifest


def _validate_authorization(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    authorization = _read_json(path)
    scope = authorization.get("authorized_scopes", {}).get("C0_RESOLUTION", {})
    if not (
        authorization.get("schema_version") == 1
        and authorization.get("authorized") is True
        and scope.get("models") == [manifest["models"]["qwen3"], manifest["models"]["gemma4"]]
        and scope.get("trajectory_count") == 4096
        and scope.get("inference") is True
        and scope.get("training") is False
        and "HELDOUT_OR_BFCL_ACCESS" in authorization.get("not_authorized", [])
    ):
        raise PermissionError("C0 outcome analysis lacks the frozen external authorization root")
    return authorization


def _validate_execution_root(path: Path, authorization_path: Path) -> dict[str, Any]:
    root = _read_json(path)
    freeze = _read_json(ANALYSIS_FREEZE_PATH)
    if not (
        root.get("protocol") == "RIST-C0-v2.2-RESOLUTION-EXECUTION-ROOT-v1"
        and root.get("source_commit") == freeze.get("source_commit")
        and root.get("resolution_freeze_sha256") == _sha256(ANALYSIS_FREEZE_PATH)
        and root.get("external_authorization_sha256") == _sha256(authorization_path)
        and root.get("trajectory_count") == 4096
        and root.get("outcome_analysis_authorized") is True
        and root.get("gpu_permitted") is False
        and root.get("training_permitted") is False
        and root.get("heldout_permitted") is False
        and root.get("bfcl_permitted") is False
        and root.get("terminal_rerun_permitted") is False
    ):
        raise PermissionError("C0 outcome analysis lacks a valid pre-access execution root")
    return root


def _factor_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_cell: dict[str, set[str]] = {}
    for row in rows:
        cell = str(row.get("structural_cell"))
        encoded = json.dumps(row.get("factors"), sort_keys=True, separators=(",", ":"))
        by_cell.setdefault(cell, set()).add(encoded)
    if set(by_cell) != {f"c{index:02d}" for index in range(8)} or any(
        len(values) != 1 for values in by_cell.values()
    ):
        raise ValueError("each structural cell must have one exact factor identity")
    return {cell: json.loads(next(iter(values))) for cell, values in sorted(by_cell.items())}


def run_analysis(
    *,
    collection_manifest_path: Path,
    collection_final_path: Path,
    d3_train_path: Path,
    d3_manifest_path: Path,
    authorization_path: Path,
    execution_root_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run once in a fresh directory; any post-access failure is terminal."""

    if output_dir.exists():
        raise FileExistsError("resolution output directory already exists; rerun is forbidden")
    freeze = _read_json(ANALYSIS_FREEZE_PATH)
    freeze_files = freeze.get("files", {})
    freeze_validation = _load_freeze_verifier().verify(ANALYSIS_FREEZE_PATH)
    if set(freeze_files) != EXPECTED_ANALYSIS_FILES or freeze_validation.get("passed") is not True:
        raise ValueError("resolution analysis source freeze does not verify")
    _validate_execution_root(execution_root_path, authorization_path)
    manifest = _read_json(collection_manifest_path)
    _validate_authorization(authorization_path, manifest)
    recorded_final = _read_json(collection_final_path)
    if {str(job.get("job_id")) for job in manifest.get("jobs", [])} != EXPECTED_JOBS:
        raise ValueError("resolution analysis requires the full frozen four-job factorial")
    recomputed_final = _load_finalizer().finalize(collection_manifest_path)
    if recomputed_final != recorded_final or not (
        recorded_final.get("passed") is True
        and recorded_final.get("decision") == "PASS_COLLECTION_TO_SEPARATE_RESOLUTION_ANALYSIS"
        and recorded_final.get("outcomes_inspected") is False
        and recorded_final.get("trajectory_count") == 4096
        and recorded_final.get("retry_count") == 0
    ):
        raise ValueError("recorded content-blind collection final is not an exact PASS")
    d3_bytes = d3_train_path.read_bytes()
    d3_manifest_bytes = d3_manifest_path.read_bytes()
    d3_rows, _ = _validate_d3(
        d3_train_path, d3_bytes, d3_manifest_path, d3_manifest_bytes
    )

    pool_path = Path(manifest["pool_manifest"])
    if _sha256(pool_path) != manifest.get("pool_manifest_sha256"):
        raise ValueError("fresh pool manifest hash mismatch")
    pool = _read_json(pool_path)
    if pool.get("protocol") != "RIST-C0-v2.2-FRESH-POOL" or pool.get("prior_outcomes_used") is not False:
        raise ValueError("unexpected or outcome-contaminated C0 v2.2 pool")

    split_sources: dict[str, tuple[Path, bytes, list[dict[str, Any]]]] = {}
    for split in ("calibration", "qualification"):
        split_spec = pool["splits"][split]
        source_path = pool_path.parent / split_spec["file"]
        source_bytes = source_path.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != split_spec["sha256"]:
            raise ValueError("C0 source split hash mismatch")
        split_sources[split] = (
            source_path,
            source_bytes,
            _parse_jsonl_bytes(source_bytes, source_path),
        )
    if not (
        _factor_map(split_sources["calibration"][2])
        == _factor_map(split_sources["qualification"][2])
        == _factor_map(d3_rows)
    ):
        raise ValueError("C0 and D3 structural-cell factors do not transport exactly")

    artifact_bytes: dict[str, bytes] = {}
    for job in manifest["jobs"]:
        for field in ("result", "trajectories", "raw_journal"):
            artifact_bytes[f"{job['job_id']}:{field}"] = Path(job[field]).read_bytes()

    output_dir.mkdir(parents=False)
    bindings = {
        "collection_manifest": {"path": str(collection_manifest_path), "sha256": _sha256(collection_manifest_path)},
        "collection_final": {"path": str(collection_final_path), "sha256": _sha256(collection_final_path)},
        "pool_manifest": {"path": str(pool_path), "sha256": _sha256(pool_path)},
        "d3_train": {"path": str(d3_train_path), "sha256": hashlib.sha256(d3_bytes).hexdigest()},
        "d3_manifest": {"path": str(d3_manifest_path), "sha256": hashlib.sha256(d3_manifest_bytes).hexdigest()},
        "evaluator": {"path": str(STAGE_ROOT.parent / "D4_EVAL/evaluate_checkpoint.py"), "sha256": _sha256(STAGE_ROOT.parent / "D4_EVAL/evaluate_checkpoint.py")},
        "analysis_freeze": {"path": str(ANALYSIS_FREEZE_PATH), "sha256": _sha256(ANALYSIS_FREEZE_PATH)},
        "external_authorization": {"path": str(authorization_path), "sha256": _sha256(authorization_path)},
        "execution_root": {"path": str(execution_root_path), "sha256": _sha256(execution_root_path)},
    }
    for split, (path, payload, _) in split_sources.items():
        bindings[f"c0_source:{split}"] = {
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    for job in manifest["jobs"]:
        for field in ("result", "trajectories", "raw_journal"):
            path = Path(job[field])
            key = f"{job['job_id']}:{field}"
            bindings[key] = {
                "path": str(path),
                "sha256": hashlib.sha256(artifact_bytes[key]).hexdigest(),
            }
    receipt = {
        "protocol": "RIST-C0-v2.2-OUTCOME-ACCESS-RECEIPT-v1",
        "collection_decision": recorded_final["decision"],
        "collection_outcomes_inspected": recorded_final["outcomes_inspected"],
        "artifact_bindings": bindings,
        "outcome_access_authorized": True,
        "rerun_permitted": False,
        "heldout_accessed": False,
        "bfcl_accessed": False,
        "training_performed": False,
        "gpu_used": False,
    }
    _exclusive_write(output_dir / "OUTCOME_ACCESS_RECEIPT.json", _json_bytes(receipt))

    try:
        replayed: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for job in manifest["jobs"]:
            family = str(job["family"])
            split = str(job["split"])
            split_spec = pool["splits"][split]
            source_path, _, source_rows = split_sources[split]
            rows = replay_outcomes(
                source_rows=source_rows,
                rollout_seeds=split_spec["rollout_seeds"],
                split=split,
                trajectories=_parse_jsonl_bytes(
                    artifact_bytes[f"{job['job_id']}:trajectories"], Path(job["trajectories"])
                ),
                journal_rows=_parse_jsonl_bytes(
                    artifact_bytes[f"{job['job_id']}:raw_journal"], Path(job["raw_journal"])
                ),
            )
            replayed.setdefault(family, {})[split] = rows

        families = []
        jobs_by_family = {str(job["family"]): job for job in manifest["jobs"] if job["split"] == "calibration"}
        for family in sorted(replayed):
            result = calibrate_family(
                replayed[family]["calibration"],
                replayed[family]["qualification"],
                family,
                str(jobs_by_family[family]["checkpoint"]),
            )
            families.append(result)
            _exclusive_write(output_dir / f"{family}_resolution.json", _json_bytes(result))
        common = combine_families(families)
        _exclusive_write(output_dir / "COMMON_RESOLUTION_MAP.json", _json_bytes(common))
        if common["passed"] is not True:
            final = {
                "protocol": "RIST-C0-v2.2-RESOLUTION-FINAL-v1",
                "passed": False,
                "decision": common["decision"],
                "collection_final_sha256": bindings["collection_final"]["sha256"],
                "outcome_access_receipt_sha256": _sha256(output_dir / "OUTCOME_ACCESS_RECEIPT.json"),
                "common_resolution_sha256": _sha256(output_dir / "COMMON_RESOLUTION_MAP.json"),
                "rerun_permitted": False,
                "heldout_accessed": False,
                "bfcl_accessed": False,
                "training_performed": False,
                "gpu_used": False,
            }
            _exclusive_write(output_dir / "FINAL_RESULT.json", _json_bytes(final))
            return final

        filtered_rows = filter_train_rows(d3_rows, common["common_resolution_map"])
        filtered_dir = output_dir / "filtered_train"
        filtered_dir.mkdir()
        filtered_bytes = _jsonl_bytes(filtered_rows)
        _exclusive_write(filtered_dir / "train.jsonl", filtered_bytes)
        filtered_manifest = {
            "protocol": "RIST-C0-v2.2-FILTERED-TRAIN-v1",
            "source_sha256": bindings["d3_train"]["sha256"],
            "common_resolution_sha256": _sha256(output_dir / "COMMON_RESOLUTION_MAP.json"),
            "train_sha256": hashlib.sha256(filtered_bytes).hexdigest(),
            "selected_cells": sorted(common["common_resolution_map"]),
            "band_cell_counts": common["band_cell_counts"],
            "task_count": len(filtered_rows),
            "individual_outcome_selection": False,
            "heldout_accessed": False,
        }
        _exclusive_write(filtered_dir / "manifest.json", _json_bytes(filtered_manifest))

        selected_cell, e1_rows = select_e1_rows(filtered_rows, common["common_resolution_map"])
        e1_dir = output_dir / "e1"
        (e1_dir / "data").mkdir(parents=True)
        e1_bytes = _jsonl_bytes(e1_rows)
        _exclusive_write(e1_dir / "data/train.jsonl", e1_bytes)
        final = {
            "protocol": "RIST-C0-v2.2-RESOLUTION-FINAL-v1",
            "passed": True,
            "decision": "PASS_C0_V2_2_TO_FILTERED_TRAIN",
            "collection_final_sha256": bindings["collection_final"]["sha256"],
            "outcome_access_receipt_sha256": _sha256(output_dir / "OUTCOME_ACCESS_RECEIPT.json"),
            "common_resolution_sha256": filtered_manifest["common_resolution_sha256"],
            "filtered_manifest_sha256": _sha256(filtered_dir / "manifest.json"),
            "e1_validation_required": True,
            "rerun_permitted": False,
            "heldout_accessed": False,
            "bfcl_accessed": False,
            "training_performed": False,
            "gpu_used": False,
        }
        _exclusive_write(output_dir / "FINAL_RESULT.json", _json_bytes(final))
        admission = {
            "protocol": "RIST-E1-ADMISSION-v2.2",
            "decision": "PASS_C0_V2_2_TO_E1_CAPACITY",
            "selection_rule": "LEXICOGRAPHIC_FIRST_COMMON_HIGH_CELL",
            "selection_uses_individual_outcomes": False,
            "selected_cell": selected_cell,
            "task_count": 4,
            "task_ids": sorted(str(row["id"]) for row in e1_rows),
            "task_signatures": sorted(str(row["task_signature"]) for row in e1_rows),
            "capacity_train_sha256": hashlib.sha256(e1_bytes).hexdigest(),
            "filtered_train_sha256": filtered_manifest["train_sha256"],
            "filtered_manifest_sha256": _sha256(filtered_dir / "manifest.json"),
            "common_resolution_sha256": filtered_manifest["common_resolution_sha256"],
            "d3_source_sha256": bindings["d3_train"]["sha256"],
            "collection_manifest_sha256": bindings["collection_manifest"]["sha256"],
            "collection_final_sha256": bindings["collection_final"]["sha256"],
            "outcome_access_receipt_sha256": _sha256(output_dir / "OUTCOME_ACCESS_RECEIPT.json"),
            "analysis_freeze_sha256": bindings["analysis_freeze"]["sha256"],
            "external_authorization_sha256": bindings["external_authorization"]["sha256"],
            "execution_root_sha256": bindings["execution_root"]["sha256"],
            "resolution_final_sha256": _sha256(output_dir / "FINAL_RESULT.json"),
            "models": manifest["models"],
            "model_revisions": manifest["model_revisions"],
            "tokenizer_snapshot_sha256": manifest["tokenizer_snapshot_sha256"],
            "gpu_uuid": recorded_final["gpu_uuid"],
            "heldout_accessed": False,
            "bfcl_accessed": False,
            "training_performed": False,
        }
        _exclusive_write(e1_dir / "E1_ADMISSION.json", _json_bytes(admission))
        return final
    except Exception as exc:
        failure_path = output_dir / "FINAL_RESULT.json"
        if not failure_path.exists():
            failure = {
                "protocol": "RIST-C0-v2.2-RESOLUTION-FINAL-v1",
                "passed": False,
                "decision": "KILL_C0_V2_2_INVALID_OUTCOME_EVIDENCE",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "outcome_access_receipt_sha256": _sha256(output_dir / "OUTCOME_ACCESS_RECEIPT.json"),
                "rerun_permitted": False,
                "training_performed": False,
                "gpu_used": False,
            }
            _exclusive_write(failure_path, _json_bytes(failure))
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-manifest", type=Path, required=True)
    parser.add_argument("--collection-final", type=Path, required=True)
    parser.add_argument("--d3-train", type=Path, required=True)
    parser.add_argument("--d3-manifest", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_analysis(
        collection_manifest_path=args.collection_manifest,
        collection_final_path=args.collection_final,
        d3_train_path=args.d3_train,
        d3_manifest_path=args.d3_manifest,
        authorization_path=args.authorization,
        execution_root_path=args.execution_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

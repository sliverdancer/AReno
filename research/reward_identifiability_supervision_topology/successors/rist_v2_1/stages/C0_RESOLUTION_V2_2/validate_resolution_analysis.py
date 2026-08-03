"""Independently replay, validate, and seal a terminal C0 v2.2 analysis."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from replay_resolution_outcomes import replay_outcomes
from resolution_analysis import calibrate_family, combine_families, filter_train_rows, select_e1_rows


STAGE_ROOT = Path(__file__).resolve().parent
FINALIZER_PATH = STAGE_ROOT / "finalize_scientific_collection.py"
FREEZE_PATH = STAGE_ROOT / "RESOLUTION_CPU_FREEZE.json"
FREEZE_VERIFIER_PATH = STAGE_ROOT / "verify_resolution_cpu_freeze.py"
EXPECTED_JOBS = {
    "qwen3-calibration",
    "qwen3-qualification",
    "gemma4-calibration",
    "gemma4-qualification",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _jsonl_bytes(payload: bytes, source: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in payload.decode().splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSONL objects: {source}")
    return rows


def _encode_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _load_finalizer():
    spec = importlib.util.spec_from_file_location("rist_c0_v2_2_validation_finalizer", FINALIZER_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("C0 v2.2 collection finalizer is unavailable")
    spec.loader.exec_module(module)
    return module


def _verify_source_freeze() -> None:
    spec = importlib.util.spec_from_file_location("rist_c0_v2_2_validation_freeze", FREEZE_VERIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("C0 v2.2 resolution freeze verifier is unavailable")
    spec.loader.exec_module(module)
    if module.verify(FREEZE_PATH).get("passed") is not True:
        raise PermissionError("resolution validation requires the commit-backed analysis freeze")


def _factor_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    by_cell: dict[str, set[str]] = {}
    for row in rows:
        by_cell.setdefault(str(row.get("structural_cell")), set()).add(
            json.dumps(row.get("factors"), sort_keys=True, separators=(",", ":"))
        )
    if set(by_cell) != {f"c{index:02d}" for index in range(8)} or any(
        len(values) != 1 for values in by_cell.values()
    ):
        raise ValueError("structural-cell factor identity is not unique")
    return {cell: next(iter(values)) for cell, values in sorted(by_cell.items())}


def _validate_d3(rows: list[dict[str, Any]], source_sha: str, manifest: dict[str, Any]) -> None:
    signatures = sorted(str(row.get("task_signature")) for row in rows)
    if not (
        manifest.get("protocol") == "RIST-D3-v2.1"
        and manifest.get("split") == "train"
        and manifest.get("count") == 32
        and manifest.get("sha256") == source_sha
        and manifest.get("task_signatures") == signatures
        and manifest.get("heldout_data_opened") is False
        and manifest.get("other_split_content_opened") is False
        and manifest.get("training_performed") is False
        and len(rows) == 32
        and len(set(signatures)) == 32
    ):
        raise ValueError("D3 train contract does not reproduce")


def validate(output_dir: Path, *, write_receipt: bool = True) -> dict[str, Any]:
    _verify_source_freeze()
    receipt_path = output_dir / "OUTCOME_ACCESS_RECEIPT.json"
    final_path = output_dir / "FINAL_RESULT.json"
    receipt = _json(receipt_path)
    final = _json(final_path)
    if not (
        receipt.get("protocol") == "RIST-C0-v2.2-OUTCOME-ACCESS-RECEIPT-v1"
        and receipt.get("outcome_access_authorized") is True
        and receipt.get("rerun_permitted") is False
        and receipt.get("heldout_accessed") is False
        and receipt.get("bfcl_accessed") is False
        and receipt.get("training_performed") is False
        and receipt.get("gpu_used") is False
        and final.get("protocol") == "RIST-C0-v2.2-RESOLUTION-FINAL-v1"
        and final.get("outcome_access_receipt_sha256") == _sha(receipt_path)
    ):
        raise ValueError("invalid resolution receipt or terminal result")

    bindings = receipt.get("artifact_bindings", {})
    required_bindings = {
        "collection_manifest",
        "collection_final",
        "pool_manifest",
        "d3_train",
        "d3_manifest",
        "evaluator",
        "analysis_freeze",
        "external_authorization",
        "execution_root",
        "c0_source:calibration",
        "c0_source:qualification",
    } | {
        f"{job}:{field}"
        for job in EXPECTED_JOBS
        for field in ("result", "trajectories", "raw_journal")
    }
    if set(bindings) != required_bindings:
        raise ValueError("resolution receipt has an incomplete artifact set")
    bound_bytes = {}
    for name, binding in bindings.items():
        path = Path(binding["path"])
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != binding["sha256"]:
            raise ValueError(f"bound artifact changed: {name}")
        bound_bytes[name] = payload

    collection_manifest_path = Path(bindings["collection_manifest"]["path"])
    recorded_collection = json.loads(bound_bytes["collection_final"])
    recomputed_collection = _load_finalizer().finalize(collection_manifest_path)
    for name, binding in bindings.items():
        if Path(binding["path"]).read_bytes() != bound_bytes[name]:
            raise ValueError(f"bound artifact changed during collection finalization: {name}")
    if recomputed_collection != recorded_collection or not (
        recorded_collection.get("passed") is True
        and recorded_collection.get("decision") == "PASS_COLLECTION_TO_SEPARATE_RESOLUTION_ANALYSIS"
        and recorded_collection.get("outcomes_inspected") is False
        and recorded_collection.get("trajectory_count") == 4096
        and recorded_collection.get("retry_count") == 0
    ):
        raise ValueError("collection final does not independently reproduce exact PASS")

    manifest = json.loads(bound_bytes["collection_manifest"])
    pool = json.loads(bound_bytes["pool_manifest"])
    if not (
        manifest.get("protocol") == "RIST-C0-v2.2-SCIENTIFIC-COLLECTION-MANIFEST-v1"
        and manifest.get("job_count") == 4
        and manifest.get("trajectory_count") == 4096
        and {str(job.get("job_id")) for job in manifest.get("jobs", [])} == EXPECTED_JOBS
        and pool.get("protocol") == "RIST-C0-v2.2-FRESH-POOL"
        and pool.get("prior_outcomes_used") is False
        and manifest.get("pool_manifest_sha256") == bindings["pool_manifest"]["sha256"]
    ):
        raise ValueError("collection or pool identity does not reproduce")
    authorization = json.loads(bound_bytes["external_authorization"])
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
        raise PermissionError("external C0 authorization does not reproduce")
    execution_root = json.loads(bound_bytes["execution_root"])
    if not (
        execution_root.get("protocol") == "RIST-C0-v2.2-RESOLUTION-EXECUTION-ROOT-v1"
        and execution_root.get("source_commit") == json.loads(bound_bytes["analysis_freeze"])["source_commit"]
        and execution_root.get("resolution_freeze_sha256") == bindings["analysis_freeze"]["sha256"]
        and execution_root.get("external_authorization_sha256") == bindings["external_authorization"]["sha256"]
        and execution_root.get("outcome_analysis_authorized") is True
        and execution_root.get("gpu_permitted") is False
        and execution_root.get("training_permitted") is False
        and execution_root.get("heldout_permitted") is False
        and execution_root.get("bfcl_permitted") is False
    ):
        raise PermissionError("pre-access execution root does not reproduce")

    d3_rows = _jsonl_bytes(bound_bytes["d3_train"], Path(bindings["d3_train"]["path"]))
    _validate_d3(d3_rows, bindings["d3_train"]["sha256"], json.loads(bound_bytes["d3_manifest"]))
    source_rows = {
        split: _jsonl_bytes(
            bound_bytes[f"c0_source:{split}"], Path(bindings[f"c0_source:{split}"]["path"])
        )
        for split in ("calibration", "qualification")
    }
    if not (_factor_map(source_rows["calibration"]) == _factor_map(source_rows["qualification"]) == _factor_map(d3_rows)):
        raise ValueError("C0-to-D3 factor transport does not reproduce")

    replayed: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for job in manifest["jobs"]:
        family = str(job["family"])
        split = str(job["split"])
        split_spec = pool["splits"][split]
        if split_spec["sha256"] != bindings[f"c0_source:{split}"]["sha256"]:
            raise ValueError("pool source hash does not match access receipt")
        replayed.setdefault(family, {})[split] = replay_outcomes(
            source_rows=source_rows[split],
            rollout_seeds=split_spec["rollout_seeds"],
            split=split,
            trajectories=_jsonl_bytes(
                bound_bytes[f"{job['job_id']}:trajectories"], Path(job["trajectories"])
            ),
            journal_rows=_jsonl_bytes(
                bound_bytes[f"{job['job_id']}:raw_journal"], Path(job["raw_journal"])
            ),
        )
    jobs = {str(job["family"]): job for job in manifest["jobs"] if job["split"] == "calibration"}
    families = []
    for family in sorted(replayed):
        expected = calibrate_family(
            replayed[family]["calibration"], replayed[family]["qualification"],
            family, str(jobs[family]["checkpoint"]),
        )
        if _json(output_dir / f"{family}_resolution.json") != expected:
            raise ValueError("family resolution result does not reproduce")
        families.append(expected)
    common = combine_families(families)
    common_path = output_dir / "COMMON_RESOLUTION_MAP.json"
    if _json(common_path) != common or final.get("common_resolution_sha256") != _sha(common_path):
        raise ValueError("cross-family resolution result does not reproduce")

    if final.get("passed") is not True:
        allowed = {"KILL_C0_FAMILY_RESOLUTION", "KILL_C0_NO_COMMON_BANDS", "KILL_C0_V2_2_INVALID_OUTCOME_EVIDENCE"}
        expected_kill = common.get("decision")
        if (
            final.get("decision") not in allowed
            or common.get("passed") is True
            or final.get("decision") != expected_kill
        ):
            raise ValueError("terminal KILL is inconsistent with reproduced resolution")
        result = {
            "protocol": "RIST-C0-v2.2-RESOLUTION-VALIDATION-v1",
            "valid": True,
            "passed": False,
            "decision": final["decision"],
            "e1_training_permitted": False,
        }
        if write_receipt:
            _exclusive_json(output_dir / "VALIDATION_RESULT.json", result)
        return result

    if not (
        final.get("decision") == "PASS_C0_V2_2_TO_FILTERED_TRAIN"
        and final.get("e1_validation_required") is True
        and final.get("rerun_permitted") is False
        and final.get("heldout_accessed") is False
        and final.get("bfcl_accessed") is False
        and final.get("training_performed") is False
        and final.get("gpu_used") is False
        and common.get("passed") is True
    ):
        raise ValueError("PASS terminal result is not exact")

    filtered = filter_train_rows(d3_rows, common["common_resolution_map"])
    filtered_path = output_dir / "filtered_train/train.jsonl"
    filtered_manifest_path = output_dir / "filtered_train/manifest.json"
    filtered_manifest = _json(filtered_manifest_path)
    if filtered_path.read_bytes() != _encode_jsonl(filtered):
        raise ValueError("filtered D3 rows do not reproduce")
    if not (
        filtered_manifest.get("train_sha256") == _sha(filtered_path)
        and final.get("filtered_manifest_sha256") == _sha(filtered_manifest_path)
        and filtered_manifest.get("common_resolution_sha256") == _sha(common_path)
        and filtered_manifest.get("individual_outcome_selection") is False
    ):
        raise ValueError("filtered D3 manifest is not bound to reproduced evidence")

    selected_cell, e1_rows = select_e1_rows(filtered, common["common_resolution_map"])
    e1_path = output_dir / "e1/data/train.jsonl"
    admission_path = output_dir / "e1/E1_ADMISSION.json"
    if e1_path.read_bytes() != _encode_jsonl(e1_rows):
        raise ValueError("E1 capacity rows do not reproduce")
    expected_admission = {
        "protocol": "RIST-E1-ADMISSION-v2.2",
        "decision": "PASS_C0_V2_2_TO_E1_CAPACITY",
        "selection_rule": "LEXICOGRAPHIC_FIRST_COMMON_HIGH_CELL",
        "selection_uses_individual_outcomes": False,
        "selected_cell": selected_cell,
        "task_count": 4,
        "task_ids": sorted(str(row["id"]) for row in e1_rows),
        "task_signatures": sorted(str(row["task_signature"]) for row in e1_rows),
        "capacity_train_sha256": _sha(e1_path),
        "filtered_train_sha256": filtered_manifest["train_sha256"],
        "filtered_manifest_sha256": _sha(filtered_manifest_path),
        "common_resolution_sha256": _sha(common_path),
        "d3_source_sha256": bindings["d3_train"]["sha256"],
        "collection_manifest_sha256": bindings["collection_manifest"]["sha256"],
        "collection_final_sha256": bindings["collection_final"]["sha256"],
        "outcome_access_receipt_sha256": _sha(receipt_path),
        "analysis_freeze_sha256": bindings["analysis_freeze"]["sha256"],
        "external_authorization_sha256": bindings["external_authorization"]["sha256"],
        "execution_root_sha256": bindings["execution_root"]["sha256"],
        "resolution_final_sha256": _sha(final_path),
        "models": manifest["models"],
        "model_revisions": manifest["model_revisions"],
        "tokenizer_snapshot_sha256": manifest["tokenizer_snapshot_sha256"],
        "gpu_uuid": recorded_collection["gpu_uuid"],
        "heldout_accessed": False,
        "bfcl_accessed": False,
        "training_performed": False,
    }
    if _json(admission_path) != expected_admission:
        raise ValueError("E1 admission does not exactly reproduce")
    result = {
        "protocol": "RIST-C0-v2.2-RESOLUTION-VALIDATION-v1",
        "valid": True,
        "passed": True,
        "decision": "PASS_C0_V2_2_TO_E1_CAPACITY",
        "resolution_final_sha256": _sha(final_path),
        "e1_admission_sha256": _sha(admission_path),
        "capacity_train_sha256": _sha(e1_path),
        "analysis_freeze_sha256": bindings["analysis_freeze"]["sha256"],
        "external_authorization_sha256": bindings["external_authorization"]["sha256"],
        "execution_root_sha256": bindings["execution_root"]["sha256"],
        "trajectory_count": 4096,
        "selected_cell": selected_cell,
        "e1_training_permitted": True,
        "heldout_accessed": False,
        "bfcl_accessed": False,
        "training_performed": False,
        "gpu_used": False,
    }
    if write_receipt:
        _exclusive_json(output_dir / "VALIDATION_RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

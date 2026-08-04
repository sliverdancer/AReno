"""One-shot C0 v2.3 qualification transport analysis after calibration GO."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

STAGE_ROOT = Path(__file__).resolve().parent


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, STAGE_ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"C0 v2.3 module is unavailable: {filename}")
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _transport_family(
    *,
    family: str,
    checkpoint: str,
    calibration: dict[str, Any],
    qualification: dict[str, Any],
    frozen_candidates: dict[str, str],
) -> dict[str, Any]:
    """Apply qualification only to calibration-selected whole cells."""
    resolution_map = {}
    transport = {}
    for cell, band in sorted(frozen_candidates.items()):
        development = calibration["cells"][cell]["classification"]
        confirmation = qualification["cells"][cell]["classification"]
        expected = "collapsed" if band == "low" else "resolved"
        passed = development == expected and confirmation == expected
        transport[cell] = {
            "calibration": development,
            "qualification": confirmation,
            "frozen_band": band,
            "passed": passed,
        }
        if passed:
            resolution_map[cell] = band
    counts = {
        band: sum(value == band for value in resolution_map.values())
        for band in ("low", "high")
    }
    return {
        "protocol": "RIST-C0-v2.3-RESOLUTION-FAMILY-v1",
        "family": family,
        "checkpoint": checkpoint,
        "calibration": calibration,
        "qualification": qualification,
        "frozen_candidate_map": frozen_candidates,
        "transport": transport,
        "resolution_map": resolution_map,
        "band_cell_counts": counts,
        "passed": counts["low"] >= 2 and counts["high"] >= 2,
    }


def analyze(manifest_path: Path, admission_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError("qualification outcome directory must be fresh")
    output_dir.mkdir(parents=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("protocol") == "RIST-C0-v2.3-SCIENTIFIC-COLLECTION-MANIFEST-v1"
        and manifest.get("split") == "qualification"
        and manifest.get("calibration_permitted") is False
        and manifest.get("qualification_permitted") is True
        and admission.get("protocol") == "RIST-C0-v2.3-CALIBRATION-ADMISSION-v1"
        and admission.get("passed") is True
        and admission.get("decision") == "PASS_CALIBRATION_TO_QUALIFICATION"
        and admission.get("qualification_accessed") is False
    ):
        raise PermissionError("qualification requires its exact calibration GO admission")
    binding = manifest["canary_gate"]["evidence_files"].get("calibration_admission", {})
    if binding.get("sha256") != _sha256(admission_path):
        raise ValueError("qualification manifest is not bound to this calibration admission")

    finalizer = _load("rist_c0_v2_3_qualification_finalizer", "finalize_scientific_collection.py")
    collection_final = finalizer.finalize(manifest_path)
    if not (
        collection_final.get("passed") is True
        and collection_final.get("decision")
        == "PASS_QUALIFICATION_COLLECTION_TO_SEPARATE_ANALYSIS"
        and collection_final.get("outcomes_inspected") is False
    ):
        raise PermissionError("qualification outcomes remain sealed until collection PASS")

    pool_path = Path(manifest["pool_manifest"])
    if _sha256(pool_path) != manifest["pool_manifest_sha256"]:
        raise ValueError("pool manifest hash mismatch")
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    spec = pool["splits"]["qualification"]
    source_path = pool_path.parent / spec["file"]
    if _sha256(source_path) != spec["sha256"]:
        raise ValueError("qualification source hash mismatch")
    source_rows = _read_jsonl(source_path)
    receipt = {
        "protocol": "RIST-C0-v2.3-QUALIFICATION-OUTCOME-ACCESS-v1",
        "collection_manifest_sha256": _sha256(manifest_path),
        "calibration_admission_sha256": _sha256(admission_path),
        "pool_manifest_sha256": _sha256(pool_path),
        "qualification_source_sha256": _sha256(source_path),
        "heldout_accessed": False,
        "bfcl_accessed": False,
        "training_performed": False,
    }
    _exclusive_json(output_dir / "OUTCOME_ACCESS_RECEIPT.json", receipt)

    replay = _load("rist_c0_v2_3_qualification_replay", "replay_resolution_outcomes.py")
    resolution = _load("rist_c0_v2_3_qualification_resolution", "resolution_analysis.py")
    family_results = []
    for job in manifest["jobs"]:
        family = str(job["family"])
        rows = replay.replay_outcomes(
            source_rows=source_rows,
            rollout_seeds=[int(seed) for seed in spec["rollout_seeds"]],
            split="qualification",
            trajectories=_read_jsonl(Path(job["trajectories"])),
            journal_rows=_read_jsonl(Path(job["raw_journal"])),
        )
        qualification = resolution.summarize_split(rows, "qualification")
        calibration = admission["families"][family]
        family_results.append(_transport_family(
            family=family,
            checkpoint=manifest["models"][family],
            calibration=calibration,
            qualification=qualification,
            frozen_candidates=admission["common_candidate_map"],
        ))
    combined = resolution.combine_families(family_results)
    passed = combined.get("passed") is True
    result = {
        "protocol": "RIST-C0-v2.3-QUALIFICATION-TRANSPORT-v1",
        "passed": passed,
        "decision": "PASS_C0_V2_3_TO_E1_CAPACITY" if passed else "KILL_C0_V2_3_TRANSPORT",
        "collection_manifest_sha256": _sha256(manifest_path),
        "calibration_admission_sha256": _sha256(admission_path),
        "pool_manifest_sha256": _sha256(pool_path),
        "families": family_results,
        "common_resolution_map": combined.get("common_resolution_map", {}),
        "band_cell_counts": combined.get("band_cell_counts", {"low": 0, "high": 0}),
        "whole_cell_selection_only": True,
        "heldout_accessed": False,
        "bfcl_accessed": False,
        "training_performed": False,
        "outcome_access_receipt_sha256": _sha256(output_dir / "OUTCOME_ACCESS_RECEIPT.json"),
    }
    _exclusive_json(output_dir / "QUALIFICATION_TRANSPORT_RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--calibration-admission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze(args.manifest, args.calibration_admission, args.output_dir)
    except Exception as exc:
        if args.output_dir.is_dir() and not (args.output_dir / "QUALIFICATION_TRANSPORT_RESULT.json").exists():
            _exclusive_json(args.output_dir / "TERMINAL_RESULT.json", {
                "protocol": "RIST-C0-v2.3-QUALIFICATION-TERMINAL-v1",
                "passed": False,
                "decision": "KILL_C0_V2_3_QUALIFICATION_PROTOCOL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "repair_or_rerun_permitted": False,
            })
        raise
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

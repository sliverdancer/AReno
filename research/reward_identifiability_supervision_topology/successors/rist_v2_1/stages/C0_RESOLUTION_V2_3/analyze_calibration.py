"""Irreversibly analyze only C0 v2.3 calibration outcomes and freeze cell candidates."""

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


def analyze(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError("calibration outcome directory must be fresh")
    output_dir.mkdir(parents=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        manifest.get("protocol") == "RIST-C0-v2.3-SCIENTIFIC-COLLECTION-MANIFEST-v1"
        and manifest.get("split") == "calibration"
        and manifest.get("calibration_permitted") is True
        and manifest.get("qualification_permitted") is False
        and manifest.get("job_count") == 2
        and manifest.get("trajectory_count") == 2048
    ):
        raise PermissionError("calibration analysis requires the calibration-only manifest")
    finalizer = _load("rist_c0_v2_3_calibration_finalizer", "finalize_scientific_collection.py")
    collection_final = finalizer.finalize(manifest_path)
    if not (
        collection_final.get("passed") is True
        and collection_final.get("decision")
        == "PASS_CALIBRATION_COLLECTION_TO_SEPARATE_ANALYSIS"
        and collection_final.get("outcomes_inspected") is False
    ):
        raise PermissionError("calibration outcomes remain sealed until content-blind collection PASS")

    pool_path = Path(manifest["pool_manifest"])
    if _sha256(pool_path) != manifest["pool_manifest_sha256"]:
        raise ValueError("pool manifest hash mismatch")
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    spec = pool["splits"]["calibration"]
    source_path = pool_path.parent / spec["file"]
    if _sha256(source_path) != spec["sha256"]:
        raise ValueError("calibration source hash mismatch")
    source_rows = _read_jsonl(source_path)

    receipt = {
        "protocol": "RIST-C0-v2.3-CALIBRATION-OUTCOME-ACCESS-v1",
        "collection_manifest_sha256": _sha256(manifest_path),
        "collection_final_sha256": hashlib.sha256(
            (json.dumps(collection_final, indent=2, sort_keys=True) + "\n").encode()
        ).hexdigest(),
        "pool_manifest_sha256": _sha256(pool_path),
        "gpu_uuid": manifest["canary_gate"]["gpu_identity"]["gpu_uuid"],
        "calibration_source_sha256": _sha256(source_path),
        "qualification_accessed": False,
        "heldout_accessed": False,
        "bfcl_accessed": False,
        "training_performed": False,
    }
    _exclusive_json(output_dir / "OUTCOME_ACCESS_RECEIPT.json", receipt)

    replay = _load("rist_c0_v2_3_calibration_replay", "replay_resolution_outcomes.py")
    resolution = _load("rist_c0_v2_3_calibration_resolution", "resolution_analysis.py")
    families = {}
    for job in manifest["jobs"]:
        family = str(job["family"])
        rows = replay.replay_outcomes(
            source_rows=source_rows,
            rollout_seeds=[int(seed) for seed in spec["rollout_seeds"]],
            split="calibration",
            trajectories=_read_jsonl(Path(job["trajectories"])),
            journal_rows=_read_jsonl(Path(job["raw_journal"])),
        )
        families[family] = resolution.summarize_split(rows, "calibration")

    maps = []
    for family in ("qwen3", "gemma4"):
        cells = families[family]["cells"]
        maps.append({
            cell: "low" if row["classification"] == "collapsed" else "high"
            for cell, row in cells.items()
            if row["classification"] in {"collapsed", "resolved"}
        })
    common = {
        cell: maps[0][cell]
        for cell in sorted(set(maps[0]) & set(maps[1]))
        if maps[0][cell] == maps[1][cell]
    }
    counts = {band: sum(value == band for value in common.values()) for band in ("low", "high")}
    passed = counts["low"] >= 2 and counts["high"] >= 2
    result = {
        "protocol": "RIST-C0-v2.3-CALIBRATION-ADMISSION-v1",
        "passed": passed,
        "decision": "PASS_CALIBRATION_TO_QUALIFICATION" if passed else "KILL_C0_V2_3_CALIBRATION",
        "collection_manifest_sha256": _sha256(manifest_path),
        "pool_manifest_sha256": _sha256(pool_path),
        "gpu_uuid": manifest["canary_gate"]["gpu_identity"]["gpu_uuid"],
        "families": families,
        "common_candidate_map": common,
        "common_candidate_counts": counts,
        "whole_cell_selection_only": True,
        "qualification_accessed": False,
        "heldout_accessed": False,
        "bfcl_accessed": False,
        "training_performed": False,
        "outcome_access_receipt_sha256": _sha256(output_dir / "OUTCOME_ACCESS_RECEIPT.json"),
    }
    _exclusive_json(output_dir / "CALIBRATION_ADMISSION.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze(args.manifest, args.output_dir)
    except Exception as exc:
        if args.output_dir.is_dir() and not (args.output_dir / "CALIBRATION_ADMISSION.json").exists():
            _exclusive_json(args.output_dir / "TERMINAL_RESULT.json", {
                "protocol": "RIST-C0-v2.3-CALIBRATION-TERMINAL-v1",
                "passed": False,
                "decision": "KILL_C0_V2_3_CALIBRATION_PROTOCOL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "repair_or_rerun_permitted": False,
            })
        raise
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

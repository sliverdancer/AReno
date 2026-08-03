"""Validate four C0 jobs, calibrate both families, intersect, and filter train."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

STAGE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[6]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"C0 module unavailable: {name}")
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _verify_job(
    manifest: dict[str, Any],
    family: str,
    split: str,
    result_path: Path,
    journal_path: Path,
    validator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = json.loads(result_path.read_text())
    spec = manifest["splits"][split]
    source_path = REPO_ROOT / manifest["source_data_dir"] / spec["file"]
    if _sha(source_path) != spec["sha256"]:
        raise ValueError("C0 source split hash changed before finalization")
    validation = validator.validate_collection(
        manifest,
        family,
        split,
        result,
        _jsonl(journal_path),
        _jsonl(source_path),
        _sha(journal_path),
    )
    validation.update(
        {
            "result_sha256": _sha(result_path),
            "journal_sha256": _sha(journal_path),
        }
    )
    if validation["passed"] is not True:
        raise ValueError(f"C0 {family}/{split} evidence validation failed")
    return result, validation


def _verify_ledger(
    ledger_path: Path,
    family: str,
    result_path: Path,
    journal_path: Path,
) -> dict[str, Any]:
    ledger = json.loads(ledger_path.read_text())
    passed = all(
        (
            ledger.get("protocol") == "RIST-C0-v2.1",
            ledger.get("family") == family,
            ledger.get("split") == "qualification",
            ledger.get("consumed") is True,
            ledger.get("result_complete") is True,
            ledger.get("result_sha256") == _sha(result_path),
            ledger.get("journal_sha256") == _sha(journal_path),
        )
    )
    if not passed:
        raise ValueError(f"C0 {family} qualification ledger mismatch")
    return {"passed": True, "ledger_sha256": _sha(ledger_path)}


def finalize(
    manifest: dict[str, Any],
    bindings: dict[str, dict[str, Path]],
    output_dir: Path,
) -> dict[str, Any]:
    validator = _load("rist_c0_finalize_validator", STAGE / "validate_collection_evidence.py")
    calibrator = _load("rist_c0_finalize_calibrator", STAGE / "calibrate_resolution.py")
    combiner = _load("rist_c0_finalize_combiner", STAGE / "combine_family_maps.py")
    filterer = _load("rist_c0_finalize_filter", STAGE / "filter_train_pool.py")
    output_dir.mkdir(parents=True, exist_ok=False)
    family_results = []
    evidence_summary = {}
    for family in ("qwen3", "gemma4"):
        calibration, calibration_validation = _verify_job(
            manifest,
            family,
            "calibration",
            bindings[family]["calibration_result"],
            bindings[family]["calibration_journal"],
            validator,
        )
        qualification, qualification_validation = _verify_job(
            manifest,
            family,
            "qualification",
            bindings[family]["qualification_result"],
            bindings[family]["qualification_journal"],
            validator,
        )
        ledger = _verify_ledger(
            bindings[family]["qualification_ledger"],
            family,
            bindings[family]["qualification_result"],
            bindings[family]["qualification_journal"],
        )
        calibrated = calibrator.calibrate_checkpoint(
            calibration["trajectories"],
            qualification["trajectories"],
            manifest["models"][family],
        )
        family_path = output_dir / f"{family}_resolution.json"
        family_path.write_text(json.dumps(calibrated, indent=2, sort_keys=True) + "\n")
        family_results.append(calibrated)
        evidence_summary[family] = {
            "calibration": calibration_validation,
            "qualification": qualification_validation,
            "qualification_ledger": ledger,
            "resolution_sha256": _sha(family_path),
        }
    combined = combiner.combine_maps(family_results)
    combined_path = output_dir / "common_resolution_map.json"
    combined_path.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    filtered = None
    if combined["passed"] is True:
        train_path = (
            REPO_ROOT
            / "research/reward_identifiability_supervision_topology/successors/rist_v2_1"
            / "stages/D3/data/train.jsonl"
        )
        filtered = filterer.filter_train(train_path, combined, output_dir / "filtered" / "data")
    result = {
        "protocol": "RIST-C0-FINAL-v2.1",
        "source_commit": manifest["source_commit"],
        "evidence": evidence_summary,
        "common_resolution_map_sha256": _sha(combined_path),
        "combined": combined,
        "filtered_train": filtered,
        "passed": combined["passed"] is True and filtered is not None,
        "decision": (
            "PASS_C0_TO_FILTERED_TRAIN"
            if combined["passed"] is True and filtered is not None
            else combined["decision"]
        ),
    }
    (output_dir / "FINAL_RESULT.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    for family in ("qwen", "gemma"):
        parser.add_argument(f"--{family}-calibration-result", type=Path, required=True)
        parser.add_argument(f"--{family}-calibration-journal", type=Path, required=True)
        parser.add_argument(f"--{family}-qualification-result", type=Path, required=True)
        parser.add_argument(f"--{family}-qualification-journal", type=Path, required=True)
        parser.add_argument(f"--{family}-qualification-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    bindings = {
        "qwen3": {
            "calibration_result": args.qwen_calibration_result,
            "calibration_journal": args.qwen_calibration_journal,
            "qualification_result": args.qwen_qualification_result,
            "qualification_journal": args.qwen_qualification_journal,
            "qualification_ledger": args.qwen_qualification_ledger,
        },
        "gemma4": {
            "calibration_result": args.gemma_calibration_result,
            "calibration_journal": args.gemma_calibration_journal,
            "qualification_result": args.gemma_qualification_result,
            "qualification_journal": args.gemma_qualification_journal,
            "qualification_ledger": args.gemma_qualification_ledger,
        },
    }
    result = finalize(json.loads(args.manifest.read_text()), bindings, args.output_dir)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

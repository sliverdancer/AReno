"""Assemble one normal RIST training run into the frozen P4 result schema."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

CHECKPOINT_STEPS = (25, 50, 75, 100)


def _load_evidence_validator():
    path = Path(__file__).with_name("run_evidence_manifest.py")
    spec = importlib.util.spec_from_file_location("rist_p4_run_evidence", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("P4 run-evidence validator is unavailable")
    spec.loader.exec_module(module)
    return module


def _selected_success(
    evaluation: dict[str, Any], resolution_map: dict[str, str]
) -> tuple[float, dict[str, float]]:
    selected = [
        row
        for row in evaluation["trajectories"]
        if row["structural_cell"] in resolution_map
    ]
    if not selected:
        raise ValueError("evaluation contains no preselected resolution cells")
    by_band = {}
    for band in ("low", "high"):
        rows = [
            row
            for row in selected
            if resolution_map[row["structural_cell"]] == band
        ]
        if not rows:
            raise ValueError(f"evaluation contains no {band} resolution rows")
        by_band[band] = sum(int(row["strict_success"]) for row in rows) / len(rows)
    overall = sum(int(row["strict_success"]) for row in selected) / len(selected)
    return overall, by_band


def assemble_run(
    run_design: dict[str, Any],
    training: dict[str, Any],
    dev_results: list[dict[str, Any]],
    confirmatory: dict[str, Any],
    resolution_map_result: dict[str, Any],
    evidence_manifest: Path,
    evidence_root: Path,
    evidence_relative_path: str,
) -> dict[str, Any]:
    if resolution_map_result.get("passed") is not True:
        raise ValueError("run assembly requires a passed common resolution map")
    resolution_map = {
        str(cell): str(band)
        for cell, band in resolution_map_result["common_resolution_map"].items()
    }
    if training.get("protocol") != "RIST-P4-v2.1":
        raise ValueError("unexpected training evidence protocol")
    points = training.get("points")
    if [int(point["checkpoint_step"]) for point in points] != list(CHECKPOINT_STEPS):
        raise ValueError("training evidence misses frozen checkpoint steps")
    if len(dev_results) != 4:
        raise ValueError("run assembly requires four development evaluations")
    dev_by_step = {}
    for evaluation in dev_results:
        if evaluation.get("split") != "dev_curve" or evaluation.get("complete") is not True:
            raise ValueError("development evaluation is incomplete")
        marker = "-dev-step-"
        checkpoint_id = str(evaluation["checkpoint_id"])
        if marker not in checkpoint_id:
            raise ValueError("development checkpoint id is malformed")
        step = int(checkpoint_id.rsplit(marker, 1)[1])
        if step in dev_by_step:
            raise ValueError("duplicate development checkpoint step")
        dev_by_step[step] = evaluation
    if set(dev_by_step) != set(CHECKPOINT_STEPS):
        raise ValueError("development evaluations miss frozen steps")
    if confirmatory.get("split") != "confirmatory" or confirmatory.get("complete") is not True:
        raise ValueError("confirmatory evaluation is incomplete")
    expected_prefix = f"{run_design['run_id']}-"
    if any(not str(row["checkpoint_id"]).startswith(expected_prefix) for row in dev_results):
        raise ValueError("development result belongs to a different run")
    if not str(confirmatory["checkpoint_id"]).startswith(expected_prefix):
        raise ValueError("confirmatory result belongs to a different run")

    curve = []
    for point in points:
        step = int(point["checkpoint_step"])
        success, by_band = _selected_success(dev_by_step[step], resolution_map)
        curve.append(
            {
                "step": step,
                "cumulative_trainable_tokens": int(point["cumulative_trainable_tokens"]),
                "strict_success": success,
                "strict_success_by_resolution": by_band,
                "mean_training_reward_diagnostic": float(point["mean_training_reward"]),
            }
        )
    confirmatory_success, confirmatory_by_band = _selected_success(
        confirmatory, resolution_map
    )
    if not evidence_manifest.is_file():
        raise FileNotFoundError("run evidence manifest is missing")
    manifest_payload = json.loads(evidence_manifest.read_text())
    verification = _load_evidence_validator().validate_manifest(
        manifest_payload, evidence_root, str(run_design["run_id"])
    )
    if verification.get("passed") is not True:
        raise ValueError("run evidence manifest did not pass")
    return {
        "run_id": run_design["run_id"],
        "family": run_design["family"],
        "algorithm": run_design["algorithm"],
        "arm": run_design["arm"],
        "seed": run_design["seed"],
        "catastrophic": False,
        "nonzero_advantage_groups": int(training["total_nonzero_advantage_groups"]),
        "confirmatory_strict_success": confirmatory_success,
        "confirmatory_strict_success_by_resolution": confirmatory_by_band,
        "curve": curve,
        "raw_evidence_path": evidence_relative_path,
        "raw_evidence_sha256": hashlib.sha256(evidence_manifest.read_bytes()).hexdigest(),
        "raw_evidence_artifact_count": verification["artifact_count"],
        "resolution_map_sha256": hashlib.sha256(
            (json.dumps(resolution_map_result, sort_keys=True) + "\n").encode()
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-design", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--dev-result", type=Path, action="append", required=True)
    parser.add_argument("--confirmatory", type=Path, required=True)
    parser.add_argument("--resolution-map", type=Path, required=True)
    parser.add_argument("--evidence-manifest", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--evidence-relative-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assemble_run(
        json.loads(args.run_design.read_text()),
        json.loads(args.training.read_text()),
        [json.loads(path.read_text()) for path in args.dev_result],
        json.loads(args.confirmatory.read_text()),
        json.loads(args.resolution_map.read_text()),
        args.evidence_manifest,
        args.evidence_root,
        args.evidence_relative_path,
    )
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

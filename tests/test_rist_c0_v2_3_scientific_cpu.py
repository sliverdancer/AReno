"""CPU-only regression tests for the split C0 v2.3 scientific protocol."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    REPO_ROOT
    / "research/reward_identifiability_supervision_topology/successors/rist_v2_1/stages/C0_RESOLUTION_V2_3"
)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, STAGE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _identity(builder, family: str, gpu_uuid: str) -> dict:
    return {
        "family": family,
        "checkpoint": builder.MODELS[family],
        "model_revision": builder.REVISIONS[family],
        "tokenizer_snapshot_sha256": builder.TOKENIZER_SNAPSHOTS[family],
        "source_commit": _head(),
        "gpu_uuid": gpu_uuid,
        "gpu_total_memory_gib": 80.0,
    }


def _evidence(tmp_path: Path):
    capacity_source = STAGE / "gpu_bind_20260804_a800/capacity_20260804_v1/CAPACITY_CANARY_RESULT.json"
    shutdown_source = STAGE / "gpu_bind_20260804_a800/clean_shutdown_20260804_v1/CLEAN_SHUTDOWN_GPU_RESULT.json"
    capacity = tmp_path / "capacity.json"
    shutdown = tmp_path / "shutdown.json"
    capacity.write_bytes(capacity_source.read_bytes())
    shutdown_value = json.loads(shutdown_source.read_text())
    shutdown_value["gpu_uuid"] = "GPU-scientific-test"
    shutdown.write_text(json.dumps(shutdown_value, sort_keys=True) + "\n")
    return capacity, shutdown


def _receipt(builder, tmp_path: Path, family: str, gpu_uuid: str) -> Path:
    embedded = b'{"protocol":"RIST-C0-v2.3-SERVING-IDENTITY-v1"}\n'
    body = {
        "protocol": "RIST-C0-v2.3-DEPLOYMENT-RECEIPT-v1",
        "authority_sha256": "1" * 64,
        "bindings": {
            "control_commit": _head(),
            "runtime_commit": _head(),
            "manifest_sha256": hashlib.sha256(embedded).hexdigest(),
            "model_revision": builder.REVISIONS[family],
            "gpu_uuid": gpu_uuid,
            "extension_sha256": "2" * 64,
        },
        "manifest_base64": base64.b64encode(embedded).decode(),
        "launcher_command": ["python", "-m", "areno.cli.main", "serve", "127.0.0.1"],
    }
    body["receipt_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path = tmp_path / f"{family}-receipt.json"
    path.write_text(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def test_calibration_manifest_is_two_family_only_and_seals_qualification(tmp_path, monkeypatch):
    builder = _load("rist_c0_v2_3_builder_cal", "build_scientific_collection_manifest.py")
    monkeypatch.setattr(builder, "_require_head", lambda commit: commit)
    capacity, shutdown = _evidence(tmp_path)
    receipts = {family: _receipt(builder, tmp_path, family, "GPU-scientific-test") for family in builder.FAMILIES}
    manifest = builder.build_manifest(
        split="calibration",
        pool_manifest_path=STAGE / "data/manifest.json",
        capacity_result_path=capacity,
        clean_shutdown_result_path=shutdown,
        runtime_identities={
            family: _identity(builder, family, "GPU-scientific-test")
            for family in builder.FAMILIES
        },
        deployment_receipts=receipts,
        source_commit=_head(),
        output_root=tmp_path / "out",
    )
    assert manifest["job_count"] == 2
    assert manifest["trajectory_count"] == 2048
    assert {job["job_id"] for job in manifest["jobs"]} == {
        "qwen3-calibration",
        "gemma4-calibration",
    }
    assert manifest["calibration_permitted"] is True
    assert manifest["qualification_permitted"] is False
    assert manifest["heldout_permitted"] is False
    assert manifest["bfcl_permitted"] is False
    assert manifest["training_permitted"] is False


def test_qualification_manifest_requires_exact_calibration_go(tmp_path, monkeypatch):
    builder = _load("rist_c0_v2_3_builder_qual", "build_scientific_collection_manifest.py")
    monkeypatch.setattr(builder, "_require_head", lambda commit: commit)
    capacity, shutdown = _evidence(tmp_path)
    receipts = {family: _receipt(builder, tmp_path, family, "GPU-scientific-test") for family in builder.FAMILIES}
    kwargs = dict(
        split="qualification",
        pool_manifest_path=STAGE / "data/manifest.json",
        capacity_result_path=capacity,
        clean_shutdown_result_path=shutdown,
        runtime_identities={
            family: _identity(builder, family, "GPU-scientific-test")
            for family in builder.FAMILIES
        },
        deployment_receipts=receipts,
        source_commit=_head(),
        output_root=tmp_path / "out",
    )
    with pytest.raises(PermissionError, match="sealed"):
        builder.build_manifest(**kwargs)
    admission = tmp_path / "admission.json"
    admission.write_text(json.dumps({
        "protocol": "RIST-C0-v2.3-CALIBRATION-ADMISSION-v1",
        "passed": True,
        "decision": "PASS_CALIBRATION_TO_QUALIFICATION",
        "pool_manifest_sha256": builder._sha256(STAGE / "data/manifest.json"),
        "qualification_accessed": False,
        "common_candidate_counts": {"low": 2, "high": 2},
    }) + "\n")
    manifest = builder.build_manifest(**kwargs, calibration_admission_path=admission)
    assert {job["job_id"] for job in manifest["jobs"]} == {
        "qwen3-qualification",
        "gemma4-qualification",
    }
    assert manifest["calibration_permitted"] is False
    assert manifest["qualification_permitted"] is True


def test_manifest_rejects_capacity_boundary_crossing(tmp_path, monkeypatch):
    builder = _load("rist_c0_v2_3_builder_boundary", "build_scientific_collection_manifest.py")
    monkeypatch.setattr(builder, "_require_head", lambda commit: commit)
    capacity, shutdown = _evidence(tmp_path)
    receipts = {family: _receipt(builder, tmp_path, family, "GPU-scientific-test") for family in builder.FAMILIES}
    value = json.loads(capacity.read_text())
    value["outcomes_inspected"] = True
    capacity.write_text(json.dumps(value) + "\n")
    with pytest.raises(PermissionError, match="forbidden boundary"):
        builder.build_manifest(
            split="calibration",
            pool_manifest_path=STAGE / "data/manifest.json",
            capacity_result_path=capacity,
            clean_shutdown_result_path=shutdown,
            runtime_identities={
                family: _identity(builder, family, "GPU-scientific-test")
                for family in builder.FAMILIES
            },
            deployment_receipts=receipts,
            source_commit=_head(),
            output_root=tmp_path / "out",
        )


def test_source_commit_gate_rejects_dirty_worktree(monkeypatch):
    builder = _load("rist_c0_v2_3_builder_dirty", "build_scientific_collection_manifest.py")
    responses = iter((SimpleNamespace(stdout="a" * 40 + "\n"), SimpleNamespace(stdout=" M changed.py\n")))
    monkeypatch.setattr(builder.subprocess, "run", lambda *args, **kwargs: next(responses))
    with pytest.raises(ValueError, match="clean checked-out worktree"):
        builder._require_head("a" * 40)


def test_deployment_receipt_rejects_two_worktree_commits(tmp_path):
    builder = _load("rist_c0_v2_3_builder_receipt", "build_scientific_collection_manifest.py")
    path = _receipt(builder, tmp_path, "qwen3", "GPU-scientific-test")
    value = json.loads(path.read_text())
    value["bindings"]["runtime_commit"] = "b" * 40
    body = {key: value[key] for key in (
        "protocol", "authority_sha256", "bindings", "manifest_base64", "launcher_command"
    )}
    value["receipt_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="one frozen commit"):
        builder._validate_deployment_receipt(
            path, "qwen3", _head(), "GPU-scientific-test"
        )


def test_scientific_runner_requires_one_exact_accept(tmp_path):
    runner = _load("rist_c0_v2_3_runner_ledger", "run_scientific_job.py")
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(
        json.dumps({"event": "prelaunch_decision", "decision": "ACCEPT"}) + "\n"
        + json.dumps({"event": "launch_attempt"}) + "\n"
    )
    assert runner._ledger_accepted(ledger) is True
    ledger.write_text(ledger.read_text() + json.dumps({"event": "prelaunch_decision", "decision": "REJECT"}) + "\n")
    assert runner._ledger_accepted(ledger) is False


def _resolution_rows(split: str) -> list[dict]:
    rows = []
    for cell_index in range(8):
        for task_index in range(4):
            for seed in range(1, 33):
                high = cell_index >= 4
                rows.append({
                    "split": split,
                    "strict_success": int(high and seed % 2 == 0),
                    "rollout_seed": seed,
                    "structural_cell": f"c{cell_index:02d}",
                    "task_id": f"{cell_index}-{task_index}",
                })
    return rows


def test_resolution_rule_finds_four_low_and_four_high_cells():
    resolution = _load("rist_c0_v2_3_resolution_rules", "resolution_analysis.py")
    summary = resolution.summarize_split(_resolution_rows("calibration"), "calibration")
    assert {cell for cell, row in summary["cells"].items() if row["classification"] == "collapsed"} == {
        "c00", "c01", "c02", "c03"
    }
    assert {cell for cell, row in summary["cells"].items() if row["classification"] == "resolved"} == {
        "c04", "c05", "c06", "c07"
    }


def test_cross_family_transport_requires_two_cells_per_band():
    resolution = _load("rist_c0_v2_3_cross_family", "resolution_analysis.py")
    mapping = {f"c{index:02d}": "low" if index < 4 else "high" for index in range(8)}
    rows = [
        {"family": family, "checkpoint": family, "resolution_map": mapping, "passed": True}
        for family in ("qwen3", "gemma4")
    ]
    result = resolution.combine_families(rows)
    assert result["passed"] is True
    assert result["band_cell_counts"] == {"low": 4, "high": 4}
    rows[1]["resolution_map"] = {"c00": "low", "c04": "high"}
    assert resolution.combine_families(rows)["passed"] is False


def test_qualification_cannot_admit_cells_not_selected_by_calibration():
    analyzer = _load("rist_c0_v2_3_qualification_selection", "analyze_qualification.py")
    cells = {
        f"c{index:02d}": {"classification": "collapsed" if index < 4 else "resolved"}
        for index in range(8)
    }
    frozen = {"c00": "low", "c01": "low", "c04": "high", "c05": "high"}
    result = analyzer._transport_family(
        family="qwen3",
        checkpoint="test",
        calibration={"cells": cells},
        qualification={"cells": cells},
        frozen_candidates=frozen,
    )
    assert result["passed"] is True
    assert result["resolution_map"] == frozen
    assert "c02" not in result["transport"]

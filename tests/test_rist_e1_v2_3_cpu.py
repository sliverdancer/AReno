"""CPU-only tests for the C0 v2.3 to E1 admission boundary."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    REPO_ROOT
    / "research/reward_identifiability_supervision_topology/successors/rist_v2_1/stages/E1_V2_3"
)
D3 = (
    REPO_ROOT
    / "research/reward_identifiability_supervision_topology/successors/rist_v2_1/stages/D3"
)


def _load():
    path = STAGE / "build_admission.py"
    spec = importlib.util.spec_from_file_location("rist_e1_v2_3_admission_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_deployment():
    path = STAGE / "build_deployment_manifest.py"
    spec = importlib.util.spec_from_file_location("rist_e1_v2_3_deployment_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _transport() -> dict:
    mapping = {f"c{index:02d}": "low" if index < 4 else "high" for index in range(8)}
    return {
        "protocol": "RIST-C0-v2.3-QUALIFICATION-TRANSPORT-v1",
        "passed": True,
        "decision": "PASS_C0_V2_3_TO_E1_CAPACITY",
        "whole_cell_selection_only": True,
        "heldout_accessed": False,
        "bfcl_accessed": False,
        "training_performed": False,
        "gpu_uuid": "GPU-test",
        "common_resolution_map": mapping,
        "band_cell_counts": {"low": 4, "high": 4},
        "families": [
            {"family": "qwen3", "passed": True},
            {"family": "gemma4", "passed": True},
        ],
    }


def _write_transport(tmp_path: Path, value: dict | None = None) -> Path:
    path = tmp_path / "transport.json"
    path.write_text(json.dumps(value or _transport(), sort_keys=True) + "\n")
    return path


def test_admission_retains_whole_cells_and_selects_first_high(tmp_path):
    builder = _load()
    result = builder.build_admission(
        transport_result_path=_write_transport(tmp_path),
        d3_manifest_path=D3 / "data/manifest.json",
        d3_train_path=D3 / "data/train.jsonl",
        output_root=tmp_path / "e1",
        source_commit="a" * 40,
    )
    assert result["passed"] is True
    assert result["filtered_task_count"] == 32
    assert result["capacity_task_count"] == 4
    assert result["selected_capacity_cell"] == "c04"
    assert result["selection_uses_individual_outcomes"] is False
    assert result["execution_authorized"] is False
    assert result["training_permitted"] is False
    rows = [json.loads(line) for line in Path(result["capacity_train_path"]).read_text().splitlines()]
    assert len(rows) == 4
    assert {row["structural_cell"] for row in rows} == {"c04"}
    assert {row["resolution_band"] for row in rows} == {"high"}
    data_manifest = json.loads(Path(result["capacity_data_manifest_path"]).read_text())
    assert data_manifest["protocol"] == "RIST-E1-CAPACITY-DATA-v2.1"
    assert data_manifest["selected_cell"] == "c04"
    assert data_manifest["task_count"] == 4
    resolution = json.loads(Path(result["resolution_result_path"]).read_text())
    assert resolution["protocol"] == "RIST-C0-v2.3-QUALIFICATION-TRANSPORT-v1"


def test_calibration_only_result_cannot_open_e1(tmp_path):
    builder = _load()
    value = _transport()
    value["protocol"] = "RIST-C0-v2.3-CALIBRATION-ADMISSION-v1"
    with pytest.raises(PermissionError, match="qualification transport PASS"):
        builder.build_admission(
            transport_result_path=_write_transport(tmp_path, value),
            d3_manifest_path=D3 / "data/manifest.json",
            d3_train_path=D3 / "data/train.jsonl",
            output_root=tmp_path / "e1",
            source_commit="a" * 40,
        )


def test_partial_family_result_cannot_open_e1(tmp_path):
    builder = _load()
    value = _transport()
    value["families"] = [{"family": "qwen3", "passed": True}]
    with pytest.raises(PermissionError, match="both family"):
        builder.build_admission(
            transport_result_path=_write_transport(tmp_path, value),
            d3_manifest_path=D3 / "data/manifest.json",
            d3_train_path=D3 / "data/train.jsonl",
            output_root=tmp_path / "e1",
            source_commit="a" * 40,
        )


def test_fewer_than_two_cells_per_band_cannot_open_e1(tmp_path):
    builder = _load()
    value = _transport()
    value["common_resolution_map"] = {"c00": "low", "c04": "high"}
    value["band_cell_counts"] = {"low": 1, "high": 1}
    with pytest.raises(PermissionError, match="two transported"):
        builder.build_admission(
            transport_result_path=_write_transport(tmp_path, value),
            d3_manifest_path=D3 / "data/manifest.json",
            d3_train_path=D3 / "data/train.jsonl",
            output_root=tmp_path / "e1",
            source_commit="a" * 40,
        )


def _deployment_fixture(tmp_path: Path):
    admission_builder = _load()
    admission_builder.build_admission(
        transport_result_path=_write_transport(tmp_path),
        d3_manifest_path=D3 / "data/manifest.json",
        d3_train_path=D3 / "data/train.jsonl",
        output_root=tmp_path / "e1-admission",
        source_commit="a" * 40,
    )
    admission_path = tmp_path / "e1-admission/E1_ADMISSION.json"
    identities = {}
    for family in ("qwen3", "gemma4"):
        model_path = tmp_path / f"model-{family}"
        model_path.mkdir()
        value = {
            "protocol": "RIST-E1-v2.3-RUNTIME-IDENTITY-v1",
            "family": family,
            "checkpoint": admission_builder.MODELS[family],
            "model_revision": admission_builder.REVISIONS[family],
            "tokenizer_snapshot_sha256": admission_builder.TOKENIZER_SNAPSHOTS[family],
            "model_weights_sha256": "1" * 64,
            "snapshot_verification_sha256": "2" * 64,
            "model_path": str(model_path),
            "gpu_name": "NVIDIA A800 80GB PCIe",
            "gpu_uuid": "GPU-test",
            "gpu_total_memory_gib": 80.0,
            "driver_version": "test",
            "cuda_version": "test",
            "torch_version": "test",
            "source_commit": "a" * 40,
            "control_commit": "a" * 40,
            "runtime_commit": "a" * 40,
            "extension_import_sha256": "3" * 64,
        }
        path = tmp_path / f"identity-{family}.json"
        path.write_text(json.dumps(value, sort_keys=True) + "\n")
        identities[family] = path
    return admission_path, identities


def test_deployment_manifest_is_exact_four_job_one_step_matrix(tmp_path):
    builder = _load_deployment()
    admission_path, identities = _deployment_fixture(tmp_path)
    manifest = builder.build_manifest(
        admission_path=admission_path,
        runtime_identity_paths=identities,
        run_root=tmp_path / "runs",
        manifest_output_path=tmp_path / "manifest.json",
        source_commit="a" * 40,
    )
    assert manifest["successor_protocol"] == "RIST-E1-v2.3-DEPLOYMENT-BOUND-v1"
    assert manifest["execution_authorized"] is True
    assert manifest["gpu_training_authorized"] is True
    assert manifest["heldout_permitted"] is False
    assert manifest["bfcl_permitted"] is False
    assert manifest["job_count"] == 4
    assert {(row["family"], row["algorithm"], row["arm"]) for row in manifest["jobs"]} == {
        (family, algorithm, "AF")
        for family in ("qwen3", "gemma4")
        for algorithm in ("gspo", "grpo")
    }
    assert all(row["max_steps"] == 1 for row in manifest["jobs"])
    assert all(row["minimum_gpu_memory_gib"] == 79 for row in manifest["jobs"])
    assert manifest["models"]["qwen3"]["minimum_gpu_memory_gib"] == 24
    assert manifest["models"]["gemma4"]["minimum_gpu_memory_gib"] == 48
    assert all(
        row["deployment_gpu_memory_floor_gib"] == 79
        for row in manifest["models"].values()
    )


def test_deployment_rejects_gpu_uuid_mismatch(tmp_path):
    builder = _load_deployment()
    admission_path, identities = _deployment_fixture(tmp_path)
    value = json.loads(identities["gemma4"].read_text())
    value["gpu_uuid"] = "GPU-other"
    identities["gemma4"].write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="gemma4 runtime identity"):
        builder.build_manifest(
            admission_path=admission_path,
            runtime_identity_paths=identities,
            run_root=tmp_path / "runs",
            manifest_output_path=tmp_path / "manifest.json",
            source_commit="a" * 40,
        )


def test_deployment_rejects_less_than_79_gib(tmp_path):
    builder = _load_deployment()
    admission_path, identities = _deployment_fixture(tmp_path)
    value = json.loads(identities["qwen3"].read_text())
    value["gpu_total_memory_gib"] = 48.0
    identities["qwen3"].write_text(json.dumps(value) + "\n")
    with pytest.raises(ValueError, match="qwen3 runtime identity"):
        builder.build_manifest(
            admission_path=admission_path,
            runtime_identity_paths=identities,
            run_root=tmp_path / "runs",
            manifest_output_path=tmp_path / "manifest.json",
            source_commit="a" * 40,
        )

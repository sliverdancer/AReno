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

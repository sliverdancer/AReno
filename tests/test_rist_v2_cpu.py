from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RIST_ROOT = REPO_ROOT / "research" / "reward_identifiability_supervision_topology"
V2_ROOT = RIST_ROOT / "successors" / "rist_v2"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_rist_v2_d0_reproduces_model_conditional_miscalibration():
    analyzer = _load_module(
        "rist_v2_d0_analyzer",
        V2_ROOT / "stages" / "D0" / "analyze_v1_1_development.py",
    )
    result = analyzer.analyze(
        RIST_ROOT / "stages" / "P1" / "data" / "qualification.jsonl",
        RIST_ROOT
        / "successors"
        / "rist_v1_1"
        / "stages"
        / "P2_1"
        / "gpu_run_20260802"
        / "evidence"
        / "qwen3_0_6b_result.json",
    )

    assert result["stage_status"] == "PASS"
    assert result["scientific_effect_estimated"] is False
    assert result["heldout_data_opened"] is False
    assert result["strata"]["high"]["mixed_group_count"] == 0
    assert result["strata"]["low"]["mixed_group_count"] == 6
    assert all(result["gates"].values())


def test_rist_v2_calibration_propagates_uncertainty_and_fails_closed():
    calibration = _load_module(
        "rist_v2_calibration",
        V2_ROOT / "stages" / "D1" / "calibration.py",
    )

    balanced = calibration.summarize_task(64, 128)
    saturated = calibration.summarize_task(128, 128)
    uncertain = calibration.summarize_task(0, 8)

    assert balanced["classification"] == "resolved"
    assert saturated["classification"] == "collapsed"
    assert uncertain["classification"] == "transition"
    assert balanced["mixed_interval"][0] >= 0.5
    assert saturated["mixed_interval"][1] <= 0.25


def test_rist_v2_cell_summary_requires_replication_and_consistency():
    calibration = _load_module(
        "rist_v2_calibration_cells",
        V2_ROOT / "stages" / "D1" / "calibration.py",
    )

    resolved = calibration.summarize_cell([(64, 128)] * 4)
    heterogeneous = calibration.summarize_cell(
        [(64, 128), (64, 128), (128, 128), (128, 128)]
    )

    assert resolved["cell_classification"] == "resolved"
    assert heterogeneous["cell_classification"] == "heterogeneous"

    try:
        calibration.summarize_cell([(64, 128)] * 3)
    except ValueError as error:
        assert "at least four" in str(error)
    else:
        raise AssertionError("under-replicated cells must be rejected")


def test_rist_v2_d2_frozen_evaluator_records_terminal_representation_failure(tmp_path):
    d2_root = V2_ROOT / "stages" / "D2"
    generator = _load_module("rist_v2_d2_generator", d2_root / "task_generator.py")
    first = generator.generate_splits(seed=41)
    repeated = generator.generate_splits(seed=41)

    assert first == repeated
    assert {split: len(rows) for split, rows in first.items()} == {
        "calibration": 32,
        "qualification": 32,
        "heldout": 32,
    }
    signatures = []
    for rows in first.values():
        signatures.extend(row["task_signature"] for row in rows)
        assert all("reward_resolution_stratum" not in row for row in rows)
    assert len(signatures) == len(set(signatures))

    data_dir = tmp_path / "pool"
    generator.write_splits(data_dir, seed=41)
    evaluator = _load_module("rist_v2_d2_evaluator", d2_root / "evaluate_pool.py")
    result = evaluator.evaluate(data_dir)
    assert result["stage_status"] == "KILL"
    assert result["decision"] == "KILL_RIST_V2_D2_STRUCTURAL_POOL"
    assert result["heldout_data_opened"] is False
    assert result["gates"]["byte_identical_regeneration"] is False
    assert all(
        passed
        for gate, passed in result["gates"].items()
        if gate != "byte_identical_regeneration"
    )

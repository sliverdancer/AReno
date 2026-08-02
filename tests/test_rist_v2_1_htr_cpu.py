from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
D2_1 = (
    REPO_ROOT
    / "research"
    / "reward_identifiability_supervision_topology"
    / "successors"
    / "rist_v2_1"
    / "stages"
    / "D2_1"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_v2_1_baseline_reconstructs_archived_seven_of_eight_without_heldout():
    harness = _load_module("rist_v2_1_htr_eval", D2_1 / "htr_eval.py")
    result = harness.evaluate("dev", D2_1 / "baseline_comparator.py")

    assert result["score"] == 7
    assert result["max_score"] == 8
    assert result["gates"]["byte_identical_regeneration"] is False
    assert result["heldout_data_opened"] is False
    assert result["gpu_used"] is False


def test_v2_1_test_phase_requires_one_shot_ledger(tmp_path):
    harness = _load_module("rist_v2_1_htr_eval_ledger", D2_1 / "htr_eval.py")
    candidate = D2_1 / "baseline_comparator.py"

    try:
        harness.evaluate("test", candidate)
    except ValueError as error:
        assert "one-shot-ledger" in str(error)
    else:
        raise AssertionError("test must fail closed without a one-shot ledger")

    ledger = tmp_path / "consumed.json"
    first = harness.evaluate("test", candidate, ledger)
    assert first["phase"] == "test"
    assert ledger.exists()
    try:
        harness.evaluate("test", candidate, ledger)
    except FileExistsError:
        pass
    else:
        raise AssertionError("qualification test may be consumed only once")

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    REPO_ROOT
    / "research/reward_identifiability_supervision_topology/successors/rist_v2_1/stages/C0_RESOLUTION_V2_3"
)
V2_2_MANIFEST = STAGE.parent / "C0_RESOLUTION_V2_2/data/manifest.json"


def _load_builder():
    path = STAGE / "build_fresh_pool.py"
    spec = importlib.util.spec_from_file_location("rist_c0_v2_3_pool_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _signatures(manifest: dict) -> set[str]:
    return {
        str(signature)
        for split in manifest["splits"].values()
        for signature in split["task_signatures"]
    }


def test_v2_3_pool_uses_fresh_fixed_seeds_and_balanced_splits(tmp_path):
    builder = _load_builder()
    manifest = builder.write_pool(tmp_path / "data")

    assert manifest["protocol"] == "RIST-C0-v2.3-FRESH-POOL"
    assert manifest["pool_seed"] == 2026080417
    assert manifest["capacity_canary_seed"] == 2026080418
    assert manifest["previous_c0_task_pools_retired"] == ["v2.1", "v2.2"]
    assert manifest["splits"]["capacity_canary"]["task_count"] == 1
    assert manifest["splits"]["calibration"]["task_count"] == 32
    assert manifest["splits"]["qualification"]["task_count"] == 32
    assert manifest["splits"]["calibration"]["trajectory_count"] == 1024
    assert manifest["splits"]["qualification"]["trajectory_count"] == 1024
    assert manifest["splits"]["capacity_canary"]["trajectory_count"] == 8


def test_v2_3_pool_is_signature_and_rollout_disjoint_from_v2_2():
    builder = _load_builder()
    rows = builder.build_rows()
    signatures = {
        str(row["task_signature"])
        for split_rows in rows.values()
        for row in split_rows
    }
    prior = json.loads(V2_2_MANIFEST.read_text())

    assert len(signatures) == 65
    assert not signatures & _signatures(prior)
    current_rollouts = {
        seed for values in builder.ROLLOUT_SEEDS.values() for seed in values
    }
    prior_rollouts = {
        seed
        for split in prior["splits"].values()
        for seed in split["rollout_seeds"]
    }
    assert not current_rollouts & prior_rollouts


def test_v2_3_pool_generation_is_byte_reproducible(tmp_path):
    builder = _load_builder()
    first = tmp_path / "first"
    second = tmp_path / "second"
    builder.write_pool(first)
    builder.write_pool(second)

    for name in (
        "capacity_canary.jsonl",
        "calibration.jsonl",
        "qualification.jsonl",
        "manifest.json",
    ):
        assert (first / name).read_bytes() == (second / name).read_bytes()

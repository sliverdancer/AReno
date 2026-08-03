from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE = (
    REPO_ROOT
    / "research/reward_identifiability_supervision_topology/successors/rist_v2_1"
    / "stages/C0_RESOLUTION_V2_2"
)


def _load(name: str, filename: str):
    if str(STAGE) not in sys.path:
        sys.path.insert(0, str(STAGE))
    spec = importlib.util.spec_from_file_location(name, STAGE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _resolution_rows(split: str, low_cells: set[str]) -> list[dict]:
    rows = []
    for cell_index in range(8):
        cell = f"c{cell_index:02d}"
        for task_index in range(4):
            for seed in range(32):
                reward = 0 if cell in low_cells else seed % 2
                rows.append(
                    {
                        "split": split,
                        "structural_cell": cell,
                        "task_id": f"{split}-{cell}-{task_index}",
                        "rollout_seed": seed,
                        "strict_success": reward,
                    }
                )
    return rows


def _invalid_replay_fixture():
    source = [
        json.loads(line)
        for line in (STAGE / "data/calibration.jsonl").read_text().splitlines()
        if line
    ]
    seeds = list(range(32))
    trajectories = []
    journal = []
    for task in source:
        for seed in seeds:
            trajectories.append(
                {
                    "task_id": task["id"],
                    "task_signature": task["task_signature"],
                    "structural_cell": task["structural_cell"],
                    "rollout_seed": seed,
                    "actions": [],
                    "strict_success": 0,
                    "completed_turns": 0,
                    "invalid_reason": "CALL_COUNT",
                    "raw_response_count": 1,
                    "split": "calibration",
                }
            )
            journal.append(
                {
                    "task_id": task["id"],
                    "task_signature": task["task_signature"],
                    "rollout_seed": seed,
                    "turn_index": 0,
                    "raw_response": {"choices": [{"message": {"tool_calls": []}}]},
                }
            )
    return source, seeds, trajectories, journal


def test_resolution_family_transport_and_cross_family_gate():
    analysis = _load("rist_c0_v2_2_resolution_core", "resolution_analysis.py")
    low = {"c00", "c01", "c02", "c03"}
    family_a = analysis.calibrate_family(
        _resolution_rows("calibration", low),
        _resolution_rows("qualification", low),
        "qwen3",
        "qwen",
    )
    family_b = analysis.calibrate_family(
        _resolution_rows("calibration", low),
        _resolution_rows("qualification", low),
        "gemma4",
        "gemma",
    )
    common = analysis.combine_families([family_a, family_b])

    assert family_a["passed"] is True
    assert family_a["band_cell_counts"] == {"low": 4, "high": 4}
    assert common["passed"] is True
    assert common["decision"] == "PASS_C0_V2_2_TO_FILTERED_TRAIN"


def test_resolution_transport_disagreement_cannot_pass():
    analysis = _load("rist_c0_v2_2_resolution_disagree", "resolution_analysis.py")
    calibration = _resolution_rows("calibration", {"c00", "c01", "c02", "c03"})
    qualification = _resolution_rows("qualification", {"c04", "c05", "c06", "c07"})
    result = analysis.calibrate_family(calibration, qualification, "qwen3", "qwen")

    assert result["passed"] is False
    assert result["resolution_map"] == {}


def test_structural_filter_and_e1_selection_keep_whole_cells():
    analysis = _load("rist_c0_v2_2_resolution_filter", "resolution_analysis.py")
    source = [
        json.loads(line)
        for line in (STAGE.parent / "D3/data/train.jsonl").read_text().splitlines()
        if line
    ]
    common = {"c00": "low", "c01": "low", "c06": "high", "c07": "high"}
    filtered = analysis.filter_train_rows(source, common)
    selected_cell, e1 = analysis.select_e1_rows(filtered, common)

    assert len(filtered) == 16
    assert selected_cell == "c06"
    assert len(e1) == 4
    assert {row["structural_cell"] for row in e1} == {"c06"}
    assert all(row["resolution_band"] == "high" for row in e1)


def test_replay_accepts_exact_grid_and_rejects_tampered_outcome_types():
    replay = _load("rist_c0_v2_2_replay", "replay_resolution_outcomes.py")
    source, seeds, trajectories, journal = _invalid_replay_fixture()
    derived = replay.replay_outcomes(
        source_rows=source,
        rollout_seeds=seeds,
        split="calibration",
        trajectories=trajectories,
        journal_rows=journal,
    )
    assert len(derived) == 1024
    assert all(row["strict_success"] == 0 for row in derived)

    tampered = copy.deepcopy(trajectories)
    tampered[0]["completed_turns"] = 0.0
    with pytest.raises(ValueError, match="disagrees"):
        replay.replay_outcomes(
            source_rows=source,
            rollout_seeds=seeds,
            split="calibration",
            trajectories=tampered,
            journal_rows=journal,
        )

    missing = copy.deepcopy(trajectories)
    del missing[0]["invalid_reason"]
    with pytest.raises(ValueError, match="lacks"):
        replay.replay_outcomes(
            source_rows=source,
            rollout_seeds=seeds,
            split="calibration",
            trajectories=missing,
            journal_rows=journal,
        )


@pytest.mark.parametrize("bad_seed", [True, 1.5, "1"])
def test_replay_rejects_non_exact_seed_types(bad_seed):
    replay = _load(f"rist_c0_v2_2_replay_seed_{type(bad_seed).__name__}", "replay_resolution_outcomes.py")
    source, seeds, trajectories, journal = _invalid_replay_fixture()
    seeds[0] = bad_seed
    with pytest.raises(ValueError, match="exact integers"):
        replay.replay_outcomes(
            source_rows=source,
            rollout_seeds=seeds,
            split="calibration",
            trajectories=trajectories,
            journal_rows=journal,
        )


def test_replay_rejects_duplicate_source_id_even_with_32_unique_signatures():
    replay = _load("rist_c0_v2_2_replay_duplicate", "replay_resolution_outcomes.py")
    source, seeds, trajectories, journal = _invalid_replay_fixture()
    source[1]["id"] = source[0]["id"]
    with pytest.raises(ValueError, match="exactly 32 source"):
        replay.replay_outcomes(
            source_rows=source,
            rollout_seeds=seeds,
            split="calibration",
            trajectories=trajectories,
            journal_rows=journal,
        )


def _response(action: dict) -> dict:
    return {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call",
                            "function": {
                                "name": action["name"],
                                "arguments": json.dumps(action["arguments"]),
                            },
                        }
                    ]
                }
            }
        ]
    }


def _write_job(root: Path, source: list[dict], seeds: list[int], family: str, split: str) -> dict:
    root.mkdir()
    trajectories = []
    journal = []
    for task in source:
        low = task["structural_cell"] in {"c00", "c01", "c02", "c03"}
        for seed_index, seed in enumerate(seeds):
            succeeds = not low and seed_index % 2 == 1
            actions = task["oracle_actions"] if succeeds else []
            responses = [_response(action) for action in actions]
            if not succeeds:
                responses = [{"choices": [{"message": {"tool_calls": []}}]}]
            trajectories.append(
                {
                    "task_id": task["id"],
                    "task_signature": task["task_signature"],
                    "structural_cell": task["structural_cell"],
                    "rollout_seed": seed,
                    "actions": actions,
                    "strict_success": int(succeeds),
                    "completed_turns": len(actions),
                    "invalid_reason": None if succeeds else "CALL_COUNT",
                    "raw_response_count": len(responses),
                    "split": split,
                }
            )
            for turn_index, response in enumerate(responses):
                journal.append(
                    {
                        "task_id": task["id"],
                        "task_signature": task["task_signature"],
                        "rollout_seed": seed,
                        "turn_index": turn_index,
                        "raw_response": response,
                    }
                )
    trajectory_path = root / "trajectories.jsonl"
    journal_path = root / "raw.jsonl"
    result_path = root / "result.json"
    trajectory_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in trajectories))
    journal_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in journal))
    result_path.write_text("{}\n")
    return {
        "job_id": f"{family}-{split}",
        "family": family,
        "split": split,
        "checkpoint": family,
        "result": str(result_path),
        "trajectories": str(trajectory_path),
        "raw_journal": str(journal_path),
    }


def test_full_resolution_run_is_one_shot_and_independently_reproducible(tmp_path, monkeypatch):
    runner = _load("rist_c0_v2_2_run", "run_resolution_analysis.py")
    validator = _load("rist_c0_v2_2_validate", "validate_resolution_analysis.py")
    gate = _load("rist_e1_c0_gate", "../E1/validate_c0_admission.py")
    pool_path = STAGE / "data/manifest.json"
    pool = json.loads(pool_path.read_text())
    jobs = []
    for family in ("qwen3", "gemma4"):
        for split in ("calibration", "qualification"):
            source = [
                json.loads(line)
                for line in (STAGE / "data" / pool["splits"][split]["file"]).read_text().splitlines()
                if line
            ]
            jobs.append(
                _write_job(
                    tmp_path / f"{family}-{split}",
                    source,
                    pool["splits"][split]["rollout_seeds"],
                    family,
                    split,
                )
            )
    manifest = {
        "protocol": "RIST-C0-v2.2-SCIENTIFIC-COLLECTION-MANIFEST-v1",
        "job_count": 4,
        "trajectory_count": 4096,
        "pool_manifest": str(pool_path),
        "pool_manifest_sha256": runner._sha256(pool_path),
        "models": {"qwen3": "Qwen/Qwen3-0.6B", "gemma4": "google/gemma-4-E2B-it"},
        "model_revisions": {"qwen3": "qrev", "gemma4": "grev"},
        "tokenizer_snapshot_sha256": {"qwen3": "qhash", "gemma4": "ghash"},
        "jobs": jobs,
    }
    manifest_path = tmp_path / "collection.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    final = {
        "passed": True,
        "decision": "PASS_COLLECTION_TO_SEPARATE_RESOLUTION_ANALYSIS",
        "outcomes_inspected": False,
        "trajectory_count": 4096,
        "retry_count": 0,
        "gpu_uuid": "GPU-test",
    }
    final_path = tmp_path / "collection_final.json"
    final_path.write_text(json.dumps(final, sort_keys=True) + "\n")

    class Finalizer:
        @staticmethod
        def finalize(_):
            return final

    monkeypatch.setattr(runner, "_load_finalizer", lambda: Finalizer)
    monkeypatch.setattr(validator, "_load_finalizer", lambda: Finalizer)
    output = tmp_path / "resolution"
    result = runner.run_analysis(
        collection_manifest_path=manifest_path,
        collection_final_path=final_path,
        d3_train_path=STAGE.parent / "D3/data/train.jsonl",
        d3_manifest_path=STAGE.parent / "D3/data/manifest.json",
        authorization_path=STAGE.parents[1] / "GPU_AUTHORIZATION_20260803.json",
        output_dir=output,
    )
    monkeypatch.setattr(gate, "_load_validator", lambda: validator)
    with pytest.raises(FileNotFoundError):
        gate.validate_c0_admission(output)
    validation = validator.validate(output)
    e1_gate = gate.validate_c0_admission(output)

    assert result["decision"] == "PASS_C0_V2_2_TO_FILTERED_TRAIN"
    assert validation["valid"] is True
    assert validation["trajectory_count"] == 4096
    assert e1_gate["decision"] == "PASS_C0_GATE_TO_DEPLOYMENT_BOUND_E1"
    assert e1_gate["training_execution_authorized"] is False
    assert len((output / "e1/data/train.jsonl").read_text().splitlines()) == 4
    with pytest.raises(FileExistsError, match="rerun is forbidden"):
        runner.run_analysis(
            collection_manifest_path=manifest_path,
            collection_final_path=final_path,
            d3_train_path=STAGE.parent / "D3/data/train.jsonl",
            d3_manifest_path=STAGE.parent / "D3/data/manifest.json",
            authorization_path=STAGE.parents[1] / "GPU_AUTHORIZATION_20260803.json",
            output_dir=output,
        )


def test_validator_rejects_coordinated_admission_identity_tamper(tmp_path, monkeypatch):
    runner = _load("rist_c0_v2_2_run_tamper", "run_resolution_analysis.py")
    validator = _load("rist_c0_v2_2_validate_tamper", "validate_resolution_analysis.py")
    pool_path = STAGE / "data/manifest.json"
    pool = json.loads(pool_path.read_text())
    jobs = []
    for family in ("qwen3", "gemma4"):
        for split in ("calibration", "qualification"):
            source = [
                json.loads(line)
                for line in (STAGE / "data" / pool["splits"][split]["file"]).read_text().splitlines()
                if line
            ]
            jobs.append(_write_job(tmp_path / f"{family}-{split}", source, pool["splits"][split]["rollout_seeds"], family, split))
    manifest = {
        "protocol": "RIST-C0-v2.2-SCIENTIFIC-COLLECTION-MANIFEST-v1",
        "job_count": 4,
        "trajectory_count": 4096,
        "pool_manifest": str(pool_path),
        "pool_manifest_sha256": runner._sha256(pool_path),
        "models": {"qwen3": "Qwen/Qwen3-0.6B", "gemma4": "google/gemma-4-E2B-it"},
        "model_revisions": {"qwen3": "qrev", "gemma4": "grev"},
        "tokenizer_snapshot_sha256": {"qwen3": "qhash", "gemma4": "ghash"},
        "jobs": jobs,
    }
    manifest_path = tmp_path / "collection.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    final = {"passed": True, "decision": "PASS_COLLECTION_TO_SEPARATE_RESOLUTION_ANALYSIS", "outcomes_inspected": False, "trajectory_count": 4096, "retry_count": 0, "gpu_uuid": "GPU-test"}
    final_path = tmp_path / "collection_final.json"
    final_path.write_text(json.dumps(final, sort_keys=True) + "\n")

    class Finalizer:
        @staticmethod
        def finalize(_):
            return final

    monkeypatch.setattr(runner, "_load_finalizer", lambda: Finalizer)
    monkeypatch.setattr(validator, "_load_finalizer", lambda: Finalizer)
    output = tmp_path / "resolution"
    runner.run_analysis(
        collection_manifest_path=manifest_path,
        collection_final_path=final_path,
        d3_train_path=STAGE.parent / "D3/data/train.jsonl",
        d3_manifest_path=STAGE.parent / "D3/data/manifest.json",
        authorization_path=STAGE.parents[1] / "GPU_AUTHORIZATION_20260803.json",
        output_dir=output,
    )
    admission_path = output / "e1/E1_ADMISSION.json"
    admission = json.loads(admission_path.read_text())
    admission["gpu_uuid"] = "GPU-attacker"
    admission["models"] = {"qwen3": "wrong", "gemma4": "wrong"}
    admission_path.write_text(json.dumps(admission, indent=2, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="exactly reproduce"):
        validator.validate(output, write_receipt=False)

"""CPU-only tests for the X4 Tau3 powered-stability design."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
STAGE = (
    REPO
    / "research/reward_identifiability_supervision_topology/successors/rist_v2_1/"
    "stages/X4_TAU3"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _records(seed_bank, effects):
    rows = []
    for seed, effect in zip(seed_bank["seeds"][: len(effects)], effects, strict=True):
        for family in ("qwen3", "gemma4"):
            for algorithm in ("gspo", "grpo"):
                for arm in ("AF", "LF", "AN", "LN"):
                    value = 0.50 + effect if arm == "AF" else 0.50
                    rows.append(
                        {
                            "run_id": f"{seed}-{family}-{algorithm}-{arm}",
                            "seed": seed,
                            "family": family,
                            "algorithm": algorithm,
                            "arm": arm,
                            "catastrophic": False,
                            "strict_success": value,
                            "token_auc": value,
                            "token_endpoint": value,
                            "token_support_fraction": 0.80,
                        }
                    )
    return rows


def _validation(analyzer, records):
    return {"passed": True, "records_sha256": analyzer._canonical_sha256(records)}


def test_x4_seed_bank_is_outcome_blind_ordered_32_with_12_floor():
    freezer = _load("x4_seed_freezer", STAGE / "freeze_seed_bank.py")
    first = freezer.build_seed_bank()
    second = freezer.build_seed_bank()
    assert first == second
    assert first["seed_count"] == 32
    assert len(first["planning_seeds"]) == 12
    assert first["planning_seeds"] == first["seeds"][:12]
    assert first["expansion_seeds"] == first["seeds"][12:]
    assert len(set(first["seeds"])) == 32
    assert set(first["seeds"]).isdisjoint(first["pilot_seeds_excluded"])
    assert first["outcomes_read"] is False
    assert first["power_plan"]["maximum_seed_count"] == 32


def test_x4_manifest_freezes_complete_initial_and_ceiling_matrices(tmp_path):
    builder = _load("x4_manifest_builder", STAGE / "build_execution_manifest.py")
    manifest = builder.build_manifest(tmp_path / "runs")
    assert manifest["job_count"] == 512
    assert manifest["planning_job_count"] == 192
    assert manifest["maximum_episode_count"] == 102_400
    assert manifest["execution_authorized"] is False
    assert manifest["heldout_permitted"] is False
    assert manifest["bfcl_permitted"] is False
    assert len(manifest["required_training_task_ids"]) == 22
    assert manifest["x3_pilot_prerequisite"]["validation_sha256"] is None
    planning = [job for job in manifest["jobs"] if job["seed_tranche"] == "planning"]
    assert len(planning) == 192
    assert len(
        {
            (job["seed"], job["family"], job["algorithm"], job["arm"])
            for job in manifest["jobs"]
        }
    ) == 512
    assert all(job["expected_episode_count"] == 200 for job in manifest["jobs"])


def test_x4_analysis_expands_kills_or_passes_only_from_actual_seed_sd():
    freezer = _load("x4_seed_for_analysis", STAGE / "freeze_seed_bank.py")
    analyzer = _load("x4_stability_analysis", STAGE / "analyze_stability.py")
    bank = freezer.build_seed_bank()

    small = [0.080 + (index % 3 - 1) * 0.002 for index in range(12)]
    records = _records(bank, small)
    passed = analyzer.analyze(records, bank, _validation(analyzer, records))
    assert passed["decision"] == "PASS_X4_POWERED_STABLE_INTERACTION"
    assert passed["powered"] is True
    assert passed["actual_n"] >= passed["required_n"] == 12

    moderate = [0.05 + (0.08 if index % 2 else -0.08) for index in range(12)]
    records = _records(bank, moderate)
    expansion = analyzer.analyze(records, bank, _validation(analyzer, records))
    assert expansion["decision"] == "EXPAND_X4_IN_FROZEN_SEED_ORDER"
    assert 12 < expansion["required_n"] <= 32
    assert expansion["next_seeds"] == bank["seeds"][12 : expansion["required_n"]]

    high = [0.45 if index % 2 else -0.45 for index in range(12)]
    records = _records(bank, high)
    killed = analyzer.analyze(records, bank, _validation(analyzer, records))
    assert killed["decision"] == "KILL_X4_VARIANCE_EXCEEDS_FROZEN_32_SEED_BANK"
    assert killed["required_n"] > 32


def test_x4_evidence_validator_requires_x3_and_resolved_authority(tmp_path):
    builder = _load("x4_manifest_for_validation", STAGE / "build_execution_manifest.py")
    validator = _load("x4_evidence_validator", STAGE / "validate_evidence.py")
    manifest = builder.build_manifest(tmp_path / "runs")
    x3_path = tmp_path / "x3.json"
    x3_path.write_text(json.dumps({"status": "PASS", "job_count": 16, "episode_count": 3200}))
    manifest["x3_pilot_prerequisite"]["validation_sha256"] = hashlib.sha256(
        x3_path.read_bytes()
    ).hexdigest()
    with pytest.raises(ValueError, match="separately authorized"):
        validator.validate(manifest, tmp_path / "evidence", x3_path, 12)


def test_x4_evidence_validator_accepts_exact_resolved_planning_prefix(
    tmp_path, monkeypatch
):
    builder = _load("x4_manifest_resolved", STAGE / "build_execution_manifest.py")
    validator = _load("x4_evidence_resolved", STAGE / "validate_evidence.py")
    manifest = builder.build_manifest(tmp_path / "runs")
    x3_path = tmp_path / "x3.json"
    x3_path.write_text(json.dumps({"status": "PASS", "job_count": 16, "episode_count": 3200}))
    manifest["x3_pilot_prerequisite"]["validation_sha256"] = hashlib.sha256(
        x3_path.read_bytes()
    ).hexdigest()
    manifest["execution_authorized"] = True
    manifest["user_simulator_binding"] = {
        "model": "sim",
        "revision": "sim-rev",
        "runtime_value": "sim@rev",
        "authorization_sha256": "a" * 64,
    }
    manifest["family_bindings"] = {
        family: {
            "revision": f"{family}-rev",
            "tokenizer_revision": f"{family}-tok",
            "weights_manifest_sha256": digest * 64,
        }
        for family, digest in (("qwen3", "b"), ("gemma4", "c"))
    }

    class X3Validator:
        @staticmethod
        def _validate_journals(*args, **kwargs):
            del args, kwargs
            return set(manifest["required_training_task_ids"])

    monkeypatch.setattr(validator, "_load_x3_validator", lambda: X3Validator)
    root = tmp_path / "evidence"
    planning = set(manifest["planning_seeds"])
    for job in manifest["jobs"]:
        if job["seed"] not in planning:
            continue
        run_root = root / job["run_id"]
        run_root.mkdir(parents=True)
        raw = run_root / "raw_events.jsonl"
        reward = run_root / "reward_events.jsonl"
        raw.write_text("")
        reward.write_text("")
        evidence = {
            "protocol": manifest["protocol"],
            "run_id": job["run_id"],
            "source_commit": manifest["source_commit"],
            "family": job["family"],
            "algorithm": job["algorithm"],
            "arm": job["arm"],
            "seed": job["seed"],
            "completed": True,
            "policy_retry_count": 0,
            "user_retry_count": 0,
            "dataset_sha256": job["dataset_sha256"],
            "source_file_sha256": manifest["source_file_sha256"],
            "model": manifest["family_bindings"][job["family"]],
            "user_simulator": manifest["user_simulator_binding"],
            "raw_events_sha256": hashlib.sha256(b"").hexdigest(),
            "reward_events_sha256": hashlib.sha256(b"").hexdigest(),
            "metrics_manifest_sha256": "d" * 64,
            "checkpoint_manifest_sha256": "e" * 64,
            "source_archive_sha256": "f" * 64,
            "catastrophic": False,
            "strict_success": 0.6,
            "token_auc": 0.6,
            "token_endpoint": 0.6,
            "token_support_fraction": 0.8,
        }
        (run_root / "run_evidence.json").write_text(json.dumps(evidence))
    result = validator.validate(manifest, root, x3_path, 12)
    assert result["passed"] is True
    assert result["job_count"] == 192
    assert result["episode_count"] == 38_400
    assert len(result["records"]) == 192


def test_x4_analysis_rejects_nonprefix_or_incomplete_seed_blocks():
    freezer = _load("x4_seed_rejection", STAGE / "freeze_seed_bank.py")
    analyzer = _load("x4_analysis_rejection", STAGE / "analyze_stability.py")
    bank = freezer.build_seed_bank()
    records = _records(bank, [0.08] * 12)
    records.pop()
    with pytest.raises(ValueError, match="exact paired-seed factorial"):
        analyzer.analyze(records, bank, _validation(analyzer, records))

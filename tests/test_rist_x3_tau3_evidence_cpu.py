"""CPU-only regression checks for the frozen RIST X3 Tau3 evidence boundary."""

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
    "stages/X3_TAU3"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_x3_manifest_binds_source_and_keeps_simulator_unresolved(tmp_path):
    builder = _load("x3_manifest_new", STAGE / "build_execution_manifest.py")
    manifest = builder.build_manifest(tmp_path / "runs")
    assert manifest["job_count"] == 16
    assert manifest["expected_episode_count"] == 3200
    assert len(manifest["required_training_task_ids"]) == 22
    assert all(len(value) == 64 for value in manifest["source_file_sha256"].values())
    assert manifest["user_simulator_revision"] is None
    assert manifest["user_simulator_authorized"] is False
    for job in manifest["jobs"]:
        environment = job["required_environment"]
        assert environment["RIST_X3_RUN_ID"] == job["run_id"]
        assert set(environment) == {
            "RIST_X3_RUN_ID",
            "RIST_X3_SOURCE_COMMIT",
            "RIST_TAU3_USER_LLM",
            "RIST_TAU3_USER_LLM_REVISION",
            "RIST_TAU3_USER_LLM_AUTHORIZATION_SHA256",
            "RIST_RAW_JOURNAL_PATH",
            "RIST_REWARD_JOURNAL_PATH",
        }


def test_x3_runtime_requires_journals_and_immutable_identity(monkeypatch):
    runner = _load(
        "x3_runner_new", REPO / "examples/agentic/rist_v2_1_tau3/run_agent.py"
    )
    monkeypatch.delenv("RIST_RAW_JOURNAL_PATH", raising=False)
    with pytest.raises(RuntimeError, match="RIST_RAW_JOURNAL_PATH"):
        runner._required_environment("RIST_RAW_JOURNAL_PATH")
    monkeypatch.setenv("RIST_TAU3_USER_LLM_AUTHORIZATION_SHA256", "not-a-digest")
    with pytest.raises(RuntimeError, match="64 lowercase hex"):
        runner._required_digest_environment(
            "RIST_TAU3_USER_LLM_AUTHORIZATION_SHA256", 64
        )
    episode = runner._episode_id("run", 3, "17", 2)
    assert episode == hashlib.sha256(b"run\x003\x0017\x002").hexdigest()
    assert episode != runner._episode_id("run", 4, "17", 2)


def test_x3_journal_validator_reconstructs_bound_episode_hash():
    validator = _load("x3_validator_new", STAGE / "validate_pilot_evidence.py")
    job = {
        "run_id": "run-1",
        "max_steps": 1,
        "group_size": 1,
        "expected_episode_count": 1,
    }
    source_commit = "a" * 40
    identity = {
        "run_id": "run-1",
        "task_id": "airline:17",
        "training_step": 0,
        "prompt_index": 0,
        "sample_index": 0,
    }
    identity["episode_id"] = validator._expected_episode_id(job, identity)
    events = [
        {"event_index": 0, "phase": "policy_response", "raw_response": {"id": "x"}},
        {
            "event_index": 1,
            "phase": "environment_step",
            "reward": 1.0,
            "terminated": True,
            "simulation_run": {"reward": 1},
        },
    ]
    raw_rows = [{**identity, **event} for event in events]
    digest = hashlib.sha256(
        json.dumps(events, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    reward_rows = [
        {
            **identity,
            "source_commit": source_commit,
            "domain": "airline",
            "reward": 1.0,
            "evaluator": "tau2.EvaluationType.ALL",
            "user_simulator": "sim@rev",
            "user_simulator_revision": "rev",
            "user_simulator_authorization_sha256": "b" * 64,
            "user_seed": 1,
            "policy_retry_count": 0,
            "user_retry_count": 0,
            "runtime_evidence_sha256": digest,
        }
    ]
    assert validator._validate_journals(
        raw_rows,
        reward_rows,
        job,
        {"airline:17"},
        "sim@rev",
        "rev",
        "b" * 64,
        source_commit,
    ) == {"airline:17"}
    reward_rows[0]["episode_id"] = "0" * 64
    with pytest.raises(ValueError, match="episode ID"):
        validator._validate_journals(
            raw_rows,
            reward_rows,
            job,
            {"airline:17"},
            "sim@rev",
            "rev",
            "b" * 64,
            source_commit,
        )

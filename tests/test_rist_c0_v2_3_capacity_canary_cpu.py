from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / (
    "research/reward_identifiability_supervision_topology/successors/rist_v2_1/"
    "stages/C0_RESOLUTION_V2_3"
)


def _load():
    path = STAGE / "run_capacity_canary.py"
    spec = importlib.util.spec_from_file_location("rist_c0_v23_capacity", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest() -> dict:
    return json.loads((STAGE / "data/manifest.json").read_text(encoding="utf-8"))


def _identity(family: str = "qwen3") -> dict:
    return {
        "family": family,
        "gpu_uuid": "GPU-test",
        "gpu_total_memory_gib": 79.25,
        "model_revision": "frozen-revision",
    }


def test_v2_3_capacity_canary_sends_exact_fresh_seeds_without_outcome_use(tmp_path):
    module = _load()
    requests = []
    journal = tmp_path / "journal.jsonl"

    result = module.run_canary(
        _manifest(),
        STAGE / "data",
        "qwen3",
        _identity(),
        journal,
        lambda payload: requests.append(payload) or {"choices": [{"message": {}}]},
        lambda: 321.0,
    )

    expected = list(range(18001, 18009))
    assert result["protocol"] == "RIST-C0-v2.3-CAPACITY-CANARY-v1"
    assert result["complete"] is True
    assert result["request_seeds"] == expected
    assert result["response_count"] == 8
    assert result["retry_count"] == 0
    assert result["outcomes_inspected"] is False
    assert result["scientific_result"] is False
    assert sorted(request["seed"] for request in requests) == expected
    assert all(request["tool_choice"] == "required" for request in requests)
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [row["request_seed"] for row in rows] == expected


def test_v2_3_capacity_canary_rejects_retired_protocol_before_requests(tmp_path):
    module = _load()
    manifest = _manifest()
    manifest["protocol"] = "RIST-C0-v2.2-FRESH-POOL"
    calls = []

    with pytest.raises(ValueError, match="unexpected C0 v2.3"):
        module.run_canary(
            manifest,
            STAGE / "data",
            "qwen3",
            _identity(),
            tmp_path / "journal.jsonl",
            calls.append,
            lambda: 0.0,
        )

    assert calls == []


def test_v2_3_capacity_canary_rejects_existing_journal_before_requests(tmp_path):
    module = _load()
    journal = tmp_path / "journal.jsonl"
    journal.write_text("prior\n", encoding="utf-8")
    calls = []

    with pytest.raises(FileExistsError, match="must be fresh"):
        module.run_canary(
            _manifest(),
            STAGE / "data",
            "gemma4",
            _identity("gemma4"),
            journal,
            calls.append,
            lambda: 0.0,
        )

    assert calls == []


def test_v2_3_capacity_canary_rejects_hash_or_gpu_identity_before_requests(tmp_path):
    module = _load()
    manifest = _manifest()
    manifest["splits"]["capacity_canary"]["sha256"] = "0" * 64
    calls = []
    with pytest.raises(ValueError, match="source hash mismatch"):
        module.run_canary(
            manifest,
            STAGE / "data",
            "qwen3",
            _identity(),
            tmp_path / "hash-journal.jsonl",
            calls.append,
            lambda: 0.0,
        )
    assert calls == []

    identity = _identity()
    identity["gpu_total_memory_gib"] = 24.0
    with pytest.raises(ValueError, match="at least 48 GB"):
        module.run_canary(
            _manifest(),
            STAGE / "data",
            "qwen3",
            identity,
            tmp_path / "gpu-journal.jsonl",
            calls.append,
            lambda: 0.0,
        )
    assert calls == []

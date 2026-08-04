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
    receipt = json.loads(_receipt_bytes(family))
    return {
        "family": family,
        **receipt["bindings"],
        "gpu_total_memory_mib": 81920,
    }


def _manifest_bytes() -> bytes:
    return (STAGE / "data/manifest.json").read_bytes()


def _binding_bytes() -> bytes:
    return (STAGE / "gpu_bind_20260804_a800/POST_RENT_BINDING.json").read_bytes()


def _receipt_bytes(family: str) -> bytes:
    return (STAGE / f"gpu_bind_20260804_a800/{family}_receipt.json").read_bytes()


def _run(module, tmp_path, family="qwen3", **overrides):
    values = {
        "pool_manifest_bytes": _manifest_bytes(),
        "data_dir": STAGE / "data",
        "family": family,
        "runtime_identity": _identity(family),
        "binding_bytes": _binding_bytes(),
        "receipt_bytes": _receipt_bytes(family),
        "journal_path": tmp_path / "journal.jsonl",
        "post_json": lambda _payload: {"choices": [{"message": {}}]},
        "live_gpu_uuid": lambda: "GPU-10daada5-2347-5f45-92ae-cb5c3ddb646c",
        "live_gpu_total_memory_mib": lambda: 81920,
        "memory_used_mib": lambda: 321.0,
    }
    values.update(overrides)
    return module.run_canary(**values)


def test_v2_3_capacity_canary_sends_exact_fresh_seeds_without_outcome_use(tmp_path):
    module = _load()
    requests = []
    journal = tmp_path / "journal.jsonl"

    result = _run(
        module,
        tmp_path,
        journal_path=journal,
        post_json=lambda payload: requests.append(payload) or {"choices": [{"message": {}}]},
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


def test_v2_3_capacity_canary_rejects_retired_protocol_before_requests(tmp_path, monkeypatch):
    module = _load()
    manifest = _manifest()
    manifest["protocol"] = "RIST-C0-v2.2-FRESH-POOL"
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    binding = json.loads(_binding_bytes())
    receipt = json.loads(_receipt_bytes("qwen3"))
    monkeypatch.setattr(
        module,
        "_validate_receipt",
        lambda *_args: (manifest, binding, receipt),
    )
    calls = []

    with pytest.raises(ValueError, match="unexpected C0 v2.3"):
        _run(
            module,
            tmp_path,
            pool_manifest_bytes=manifest_bytes,
            post_json=calls.append,
        )

    assert calls == []


def test_v2_3_capacity_canary_rejects_existing_journal_before_requests(tmp_path):
    module = _load()
    journal = tmp_path / "journal.jsonl"
    journal.write_text("prior\n", encoding="utf-8")
    calls = []

    with pytest.raises(FileExistsError, match="must be fresh"):
        _run(
            module,
            tmp_path,
            family="gemma4",
            journal_path=journal,
            post_json=calls.append,
        )

    assert calls == []


def test_v2_3_capacity_canary_rejects_hash_or_gpu_identity_before_requests(tmp_path):
    module = _load()
    calls = []
    with pytest.raises(ValueError, match="pool manifest SHA mismatch"):
        _run(
            module,
            tmp_path,
            pool_manifest_bytes=_manifest_bytes() + b" ",
            post_json=calls.append,
        )
    assert calls == []

    identity = _identity()
    identity["gpu_uuid"] = "GPU-forged"
    with pytest.raises(ValueError, match="runtime identity mismatch: gpu_uuid"):
        _run(
            module,
            tmp_path,
            runtime_identity=identity,
            journal_path=tmp_path / "gpu-journal.jsonl",
            post_json=calls.append,
        )
    assert calls == []


def test_v2_3_capacity_canary_requires_exact_seed_bank_even_if_hash_constant_is_rebound(
    tmp_path, monkeypatch
):
    module = _load()
    manifest = _manifest()
    manifest["splits"]["capacity_canary"]["rollout_seeds"] = list(range(28001, 28009))
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    binding = json.loads(_binding_bytes())
    receipt = json.loads(_receipt_bytes("qwen3"))
    monkeypatch.setattr(
        module,
        "_validate_receipt",
        lambda *_args: (manifest, binding, receipt),
    )
    with pytest.raises(ValueError, match="exact eight frozen request seeds"):
        _run(module, tmp_path, pool_manifest_bytes=manifest_bytes)


def test_v2_3_capacity_canary_atomically_reserves_journal(tmp_path, monkeypatch):
    module = _load()
    flags = []
    real_open = module.os.open

    def recording_open(path, value, mode=0o777):
        flags.append(value)
        return real_open(path, value, mode)

    monkeypatch.setattr(module.os, "open", recording_open)
    _run(module, tmp_path)
    assert flags[0] & module.os.O_EXCL


def test_v2_3_capacity_canary_rejects_receipt_or_live_gpu_mismatch(tmp_path):
    module = _load()
    tampered = json.loads(_receipt_bytes("qwen3"))
    tampered["bindings"]["gpu_uuid"] = "GPU-forged"
    tampered_bytes = (json.dumps(tampered, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with pytest.raises(ValueError, match="receipt artifact SHA mismatch"):
        _run(module, tmp_path, receipt_bytes=tampered_bytes)

    with pytest.raises(ValueError, match="live GPU UUID mismatch"):
        _run(module, tmp_path, live_gpu_uuid=lambda: "GPU-other")

    with pytest.raises(ValueError, match="live GPU memory mismatch"):
        _run(module, tmp_path, live_gpu_total_memory_mib=lambda: 49152)

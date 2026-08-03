from __future__ import annotations

import importlib.util
import json
import re
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
E1 = REPO_ROOT / "research/reward_identifiability_supervision_topology/successors/rist_v2_1/stages/E1"
RUNTIME_COMMIT = "f00b688dff3e4ad35229d059536c975d4c56594f"


def _load():
    spec = importlib.util.spec_from_file_location("rist_e1_v2_2_deployment", E1 / "build_v2_2_deployment_manifest.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _gate(model_path: Path) -> dict:
    return {
        "models": {"qwen3": "Qwen/Qwen3-0.6B", "gemma4": "google/gemma-4-E2B-it"},
        "model_revisions": {"qwen3": "qrev", "gemma4": "grev"},
        "tokenizer_snapshot_sha256": {"qwen3": "a" * 64, "gemma4": "d" * 64},
        "collection_gpu_uuid": "GPU-test",
        "capacity_train_path": str(model_path / "train.jsonl"),
        "capacity_train_sha256": "a" * 64,
    }


def _identity(family: str, model_path: Path) -> dict:
    checkpoints = {"qwen3": "Qwen/Qwen3-0.6B", "gemma4": "google/gemma-4-E2B-it"}
    revisions = {"qwen3": "qrev", "gemma4": "grev"}
    tokenizers = {"qwen3": "a" * 64, "gemma4": "d" * 64}
    return {
        "protocol": "RIST-E1-RUNTIME-IDENTITY-v2.2",
        "family": family,
        "checkpoint": checkpoints[family],
        "model_revision": revisions[family],
        "tokenizer_snapshot_sha256": tokenizers[family],
        "model_path": str(model_path),
        "gpu_uuid": "GPU-test",
        "gpu_total_memory_gib": 80.0,
        "model_weights_sha256": "b" * 64,
        "snapshot_verification_sha256": "e" * 64,
        "extension_import_sha256": "c" * 64,
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "driver_version": "570.0",
        "cuda_version": "12.8",
        "torch_version": "2.8.0+cu128",
        "source_commit": RUNTIME_COMMIT,
    }


def test_runtime_identity_requires_same_c0_gpu_and_48_gib(tmp_path):
    module = _load()
    model_path = tmp_path / "model"
    model_path.mkdir()
    gate = _gate(tmp_path)
    identity = _identity("gemma4", model_path)
    module._validate_runtime_identity("gemma4", identity, gate, RUNTIME_COMMIT)

    identity["gpu_uuid"] = "GPU-other"
    with pytest.raises(ValueError, match="deployment-bound"):
        module._validate_runtime_identity("gemma4", identity, gate, RUNTIME_COMMIT)
    identity = _identity("gemma4", model_path)
    identity["gpu_total_memory_gib"] = 47.9
    with pytest.raises(ValueError, match="deployment-bound"):
        module._validate_runtime_identity("gemma4", identity, gate, RUNTIME_COMMIT)

    identity = _identity("gemma4", model_path)
    identity["model_weights_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="deployment-bound"):
        module._validate_runtime_identity("gemma4", identity, gate, RUNTIME_COMMIT)


def test_recursive_replacement_removes_all_execution_tokens():
    module = _load()
    value = {"command": ["{MODEL}", "x={GPU}"], "nested": {"path": "{MODEL}/a"}}
    replaced = module._replace(value, {"{MODEL}": "/models/q", "{GPU}": "GPU-1"})
    assert replaced == {
        "command": ["/models/q", "x=GPU-1"],
        "nested": {"path": "/models/q/a"},
    }


def test_e1_authorization_rejects_missing_algorithm(tmp_path):
    module = _load()
    authorization = {
        "schema_version": 1,
        "authorized": True,
        "authorized_scopes": {
            "E1_QWEN_CAPACITY": {"algorithms": ["gspo"], "optimizer_steps_per_algorithm": 1, "training": True},
            "E1_GEMMA_CAPACITY": {"algorithms": ["gspo", "grpo"], "optimizer_steps_per_algorithm": 1, "training": True},
        },
        "not_authorized": ["P3_48_RUN_DIAGNOSTIC_TRAINING", "HELDOUT_OR_BFCL_ACCESS"],
    }
    path = tmp_path / "authorization.json"
    path.write_text(__import__("json").dumps(authorization))
    with pytest.raises(PermissionError, match="exact frozen"):
        module._validate_e1_authorization(path)


def test_builder_identity_contract_covers_downstream_evidence_fields():
    module = _load()
    spec = importlib.util.spec_from_file_location(
        "rist_e1_v2_2_downstream_validator", E1 / "validate_capacity_evidence.py"
    )
    downstream = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(downstream)
    assert set(downstream.IDENTITY_FIELDS) <= module.REQUIRED_RUNTIME_IDENTITY_FIELDS
    assert {
        "snapshot_verification_sha256",
        "extension_import_sha256",
        "model_path",
    } <= module.REQUIRED_RUNTIME_IDENTITY_FIELDS


def test_builder_emits_concrete_manifest_matching_downstream_contract(tmp_path, monkeypatch):
    module = _load()
    model_paths = {family: tmp_path / family for family in ("qwen3", "gemma4")}
    for path in model_paths.values():
        path.mkdir()
    capacity = tmp_path / "train.jsonl"
    capacity.write_text("{}\n")
    (tmp_path / "e1").mkdir()
    for relative in ("VALIDATION_RESULT.json", "FINAL_RESULT.json", "e1/E1_ADMISSION.json"):
        (tmp_path / relative).write_text("{}\n")
    gate = {
        "models": {"qwen3": "Qwen/Qwen3-0.6B", "gemma4": "google/gemma-4-E2B-it"},
        "model_revisions": {
            "qwen3": "c1899de289a04d12100db370d81485cdf75e47ca",
            "gemma4": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        },
        "tokenizer_snapshot_sha256": {
            "qwen3": "c83c7f983e1204841852a4cb47cff31dfd829437c80dccc55dd52d0c8fe532b1",
            "gemma4": "7c813a44e67aa09d81001db777c261d858417b45ce0204bc6a32f0b7b96720f7",
        },
        "collection_gpu_uuid": "GPU-test",
        "capacity_train_path": str(capacity),
        "capacity_train_sha256": module._sha(capacity),
    }
    identities = {}
    identity_paths = {}
    for family in ("qwen3", "gemma4"):
        identity = _identity(family, model_paths[family])
        identity["model_revision"] = gate["model_revisions"][family]
        identity["tokenizer_snapshot_sha256"] = gate["tokenizer_snapshot_sha256"][family]
        identities[family] = identity
        path = tmp_path / f"{family}-identity.json"
        path.write_text(json.dumps(identity, sort_keys=True) + "\n")
        identity_paths[family] = path

    original_load = module._load

    def fake_load(name, path):
        if path == module.C0_GATE:
            return SimpleNamespace(validate_c0_admission=lambda _: gate)
        return original_load(name, path)

    monkeypatch.setattr(module, "_load", fake_load)
    output = tmp_path / "e1-manifest.json"
    manifest = module.build_deployment_manifest(
        resolution_root=tmp_path,
        authorization_path=E1.parents[1] / "GPU_AUTHORIZATION_20260803.json",
        runtime_identity_paths=identity_paths,
        run_root=tmp_path / "runs",
        manifest_output_path=output,
    )

    assert manifest["source_commit"] == RUNTIME_COMMIT
    assert manifest["execution_authorized"] is True
    assert manifest["gpu_training_authorized"] is True
    assert not re.findall(r"\{[A-Z0-9_]+\}|__[A-Z0-9_]+__", json.dumps(manifest))
    for job in manifest["jobs"]:
        command = job["reload_client_command_template"]
        identity_path = command[command.index("--runtime-identity") + 1]
        assert identity_path == str(tmp_path / "runs" / job["run_id"] / "reload_runtime_identity.json")
        assert job["runtime_identity_sha256"] == manifest["runtime_identity_sha256"][job["family"]]

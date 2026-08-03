from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
E1 = REPO_ROOT / "research/reward_identifiability_supervision_topology/successors/rist_v2_1/stages/E1"


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
        "tokenizer_snapshot_sha256": {"qwen3": "qhash", "gemma4": "ghash"},
        "collection_gpu_uuid": "GPU-test",
        "capacity_train_path": str(model_path / "train.jsonl"),
        "capacity_train_sha256": "a" * 64,
    }


def _identity(family: str, model_path: Path) -> dict:
    checkpoints = {"qwen3": "Qwen/Qwen3-0.6B", "gemma4": "google/gemma-4-E2B-it"}
    revisions = {"qwen3": "qrev", "gemma4": "grev"}
    tokenizers = {"qwen3": "qhash", "gemma4": "ghash"}
    return {
        "family": family,
        "checkpoint": checkpoints[family],
        "model_revision": revisions[family],
        "tokenizer_snapshot_sha256": tokenizers[family],
        "model_path": str(model_path),
        "gpu_uuid": "GPU-test",
        "gpu_total_memory_gib": 80.0,
        "weights_manifest_sha256": "b" * 64,
        "extension_import_sha256": "c" * 64,
    }


def test_runtime_identity_requires_same_c0_gpu_and_48_gib(tmp_path):
    module = _load()
    model_path = tmp_path / "model"
    model_path.mkdir()
    gate = _gate(tmp_path)
    identity = _identity("gemma4", model_path)
    module._validate_runtime_identity("gemma4", identity, gate)

    identity["gpu_uuid"] = "GPU-other"
    with pytest.raises(ValueError, match="deployment-bound"):
        module._validate_runtime_identity("gemma4", identity, gate)
    identity = _identity("gemma4", model_path)
    identity["gpu_total_memory_gib"] = 47.9
    with pytest.raises(ValueError, match="deployment-bound"):
        module._validate_runtime_identity("gemma4", identity, gate)


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

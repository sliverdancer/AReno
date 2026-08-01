"""Optional post-freeze AReaL replication adapter and dynamic CPU probe."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
import types
from typing import Any

from research.silent_reward_contracts.arca import audit_case


AREAL_COMMIT = "62f955c5e0388aebc6fd58c5ad3f8fcf9d7384b8"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_areal(root: Path) -> dict[str, Any]:
    files = {
        "interaction_types": root / "areal/experimental/openai/types.py",
        "interaction_cache": root / "areal/experimental/openai/cache.py",
        "cli_args": root / "areal/api/cli_args.py",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing AReaL files: {missing}")
    types_tree = ast.parse(
        files["interaction_types"].read_text(encoding="utf-8")
    )
    string_literals = {
        node.value
        for node in ast.walk(types_tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    types_text = files["interaction_types"].read_text(encoding="utf-8")
    cache_text = files["interaction_cache"].read_text(encoding="utf-8")
    cli_text = files["cli_args"].read_text(encoding="utf-8")
    return {
        "commit": AREAL_COMMIT,
        "source_hashes": {
            name: _sha256(path) for name, path in sorted(files.items())
        },
        "facts": {
            "none_reward_maps_to_zero": (
                "self.reward if self.reward is not None else 0.0" in types_text
            ),
            "tensor_output_has_reward_status": "reward_status" in string_literals,
            "discount_path_warns_missing_reward": (
                "does not have a reward set" in cache_text
            ),
            "singleton_group_native_warning": (
                "singleton group centering erases the task reward" in cli_text
            ),
        },
    }


def execute_areal_probe(root: Path) -> dict[str, Any]:
    import logging as python_logging
    import torch

    source = root / "areal/experimental/openai/types.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    stubs = {
        "openai": types.ModuleType("openai"),
        "openai.types": types.ModuleType("openai.types"),
        "openai.types.chat": types.ModuleType("openai.types.chat"),
        "openai.types.responses": types.ModuleType("openai.types.responses"),
        "openai.types.responses.response": types.ModuleType(
            "openai.types.responses.response"
        ),
        "openai.types.responses.response_input_param": types.ModuleType(
            "openai.types.responses.response_input_param"
        ),
        "areal": types.ModuleType("areal"),
        "areal.api": types.ModuleType("areal.api"),
        "areal.utils": types.ModuleType("areal.utils"),
        "areal.utils.logging": types.ModuleType("areal.utils.logging"),
    }
    stubs["openai.types.chat"].ChatCompletion = object
    stubs["openai.types.responses.response"].Response = object
    stubs["openai.types.responses.response_input_param"].ResponseInputParam = object
    stubs["areal.api"].ModelResponse = object
    stubs["areal.utils.logging"].getLogger = python_logging.getLogger
    stubs["areal.utils"].logging = stubs["areal.utils.logging"]
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("arca_pinned_areal_types", source)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        response = types.SimpleNamespace(
            input_tokens=[1],
            output_tokens=[2],
            output_logprobs=[-0.1],
            output_versions=[1],
            input_len=1,
            output_len=1,
        )
        interaction = module.InteractionWithTokenLogpReward(
            model_response=response,
            reward=None,
            messages=[{"role": "user", "content": "p"}],
        )
        tensor_dict = interaction.to_tensor_dict()
    finally:
        sys.modules.pop("arca_pinned_areal_types", None)
        for name, old_value in previous.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value
    return {
        "input_reward": None,
        "output_reward": float(tensor_dict["rewards"][0].item()),
        "output_keys": sorted(tensor_dict),
        "structured_missing_status": any(
            "status" in key or "missing" in key for key in tensor_dict
        ),
        "torch_cpu": not tensor_dict["rewards"].is_cuda,
    }


def build_replication(root: Path) -> dict[str, Any]:
    inspection = inspect_areal(root)
    probe = execute_areal_probe(root)
    case = {
        "case_id": "ARCA-P3-AREAL-MISSING-REWARD-ZERO",
        "framework": "AReaL",
        "workload": "failure_status",
        "seed": 0,
        "payload": {
            "execution_failures": [
                {
                    "kind": "missing_reward",
                    "numeric_reward": probe["output_reward"],
                    "structured_status": (
                        "present" if probe["structured_missing_status"] else None
                    ),
                }
            ]
        },
        "expected_violations": ["ARCA-F6-STATUS"],
        "natural_case": True,
        "severity": "medium",
        "conclusion_flip": False,
    }
    predicted = audit_case(case)
    return {
        "schema_version": "arca.p3.replication.v1",
        "protocol_id": "ARCA-CPU-AUDIT-v0.1",
        "gpu_executed": False,
        "inspection": inspection,
        "dynamic_probe": probe,
        "case": case,
        "predicted_violations": predicted,
        "exact_match": predicted == case["expected_violations"],
    }


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--areal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_replication(args.areal_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Static production-source checks used by the ARCA cross-system audit."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import types
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _string_literals(path: Path) -> list[str]:
    return [
        node.value
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def _mapping_get_keys(path: Path) -> list[str]:
    keys = []
    for node in ast.walk(_tree(path)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "get" or not node.args:
            continue
        key = node.args[0]
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.append(key.value)
    return keys


def _timeout_assigns_zero(path: Path) -> bool:
    for node in ast.walk(_tree(path)):
        if not isinstance(node, ast.ExceptHandler):
            continue
        exception_name = node.type.id if isinstance(node.type, ast.Name) else None
        if exception_name != "TimeoutError":
            continue
        for child in ast.walk(node):
            if not isinstance(child, (ast.Assign, ast.AnnAssign)):
                continue
            value = child.value
            targets = child.targets if isinstance(child, ast.Assign) else [child.target]
            if (
                isinstance(value, ast.Constant)
                and value.value == 0.0
                and any(isinstance(target, ast.Name) and target.id == "score" for target in targets)
            ):
                return True
    return False


def analyze_verl_source(verl_root: Path) -> dict[str, Any]:
    """Check the pinned veRL producer/consumer contracts without importing veRL."""

    files = {
        "tool_loop": verl_root / "verl/experimental/agent_loop/tool_agent_loop.py",
        "agent_loop": verl_root / "verl/experimental/agent_loop/agent_loop.py",
        "reward_manager": verl_root / "verl/workers/reward_manager/naive.py",
        "function_tool": verl_root / "verl/tools/function_tool.py",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing pinned veRL source files: {missing}")

    producer_literals = _string_literals(files["tool_loop"])
    serializer_literals = _string_literals(files["agent_loop"])
    consumer_get_keys = _mapping_get_keys(files["reward_manager"])
    function_literals = _string_literals(files["function_tool"])
    tool_reward_produced = "tool_rewards" in producer_literals
    tool_reward_serialized = "tool_rewards" in serializer_literals
    legacy_reward_consumed = "reward_scores" in consumer_get_keys
    tool_reward_consumed = "tool_rewards" in consumer_get_keys

    return {
        "source_hashes": {
            name: sha256_file(path) for name, path in sorted(files.items())
        },
        "facts": {
            "function_tool_documents_reward_tuple": any(
                "(response, reward)" in value for value in function_literals
            ),
            "tool_rewards_produced": tool_reward_produced,
            "tool_rewards_serialized": tool_reward_serialized,
            "reward_manager_consumes_reward_scores": legacy_reward_consumed,
            "reward_manager_consumes_tool_rewards": tool_reward_consumed,
            "tool_reward_namespace_mismatch": (
                tool_reward_produced
                and tool_reward_serialized
                and legacy_reward_consumed
                and not tool_reward_consumed
            ),
            "timeout_maps_to_numeric_zero": _timeout_assigns_zero(
                files["reward_manager"]
            ),
            "timeout_status_structured": "reward_timeout" in producer_literals
            or "reward_timeout" in _string_literals(files["reward_manager"]),
        },
    }


def execute_verl_reward_manager_probe(verl_root: Path) -> dict[str, Any]:
    """Execute the pinned manager with lightweight stubs for veRL infrastructure.

    The production class body and ``__call__`` implementation are loaded from
    the pinned source file. Only Ray/DataProto registry dependencies are
    replaced; reward control flow and torch tensor assignment remain upstream
    code.
    """

    import torch

    source = verl_root / "verl/workers/reward_manager/naive.py"
    if not source.is_file():
        raise FileNotFoundError(source)

    class AbstractRewardManager:
        def _extract_reward_from_rm_scores(self, data, return_dict=False):
            return None

    class FakeData:
        def __init__(self):
            self.batch = {"responses": torch.tensor([[7]], dtype=torch.int64)}
            self.item = types.SimpleNamespace(
                batch={
                    "prompts": torch.tensor([3], dtype=torch.int64),
                    "responses": torch.tensor([7], dtype=torch.int64),
                    "attention_mask": torch.tensor([1, 1], dtype=torch.int64),
                },
                non_tensor_batch={
                    "reward_model": {"ground_truth": "ok"},
                    "data_source": "arca-probe",
                    "extra_info": {},
                    "tool_rewards": [1.0],
                    "__num_turns__": 2,
                },
            )

        def __len__(self):
            return 1

        def __getitem__(self, index):
            if index != 0:
                raise IndexError(index)
            return self.item

    class FakeTokenizer:
        def decode(self, token_ids, skip_special_tokens=True):
            return "decoded"

    def register(_name):
        return lambda cls: cls

    stubs = {
        "verl": types.ModuleType("verl"),
        "verl.utils": types.ModuleType("verl.utils"),
        "verl.utils.reward_score": types.ModuleType("verl.utils.reward_score"),
        "verl.workers": types.ModuleType("verl.workers"),
        "verl.workers.reward_manager": types.ModuleType(
            "verl.workers.reward_manager"
        ),
        "verl.workers.reward_manager.abstract": types.ModuleType(
            "verl.workers.reward_manager.abstract"
        ),
    }
    stubs["verl"].DataProto = FakeData
    stubs["verl.utils.reward_score"].default_compute_score = lambda **_: 0.0
    stubs["verl.workers.reward_manager"].register = register
    stubs["verl.workers.reward_manager.abstract"].AbstractRewardManager = (
        AbstractRewardManager
    )

    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("arca_pinned_verl_naive", source)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        captured_extra_info: dict[str, Any] = {}

        def score_probe(**kwargs):
            captured_extra_info.update(kwargs["extra_info"])
            return 0.0

        data = FakeData()
        manager = module.NaiveRewardManager(FakeTokenizer(), 0, score_probe)
        reward = manager(data)
        tool_observed = float(reward[0, 0].item())

        def timeout_probe(**_kwargs):
            raise TimeoutError("arca-probe")

        timeout_data = FakeData()
        timeout_manager = module.NaiveRewardManager(
            FakeTokenizer(), 0, timeout_probe
        )
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            timeout_result = timeout_manager(timeout_data, return_dict=True)
        timeout_score = float(timeout_result["reward_tensor"][0, 0].item())
        timeout_extra = dict(timeout_result["reward_extra_info"])
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value

    return {
        "tool_rewards_visible_to_compute_score": "tool_rewards"
        in captured_extra_info,
        "rollout_reward_scores_seen": captured_extra_info.get(
            "rollout_reward_scores"
        ),
        "tool_reward_observed_score": tool_observed,
        "tool_reward_oracle_score": 1.0,
        "timeout_observed_score": timeout_score,
        "timeout_structured": bool(timeout_extra),
        "timeout_console_marker": "assigning reward 0.0" in stdout.getvalue(),
    }

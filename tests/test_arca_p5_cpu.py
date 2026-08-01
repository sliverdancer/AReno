from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from research.silent_reward_contracts.p5.external_validation import build_artifacts


RLLM_COERCER = """
def _coerce_eval_result(result: object) -> EvalOutput:
    if isinstance(result, EvalOutput):
        return result
    if isinstance(result, bool):
        return EvalOutput(reward=1.0 if result else 0.0, is_correct=result)
    if isinstance(result, int | float):
        reward = float(result)
        return EvalOutput(reward=reward, is_correct=reward >= 1.0)
    if isinstance(result, tuple) and len(result) == 2:
        reward, is_correct = result
        return EvalOutput(reward=float(reward), is_correct=bool(is_correct))
    if isinstance(result, dict):
        reward = float(result.get("reward", 0.0))
        is_correct = result.get("is_correct", reward >= 1.0)
        signals = [Signal(name=k, value=float(v)) for k, v in result.get("signals", {}).items()]
        return EvalOutput(reward=reward, is_correct=is_correct, signals=signals, metadata=result.get("metadata", {}))
    reward = float(result)
    return EvalOutput(reward=reward, is_correct=reward >= 1.0)
"""


def _write_p5_snapshot(tmp_path: Path) -> Path:
    sources = {
        "slime/slime/rollout/forge_load.py": """
def generate_rollout(args, rollout_id, data_source, evaluation: bool = False):
    path = _resolve_path(args, rollout_id, evaluation)
    if evaluation:
        if path is None:
            return RolloutFnEvalOutput(data={})
        blob = torch.load(path, weights_only=False)
        samples = [Sample.from_dict(s) for s in blob["samples"]]
        reward_key = args.eval_reward_key or args.reward_key
        rewards = [s.reward if (not reward_key or s.reward is None) else s.reward[reward_key] for s in samples]
        return RolloutFnEvalOutput(data={"forge_eval": {
            "rewards": [r if r is not None else 0.0 for r in rewards],
            "truncated": [s.status == Sample.Status.TRUNCATED for s in samples],
            "samples": samples,
        }})
""",
        "agent-r1/recipes/webshop/webshop_agent_flow.py": """
step_info = {"error": str(exc)}
extra_fields = {"reward_extra_info": reward_extra_info}
""",
        "ragen/ragen/env/lean/env.py": """
status = "timeout" if "timed out" in str(exc).lower() else "server_error"
result = _make_result(message_objects=[{"severity": "error", "data": str(exc)}])
""",
        "rllm/rllm/eval/module_evaluator.py": RLLM_COERCER,
        "rllm/rllm/eval/evaluator_loader.py": RLLM_COERCER,
        "rllm/docs/core-concepts/tasks.mdx": """
def evaluate(metadata: dict, trajectory: dict) -> dict
Returns are coerced: `float`, `bool`, `dict`, `tuple[float, bool]`, `EvalOutput` are all accepted.
""",
        "agent-lightning/agentlightning/verl/daemon.py": """
message = "Warning: Reward is None for rollout"
sample_stat_list.append({"has_reward": final_reward_raw is not None})
""",
    }
    root = tmp_path / "upstreams"
    for relative, source in sources.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source.strip() + "\n", encoding="utf-8")

    definitions = [
        ("slime", "slime", ["slime/rollout/forge_load.py"]),
        ("Agent-R1", "agent-r1", ["recipes/webshop/webshop_agent_flow.py"]),
        ("RAGEN", "ragen", ["ragen/env/lean/env.py"]),
        (
            "rllm",
            "rllm",
            [
                "rllm/eval/module_evaluator.py",
                "rllm/eval/evaluator_loader.py",
                "docs/core-concepts/tasks.mdx",
            ],
        ),
        ("Agent-Lightning", "agent-lightning", ["agentlightning/verl/daemon.py"]),
    ]
    candidates = []
    for name, local_dir, files in definitions:
        audit_files = []
        for relative in files:
            path = root / local_dir / relative
            audit_files.append(
                {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )
        candidates.append(
            {
                "name": name,
                "local_dir": local_dir,
                "repository": f"https://example.invalid/{local_dir}.git",
                "commit": "fixture",
                "audit_files": audit_files,
            }
        )
    manifest = {
        "protocol_id": "ARCA-P5-EXTERNAL-v0.1",
        "freeze_date": "2026-08-01",
        "candidates": candidates,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def test_p5_external_artifacts_execute_upstream_function_bodies(tmp_path):
    manifest = _write_p5_snapshot(tmp_path)
    output = tmp_path / "output"
    evidence = build_artifacts(
        manifest,
        tmp_path / "upstreams",
        output,
        require_git=False,
    )
    assert evidence["summary"]["decision"] == "PASS_P5_EXTERNAL_NATURAL_TO_PAPER"
    assert evidence["summary"]["new_natural_case_count"] == 2
    assert evidence["summary"]["new_natural_framework_count"] == 2
    assert evidence["summary"]["frozen_rule_changes"] == 0
    assert evidence["summary"]["gpu_executed"] is False
    assert evidence["probes"]["slime"]["missing_reward"] == {
        "rewards": [0.0],
        "truncated": [False],
        "has_reward_field_present": False,
    }
    rllm = evidence["probes"]["rllm"]["duplicated_production_paths"]
    assert len(rllm) == 2
    assert all(row["missing_true_reward"] == 0.0 for row in rllm)
    assert all(row["missing_true_is_correct"] is True for row in rllm)
    assert json.loads((output / "external_evidence.json").read_text(encoding="utf-8")) == evidence
    with (output / "framework_inspections.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 5


def test_p5_external_artifacts_are_byte_deterministic_and_fail_closed(tmp_path):
    manifest = _write_p5_snapshot(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    build_artifacts(manifest, tmp_path / "upstreams", first, require_git=False)
    build_artifacts(manifest, tmp_path / "upstreams", second, require_git=False)
    names = (
        "external_cases.json",
        "external_metrics.json",
        "external_evidence.json",
        "external_cases.csv",
        "framework_inspections.csv",
    )
    assert all((first / name).read_bytes() == (second / name).read_bytes() for name in names)

    source = tmp_path / "upstreams/slime/slime/rollout/forge_load.py"
    source.write_text(source.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash mismatch"):
        build_artifacts(manifest, tmp_path / "upstreams", tmp_path / "drift", require_git=False)

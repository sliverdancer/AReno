from __future__ import annotations

import importlib.util
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "research/structured_action_supervision_v2"


def _load():
    spec = importlib.util.spec_from_file_location("sas_b3_signal", ROOT / "analyze_b3_signal.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_seed(path: Path, seed: int, mixed_rows: int) -> None:
    trajectories = []
    seed_index = [5101, 5202, 5303, 5404, 5505, 5606, 5707, 5808].index(seed)
    for row in range(16):
        reward = 1.0 if row < mixed_rows and seed_index % 2 == 0 else -1.0
        trajectories.append(
            {
                "row_id": f"row-{row}",
                "sampling_seed": seed,
                "strict_reward": reward,
                "first_turn_executable": True,
                "four_turn_complete": True,
                "fabricated_call_count": 0,
                "terminal_reason": "COMPLETE",
                "turns": [{"raw_response": {"choices": []}}],
            }
        )
    path.write_text(
        json.dumps(
            {
                "protocol_id": "SAS-B3-v3.0",
                "sampling_seed": seed,
                "trajectories": trajectories,
            }
        ),
        encoding="utf-8",
    )


def test_b3_signal_gate_passes_only_with_group_relative_support(tmp_path):
    module = _load()
    manifest = json.loads((ROOT / "stages/B3/manifest.json").read_text(encoding="utf-8"))
    paths = []
    for seed in manifest["sampling_seeds"]:
        path = tmp_path / f"signal_seed_{seed}.json"
        _write_seed(path, seed, mixed_rows=4)
        paths.append(path)

    result = module.analyze(paths, manifest)

    assert result["passed"]
    assert result["metrics"]["mixed_strict_success_groups"] == 4
    assert result["metrics"]["nonzero_advantage_trajectories"] == 32


def test_b3_signal_gate_kills_aggregate_only_reward_support(tmp_path):
    module = _load()
    manifest = json.loads((ROOT / "stages/B3/manifest.json").read_text(encoding="utf-8"))
    paths = []
    for seed in manifest["sampling_seeds"]:
        path = tmp_path / f"signal_seed_{seed}.json"
        _write_seed(path, seed, mixed_rows=0)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row, trajectory in enumerate(payload["trajectories"]):
            trajectory["strict_reward"] = 1.0 if row < 4 else -1.0
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths.append(path)

    result = module.analyze(paths, manifest)

    assert not result["passed"]
    assert result["metrics"]["positive_reward_rate"] == 0.25
    assert result["metrics"]["mixed_strict_success_groups"] == 0
    assert result["decision"] == "KILL_CURRENT_GSPO_PILOT_NO_WITHIN_GROUP_SIGNAL"


def test_b3_completion_hook_stays_diagnostic_on_signal_kill():
    spec = importlib.util.spec_from_file_location(
        "sas_b3_hook", ROOT / "b3_stage_completion_hook.py"
    )
    hook = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(hook)
    payload = json.loads((ROOT / "stages/B3/stage_result.json").read_text(encoding="utf-8"))

    result = hook.assess(payload)

    assert result["decision"] == "STAY_DIAGNOSTIC"
    assert result["protocol_action"] == "KILL_CURRENT_PROTOCOL"
    assert "within_group_learning_signal_valid" in result["unmet_criteria"]

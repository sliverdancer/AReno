from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ABLATION_DIR = REPO_ROOT / "examples" / "agentic" / "trainable_turns_ablation"


def _load_module(name: str, filename: str):
    path = ABLATION_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _series(module, *, steps=(0, 1)):
    result = {}
    for arm_idx, arm in enumerate(module.ARMS):
        result[arm] = {
            module.METRIC_TAGS["reward_mean"]: [
                {"step": step, "value": 0.1 * (arm_idx + step)}
                for step in steps
            ],
            module.METRIC_TAGS["reward_std"]: [
                {"step": step, "value": 0.01 * (arm_idx + step)}
                for step in steps
            ],
            module.METRIC_TAGS["trainable_tokens"]: [
                {"step": step, "value": 10 + arm_idx + step}
                for step in steps
            ],
            module.METRIC_TAGS["masked_response_tokens"]: [
                {"step": step, "value": 20 - arm_idx + step}
                for step in steps
            ],
        }
    return result


def test_ablation_artifacts_round_trip_synthetic_scalars(tmp_path):
    collector = _load_module("issue199_collect_metrics_for_tests", "collect_metrics.py")

    rows = collector.build_rows(_series(collector))
    csv_path, json_path = collector.write_artifacts(
        rows,
        tmp_path,
        metadata={"git_commit": "deadbeef"},
    )

    assert len(rows) == 6
    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert list(csv_rows[0]) == list(collector.CSV_FIELDS)
    assert csv_rows[0]["arm"] == "all_assistant"
    assert csv_rows[-1]["arm"] == "final_answer"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["metadata"]["git_commit"] == "deadbeef"
    assert payload["rows"] == rows


def test_ablation_artifacts_reject_misaligned_metric_steps():
    collector = _load_module("issue199_collect_metrics_bad_steps", "collect_metrics.py")
    series = _series(collector)
    series["final_answer"][collector.METRIC_TAGS["reward_std"]].pop()

    with pytest.raises(ValueError, match="metric steps are not aligned"):
        collector.build_rows(series)


def test_ablation_harness_dry_run_has_exactly_three_isolated_arms(tmp_path):
    script = ABLATION_DIR / "run_ablation.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--run-root",
            str(tmp_path),
            "--max-steps",
            "3",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert set(payload["commands"]) == {
        "all_assistant",
        "last_assistant",
        "final_answer",
    }
    for arm, command in payload["commands"].items():
        normalized_command = command.replace("\\", "/")
        assert f"--trainable-turns {arm}" in command
        assert f"arms/{arm}/metrics" in normalized_command
        assert "--max-steps 3" in command
        assert "--model-hub modelscope" in command
        assert "--seed 2026" in command


def test_ablation_tool_results_pair_every_returned_call_id():
    agent = _load_module("issue199_ablation_agent_for_tests", "run_agent.py")
    board = [["X", "X", "."], ["O", ".", "."], ["O", ".", "."]]
    assistant_message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-good",
                "type": "function",
                "function": {
                    "name": "choose_square",
                    "arguments": '{"square":3}',
                },
            },
            {
                "id": "call-bad-json",
                "type": "function",
                "function": {
                    "name": "choose_square",
                    "arguments": "{",
                },
            },
        ],
    }

    results = agent._tool_result_messages(assistant_message, {"board": board})

    assert [result["tool_call_id"] for result in results] == [
        "call-good",
        "call-bad-json",
    ]
    assert json.loads(results[0]["content"])["reward"] == 1.0
    assert json.loads(results[1]["content"])["error"] == "invalid JSON arguments"


def test_ablation_agent_preserves_response_content_and_tool_arguments():
    agent = _load_module("issue199_ablation_agent_response_for_tests", "run_agent.py")
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="raw content",
                    tool_calls=[
                        SimpleNamespace(
                            id="call-1",
                            type="function",
                            function=SimpleNamespace(
                                name="choose_square",
                                arguments='{"square": 9}',
                            ),
                        )
                    ],
                )
            )
        ]
    )

    message = agent._assistant_message(response)

    assert message["content"] == "raw content"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"square": 9}'

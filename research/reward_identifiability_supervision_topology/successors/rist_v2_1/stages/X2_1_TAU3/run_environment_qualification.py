"""Execute the frozen X2.1 Tau3 environment qualification once."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_qualifier():
    path = Path(__file__).parents[1] / "X0" / "qualify_replay.py"
    return _load_module("rist_x2_1_replay_qualifier", path).qualify_replays


qualify_replays = _load_qualifier()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _state_hash(environment: Any) -> str:
    return _canonical_sha256(
        {
            "agent_db_hash": environment.get_db_hash(),
            "user_db_hash": environment.get_user_db_hash(),
        }
    )


@contextmanager
def _deny_network_connections():
    original = socket.socket.connect

    def denied(_socket, address):
        raise RuntimeError(
            f"outbound network connection denied during Tau3 qualification: {address}"
        )

    socket.socket.connect = denied
    try:
        yield
    finally:
        socket.socket.connect = original


def _semantic_tool_response(response: Any) -> dict[str, Any]:
    payload = response.model_dump(mode="json")
    payload.pop("timestamp", None)
    return payload


def _canary_transcript(
    episode_id: str,
    environment_factory: Callable[[], Any],
    tool_call_factory: Callable[[], Any],
) -> list[dict[str, Any]]:
    environment = environment_factory()
    tool_call = tool_call_factory()
    before = _state_hash(environment)
    response = environment.get_response(tool_call)
    if response.error:
        raise RuntimeError(f"canary tool failed for {episode_id}: {response.content}")
    after = _state_hash(environment)
    if before == after:
        raise RuntimeError(f"canary did not mutate state for {episode_id}")
    return [
        {
            "type": "reset",
            "episode_id": episode_id,
            "state_hash": before,
            "observation": {"domain": episode_id.split(":", 1)[0]},
        },
        {
            "type": "step",
            "step_index": 0,
            "action": {
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            },
            "state_hash_before": before,
            "state_hash_after": after,
            "reward": 1.0,
            "reward_source": "strict_state_transition",
            "raw_tool_result": _semantic_tool_response(response),
            "done": True,
        },
    ]


def _parse_pytest_summary(output: str) -> dict[str, int]:
    matches = re.findall(r"(\d+)\s+(passed|failed|error|errors|skipped)", output)
    summary: dict[str, int] = {}
    for count, label in matches:
        normalized = "errors" if label in {"error", "errors"} else label
        summary[normalized] = int(count)
    if "passed" not in summary:
        raise ValueError("pytest output has no passed count")
    return summary


def _run_upstream_tests(source: Path) -> dict[str, Any]:
    wrapper = Path(__file__).with_name("network_blocked_pytest.py")
    test_paths = [
        source / "tests/test_domains/test_airline/test_tools_airline.py",
        source / "tests/test_domains/test_retail/test_tools_retail.py",
    ]
    completed = subprocess.run(
        [sys.executable, str(wrapper), *(str(path) for path in test_paths), "-q"],
        cwd=source,
        capture_output=True,
        text=True,
        check=False,
    )
    combined = completed.stdout + completed.stderr
    return {
        "return_code": completed.returncode,
        "summary": _parse_pytest_summary(combined),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "combined_sha256": hashlib.sha256(combined.encode()).hexdigest(),
    }


def collect(source: Path) -> dict[str, Any]:
    airline_tests = _load_module(
        "rist_x2_1_airline_fixture",
        source / "tests/test_domains/test_airline/test_tools_airline.py",
    )
    retail_tests = _load_module(
        "rist_x2_1_retail_fixture",
        source / "tests/test_domains/test_retail/test_tools_retail.py",
    )
    from tau2.data_model.message import ToolCall
    from tau2.domains.airline.environment import get_environment as airline_environment
    from tau2.domains.retail.environment import get_environment as retail_environment

    components = {
        "airline:fixture_cancel_reservation": (
            lambda: airline_environment(airline_tests.airline_db.__wrapped__()),
            lambda: ToolCall(
                id="x2.1-airline",
                name="cancel_reservation",
                arguments={"reservation_id": "4WQ150"},
            ),
        ),
        "retail:fixture_cancel_pending_order": (
            lambda: retail_environment(retail_tests.retail_db.__wrapped__()),
            lambda: ToolCall(
                id="x2.1-retail",
                name="cancel_pending_order",
                arguments={"order_id": "#W0000000", "reason": "no longer needed"},
            ),
        ),
    }

    upstream_runs = []
    replays = []
    with _deny_network_connections():
        for _ in range(2):
            upstream_runs.append(_run_upstream_tests(source))
            replays.append(
                {
                    episode_id: _canary_transcript(
                        episode_id, environment_factory, tool_call_factory
                    )
                    for episode_id, (
                        environment_factory,
                        tool_call_factory,
                    ) in components.items()
                }
            )

    if any(run["return_code"] != 0 for run in upstream_runs):
        raise RuntimeError("an upstream Tau3 tool-test run failed")
    if upstream_runs[0]["summary"] != upstream_runs[1]["summary"]:
        raise RuntimeError("upstream Tau3 tool-test counts differ across runs")
    partitions = {
        "development": sorted(components),
        "training": [],
        "confirmatory": [],
    }
    qualification = qualify_replays(replays[0], replays[1], partitions)
    return {
        **qualification,
        "protocol": "RIST-X2.1-TAU3-ENVIRONMENT-v1",
        "tau3_tag": "v1.0.1",
        "tau3_commit": "fc0055dc4e0a316c3f83133267fbd6faaa770992",
        "upstream_test_runs": upstream_runs,
        "outbound_connections_blocked": True,
        "tau3_upstream_task_test_split_opened": False,
        "bfcl_content_opened": False,
        "model_or_tokenizer_accessed": False,
        "inference_run": False,
        "training_run": False,
        "gpu_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.source.resolve())
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

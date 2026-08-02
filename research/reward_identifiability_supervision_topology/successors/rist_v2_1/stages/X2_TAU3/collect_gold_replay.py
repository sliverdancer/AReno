"""Collect two network-blocked Tau3 gold-action replays and qualify them."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import socket
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable


def _load_qualifier():
    path = Path(__file__).parents[1] / "X0" / "qualify_replay.py"
    spec = importlib.util.spec_from_file_location("rist_x2_replay_qualifier", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen replay qualifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.qualify_replays


qualify_replays = _load_qualifier()


def _json_value(value: Any) -> Any:
    def fallback(item: Any) -> Any:
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json")
        return str(item)

    return json.loads(json.dumps(value, default=fallback, sort_keys=True))


def _state_hash(environment: Any) -> str:
    payload = {
        "agent_db_hash": environment.get_db_hash(),
        "user_db_hash": environment.get_user_db_hash(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


@contextmanager
def _deny_network_connections():
    original = socket.socket.connect

    def denied(_socket, address):
        raise RuntimeError(f"outbound network connection denied during Tau3 replay: {address}")

    socket.socket.connect = denied
    try:
        yield
    finally:
        socket.socket.connect = original


def replay_task(task: Any, environment_factory: Callable[[], Any], domain: str):
    actions = list(task.evaluation_criteria.actions or [])
    if not actions:
        raise ValueError(f"{domain}:{task.id} has no reference actions")
    environment = environment_factory()
    initial = task.initial_state
    environment.set_state(
        initialization_data=initial.initialization_data if initial else None,
        initialization_actions=initial.initialization_actions if initial else None,
        message_history=list(initial.message_history or []) if initial else [],
        strict=True,
    )
    episode_id = f"{domain}:{task.id}"
    before = _state_hash(environment)
    events = [
        {
            "type": "reset",
            "episode_id": episode_id,
            "state_hash": before,
            "observation": {
                "domain": domain,
                "reference_action_count": len(actions),
            },
        }
    ]
    for index, action in enumerate(actions):
        raw_result = environment.make_tool_call(
            tool_name=action.name,
            requestor=action.requestor,
            **action.arguments,
        )
        after = _state_hash(environment)
        events.append(
            {
                "type": "step",
                "step_index": index,
                "action": {
                    "name": action.name,
                    "arguments": _json_value(action.arguments),
                },
                "state_hash_before": before,
                "state_hash_after": after,
                "reward": 1.0 if index == len(actions) - 1 else 0.0,
                "reward_source": "strict_state_transition",
                "raw_tool_result": _json_value(raw_result),
                "done": index == len(actions) - 1,
            }
        )
        before = after
    return events


def collect(partition_manifest: dict[str, Any]) -> dict[str, Any]:
    from tau2.domains.airline.environment import (
        get_environment as get_airline_environment,
        get_tasks as get_airline_tasks,
    )
    from tau2.domains.retail.environment import (
        get_environment as get_retail_environment,
        get_tasks as get_retail_tasks,
    )

    components = {
        "airline": (get_airline_environment, get_airline_tasks),
        "retail": (get_retail_environment, get_retail_tasks),
    }
    replays = []
    with _deny_network_connections():
        for _ in range(2):
            rows = {}
            for domain, (environment_factory, task_loader) in components.items():
                tasks = {str(task.id): task for task in task_loader("train")}
                for namespaced_id in partition_manifest["partitions"]["development"]:
                    row_domain, task_id = namespaced_id.split(":", 1)
                    if row_domain == domain:
                        rows[namespaced_id] = replay_task(
                            tasks[task_id], environment_factory, domain
                        )
            replays.append(rows)
    qualification = qualify_replays(
        replays[0],
        replays[1],
        partition_manifest["partitions"],
    )
    return {
        **qualification,
        "tau3_tag": partition_manifest["tau3_tag"],
        "tau3_commit": partition_manifest["tau3_commit"],
        "domains": sorted(components),
        "outbound_connections_blocked": True,
        "tau3_upstream_test_retired": True,
        "bfcl_content_opened": False,
        "model_accessed": False,
        "inference_run": False,
        "training_run": False,
        "gpu_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(json.loads(args.partitions.read_text(encoding="utf-8")))
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

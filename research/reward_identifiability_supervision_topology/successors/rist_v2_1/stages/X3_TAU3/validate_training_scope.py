"""Validate the frozen X3 task scope without emitting task content."""

from __future__ import annotations

import argparse
import json
import socket
from collections import Counter
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _deny_network_connections():
    original = socket.socket.connect

    def denied(_socket, address):
        raise RuntimeError(f"network denied during X3 scope validation: {address}")

    socket.socket.connect = denied
    try:
        yield
    finally:
        socket.socket.connect = original


def validate(partitions: dict) -> dict:
    from tau2.domains.airline.environment import get_tasks as airline_tasks
    from tau2.domains.retail.environment import get_tasks as retail_tasks

    loaders = {"airline": airline_tasks, "retail": retail_tasks}
    basis_counts = {}
    selected_counts = {}
    with _deny_network_connections():
        for domain, loader in loaders.items():
            allowed_ids = {
                value.split(":", 1)[1]
                for value in partitions["partitions"]["training"]
                if value.startswith(f"{domain}:")
            }
            tasks = {str(task.id): task for task in loader("train")}
            if not allowed_ids <= set(tasks):
                raise ValueError(f"{domain} training partition references missing tasks")
            counts = Counter(
                "+".join(
                    sorted(
                        reward.value
                        for reward in tasks[task_id].evaluation_criteria.reward_basis
                    )
                )
                for task_id in allowed_ids
            )
            basis_counts[domain] = dict(sorted(counts.items()))
            selected_counts[domain] = len(allowed_ids)
    if basis_counts["airline"] != {"COMMUNICATE+DB": 22}:
        raise ValueError("airline reward basis is not the frozen strict scope")
    if basis_counts["retail"].get("DB+NL_ASSERTION") != 65:
        raise ValueError("retail NL_ASSERTION confound count changed")
    return {
        "protocol": "RIST-X3-TAU3-SCOPE-v1",
        "tau3_tag": partitions["tau3_tag"],
        "tau3_commit": partitions["tau3_commit"],
        "training_partition_counts": selected_counts,
        "reward_basis_counts": basis_counts,
        "opened_domain": "airline",
        "opened_training_task_count": 22,
        "blocked_domain": "retail",
        "blocked_training_task_count": 66,
        "blocked_reason": "NL_ASSERTION invokes an additional LLM judge",
        "retired_source_file_parsed": True,
        "retired_task_selected_or_copied": False,
        "task_content_emitted": False,
        "outbound_connections_blocked": True,
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
    result = validate(json.loads(args.partitions.read_text()))
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

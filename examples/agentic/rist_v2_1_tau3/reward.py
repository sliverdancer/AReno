"""Read the exact upstream Tau3 reward recorded for one rollout sample."""

from __future__ import annotations

import json
import os
from pathlib import Path

RUNTIME_KEY = "_rist_tau3_runtime_by_sample"


def reward_fn(record) -> float:
    sample_index = str(int(record.metadata["sample_index"]))
    runtime = record.source_record.get(RUNTIME_KEY)
    if not isinstance(runtime, dict) or sample_index not in runtime:
        raise ValueError("Tau3 runtime reward metadata is missing for this sample")
    result = runtime[sample_index]
    if result.get("domain") != record.source_record.get("domain") or result.get(
        "task_id"
    ) != record.source_record.get("task_id"):
        raise ValueError("Tau3 runtime reward belongs to a different task")
    reward = result.get("reward")
    if type(reward) not in (int, float) or float(reward) not in {0.0, 1.0}:
        raise ValueError("Tau3 upstream reward must be binary")
    if result.get("evaluator") != "tau2.EvaluationType.ALL":
        raise ValueError("Tau3 reward must come from the frozen upstream evaluator")
    _append_reward_event(record, result)
    return float(reward)


def _append_reward_event(record, result: dict) -> None:
    journal = os.environ.get("RIST_REWARD_JOURNAL_PATH")
    if not journal:
        if result.get("run_id"):
            raise RuntimeError("RIST_REWARD_JOURNAL_PATH is required by the frozen Tau3 run")
        return
    payload = {
        "run_id": result["run_id"],
        "source_commit": result["source_commit"],
        "episode_id": result["episode_id"],
        "task_id": record.source_record["id"],
        "domain": result["domain"],
        "training_step": int(result["training_step"]),
        "prompt_index": int(record.metadata["prompt_index"]),
        "sample_index": int(record.metadata["sample_index"]),
        "reward": float(result["reward"]),
        "terminated": bool(result["terminated"]),
        "truncated": bool(result["truncated"]),
        "evaluator": result["evaluator"],
        "user_simulator": result["user_simulator"],
        "user_simulator_revision": result["user_simulator_revision"],
        "user_simulator_authorization_sha256": result[
            "user_simulator_authorization_sha256"
        ],
        "user_seed": int(result["user_seed"]),
        "policy_retry_count": int(result["policy_retry_count"]),
        "user_retry_count": int(result["user_retry_count"]),
        "runtime_evidence_sha256": result["runtime_evidence_sha256"],
    }
    descriptor = os.open(Path(journal), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, (json.dumps(payload, sort_keys=True) + "\n").encode())
    finally:
        os.close(descriptor)

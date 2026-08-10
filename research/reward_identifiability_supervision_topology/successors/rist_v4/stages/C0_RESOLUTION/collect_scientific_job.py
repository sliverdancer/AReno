"""Collect one RIST C0 v4 scientific job through the v4 request contract.

This module is self-contained within the v4 namespace. It intentionally does
not import v2.3 collector/finalizer modules or the v2.1 D4 evaluator.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

PROTOCOL = "RIST-C0-v4.0-SCIENTIFIC-JOB-EVIDENCE-v1"
MANIFEST_PROTOCOL = "RIST-C0-v4.0-STAGE-MANIFEST-v1"
POOL_PROTOCOL = "RIST-C0-v4.0-FRESH-POOL-v1"
SYSTEM_PROMPT = (
    "Complete the registry route using exactly one tool call per turn. "
    "Choose only from the offered tools and pass exactly one code argument."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        for row in rows:
            os.write(descriptor, (json.dumps(row, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    temporary.replace(path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, (json.dumps(row, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_jsonl_locked(path: Path, row: dict[str, Any], lock: Any | None) -> None:
    if lock is None:
        _append_jsonl(path, row)
        return
    with lock:
        _append_jsonl(path, row)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _job(manifest: dict[str, Any], job_id: str) -> dict[str, Any]:
    matches = [row for row in manifest.get("jobs", []) if row.get("job_id") == job_id]
    if len(matches) != 1:
        raise ValueError("RIST C0 v4 job id must resolve exactly once")
    return matches[0]


def _turn_instruction(turn: dict[str, Any]) -> str:
    candidates = ", ".join(
        f"{candidate['label']}:{candidate['code']}"
        for candidate in turn["candidate_records"]
    )
    return (
        f"Turn {int(turn['turn_index']) + 1}: offered tools="
        + ", ".join(turn["offered_tools"])
        + f"; target_label={turn['target_label']}; candidates={candidates}."
    )


def _render_initial_prompt(task: dict[str, Any]) -> str:
    contract = task["prompt_contract"]
    return "\n".join(
        [str(contract["instruction"])]
        + [_turn_instruction(turn) for turn in contract["initial_visible_turns"]]
    )


def _tool_schema(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Perform the {name} registry operation.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    }


def _visible_turn(turn: dict[str, Any]) -> dict[str, Any]:
    return {
        key: turn[key]
        for key in (
            "turn_index",
            "offered_tools",
            "target_label",
            "candidate_records",
            "selection_rule",
            "depends_on_previous_observation",
        )
    }


def _request_seed(base_seed: int, task_signature: str, turn_index: int) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{task_signature}:{turn_index}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def parse_call(message: dict[str, Any], turn: dict[str, Any]) -> dict[str, Any]:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        return {"valid": False, "reason": "CALL_COUNT", "call": None}
    call = calls[0]
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict):
        return {"valid": False, "reason": "MALFORMED_CALL", "call": None}
    name = function.get("name")
    if name not in turn["offered_tools"]:
        return {"valid": False, "reason": "UNLISTED_TOOL", "call": None}
    if name != turn["expected_tool"]:
        return {"valid": False, "reason": "WRONG_TOOL", "call": None}
    try:
        arguments = json.loads(function.get("arguments"))
    except (TypeError, json.JSONDecodeError):
        return {"valid": False, "reason": "INVALID_JSON", "call": None}
    if (
        not isinstance(arguments, dict)
        or set(arguments) != {"code"}
        or not isinstance(arguments["code"], str)
    ):
        return {"valid": False, "reason": "INVALID_ARGUMENT_SCHEMA", "call": None}
    target = next(
        candidate["code"]
        for candidate in turn["candidate_records"]
        if candidate["label"] == turn["target_label"]
    )
    if arguments["code"] != target:
        return {"valid": False, "reason": "WRONG_CODE", "call": None}
    return {
        "valid": True,
        "reason": "VALID",
        "call": {
            "id": call.get("id"),
            "name": name,
            "arguments": arguments,
        },
    }


def run_trajectory(
    task: dict[str, Any],
    rollout_seed: int,
    sampling: dict[str, Any],
    post_json: Callable[[dict[str, Any]], dict[str, Any]],
    journal_path: Path,
    journal_lock: Any | None = None,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _render_initial_prompt(task)},
    ]
    actions = []
    invalid_reason = None
    for turn_index, turn in enumerate(task["turns"]):
        turn_messages = [
            *messages,
            {"role": "user", "content": _turn_instruction(turn)},
        ]
        payload = {
            "model": "policy",
            "messages": turn_messages,
            "tools": [_tool_schema(name) for name in turn["offered_tools"]],
            "tool_choice": "required",
            "stream": False,
            "temperature": float(sampling["temperature"]),
            "top_p": float(sampling["top_p"]),
            "max_tokens": int(sampling["max_tokens"]),
            "seed": _request_seed(rollout_seed, task["task_signature"], turn_index),
        }
        response = post_json(payload)
        _append_jsonl_locked(
            journal_path,
            {
                "task_id": task["id"],
                "task_signature": task["task_signature"],
                "rollout_seed": rollout_seed,
                "turn_index": turn_index,
                "raw_response": response,
            },
            journal_lock,
        )
        message = response["choices"][0]["message"]
        parsed = parse_call(message, turn)
        if not parsed["valid"]:
            invalid_reason = parsed["reason"]
            break
        call = parsed["call"]
        actions.append({"name": call["name"], "arguments": call["arguments"]})
        next_turn = (
            _visible_turn(task["turns"][turn_index + 1])
            if turn_index + 1 < len(task["turns"])
            else None
        )
        messages = [
            *turn_messages,
            message,
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["name"],
                "content": json.dumps(
                    {"accepted": True, "next_turn": next_turn}, sort_keys=True
                ),
            },
        ]
    strict_success = actions == task["oracle_actions"]
    return {
        "task_id": task["id"],
        "task_signature": task["task_signature"],
        "structural_cell": task["structural_cell"],
        "rollout_seed": rollout_seed,
        "actions": actions,
        "strict_success": int(strict_success),
        "completed_turns": len(actions),
        "invalid_reason": invalid_reason,
        "raw_response_count": len(actions) + (1 if invalid_reason is not None else 0),
    }


def _post_json(base_url: str, api_key: str, timeout_seconds: float, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc


def _validate_gate(manifest: dict[str, Any], job: dict[str, Any], identity: dict[str, Any]) -> None:
    if manifest.get("protocol") != MANIFEST_PROTOCOL:
        raise ValueError("unexpected RIST C0 v4 scientific manifest")
    if manifest.get("split") not in {"capacity_canary", "calibration", "qualification"}:
        raise ValueError("v4 collection requires capacity_canary, calibration, or qualification")
    if manifest.get("retry_permitted") is not False or job.get("max_retries") != 0:
        raise PermissionError("v4 scientific collection forbids retries")
    if any(
        manifest.get(key) is not False
        for key in (
            "heldout_permitted",
            "bfcl_permitted",
            "training_permitted",
            "prior_lineage_outcomes_permitted",
        )
    ):
        raise PermissionError("v4 scientific collection crosses a forbidden boundary")
    if manifest["split"] == "calibration" and not (
        manifest.get("calibration_permitted") is True
        and manifest.get("qualification_permitted") is False
        and manifest.get("capacity_canary_permitted") is False
    ):
        raise PermissionError("invalid v4 calibration boundary")
    if manifest["split"] == "qualification" and not (
        manifest.get("calibration_permitted") is False
        and manifest.get("qualification_permitted") is True
        and manifest.get("capacity_canary_permitted") is False
    ):
        raise PermissionError("invalid v4 qualification boundary")
    if manifest["split"] == "capacity_canary" and not (
        manifest.get("capacity_canary_permitted") is True
        and manifest.get("calibration_permitted") is False
        and manifest.get("qualification_permitted") is False
    ):
        raise PermissionError("invalid v4 capacity canary boundary")

    expected_identity = job.get("runtime_identity")
    if not isinstance(expected_identity, dict):
        raise ValueError("v4 job must bind runtime identity")
    frozen_required = {
        "family",
        "source_commit",
        "model_revision",
        "gpu_uuid",
        "interpreter_realpath",
        "interpreter_sha256",
        "interpreter_version",
    }
    live_required = {*frozen_required, "deployment_receipt_sha256"}
    if not frozen_required.issubset(expected_identity):
        raise ValueError("v4 frozen runtime identity binding is incomplete")
    if "deployment_receipt_sha256" in expected_identity:
        raise ValueError("v4 manifest cannot pre-bind deployment receipt SHA")
    if not live_required.issubset(identity):
        raise ValueError("v4 live runtime identity binding is incomplete")
    if any(identity.get(key) != expected_identity[key] for key in frozen_required):
        raise ValueError("v4 live runtime identity does not match frozen job")
    receipt_sha = identity.get("deployment_receipt_sha256")
    if not isinstance(receipt_sha, str) or len(receipt_sha) != 64:
        raise ValueError("v4 live deployment receipt SHA-256 is required")
    if identity.get("family") != job.get("family"):
        raise ValueError("v4 runtime identity family mismatch")


def collect_job(
    *,
    manifest: dict[str, Any],
    manifest_sha256: str,
    job_id: str,
    runtime_identity: dict[str, Any],
    pool_manifest: dict[str, Any],
    result_path: Path,
    trajectory_path: Path,
    journal_path: Path,
    post_json: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    job = _job(manifest, job_id)
    _validate_gate(manifest, job, runtime_identity)
    if pool_manifest.get("protocol") != POOL_PROTOCOL:
        raise ValueError("unexpected RIST C0 v4 pool manifest")
    if any(path.exists() for path in (result_path, trajectory_path, journal_path)):
        raise FileExistsError("v4 scientific evidence paths must be fresh")

    task_path = Path(job["task_file"])
    if _sha256(task_path) != job["task_file_sha256"]:
        raise ValueError("v4 scientific split hash mismatch")
    tasks = _read_jsonl(task_path)
    seeds = [int(seed) for seed in job["rollout_seeds"]]
    expected_count = len(tasks) * len(seeds)
    if expected_count != int(job["trajectory_count"]):
        raise ValueError("v4 scientific job trajectory count mismatch")
    if expected_count > 1024:
        raise ValueError("v4 scientific job exceeds one family split")
    concurrency = int(job["concurrency"])
    if concurrency <= 0 or concurrency != int(manifest.get("request_concurrency", concurrency)):
        raise ValueError("v4 request concurrency mismatch")
    sampling = manifest.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError("v4 sampling contract is missing")

    journal_lock = threading.Lock()
    trajectories: list[dict[str, Any]] = []
    infrastructure_error = None
    for task in tasks:
        def run_one(seed: int) -> dict[str, Any]:
            return run_trajectory(task, seed, sampling, post_json, journal_path, journal_lock)

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {seed: pool.submit(run_one, seed) for seed in seeds}
            for seed in seeds:
                if infrastructure_error is not None:
                    futures[seed].cancel()
                    continue
                try:
                    row = futures[seed].result()
                    row["split"] = str(job["split"])
                    trajectories.append(row)
                except Exception as exc:
                    infrastructure_error = {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "task_id": task["id"],
                        "rollout_seed": seed,
                    }
                    for future in futures.values():
                        future.cancel()
        if infrastructure_error is not None:
            break

    complete = infrastructure_error is None and len(trajectories) == expected_count
    _write_jsonl(trajectory_path, trajectories)
    result = {
        "protocol": PROTOCOL,
        "job_id": job_id,
        "family": job["family"],
        "split": job["split"],
        "runtime_identity": runtime_identity,
        "collection_manifest_sha256": manifest_sha256,
        "expected_trajectory_count": expected_count,
        "trajectory_count": len(trajectories),
        "request_concurrency": concurrency,
        "retry_count": 0,
        "infrastructure_error": infrastructure_error,
        "complete": complete,
        "outcomes_inspected_by_evidence_chain": False,
        "trajectory_artifact_sha256": _sha256(trajectory_path),
        "raw_journal_sha256": _sha256(journal_path) if journal_path.is_file() else None,
    }
    _write_json(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()
    manifest = _read_json(args.manifest)
    pool_path = Path(manifest["pool_manifest"])
    if _sha256(pool_path) != manifest["pool_manifest_sha256"]:
        raise ValueError("v4 pool manifest hash mismatch")
    result = collect_job(
        manifest=manifest,
        manifest_sha256=_sha256(args.manifest),
        job_id=args.job_id,
        runtime_identity=_read_json(args.runtime_identity),
        pool_manifest=_read_json(pool_path),
        result_path=args.result,
        trajectory_path=args.trajectories,
        journal_path=args.journal,
        post_json=lambda payload: _post_json(args.base_url, args.api_key, args.timeout_seconds, payload),
    )
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

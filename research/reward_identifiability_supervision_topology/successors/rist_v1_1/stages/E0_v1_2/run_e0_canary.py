"""Run the frozen non-scientific RIST-E0 serving canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are executing a non-scientific serving canary. Each user turn states "
    "the exact function and exact code. Return exactly that one tool call and "
    "no plain text."
)


def collect(
    *,
    base_url: str,
    api_key: str,
    model_cell: str,
    tasks_path: Path,
    manifest_path: Path,
    output_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Collect two explicit four-turn canaries without retry or repair."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks_payload = json.loads(tasks_path.read_text(encoding="utf-8"))
    if manifest["protocol"] != "RIST-E0-v1.2":
        raise ValueError("unexpected E0 protocol")
    if _sha256(tasks_path) != manifest["canary_tasks_sha256"]:
        raise ValueError("canary task hash does not match frozen E0 manifest")
    model_cells = {model["cell"] for model in manifest["models"]}
    if model_cell not in model_cells:
        raise ValueError(f"unknown frozen E0 model cell: {model_cell}")
    validate_canary_tasks(tasks_payload)

    trajectories = []
    infrastructure_error = None
    started = time.monotonic()
    for task in tasks_payload["tasks"]:
        try:
            trajectories.append(
                _run_trajectory(
                    base_url=base_url,
                    api_key=api_key,
                    task=task,
                    request_seed=int(manifest["request_seed"]),
                    sampling=manifest["sampling"],
                    timeout_seconds=timeout_seconds,
                )
            )
        except Exception as exc:
            infrastructure_error = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "task_id": task["id"],
            }
            break

    result = {
        "schema_version": 1,
        "protocol": "RIST-E0-v1.2",
        "scientific_interpretation": "FORBIDDEN_INFRASTRUCTURE_ONLY",
        "model_cell": model_cell,
        "canary_tasks_sha256": manifest["canary_tasks_sha256"],
        "expected_trajectories": len(tasks_payload["tasks"]),
        "expected_raw_responses": sum(
            len(task["turns"]) for task in tasks_payload["tasks"]
        ),
        "elapsed_seconds": time.monotonic() - started,
        "infrastructure_error": infrastructure_error,
        "trajectories": trajectories,
        "fabricated_call_count": 0,
        "retry_count": 0,
    }
    _write_json(output_path, result)
    return result


def validate_canary_tasks(payload: dict[str, Any]) -> None:
    """Fail closed if canary structure drifts from two explicit four-turn tasks."""

    if payload.get("protocol") != "RIST-E0-v1.2":
        raise ValueError("unexpected canary task protocol")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 2:
        raise ValueError("E0 requires exactly two canary tasks")
    if {task.get("mode") for task in tasks} != {"forced", "required"}:
        raise ValueError("E0 requires one forced and one required task")
    if len({task.get("id") for task in tasks}) != 2:
        raise ValueError("canary task IDs must be unique")
    for task in tasks:
        turns = task.get("turns")
        if not isinstance(turns, list) or len(turns) != 4:
            raise ValueError("every E0 canary task must have four turns")
        for turn in turns:
            offered = turn.get("offered_tools")
            candidates = turn.get("argument_candidates")
            if turn.get("expected_tool") not in offered:
                raise ValueError("expected tool must be offered")
            if turn.get("expected_code") not in candidates:
                raise ValueError("expected code must be allowed")
            if task["mode"] == "forced" and len(offered) != 1:
                raise ValueError("forced canary turns expose one tool")
            if task["mode"] == "required" and len(offered) != 2:
                raise ValueError("required canary turns expose one distractor")


def parse_tool_call(message: dict[str, Any], offered_tools: set[str]) -> dict[str, Any]:
    """Validate one raw tool call without synthesizing missing content."""

    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return {"valid": False, "reason": "MISSING_TOOL_CALL", "call": None}
    if len(calls) != 1:
        return {"valid": False, "reason": "MULTIPLE_TOOL_CALLS", "call": None}
    raw_call = calls[0]
    function = raw_call.get("function") if isinstance(raw_call, dict) else None
    if not isinstance(function, dict):
        return {"valid": False, "reason": "MALFORMED_TOOL_CALL", "call": None}
    name = function.get("name")
    if name not in offered_tools:
        return {"valid": False, "reason": "UNLISTED_TOOL", "call": None}
    try:
        arguments = json.loads(function.get("arguments"))
    except (TypeError, json.JSONDecodeError):
        return {"valid": False, "reason": "INVALID_JSON_ARGUMENTS", "call": None}
    if not isinstance(arguments, dict) or set(arguments) != {"code"}:
        return {"valid": False, "reason": "INVALID_ARGUMENT_SCHEMA", "call": None}
    if not isinstance(arguments["code"], str):
        return {"valid": False, "reason": "INVALID_CODE_TYPE", "call": None}
    return {
        "valid": True,
        "reason": "VALID",
        "call": {
            "id": raw_call.get("id"),
            "name": name,
            "arguments": arguments,
            "raw_arguments": function.get("arguments"),
        },
    }


def _run_trajectory(
    *,
    base_url: str,
    api_key: str,
    task: dict[str, Any],
    request_seed: int,
    sampling: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["prompt"]},
    ]
    records = []
    raw_responses = []
    for turn_index, turn in enumerate(task["turns"]):
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Canary turn {turn_index + 1}: call function "
                    f"{turn['expected_tool']} with code {turn['expected_code']}."
                ),
            }
        )
        tools = [
            _tool_schema(name, turn["argument_candidates"])
            for name in turn["offered_tools"]
        ]
        tool_choice: Any = "required"
        if task["mode"] == "forced":
            tool_choice = {
                "type": "function",
                "function": {"name": turn["expected_tool"]},
            }
        payload = {
            "model": "policy",
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "stream": False,
            "temperature": float(sampling["temperature"]),
            "top_p": float(sampling["top_p"]),
            "max_tokens": int(sampling["max_new_tokens"]),
            "seed": _request_seed(request_seed, task["id"], turn_index),
        }
        response = _post_json(
            f"{base_url.rstrip('/')}/chat/completions",
            payload,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        raw_responses.append(response)
        message = response["choices"][0]["message"]
        parsed = parse_tool_call(message, set(turn["offered_tools"]))
        exact = False
        call = parsed["call"]
        if parsed["valid"]:
            exact = (
                call["name"] == turn["expected_tool"]
                and call["arguments"] == {"code": turn["expected_code"]}
            )
            messages.append(message)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["name"],
                    "content": json.dumps(
                        {"canary_ack": bool(exact), "turn": turn_index + 1},
                        sort_keys=True,
                    ),
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": "Invalid canary response recorded; no repair is attempted.",
                }
            )
        records.append(
            {
                "turn_index": turn_index,
                "parse_valid": bool(parsed["valid"]),
                "parse_reason": parsed["reason"],
                "exact_instruction": exact,
                "call": call,
            }
        )
    return {
        "task_id": task["id"],
        "mode": task["mode"],
        "records": records,
        "raw_response_count": len(raw_responses),
        "raw_responses": raw_responses,
    }


def _tool_schema(name: str, codes: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Non-scientific structured-output canary function.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "enum": codes}},
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    }


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc


def _request_seed(base_seed: int, task_id: str, turn_index: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{task_id}:{turn_index}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-cell", required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    result = collect(
        base_url=args.base_url,
        api_key=args.api_key,
        model_cell=args.model_cell,
        tasks_path=args.tasks,
        manifest_path=args.manifest,
        output_path=args.output,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "trajectories"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["infrastructure_error"] is None else 2


if __name__ == "__main__":
    raise SystemExit(main())

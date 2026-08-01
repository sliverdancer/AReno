"""Collect frozen RIST P2 trajectories from an OpenAI-compatible server."""

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
    "You are a registry-route agent. On each turn, return exactly one offered "
    "tool call and no plain-text answer. Use only codes available in the task "
    "prompt or revealed by successful tool observations."
)


def collect(
    *,
    base_url: str,
    api_key: str,
    model_cell: str,
    data_dir: Path,
    manifest_path: Path,
    output_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run every qualification task and seed without retry or call repair."""

    protocol = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = [
        json.loads(line)
        for line in (data_dir / "qualification.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    expected_hash = protocol["qualification_sha256"]
    if _sha256(data_dir / "qualification.jsonl") != expected_hash:
        raise ValueError("qualification data hash does not match frozen P2 manifest")
    model_cells = {model["cell"] for model in protocol["models"]}
    if model_cell not in model_cells:
        raise ValueError(f"unknown frozen model cell: {model_cell}")
    trajectories = []
    started = time.monotonic()
    infrastructure_error = None
    for task_index, task in enumerate(tasks):
        for rollout_seed in protocol["rollout_seeds"]:
            try:
                trajectories.append(
                    _run_trajectory(
                        base_url=base_url,
                        api_key=api_key,
                        task=task,
                        task_index=task_index,
                        rollout_seed=int(rollout_seed),
                        sampling=protocol["sampling"],
                        timeout_seconds=timeout_seconds,
                    )
                )
            except Exception as exc:
                infrastructure_error = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "task_index": task_index,
                    "rollout_seed": rollout_seed,
                }
                break
        if infrastructure_error is not None:
            break
    result = {
        "schema_version": 1,
        "protocol": "RIST-P2-v1.0",
        "model_cell": model_cell,
        "qualification_sha256": expected_hash,
        "elapsed_seconds": time.monotonic() - started,
        "expected_trajectories": int(protocol["trajectories_per_model"]),
        "trajectory_count": len(trajectories),
        "infrastructure_error": infrastructure_error,
        "trajectories": trajectories,
    }
    _write_json(output_path, result)
    return result


def parse_tool_call(message: dict[str, Any], offered_tools: set[str]) -> dict[str, Any]:
    """Validate one raw assistant message without synthesizing a call."""

    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        return {"valid": False, "reason": "MISSING_TOOL_CALL", "call": None}
    if len(calls) != 1:
        return {"valid": False, "reason": "MULTIPLE_TOOL_CALLS", "call": None}
    call = calls[0]
    function = call.get("function") if isinstance(call, dict) else None
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
            "id": call.get("id"),
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
    task_index: int,
    rollout_seed: int,
    sampling: dict[str, Any],
    timeout_seconds: float,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task["prompt"]},
    ]
    actions = []
    raw_responses = []
    invalid_reasons = []
    for turn_index, turn in enumerate(task["turns"]):
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Turn {turn_index + 1}: choose exactly one offered tool and "
                    "pass one allowed code."
                ),
            }
        )
        offered_tools = _offered_tool_names(task, turn)
        tools = [_tool_schema(name, turn["argument_candidates"]) for name in offered_tools]
        tool_choice: Any
        if task["factors"]["tool_choice_mode"] == "forced":
            tool_choice = {
                "type": "function",
                "function": {"name": turn["expected_tool"]},
            }
        else:
            tool_choice = "required"
        payload = {
            "model": "policy",
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "stream": False,
            "temperature": float(sampling["temperature"]),
            "top_p": float(sampling["top_p"]),
            "max_tokens": int(sampling["max_new_tokens"]),
            "seed": _request_seed(rollout_seed, task["task_signature"], turn_index),
        }
        response = _post_json(
            f"{base_url.rstrip('/')}/chat/completions",
            payload,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        raw_responses.append(response)
        message = response["choices"][0]["message"]
        parsed = parse_tool_call(message, set(offered_tools))
        if not parsed["valid"]:
            invalid_reasons.append(parsed["reason"])
            actions.append(None)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous response was invalid and is excluded from "
                        "the active tool-call context; it revealed no code."
                    ),
                }
            )
            continue
        call = parsed["call"]
        action = {"name": call["name"], "arguments": call["arguments"]}
        actions.append(action)
        correct = action == task["oracle_actions"][turn_index]
        observation = {
            "ok": correct,
            "next_code": turn["successful_observation"]["next_code"] if correct else None,
            "error": None if correct else "WRONG_TOOL_OR_CODE",
        }
        messages.append(message)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "name": call["name"],
                "content": json.dumps(observation, sort_keys=True),
            }
        )
    strict_success = actions == task["oracle_actions"]
    return {
        "task_id": task["id"],
        "task_signature": task["task_signature"],
        "factor_cell": task["factor_cell"],
        "analytic_stratum": task["reward_resolution_stratum"],
        "rollout_seed": rollout_seed,
        "actions": actions,
        "strict_reward": int(strict_success),
        "first_turn_executable": actions[0] is not None,
        "four_turn_complete": all(action is not None for action in actions),
        "invalid_reasons": invalid_reasons,
        "raw_response_count": len(raw_responses),
        "raw_responses": raw_responses,
        "fabricated_call_count": 0,
    }


def _offered_tool_names(task: dict[str, Any], turn: dict[str, Any]) -> list[str]:
    if task["factors"]["tool_choice_mode"] == "forced":
        return [turn["expected_tool"]]
    return list(turn["offered_tools"])


def _tool_schema(name: str, codes: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Advance the registry route with one allowed code.",
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


def _request_seed(base_seed: int, task_signature: str, turn_index: int) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{task_signature}:{turn_index}".encode()
    ).digest()
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
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()
    result = collect(
        base_url=args.base_url,
        api_key=args.api_key,
        model_cell=args.model_cell,
        data_dir=args.data_dir,
        manifest_path=args.manifest,
        output_path=args.output,
        timeout_seconds=args.timeout_seconds,
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key != "trajectories"
            },
            indent=2,
        )
    )
    return 0 if result["infrastructure_error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(main())

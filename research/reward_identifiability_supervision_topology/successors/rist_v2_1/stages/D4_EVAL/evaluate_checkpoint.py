"""Strictly evaluate one frozen RIST checkpoint through an OpenAI endpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

SYSTEM_PROMPT = (
    "Complete the registry route using exactly one tool call per turn. "
    "Choose only from the offered tools and pass exactly one code argument."
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_split_rows(
    data_dir: Path, split: str, expected: dict[str, Any]
) -> list[dict[str, Any]]:
    split_path = data_dir / f"{split}.jsonl"
    if split_path.is_file():
        encoded = split_path.read_bytes()
    elif split == "confirmatory" and expected.get("materialized") is False:
        builder_path = Path(__file__).with_name("build_eval_splits.py")
        spec = importlib.util.spec_from_file_location(
            "rist_v2_1_d4_in_memory_builder", builder_path
        )
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise RuntimeError("D4 in-memory split builder unavailable")
        spec.loader.exec_module(module)
        rows = module.build_rows("confirmatory")
        encoded = "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in rows
        ).encode("utf-8")
    else:
        raise FileNotFoundError(f"evaluation split unavailable: {split}")
    if hashlib.sha256(encoded).hexdigest() != expected["sha256"]:
        raise ValueError("evaluation split hash mismatch")
    return [json.loads(line) for line in encoded.decode("utf-8").splitlines() if line]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _consume_confirmatory_ledger(path: Path, checkpoint_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "RIST-D4-EVAL-v2.1",
        "split": "confirmatory",
        "checkpoint_id": checkpoint_id,
        "consumed": True,
        "consumed_at_unix_ns": time.time_ns(),
        "result_complete": False,
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, (json.dumps(payload, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_journal(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, (json.dumps(payload, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _render_initial_prompt(task: dict[str, Any]) -> str:
    contract = task["prompt_contract"]
    parts = [str(contract["instruction"])]
    for turn in contract["initial_visible_turns"]:
        parts.append(_turn_instruction(turn))
    return "\n".join(parts)


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


def _request_seed(base_seed: int, task_signature: str, turn_index: int) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{task_signature}:{turn_index}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def run_trajectory(
    task: dict[str, Any],
    rollout_seed: int,
    sampling: dict[str, Any],
    post_json: Callable[[dict[str, Any]], dict[str, Any]],
    journal_path: Path,
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
        _append_journal(
            journal_path,
            {
                "task_id": task["id"],
                "task_signature": task["task_signature"],
                "rollout_seed": rollout_seed,
                "turn_index": turn_index,
                "raw_response": response,
            },
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


def _post_json(
    base_url: str, api_key: str, timeout_seconds: float, payload: dict[str, Any]
) -> dict[str, Any]:
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


def collect(
    *,
    data_dir: Path,
    split: str,
    checkpoint_id: str,
    output_path: Path,
    journal_path: Path,
    post_json: Callable[[dict[str, Any]], dict[str, Any]],
    ledger_path: Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != "RIST-D4-EVAL-v2.1" or split not in manifest["splits"]:
        raise ValueError("unexpected D4 evaluation manifest or split")
    if output_path.exists() or journal_path.exists():
        raise FileExistsError("evaluation output and journal must be fresh paths")
    if split == "confirmatory":
        if ledger_path is None:
            raise ValueError("confirmatory evaluation requires a one-shot ledger")
        _consume_confirmatory_ledger(ledger_path, checkpoint_id)
    elif ledger_path is not None:
        raise ValueError("development evaluation must not use confirmatory ledger")
    expected = manifest["splits"][split]
    tasks = _load_split_rows(data_dir, split, expected)
    if len(tasks) != int(expected["count"]):
        raise ValueError("evaluation task count mismatch")
    trajectories = []
    infrastructure_error = None
    for task in tasks:
        for rollout_seed in expected["rollout_seeds"]:
            try:
                trajectories.append(
                    run_trajectory(
                        task,
                        int(rollout_seed),
                        manifest["sampling"],
                        post_json,
                        journal_path,
                    )
                )
            except Exception as exc:
                infrastructure_error = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "task_id": task["id"],
                    "rollout_seed": int(rollout_seed),
                }
                break
        if infrastructure_error is not None:
            break
    successes = [int(row["strict_success"]) for row in trajectories]
    by_cell = {
        cell: sum(row["strict_success"] for row in trajectories if row["structural_cell"] == cell)
        / sum(row["structural_cell"] == cell for row in trajectories)
        for cell in sorted({row["structural_cell"] for row in trajectories})
    }
    expected_trajectory_count = int(expected["trajectory_count"])
    complete = (
        infrastructure_error is None
        and len(trajectories) == expected_trajectory_count
    )
    result = {
        "protocol": "RIST-D4-EVAL-v2.1",
        "checkpoint_id": checkpoint_id,
        "split": split,
        "split_sha256": expected["sha256"],
        "trajectory_count": len(trajectories),
        "expected_trajectory_count": expected_trajectory_count,
        "complete": complete,
        "infrastructure_error": infrastructure_error,
        "strict_success": None if not successes else sum(successes) / len(successes),
        "strict_success_by_structural_cell": by_cell,
        "retry_count": 0,
        "raw_journal_sha256": _sha256(journal_path) if journal_path.is_file() else None,
        "trajectories": trajectories,
    }
    _write_json(output_path, result)
    if ledger_path is not None:
        ledger = json.loads(ledger_path.read_text())
        ledger.update(
            {
                "result_complete": complete,
                "split_sha256": expected["sha256"],
                "result_sha256": _sha256(output_path),
                "journal_sha256": result["raw_journal_sha256"],
            }
        )
        _write_json(ledger_path, ledger)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("dev_curve", "confirmatory"), required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()
    result = collect(
        data_dir=args.data_dir,
        split=args.split,
        checkpoint_id=args.checkpoint_id,
        output_path=args.output,
        journal_path=args.journal,
        ledger_path=args.ledger,
        post_json=lambda payload: _post_json(
            args.base_url, args.api_key, args.timeout_seconds, payload
        ),
    )
    print(json.dumps({key: value for key, value in result.items() if key != "trajectories"}, indent=2))
    return 0 if result["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

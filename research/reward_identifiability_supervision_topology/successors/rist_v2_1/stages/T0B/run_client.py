"""Collect frozen T0b runtime response-token rows without retry or repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

PROTOCOL = "RIST-T0B-RUNTIME-TOKENS-v1.0"
SYSTEM_PROMPT = (
    "This is a structured tool-call calibration. On every turn, emit exactly "
    "the requested function call with its requested code and no plain text."
)


def collect(
    *,
    base_url: str,
    api_key: str,
    model_cell: str,
    tasks_path: Path,
    manifest_path: Path,
    journal_path: Path,
    result_path: Path,
    timeout_seconds: float,
    post_json: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks_payload = json.loads(tasks_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != PROTOCOL or tasks_payload.get("protocol") != PROTOCOL:
        raise ValueError("unexpected T0b protocol")
    if _sha256(tasks_path) != manifest["tasks_sha256"]:
        raise ValueError("T0b task hash mismatch")
    if model_cell not in manifest["model_order"]:
        raise ValueError("unknown T0b model cell")
    if journal_path.exists() or result_path.exists():
        raise FileExistsError("T0b evidence paths must be fresh")
    tasks = tasks_payload.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 8:
        raise ValueError("T0b requires exactly eight calibration tasks")
    request = post_json or _post_json
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    row_count = 0
    complete_tasks = 0
    failure = None
    for task in tasks:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Calibration nonce: {task['nonce']}.",
            },
        ]
        task_complete = True
        for turn in task["turns"]:
            turn_index = int(turn["turn_index"])
            name = str(turn["expected_tool"])
            code = str(turn["expected_code"])
            messages.append(
                {
                    "role": "user",
                    "content": f"Turn {turn_index + 1}: call {name} with code {code}.",
                }
            )
            payload = {
                "model": "policy",
                "messages": messages,
                "tools": [_tool_schema(name, code)],
                "tool_choice": {"type": "function", "function": {"name": name}},
                "stream": False,
                "temperature": float(manifest["sampling"]["temperature"]),
                "top_p": float(manifest["sampling"]["top_p"]),
                "max_tokens": int(manifest["sampling"]["max_new_tokens"]),
                "seed": _request_seed(
                    int(manifest["request_seed"]), str(task["id"]), turn_index
                ),
            }
            try:
                response = request(
                    f"{base_url.rstrip('/')}/chat/completions",
                    payload,
                    api_key=api_key,
                    timeout_seconds=timeout_seconds,
                )
                _append_row(
                    journal_path,
                    {
                        "task_id": task["id"],
                        "sample_index": int(task["sample_index"]),
                        "turn_index": turn_index,
                        "raw_response": response,
                    },
                )
                row_count += 1
                message = response["choices"][0]["message"]
                call = _exact_call(message, name, code)
                tokens = response.get("areno", {}).get("response_tokens")
                if call is None or not isinstance(tokens, list) or not tokens:
                    raise ValueError("response lacks exact call or actual response_tokens")
                messages.extend(
                    [
                        message,
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": name,
                            "content": json.dumps(
                                {"accepted": True, "next_turn": turn_index + 2},
                                sort_keys=True,
                            ),
                        },
                    ]
                )
            except Exception as exc:
                failure = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "task_id": task["id"],
                    "turn_index": turn_index,
                }
                task_complete = False
                break
        if task_complete:
            complete_tasks += 1
        if failure is not None:
            break
    result = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "model_cell": model_cell,
        "model_revision": manifest["model_revisions"][model_cell],
        "tasks_sha256": manifest["tasks_sha256"],
        "runtime_row_count": row_count,
        "complete_task_count": complete_tasks,
        "expected_runtime_row_count": 32,
        "failure": failure,
        "retry_count": 0,
        "elapsed_seconds": time.monotonic() - started,
        "journal_sha256": _sha256(journal_path) if journal_path.exists() else None,
        "training_performed": False,
        "heldout_opened": False,
        "bfcl_content_opened": False,
    }
    _write_json(result_path, result)
    return result


def _tool_schema(name: str, code: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Emit the requested calibration code.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "enum": [code]}},
                "required": ["code"],
                "additionalProperties": False,
            },
        },
    }


def _exact_call(message: dict[str, Any], name: str, code: str) -> dict[str, Any] | None:
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        return None
    call = calls[0]
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict) or function.get("name") != name:
        return None
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if arguments != {"code": code}:
        return None
    return call


def _append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
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
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-cell", required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()
    result = collect(
        base_url=args.base_url,
        api_key=args.api_key,
        model_cell=args.model_cell,
        tasks_path=args.tasks,
        manifest_path=args.manifest,
        journal_path=args.journal,
        result_path=args.result,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failure"] is None and result["runtime_row_count"] == 32 else 2


if __name__ == "__main__":
    raise SystemExit(main())

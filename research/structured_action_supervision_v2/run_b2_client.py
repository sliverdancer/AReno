"""Run one frozen SAS tool-readiness cell/seed against an AReno endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import b2_inference as core


def _load_game(repo_root: Path) -> Any:
    path = repo_root / "examples" / "agentic" / "shopping" / "game.py"
    spec = importlib.util.spec_from_file_location("sas_b2_game", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load game module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise RuntimeError("endpoint returned a non-object JSON response")
    return decoded, time.monotonic() - started


def _endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def run_trajectory(
    *,
    row: dict[str, Any],
    row_index: int,
    game: Any,
    endpoint: str,
    model: str,
    max_new_tokens: int,
    sampling_seed: int,
    timeout: float,
    deadline_epoch: float,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": core.SYSTEM_PROMPT},
        {"role": "user", "content": game.make_prompt(row)},
    ]
    turns: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    terminal_reason = "COMPLETE"
    for turn_index, expected_name in enumerate(core.EXPECTED_TOOLS):
        if time.time() >= deadline_epoch:
            terminal_reason = "GPU_DEADLINE_EXCEEDED"
            break
        turn_messages = [
            *messages,
            {"role": "user", "content": core.TURN_PROMPTS[expected_name]},
        ]
        tool = core.TOOL_BY_NAME[expected_name]
        tool_choice = {"type": "function", "function": {"name": expected_name}}
        seed = core.request_seed(sampling_seed, row_index, turn_index)
        payload = {
            "model": model,
            "messages": turn_messages,
            "tools": [tool],
            "tool_choice": tool_choice,
            "max_tokens": max_new_tokens,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": seed,
            "stream": False,
        }
        try:
            raw_response, latency = _post_json(endpoint, payload, timeout)
            validation = core.validate_response(raw_response, expected_name)
            turn = {
                "turn_index": turn_index,
                "expected_tool": expected_name,
                "request_seed": seed,
                "request": payload,
                "raw_response": raw_response,
                "latency_s": latency,
                "validation": validation,
            }
        except Exception as exc:
            terminal_reason = "REQUEST_ERROR"
            turns.append(
                {
                    "turn_index": turn_index,
                    "expected_tool": expected_name,
                    "request_seed": seed,
                    "request": payload,
                    "raw_response": None,
                    "latency_s": None,
                    "validation": {
                        "valid": False,
                        "reason": "REQUEST_ERROR",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                }
            )
            break
        turns.append(turn)
        if not validation["valid"]:
            terminal_reason = validation["reason"]
            break
        arguments = validation["arguments"]
        calls.append({"name": expected_name, "arguments": arguments})
        result = core.execute_call(game, expected_name, arguments, row)
        assistant = dict(validation["assistant_message"])
        tool_call_id = assistant["tool_calls"][0]["id"]
        messages = [
            *turn_messages,
            assistant,
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": expected_name,
                "content": json.dumps(result, ensure_ascii=False, sort_keys=True),
            },
        ]

    complete = len(calls) == len(core.EXPECTED_TOOLS)
    reward = core.strict_reward(game, row, calls) if complete else -1.0
    return {
        "row_index": row_index,
        "row_id": row["id"],
        "constraint_signature": row["constraint_signature"],
        "sampling_seed": sampling_seed,
        "first_turn_executable": bool(calls),
        "four_turn_complete": complete,
        "strict_reward": reward,
        "fabricated_call_count": 0,
        "terminal_reason": terminal_reason,
        "calls": calls,
        "turns": turns,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--protocol-id",
        choices=("SAS-TR-v2.0", "SAS-TR-v2.1"),
        default="SAS-TR-v2.0",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell-id", required=True, choices=("D128", "D512", "N128", "N512"))
    parser.add_argument("--split", required=True, choices=("calibration", "validation"))
    parser.add_argument("--max-new-tokens", type=int, required=True, choices=(128, 512))
    parser.add_argument("--sampling-seed", type=int, required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 16:
        raise ValueError(f"B2 split must contain exactly 16 rows, found {len(rows)}")
    game = _load_game(args.repo_root.resolve())
    endpoint = _endpoint(args.base_url)
    started = time.time()
    trajectories: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                run_trajectory,
                row=row,
                row_index=index,
                game=game,
                endpoint=endpoint,
                model=args.model,
                max_new_tokens=args.max_new_tokens,
                sampling_seed=args.sampling_seed,
                timeout=args.request_timeout_s,
                deadline_epoch=args.deadline_epoch,
            )
            for index, row in enumerate(rows)
        ]
        for future in concurrent.futures.as_completed(futures):
            trajectories.append(future.result())
    trajectories.sort(key=lambda item: item["row_index"])
    result = {
        "schema_version": 1,
        "protocol_id": args.protocol_id,
        "cell_id": args.cell_id,
        "split": args.split,
        "max_new_tokens": args.max_new_tokens,
        "sampling_seed": args.sampling_seed,
        "sampling_policy": {"temperature": 1.0, "top_p": 1.0, "top_k": -1},
        "request_seed_derivation": (
            "sha256(SAS-TR-v2.0, base_seed, row_index, turn_index) mod 2^31; "
            "preserved unchanged in SAS-TR-v2.1"
        ),
        "started_epoch": started,
        "finished_epoch": time.time(),
        "summary": core.summarize(trajectories),
        "trajectories": trajectories,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

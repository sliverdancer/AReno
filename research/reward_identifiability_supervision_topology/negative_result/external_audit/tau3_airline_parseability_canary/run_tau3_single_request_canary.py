"""Run or dry-run the Tau3 parseability canary single request.

Default behavior is safe dry-run. A real request requires the explicit
``--execute-one-request`` flag, a bound runtime receipt, OpenAI-compatible API
configuration, and a separate external authorization. This script enforces the
one-request/zero-retry boundary and writes only derived observation/finalizer
artifacts; raw response text should not be committed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
FINALIZER_PATH = HERE / "tau3_canary_finalizer.py"


def _load_finalizer():
    spec = importlib.util.spec_from_file_location("tau3_canary_finalizer_for_runner", FINALIZER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_json(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_receipt(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict):
        raise ValueError("runtime receipt must be a JSON object")
    finalizer = _load_finalizer()
    finalizer.validate_runtime_receipt(receipt)
    return receipt


def build_request_plan(receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": "RRC-TAU3-AIRLINE-PARSEABILITY-CANARY-REQUEST-PLAN-v1",
        "status": "DRY_RUN_NO_MODEL_REQUEST_SENT",
        "task_id": receipt["task_id"],
        "model_repo_or_api_id": receipt["model_repo_or_api_id"],
        "model_revision": receipt.get("model_revision"),
        "model_request_budget": 1,
        "retry_budget": 0,
        "tool_choice": "auto",
        "success_gate": "parseable_tool_call_emission_only",
        "strict_task_success_required": False,
        "reward_resolution_claim_allowed": False,
        "raw_response_commit_allowed": False,
    }


def _extract_tool_calls_from_openai_message(message: Any) -> list[dict[str, Any]]:
    calls = []
    for call in getattr(message, "tool_calls", None) or []:
        function = getattr(call, "function", None)
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)
        if not isinstance(name, str):
            continue
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if isinstance(arguments, dict):
            calls.append({"name": name, "arguments": arguments})
    return calls


def build_observation(
    *,
    receipt: dict[str, Any],
    receipt_sha256: str,
    observed_tool_calls: list[dict[str, Any]],
    raw_response_sha256: str | None,
) -> dict[str, Any]:
    observation = {
        "protocol": "RRC-TAU3-AIRLINE-PARSEABILITY-CANARY-OBSERVATION-v1",
        "runtime_receipt_sha256": receipt_sha256,
        "task_id": receipt["task_id"],
        "model_repo_or_api_id": receipt["model_repo_or_api_id"],
        "model_request_count": 1,
        "retry_count": 0,
        "observed_tool_calls": observed_tool_calls,
    }
    if raw_response_sha256:
        observation["raw_response_sha256"] = raw_response_sha256
    return observation


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_real_single_request(receipt: dict[str, Any], *, base_url: str, api_key: str) -> dict[str, Any]:
    # Import only inside the real execution path so CPU-only tests and dry-runs
    # do not require OpenAI SDK installation.
    from openai import OpenAI  # type: ignore

    client = OpenAI(base_url=base_url, api_key=api_key, max_retries=0)
    messages = [
        {
            "role": "system",
            "content": (
                "You are in a Tau3 airline tool-use canary. Emit at most one tool call. "
                "The goal is only to test parseable tool-call emission."
            ),
        },
        {
            "role": "user",
            "content": (
                "Call an available airline environment tool if appropriate for this public "
                f"canary task id: {receipt['task_id']}."
            ),
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "DB",
                "description": "Execute one read-only airline database lookup for the canary.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
    response = client.chat.completions.create(
        model=receipt["model_repo_or_api_id"],
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.0,
        max_tokens=256,
    )
    message = response.choices[0].message
    raw_summary = {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "finish_reason": response.choices[0].finish_reason,
        "content_present": bool(getattr(message, "content", None)),
        "tool_call_count": len(getattr(message, "tool_calls", None) or []),
    }
    # Hash a bounded representation, not raw text.
    raw_response_sha256 = hashlib.sha256(
        json.dumps(raw_summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "observed_tool_calls": _extract_tool_calls_from_openai_message(message),
        "raw_response_sha256": raw_response_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Write request plan only; default if execute flag is absent.")
    parser.add_argument("--execute-one-request", action="store_true")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    args = parser.parse_args()

    receipt = load_receipt(args.runtime_receipt)
    receipt_sha256 = sha256_json(receipt)
    output_dir = args.output_dir
    if not args.execute_one_request or args.dry_run:
        write_json(output_dir / "TAU3_REQUEST_PLAN_DRY_RUN.json", build_request_plan(receipt))
        return 0
    if not args.base_url or not args.api_key:
        raise SystemExit("real execution requires --base-url/OPENAI_BASE_URL and --api-key/OPENAI_API_KEY")

    result = run_real_single_request(receipt, base_url=args.base_url, api_key=args.api_key)
    observation = build_observation(
        receipt=receipt,
        receipt_sha256=receipt_sha256,
        observed_tool_calls=result["observed_tool_calls"],
        raw_response_sha256=result["raw_response_sha256"],
    )
    write_json(output_dir / "TAU3_CANARY_OBSERVATION.json", observation)
    finalizer = _load_finalizer()
    write_json(output_dir / "TAU3_CANARY_TERMINAL_FINALIZER.json", finalizer.finalize_canary(receipt, observation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

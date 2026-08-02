"""Strict dynamic-tool runner for RIST-v2.1 multi-turn training."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "Complete the registry route using exactly one tool call per turn. "
    "Choose only from the offered tools and pass exactly one code argument."
)


async def run_agent(ctx, batch):
    """Run each four-turn sample with no retry, repair, or forced correct tool."""

    try:
        import httpx
        from openai import AsyncOpenAI
        from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn
    except ImportError as exc:
        raise RuntimeError(
            "RIST-v2.1 rollout requires AReno, openai, httpx, and their runtime dependencies"
        ) from exc

    items = list(batch.iter_samples())
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=max(len(items), ctx.max_running_prompts),
            max_keepalive_connections=max(len(items), ctx.max_running_prompts),
        ),
        timeout=httpx.Timeout(900.0, connect=30.0),
    )
    client = AsyncOpenAI(
        base_url=ctx.get_base_url(),
        api_key=ctx.api_key,
        http_client=http_client,
        max_retries=0,
    )

    async def run_one(item):
        turns = []
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        for turn_index, turn_contract in enumerate(item.record["turns"]):
            tools = [_tool_schema(name) for name in turn_contract["offered_tools"]]
            turn_messages = [
                *messages,
                {"role": "user", "content": _turn_instruction(turn_contract)},
            ]
            request_kwargs = {
                "model": "policy",
                "messages": turn_messages,
                "tools": tools,
                "tool_choice": "required",
                "stream": False,
            }
            request_seed = ctx.request_seed(
                item.prompt_index, item.sample_index, turn_index
            )
            if request_seed is not None:
                request_kwargs["seed"] = request_seed
            response = await client.chat.completions.create(**request_kwargs)
            _append_raw_response(item, turn_index, response)
            turns.append(
                AgentTrajectoryTurn(
                    item=item,
                    messages=turn_messages,
                    response=response,
                    tools=tools,
                    tool_choice="required",
                )
            )
            validation = validate_response(response, turn_contract)
            if not validation["valid"]:
                break
            assistant = validation["assistant_message"]
            next_contract = (
                _visible_turn(item.record["turns"][turn_index + 1])
                if turn_index + 1 < len(item.record["turns"])
                else None
            )
            messages = [
                *turn_messages,
                assistant,
                {
                    "role": "tool",
                    "tool_call_id": assistant["tool_calls"][0]["id"],
                    "name": validation["name"],
                    "content": json.dumps(
                        {
                            "accepted": True,
                            "next_turn": next_contract,
                        },
                        sort_keys=True,
                    ),
                },
            ]
        return turns

    try:
        grouped = await asyncio.gather(*(run_one(item) for item in items))
        return AgentTrajectory(turns=[turn for group in grouped for turn in group])
    finally:
        await client.close()


def validate_response(response: Any, turn_contract: dict[str, Any]) -> dict[str, Any]:
    """Validate one exact raw call against the current contract."""

    assistant = _assistant_message(response)
    calls = assistant.get("tool_calls") or []
    if len(calls) != 1:
        return {"valid": False, "reason": "CALL_COUNT", "assistant_message": assistant}
    function = calls[0].get("function") or {}
    name = function.get("name")
    if name not in turn_contract["offered_tools"]:
        return {"valid": False, "reason": "UNLISTED_TOOL", "assistant_message": assistant}
    if name != turn_contract["expected_tool"]:
        return {"valid": False, "reason": "WRONG_TOOL", "assistant_message": assistant}
    try:
        arguments = json.loads(function.get("arguments"))
    except (TypeError, json.JSONDecodeError):
        return {"valid": False, "reason": "INVALID_JSON", "assistant_message": assistant}
    if not isinstance(arguments, dict) or set(arguments) != {"code"} or not isinstance(arguments["code"], str):
        return {"valid": False, "reason": "INVALID_ARGUMENT_SCHEMA", "assistant_message": assistant}
    target = next(
        candidate["code"]
        for candidate in turn_contract["candidate_records"]
        if candidate["label"] == turn_contract["target_label"]
    )
    if arguments["code"] != target:
        return {"valid": False, "reason": "WRONG_CODE", "assistant_message": assistant}
    return {
        "valid": True,
        "reason": "VALID",
        "assistant_message": assistant,
        "name": name,
        "arguments": arguments,
    }


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


def _turn_instruction(turn: dict[str, Any]) -> str:
    return (
        f"Turn {int(turn['turn_index']) + 1}. Offered tools: "
        + ", ".join(turn["offered_tools"])
        + f". Target label: {turn['target_label']}. Candidates: "
        + ", ".join(
            f"{candidate['label']}:{candidate['code']}"
            for candidate in turn["candidate_records"]
        )
    )


def _visible_turn(turn: dict[str, Any]) -> dict[str, Any]:
    """Return only information the environment may reveal after success."""

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


def _assistant_message(response: Any) -> dict[str, Any]:
    message = response.choices[0].message
    calls = [
        {
            "id": call.id,
            "type": call.type,
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments,
            },
        }
        for call in message.tool_calls or []
    ]
    result = {"role": "assistant", "content": message.content}
    if calls:
        result["tool_calls"] = calls
    return result


def _append_raw_response(item, turn_index: int, response: Any) -> None:
    journal = os.environ.get("RIST_RAW_JOURNAL_PATH")
    if not journal:
        return
    payload = {
        "prompt_index": item.prompt_index,
        "sample_index": item.sample_index,
        "turn_index": turn_index,
        "raw_response": response.model_dump(mode="json"),
    }
    encoded = (json.dumps(payload, sort_keys=True) + "\n").encode()
    descriptor = os.open(Path(journal), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)

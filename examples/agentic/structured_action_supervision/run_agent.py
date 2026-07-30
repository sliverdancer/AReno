"""Strict four-turn shopping runner for Structured Action-Span Supervision."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn

SHOPPING_DIR = Path(__file__).resolve().parents[1] / "shopping"
sys.path.insert(0, str(SHOPPING_DIR))
from game import check_kit, inspect_items, search_catalog_many  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are a shopping kit planner. Follow the requested four tool turns. "
    "Return exactly one tool call on every turn and never answer in plain text."
)
EXPECTED_TOOLS = (
    "search_catalog",
    "inspect_items",
    "check_kit",
    "submit_bundle",
)
TURN_PROMPTS = {
    "search_catalog": "Turn 1: search all required task categories.",
    "inspect_items": "Turn 2: inspect the candidate item ids needed to decide.",
    "check_kit": "Turn 3: check exactly one complete proposed kit.",
    "submit_bundle": "Turn 4: submit exactly one final kit.",
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": "Search compact catalog results for one or more categories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "categories": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["jacket", "shoes", "bottle"],
                        },
                    },
                    "max_price": {"type": "integer"},
                },
                "required": ["categories"],
                "additionalProperties": False,
            },
        },
    },
    *[
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["item_ids"],
                    "additionalProperties": False,
                },
            },
        }
        for name, description in (
            ("inspect_items", "Inspect full details for candidate item ids."),
            ("check_kit", "Check one proposed kit against the task constraints."),
            ("submit_bundle", "Submit the final kit item ids."),
        )
    ],
]
TOOL_BY_NAME = {tool["function"]["name"]: tool for tool in TOOLS}


@dataclass(frozen=True, slots=True)
class CallValidation:
    """Auditable result of validating one raw model response."""

    valid: bool
    reason: str
    assistant_message: dict[str, Any]
    arguments: dict[str, Any] | None = None


async def run_agent(ctx, batch):
    """Run four strict tool turns; stop on the first invalid raw response."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The SAS runner requires `openai` and `httpx`. "
            "Install them with `pip install openai`."
        ) from exc

    items = list(batch.iter_samples())
    max_connections = max(len(items), ctx.max_running_prompts)
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
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
        for turn_index, expected_name in enumerate(EXPECTED_TOOLS):
            turn_messages = [
                *messages,
                {"role": "user", "content": TURN_PROMPTS[expected_name]},
            ]
            tool = TOOL_BY_NAME[expected_name]
            tool_choice = {
                "type": "function",
                "function": {"name": expected_name},
            }
            request_kwargs = {
                "model": "policy",
                "messages": turn_messages,
                "tools": [tool],
                "tool_choice": tool_choice,
                "stream": False,
            }
            request_seed = ctx.request_seed(
                item.prompt_index,
                item.sample_index,
                turn_index,
            )
            if request_seed is not None:
                request_kwargs["seed"] = request_seed
            response = await client.chat.completions.create(
                **request_kwargs,
            )
            turns.append(
                AgentTrajectoryTurn(
                    item=item,
                    messages=turn_messages,
                    response=response,
                    tools=[tool],
                    tool_choice=tool_choice,
                )
            )
            validation = validate_response(response, expected_name)
            if not validation.valid:
                logger.info(
                    "sas_invalid_call prompt_index=%d sample_index=%d turn=%s reason=%s",
                    item.prompt_index,
                    item.sample_index,
                    expected_name,
                    validation.reason,
                )
                break
            result = execute_call(expected_name, validation.arguments or {}, item.record)
            messages = [
                *turn_messages,
                validation.assistant_message,
                {
                    "role": "tool",
                    "tool_call_id": validation.assistant_message["tool_calls"][0]["id"],
                    "name": expected_name,
                    "content": json.dumps(
                        result,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                },
            ]
        return turns

    try:
        grouped = await asyncio.gather(*(run_one(item) for item in items))
        return AgentTrajectory(
            turns=[turn for sample_turns in grouped for turn in sample_turns]
        )
    finally:
        await client.close()


def validate_response(response: Any, expected_name: str) -> CallValidation:
    """Validate exactly one expected call without filtering or synthesizing."""

    assistant_message = assistant_message_from_response(response)
    calls = assistant_message.get("tool_calls") or []
    if not calls:
        return CallValidation(False, "MISSING_TOOL_CALL", assistant_message)
    if len(calls) != 1:
        return CallValidation(False, "MULTIPLE_TOOL_CALLS", assistant_message)
    function = calls[0].get("function") or {}
    if function.get("name") != expected_name:
        return CallValidation(False, "UNEXPECTED_TOOL_NAME", assistant_message)
    raw_arguments = function.get("arguments")
    try:
        arguments = json.loads(raw_arguments)
    except (TypeError, json.JSONDecodeError):
        return CallValidation(False, "INVALID_JSON_ARGUMENTS", assistant_message)
    if not isinstance(arguments, dict):
        return CallValidation(False, "ARGUMENTS_NOT_OBJECT", assistant_message)
    if not _valid_argument_schema(expected_name, arguments):
        return CallValidation(False, "INVALID_ARGUMENT_SCHEMA", assistant_message)
    return CallValidation(True, "VALID", assistant_message, arguments)


def assistant_message_from_response(response: Any) -> dict[str, Any]:
    """Preserve raw assistant content and every returned tool call."""

    message = response.choices[0].message
    calls = []
    for call in message.tool_calls or []:
        calls.append(
            {
                "id": call.id,
                "type": call.type,
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
        )
    result: dict[str, Any] = {
        "role": "assistant",
        "content": message.content,
    }
    if calls:
        result["tool_calls"] = calls
    return result


def execute_call(name: str, arguments: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Execute one already-validated call."""

    if name == "search_catalog":
        return {
            "results_by_category": search_catalog_many(
                arguments["categories"],
                max_price=arguments.get("max_price"),
            )
        }
    if name == "inspect_items":
        return {"items": inspect_items(arguments["item_ids"])}
    if name == "check_kit":
        return {"kit": check_kit(record, arguments["item_ids"])}
    if name == "submit_bundle":
        return {"submitted": arguments["item_ids"]}
    raise ValueError(f"unsupported SAS tool: {name}")


def _valid_argument_schema(name: str, arguments: dict[str, Any]) -> bool:
    if name == "search_catalog":
        if set(arguments) - {"categories", "max_price"}:
            return False
        categories = arguments.get("categories")
        if not _unique_nonempty_strings(categories):
            return False
        if not set(categories).issubset({"jacket", "shoes", "bottle"}):
            return False
        max_price = arguments.get("max_price")
        return max_price is None or (
            isinstance(max_price, int)
            and not isinstance(max_price, bool)
            and max_price >= 0
        )
    if name in {"inspect_items", "check_kit", "submit_bundle"}:
        return set(arguments) == {"item_ids"} and _unique_nonempty_strings(
            arguments.get("item_ids")
        )
    return False


def _unique_nonempty_strings(values: Any) -> bool:
    return (
        isinstance(values, list)
        and bool(values)
        and all(isinstance(value, str) and bool(value.strip()) for value in values)
        and len(values) == len(set(values))
    )

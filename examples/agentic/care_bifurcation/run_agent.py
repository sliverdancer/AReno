"""Five-call agent loop for the controlled CARe bifurcation pilot."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from areno.api.agentic import AgentTrajectory, AgentTrajectoryTurn


logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

SYSTEM_PROMPT = (
    "You are solving a four-turn weighted binary decision task. "
    "At each turn call choose_bit exactly once with bit 0 or 1. "
    "Do not skip, repeat, or combine turns."
)
CHOOSE_BIT_TOOL = {
    "type": "function",
    "function": {
        "name": "choose_bit",
        "description": "Choose the binary action for the current turn.",
        "parameters": {
            "type": "object",
            "properties": {
                "bit": {
                    "type": "integer",
                    "enum": [0, 1],
                }
            },
            "required": ["bit"],
            "additionalProperties": False,
        },
    },
}


async def run_agent(ctx, batch):
    """Run four tool decisions followed by one plain-text summary."""

    try:
        import httpx
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The CARe bifurcation pilot requires openai and httpx."
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
        weights = tuple(int(value) for value in item.record["weights"])
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        valid_actions: list[int] = []
        turns = []
        tool_choice = {"type": "function", "function": {"name": "choose_bit"}}
        for turn_index in range(len(weights)):
            turn_messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        f"Turn {turn_index + 1}/4. Current valid actions: "
                        f"{valid_actions}. Call choose_bit once."
                    ),
                },
            ]
            response = await client.chat.completions.create(
                model="policy",
                messages=turn_messages,
                tools=[CHOOSE_BIT_TOOL],
                tool_choice=tool_choice,
                seed=ctx.request_seed(
                    item.prompt_index,
                    item.sample_index,
                    turn_index,
                ),
                stream=False,
            )
            turns.append(
                AgentTrajectoryTurn(
                    item=item,
                    messages=turn_messages,
                    response=response,
                    tools=[CHOOSE_BIT_TOOL],
                    tool_choice=tool_choice,
                )
            )
            assistant_message = _assistant_message(response)
            tool_results, accepted = _tool_result_messages(
                assistant_message,
                weights=weights,
                turn_index=turn_index,
                valid_actions=valid_actions,
            )
            if accepted is not None:
                valid_actions.append(accepted)
            messages = [*turn_messages, assistant_message, *tool_results]

        final_messages = [
            *messages,
            {
                "role": "user",
                "content": "State the four bits you chose. Do not call a tool.",
            },
        ]
        final_response = await client.chat.completions.create(
            model="policy",
            messages=final_messages,
            seed=ctx.request_seed(
                item.prompt_index,
                item.sample_index,
                "final",
            ),
            stream=False,
        )
        turns.append(
            AgentTrajectoryTurn(
                item=item,
                messages=final_messages,
                response=final_response,
            )
        )
        return turns

    logger.info("CARe bifurcation rollout start trajectories=%d", len(items))
    try:
        grouped = await asyncio.gather(*(run_one(item) for item in items))
        return AgentTrajectory(turns=[turn for group in grouped for turn in group])
    finally:
        await client.close()


def _assistant_message(response: Any) -> dict[str, Any]:
    """Preserve exact assistant content and structured tool calls."""

    message = response.choices[0].message
    tool_calls = [
        {
            "id": call.id,
            "type": call.type,
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments,
            },
        }
        for call in (message.tool_calls or [])
    ]
    result = {"role": "assistant", "content": message.content}
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def _tool_result_messages(
    assistant_message: dict[str, Any],
    *,
    weights: tuple[int, ...],
    turn_index: int,
    valid_actions: list[int],
) -> tuple[list[dict[str, Any]], int | None]:
    """Execute only one valid call while retaining an error for every call ID."""

    calls = assistant_message.get("tool_calls") or []
    accepted = None
    results = []
    for call_index, call in enumerate(calls):
        function = call.get("function") or {}
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = None
        bit = arguments.get("bit") if isinstance(arguments, dict) else None
        valid = (
            len(calls) == 1
            and call_index == 0
            and function.get("name") == "choose_bit"
            and not isinstance(bit, bool)
            and bit in (0, 1)
        )
        if valid:
            accepted = int(bit)
            prospective = [*valid_actions, accepted]
            running_score = sum(
                weight if action == 1 else -weight
                for weight, action in zip(
                    weights,
                    prospective,
                    strict=False,
                )
            )
            payload = {
                "ok": True,
                "turn_index": turn_index,
                "bit": accepted,
                "running_score": running_score,
            }
        else:
            payload = {
                "ok": False,
                "turn_index": turn_index,
                "error": "expected exactly one choose_bit call with bit 0 or 1",
            }
        results.append(
            {
                "role": "tool",
                "tool_call_id": str(call.get("id")),
                "name": str(function.get("name")),
                "content": json.dumps(payload, sort_keys=True),
            }
        )
    return results, accepted
